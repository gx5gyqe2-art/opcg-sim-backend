"""P2/Gen0: 教師ありSL価値netをMCTSの葉に載せ、L1+α-β CPU と対戦（必要条件チェック）。

docs/.../cpu_rl_pilot_plan_20260629.md P2。scale-curve が測れない唯一のもの＝「value予測の優位が
**プレイ強度**に転換されるか」を測る。GATE B のMCTS（葉=L1）の葉価値だけを**学習SL-net**に差し替え、
相手を **L1+α-β(decide_guarded)** にした対戦勝率を見る。
  - 勝率45〜50%+ → value優位のプレイ転換がRL前にほぼ証明 → P3(RL本走)へ。
  - 20%以下のボロ負け → P4(容量ラダー)でSL-netの表現力を先に診る。

注意: 統計的判定の本ゲートは **400戦CRN**（多時間・外部計算資源）。本スクリプトは harness 構築＋
**少数Nの方向性パイロット**（ここで回せる範囲）。N は明示ログし、過大解釈しない。
実行(訓練+保存): OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/p2_gen0.py --train --train-games 160 --net tests/p2_sl_net.npz
実行(対戦):     OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/p2_gen0.py --net tests/p2_sl_net.npz --pairs 12 --sims 200
"""
import argparse
import random
import time

import numpy as np

import conftest  # noqa: F401
from opcg_sim.src.core import cpu_ai
from opcg_game import OPCGGame
from az_mcts_tree import TreeMCTS
import rl_encoder as E
import rl_net as RN
import rl_datagen as G
from cpu_selfplay import _load_db


def train_sl_net(db, vocab, n_games, eps=0.3, d_emb=24, hidden=128, epochs=30, seed0=0, log=print):
    log(f"P2 訓練データ生成: self-play {n_games}局（eps={eps}）...", flush=True)
    t0 = time.perf_counter()
    data = G.generate(db, vocab, n_games, eps, 400, seed0, sample_every=2)
    if data is None:
        raise RuntimeError("訓練データ0")
    n = len(data["value"])
    log(f"  局面 {n}（{time.perf_counter()-t0:.0f}s）。net 学習...", flush=True)
    net = RN.ValueNet(len(vocab), d_emb=d_emb, hidden=hidden, feat_dim=E.feature_dim(), seed=0)
    tm, vm = RN.train(net, data, epochs=epochs, lr=2e-3, batch=256, val_frac=0.05)
    # 参考: held-out 勝者予測 sign-acc（scale-curve と同義の質指標）。
    log(f"  train_mse={tm:.3f} val_mse={vm:.3f}", flush=True)
    return net


def sl_value(game, net, vocab):
    def value(state, to_move):
        if state.winner is not None:
            return 1.0 if state.winner == to_move else -1.0
        enc = E.encode(state, to_move, vocab)
        batch = {k: enc[k][None, ...] for k in ("scalars", "field", "card_idx")}
        return float(net.predict(batch)[0])
    return value


def mcts_sl_agent(game, value_fn, sims, c_puct=1.5):
    def act(state, name, rng):
        mcts = TreeMCTS(game, value_fn=value_fn, c_puct=c_puct, n_sims=sims,
                        determinize_fn=lambda s, r: game.determinize(s, name, r), rng=rng)
        move, _, _ = mcts.run(state)
        if move is None:
            legal = game.legal_actions(state)
            move = legal[0] if legal else None
        return move
    return act


def l1_agent_factory(difficulty="hard", pimc_worlds=4, seed=0):
    """対局ごとに mem/rng を持つ L1+α-β エージェントを作る。"""
    mem = {}
    prng = random.Random(seed)

    def act(state, name, rng):
        actor = state.p1 if state.p1.name == name else state.p2
        return cpu_ai.decide_guarded(state, actor, difficulty, prng, mem, pimc_worlds=pimc_worlds)
    return act


def play_one(game, m, agentA_is_p1, sl_act, l1_factory, rng, max_steps=400):
    l1_act = l1_factory()   # 対局ごとに mem 新規
    steps = 0
    while game.winner(m) is None and not game.is_terminal(m) and steps < max_steps:
        name = game.current_player(m)
        if name is None:
            break
        a_to_move = (name == "p1") == agentA_is_p1
        move = (sl_act if a_to_move else l1_act)(m, name, rng)
        if move is None:
            break
        try:
            cpu_ai._apply_move_inplace(m, name, move)
        except Exception:
            break
        steps += 1
    return game.winner(m)


def match(game, db, sl_act, l1_factory, pairs, seed0=2000, log=print):
    """CRN: 各seedで SL=p1/L1=p2 と SL=p2/L1=p1 の2戦。SL視点 W/D/L。"""
    res = {"sl_win": 0, "draw": 0, "sl_loss": 0}
    for i in range(pairs):
        seed = seed0 + i
        for sl_is_p1 in (True, False):
            m = game.new_game(db, seed)
            rng = np.random.default_rng(seed * 7 + (0 if sl_is_p1 else 1))
            w = play_one(game, m, sl_is_p1, sl_act, l1_factory, rng)
            if w is None:
                res["draw"] += 1
            else:
                sl_won = (w == "p1") == sl_is_p1
                res["sl_win" if sl_won else "sl_loss"] += 1
        log(f"  pair {i+1}/{pairs}: {res}", flush=True)
    res["games"] = pairs * 2
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--train-games", type=int, default=160)
    ap.add_argument("--net", default="tests/p2_sl_net.npz")
    ap.add_argument("--pairs", type=int, default=12)
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--pimc", type=int, default=4, help="L1相手のPIMC世界数（本番=4）")
    ap.add_argument("--c-puct", type=float, default=1.5)
    args = ap.parse_args()

    db = _load_db()
    vocab = E.build_vocab(db)
    game = OPCGGame()

    if args.train:
        net = train_sl_net(db, vocab, args.train_games)
        net.save(args.net)
        print(f"保存: {args.net}")
        return 0

    net = RN.ValueNet.load(args.net)
    print(f"net ロード: {args.net}", flush=True)
    sl_act = mcts_sl_agent(game, sl_value(game, net, vocab), args.sims, args.c_puct)
    l1_factory = lambda: l1_agent_factory("hard", args.pimc)
    print(f"=== P2/Gen0: SL-net+MCTS(sims={args.sims}) vs L1+α-β(pimc={args.pimc}) "
          f"CRN {args.pairs}ペア×2={args.pairs*2}戦（方向性パイロット） ===", flush=True)
    t0 = time.perf_counter()
    r = match(game, db, sl_act, l1_factory, args.pairs)
    wr = (r["sl_win"] + 0.5 * r["draw"]) / r["games"]
    print(f"\nSL勝率={wr:.3f}  {r}  ({time.perf_counter()-t0:.0f}s)")
    print(f"※ N={r['games']} の方向性パイロット（統計的本ゲートは400戦CRN・外部計算資源）。")
    if wr >= 0.45:
        print("→ value優位のプレイ転換が示唆される（P3=RL本走へ進む価値）。")
    elif wr <= 0.20:
        print("→ ボロ負け＝先に P4(容量ラダー)でSL-netの表現力を診る。")
    else:
        print("→ 中間（0.20〜0.45）。Nを増やすか sims/訓練規模を上げて再評価。")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
