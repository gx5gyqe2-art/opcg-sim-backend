"""pre-flight ④(MCTS版): fingerprint value ＋ MCTS が L1 を超え・黄へ転移するか。

docs/reports/cpu_rl_preflight4_results_20260701.md の続き。1-ply では value の粗さで L1 に勝負に
ならなかった → 探索(MCTS)を戻して「fingerprint value+MCTS が L1 を超え、held-out(黄)へ転移するか」
を直接測る（本走の本命問い）。

構成:
  net0 = encoder_v2 を L1評価で bootstrap（＝value）
  player = 本番 TreeMCTS（葉=tanh(net0.value)・priors=一様・PIMC 決定化）
  判定 = MCTS(player) vs greedy-L1 の勝率を in-dist(非黄)/held-out(黄) で測る
  比較 = 同 net0 の 1-ply 貪欲（pre-flight④）は vs L1 ≈0.19 → MCTS でどこまで上がるか
"""
import argparse
import random

import numpy as np

import rl_fingerprint as FP
import rl_encoder as E
from rl_encoder_v2 import encode_v2, DIM
from probe_generalization import _leaders_split
from mini_set_trial import MLP, gen_dataset, _random_game
from pre_flight4_outcome import _score_l1, greedy_by
from cpu_selfplay import _load_db
from opcg_sim.src.core import cpu_ai
from opcg_sim.src.learned.mcts import TreeMCTS
from opcg_sim.src.learned.adapter import OPCGGame


COLOR = (7, 13)          # fingerprint の色6次元 [7:13]
NON_BEHAVIOR = (0, 21)   # trigger/action[21:50] の前まで（static/色/type/keyword）


def mask_fps(fps, zero_slices):
    """fingerprint の指定スライスをゼロマスク（脱もつれ用）。"""
    out = {}
    for k, v in fps.items():
        v2 = v.copy()
        for a, b in zero_slices:
            v2[a:b] = 0.0
        out[k] = v2
    return out


def make_value_fn(net, vocab, fps):
    def vf(state, to_move):
        if state.winner is not None:
            return 1.0 if state.winner == to_move else -1.0
        return float(np.tanh(net.value(encode_v2(state, to_move, vocab, fps))))
    return vf


def mcts_move(game, value_fn, m, me, sims, c_puct, rng, priors_fn=None):
    mcts = TreeMCTS(game, value_fn=value_fn, priors_fn=priors_fn, c_puct=c_puct, n_sims=sims,
                    determinize_fn=lambda s, r: game.determinize(s, me, r), rng=rng)
    move, _, legal = mcts.run(m)
    return move if move is not None else (legal[0] if legal else None)


def match_mcts_vs_l1(value_fn, leaders, db, vocab, fps, n_games, sims, c_puct, ply_cap, rng, nrng):
    """MCTS(player, p1) vs greedy-L1(p2)。player 側勝率を返す。"""
    game = OPCGGame(fair_determinize=True)
    wins = done = 0
    for _g in range(n_games):
        m = _random_game(leaders, db, rng)
        ply = 0
        while ply < ply_cap and m.winner is None:
            pa = m.pending_actor_action()
            if pa is None:
                break
            nm = pa[0]
            if nm == "p1":
                mv = mcts_move(game, value_fn, m, "p1", sims, c_puct, nrng)
            else:
                mv = greedy_by(lambda mm, me: _score_l1(mm, me, vocab, fps), m, nm, rng)
            if mv is None:
                break
            try:
                cpu_ai._apply_move_inplace(m, nm, mv)
            except Exception:
                break
            ply += 1
        if m.winner is not None:
            done += 1
            if m.winner == "p1":
                wins += 1
    return (wins / done if done else float("nan")), done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot-games", type=int, default=140)
    ap.add_argument("--eval-games", type=int, default=16)
    ap.add_argument("--sims", type=int, default=40)
    ap.add_argument("--c-puct", type=float, default=1.5)
    ap.add_argument("--ply-cap", type=int, default=550)
    ap.add_argument("--every", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mask", choices=["none", "color", "behavior"], default="none",
                    help="脱もつれ: color=raw色6次元を除去(①b) / behavior=trigger+actionのみ")
    ap.add_argument("--holdout", choices=["color", "leader"], default="color",
                    help="②被覆: color=黄を全除外(被覆ゼロ) / leader=黄一部だけ除外(被覆あり)")
    ap.add_argument("--holdout-k", type=int, default=8)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    nrng = np.random.default_rng(args.seed)
    db = _load_db()
    vocab = E.build_vocab(db)
    fps = FP.build_fingerprints(db)
    if args.mask == "color":
        fps = mask_fps(fps, [COLOR])
    elif args.mask == "behavior":
        fps = mask_fps(fps, [NON_BEHAVIOR])
    train_leaders, held_leaders = _leaders_split(db, args.holdout, args.holdout_k, rng)
    print(f"DIM={DIM} sims={args.sims} mask={args.mask} holdout={args.holdout} "
          f"train={len(train_leaders)} held-out(黄)={len(held_leaders)}")

    print(f"boot: L1評価ラベル {args.boot_games} games ...")
    Xb, Yb = gen_dataset(train_leaders, db, vocab, fps, args.boot_games, args.ply_cap, args.every, rng)
    net0 = MLP(DIM, seed=args.seed); net0.fit_norm(Xb, Yb)
    net0.train(Xb, Yb, epochs=args.epochs, rng=nrng)
    print(f"  boot states={len(Xb)}")
    vf = make_value_fn(net0, vocab, fps)

    print(f"③ MCTS(net0,{args.sims}sims) vs greedy-L1 各 {args.eval_games} games ...")
    wi, di = match_mcts_vs_l1(vf, train_leaders, db, vocab, fps, args.eval_games, args.sims,
                              args.c_puct, args.ply_cap, rng, nrng)
    wh, dh = match_mcts_vs_l1(vf, held_leaders, db, vocab, fps, args.eval_games, args.sims,
                              args.c_puct, args.ply_cap, rng, nrng)
    print("\n=== 結果: MCTS(net0) vs greedy-L1 の player側勝率 ===")
    print(f"  in-dist(非黄) = {wi:.3f} (n={di})   [参考: 1-ply貪欲 net0 ≈0.19]")
    print(f"  held-out(黄)  = {wh:.3f} (n={dh})   [参考: 1-ply貪欲 net0 ≈0.23]")
    print("\n判定:")
    if wi == wi and wi >= 0.45 and wh == wh and wh >= 0.42 and abs(wi - wh) < 0.18:
        print("  MCTSがvalueの粗さを洗濯しL1と互角以上・held-out追随＝fingerprint+MCTSは転移・本走GO寄り")
    elif wi == wi and wi > 0.30:
        print("  MCTSで1-plyより明確改善（探索の効き確認）。sims/世代/policy priorでさらに上げ余地")
    else:
        print("  改善弱い。sims↑ or 教師をoutcome self-play(MCTS下)へ。要再判定")


if __name__ == "__main__":
    main()
