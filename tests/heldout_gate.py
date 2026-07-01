"""汎化ゲート（v4b 実装ステップ5・本番と同じ問い）。

docs/reports/cpu_rl_frozen_design_v4b_20260701.md §ゲート。**生成デッキで訓練した value+MCTS が、
held-out 実デッキで L1 に勝てるか**を測る。合格ライン vs L1 > 0.60・SPRT で早期終了。

- 訓練データ = パラメトリック生成デッキ（`deck_generator`・実リスト不参照）で L1評価ラベル bootstrap。
- 相手 = greedy-L1（研究中の代理・同1-ply）。最終ゲートは製品L1（α-β+PIMC）だが本ハーネスは代理で高速に。
- 決定化 = `_determinize_hidden`（透視禁止・adapter 経由）。
- SPRT: H0=p0(=0.50) vs H1=p1(=0.65)、α=β=0.05。対数尤度比が境界を越えたら早期終了。
"""
import argparse
import math
import random

import numpy as np

import rl_fingerprint as FP
import rl_encoder as E
from rl_encoder_v2 import encode_v2, DIM
from mini_set_trial import MLP
from pre_flight4_outcome import _score_l1, greedy_by
from pre_flight4_mcts import make_value_fn, mcts_move, mask_fps, COLOR
from deck_generator import DeckGenerator, build_instances
import heldout_decks as HD
from cpu_selfplay import _load_db
from opcg_sim.src.core.gamestate import GameManager, Player
from opcg_sim.src.core import cpu_ai
from opcg_sim.src.learned.adapter import OPCGGame


def gen_dataset_parametric(gen, db, vocab, fps, n_games, ply_cap, every, rng):
    """パラメトリック生成デッキ同士のランダムプレイで (encode_v2, L1評価) を収集。"""
    X, Y = [], []
    for _g in range(n_games):
        lid1, d1 = gen.generate(rng); lid2, d2 = gen.generate(rng)
        l1, c1 = build_instances(db, lid1, d1, "p1")
        l2, c2 = build_instances(db, lid2, d2, "p2")
        m = GameManager(Player("p1", c1, l1), Player("p2", c2, l2)); m.start_game()
        ply = 0
        while ply < ply_cap and m.winner is None:
            pa = m.pending_actor_action()
            if pa is None:
                break
            nm = pa[0]; actor = m.p1 if m.p1.name == nm else m.p2
            legal = m.get_legal_actions(actor)
            if not legal:
                break
            if ply % every == 0 and m.turn_count >= 2:
                try:
                    X.append(encode_v2(m, nm, vocab, fps))
                    Y.append(float(cpu_ai.evaluate(m, nm)))
                except Exception:
                    pass
            try:
                cpu_ai._apply_move_inplace(m, nm, legal[rng.randrange(len(legal))])
            except Exception:
                break
            ply += 1
    return np.array(X, np.float32), np.array(Y, np.float32)


class SPRT:
    """H0: p<=p0 vs H1: p>=p1（Bernoulli）。log 尤度比で早期終了。"""
    def __init__(self, p0=0.50, p1=0.65, alpha=0.05, beta=0.05):
        self.p0, self.p1 = p0, p1
        self.lo = math.log(beta / (1 - alpha))
        self.hi = math.log((1 - beta) / alpha)
        self.llr = 0.0; self.n = 0; self.w = 0

    def update(self, win: bool):
        self.n += 1
        p0, p1 = self.p0, self.p1
        if win:
            self.w += 1
            self.llr += math.log(p1 / p0)
        else:
            self.llr += math.log((1 - p1) / (1 - p0))

    def decision(self):
        if self.llr >= self.hi:
            return "PASS"      # H1 採択（>p1 寄り）
        if self.llr <= self.lo:
            return "FAIL"      # H0 採択（<p0 寄り）
        return None            # 継続


def play_vs_l1(value_fn, deck_builder, db, vocab, fps, sims, c_puct, ply_cap, rng, nrng,
               sprt: SPRT, max_games, p1_kind="mcts"):
    """ミラー対局: 両者とも同じ held-out 実デッキを操縦。p2=greedy-L1、p1 は p1_kind:
    "mcts"=生成訓練value+MCTS（本ゲート）／"l1"=greedy-L1（先手ベースラインのコントロール）。
    デッキ強度の交絡を消し「未見の実デッキをどちらが上手く操縦するか」だけを測る。SPRTで早期終了。"""
    game = OPCGGame()
    l1_score = lambda mm, me: _score_l1(mm, me, vocab, fps)
    while sprt.n < max_games:
        l1, c1 = deck_builder("p1")                       # held-out 実デッキ（player側）
        l2, c2 = deck_builder("p2")                       # 同じ held-out 実デッキ（相手側＝ミラー）
        m = GameManager(Player("p1", c1, l1), Player("p2", c2, l2)); m.start_game()
        ply = 0
        while ply < ply_cap and m.winner is None:
            pa = m.pending_actor_action()
            if pa is None:
                break
            nm = pa[0]
            if nm == "p1":
                mv = (mcts_move(game, value_fn, m, "p1", sims, c_puct, nrng)
                      if p1_kind == "mcts" else greedy_by(l1_score, m, "p1", rng))
            else:
                mv = greedy_by(l1_score, m, nm, rng)
            if mv is None:
                break
            try:
                cpu_ai._apply_move_inplace(m, nm, mv)
            except Exception:
                break
            ply += 1
        if m.winner is None:
            continue
        sprt.update(m.winner == "p1")
        if sprt.decision() is not None:
            break
    return sprt


_GEN = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot-games", type=int, default=160)
    ap.add_argument("--sims", type=int, default=40)
    ap.add_argument("--max-games", type=int, default=60)
    ap.add_argument("--ply-cap", type=int, default=550)
    ap.add_argument("--every", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--p0", type=float, default=0.50)
    ap.add_argument("--p1", type=float, default=0.65)
    ap.add_argument("--player", choices=["mcts", "l1"], default="mcts",
                    help="p1: mcts=生成訓練value+MCTS(ゲート) / l1=greedy-L1(先手ベースライン)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    nrng = np.random.default_rng(args.seed)
    db = _load_db()
    vocab = E.build_vocab(db)
    fps = mask_fps(FP.build_fingerprints(db), [COLOR])   # 脱もつれ（raw色除去）
    _GEN["gen"] = DeckGenerator(db, seed=args.seed)

    vf = None
    if args.player == "mcts":
        print(f"boot: パラメトリック生成デッキ {args.boot_games} games（L1評価・脱もつれ表現）...")
        X, Y = gen_dataset_parametric(_GEN["gen"], db, vocab, fps, args.boot_games,
                                      args.ply_cap, args.every, rng)
        net = MLP(DIM, seed=args.seed); net.fit_norm(X, Y)
        net.train(X, Y, epochs=args.epochs, rng=nrng)
        print(f"  states={len(X)}")
        vf = make_value_fn(net, vocab, fps)

    tag = f"生成訓練value+MCTS({args.sims}sims)" if args.player == "mcts" else "greedy-L1(先手ベースライン)"
    print(f"\n=== ゲート[p1={tag}] vs greedy-L1 / held-out実デッキ・ミラー ===")
    print(f"  SPRT: H0 p<={args.p0} vs H1 p>={args.p1} (α=β=0.05, max {args.max_games}戦)\n")
    for did in HD.deck_ids():
        sprt = SPRT(args.p0, args.p1)
        play_vs_l1(vf, lambda owner, _d=did: HD.build(db, _d, owner), db, vocab, fps,
                   args.sims, 1.5, args.ply_cap, rng, nrng, sprt, args.max_games, args.player)
        wr = sprt.w / sprt.n if sprt.n else float("nan")
        dec = sprt.decision() or "INCONCLUSIVE"
        print(f"  {did:26s} 勝率={wr:.3f} (n={sprt.n})  SPRT={dec}")


if __name__ == "__main__":
    main()
