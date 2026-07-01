"""pre-flight ④: outcome-teacher viability（教師/報酬の本走前 de-risk）。

docs/reports/cpu_rl_generalization_plan_v2_20260701.md §4④/§7-2。問い:
  「self-play 方策下の対局 outcome を教師にした value は、L1模倣を超え・L1に迫る/超え・黄へ転移するか？」

最小構成:
  net0 = L1評価ラベルで bootstrap（＝L1模倣・上限≒L1）
  net_out = net0(貪欲) 同士の self-play の **対局 outcome(±1)** で学習した value
  判定 = greedy(net) vs **greedy-L1(同じ1-ply)** の勝率を in-dist(非黄)/held-out(黄) で比較
         （同条件の move-picker 対決＝learned value が L1評価より良い手を選べるか）
  期待: net0 ≈ 0.5（模倣）／net_out > 0.5 なら outcome教師が効く。held-out がそれに追随＝転移。

ランダム続行 outcome は不可（pre-flight② で確定）＝必ず貪欲value 方策下で outcome を取る。
"""
import argparse
import random

import numpy as np

import rl_fingerprint as FP
import rl_encoder as E
from rl_encoder_v2 import encode_v2, DIM
from cpu_selfplay import build_deck, _load_db
from probe_generalization import _leaders_split
from mini_set_trial import MLP, gen_dataset, _random_game, MOVE_CAP
from opcg_sim.src.core import cpu_ai


def _score_net(net, m, me, vocab, fps):
    if m.winner is not None:
        return 1e9 if m.winner == me else -1e9
    return net.value(encode_v2(m, me, vocab, fps))


def _score_l1(m, me, vocab, fps):
    if m.winner is not None:
        return 1e9 if m.winner == me else -1e9
    return float(cpu_ai.evaluate(m, me))


def greedy_by(score, m, me, rng):
    """score(clone, me) を最大化する 1-ply 貪欲手。"""
    actor = m.p1 if m.p1.name == me else m.p2
    legal = m.get_legal_actions(actor)
    if not legal:
        return None
    best, bv = None, -1e18
    for mv in legal[:MOVE_CAP]:
        clone = cpu_ai._apply_clone(m, me, mv)
        if clone is None:
            continue
        v = score(clone, me)
        if v > bv:
            bv, best = v, mv
    return best or legal[rng.randrange(len(legal))]


def self_play_outcome(net, leaders, db, vocab, fps, n_games, ply_cap, every, rng):
    """greedy(net) 同士の self-play。(encode_v2, 対局outcome±1) を収集。"""
    X, Y = [], []
    def sc(m, me):
        return _score_net(net, m, me, vocab, fps)
    for _g in range(n_games):
        m = _random_game(leaders, db, rng)
        samples = []
        ply = 0
        while ply < ply_cap and m.winner is None:
            pa = m.pending_actor_action()
            if pa is None:
                break
            nm = pa[0]
            actor = m.p1 if m.p1.name == nm else m.p2
            legal = m.get_legal_actions(actor)
            if not legal:
                break
            if ply % every == 0 and m.turn_count >= 2:
                try:
                    samples.append((encode_v2(m, nm, vocab, fps), nm))
                except Exception:
                    pass
            mv = greedy_by(sc, m, nm, rng)
            try:
                cpu_ai._apply_move_inplace(m, nm, mv)
            except Exception:
                break
            ply += 1
        w = m.winner
        if w is None:
            continue
        for x, nm in samples:
            X.append(x); Y.append(1.0 if w == nm else -1.0)
    return np.array(X, np.float32), np.array(Y, np.float32)


def match(scoreA, scoreB, leaders, db, vocab, fps, n_games, ply_cap, rng):
    """A(p1) vs B(p2) 対局。A の勝率を返す。score* は (m, me)->float。"""
    wins = done = 0
    for _g in range(n_games):
        m = _random_game(leaders, db, rng)
        ply = 0
        while ply < ply_cap and m.winner is None:
            pa = m.pending_actor_action()
            if pa is None:
                break
            nm = pa[0]
            sc = scoreA if nm == "p1" else scoreB
            mv = greedy_by(lambda mm, me: sc(mm, me), m, nm, rng)
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
    ap.add_argument("--boot-games", type=int, default=120)
    ap.add_argument("--sp-games", type=int, default=60)
    ap.add_argument("--eval-games", type=int, default=30)
    ap.add_argument("--ply-cap", type=int, default=400)
    ap.add_argument("--every", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    nrng = np.random.default_rng(args.seed)
    db = _load_db()
    vocab = E.build_vocab(db)
    fps = FP.build_fingerprints(db)
    train_leaders, held_leaders = _leaders_split(db, "color", 0, rng)
    print(f"DIM={DIM} train(非黄)={len(train_leaders)} held-out(黄)={len(held_leaders)}")

    # net0: L1評価 bootstrap（模倣・上限≒L1）
    print(f"boot: L1評価ラベル {args.boot_games} games ...")
    Xb, Yb = gen_dataset(train_leaders, db, vocab, fps, args.boot_games, args.ply_cap, args.every, rng)
    net0 = MLP(DIM, seed=args.seed); net0.fit_norm(Xb, Yb)
    net0.train(Xb, Yb, epochs=args.epochs, rng=nrng)
    print(f"  boot states={len(Xb)}")

    # net_out: net0 自己対戦の outcome で学習
    print(f"self-play(outcome): greedy(net0) 自己対戦 {args.sp_games} games ...")
    Xo, Yo = self_play_outcome(net0, train_leaders, db, vocab, fps, args.sp_games, args.ply_cap, args.every, rng)
    print(f"  outcome states={len(Xo)}  (勝敗ラベル平均={Yo.mean():+.2f})")
    net_out = MLP(DIM, seed=args.seed + 1); net_out.fit_norm(Xo, Yo)
    net_out.train(Xo, Yo, epochs=args.epochs, rng=nrng)

    def sN(net):
        return lambda m, me: _score_net(net, m, me, vocab, fps)
    def sL1():
        return lambda m, me: _score_l1(m, me, vocab, fps)

    print(f"③ vs greedy-L1（同1-ply）各 {args.eval_games} games ...")
    rows = []
    for label, leaders in (("in-dist(非黄)", train_leaders), ("held-out(黄)", held_leaders)):
        w0, d0 = match(sN(net0), sL1(), leaders, db, vocab, fps, args.eval_games, args.ply_cap, rng)
        wo, do = match(sN(net_out), sL1(), leaders, db, vocab, fps, args.eval_games, args.ply_cap, rng)
        rows.append((label, w0, d0, wo, do))

    print("\n=== 結果: greedy(net) vs greedy-L1 の net側勝率 ===")
    for label, w0, d0, wo, do in rows:
        print(f"  {label:14s} net0(模倣)={w0:.3f}(n={d0})   net_out(outcome)={wo:.3f}(n={do})")
    print("\n判定:")
    (_, wi0, _, wio, _), (_, wh0, _, who, _) = rows[0], rows[1]
    if wio > 0.55 and who > 0.52 and (wio - who) < 0.15:
        print("  outcome教師が L1模倣を超え(vs L1>0.5)・held-out にも追随＝viability OK・本走GO")
    elif wio > 0.55 and who <= 0.5:
        print("  in-dist では L1超えだが held-out で崩れる＝転移不足（②被覆/正則化/データ増を本走前に）")
    elif wio <= 0.52:
        print("  outcome教師が L1模倣を超えない＝報酬/探索/スケール不足（sp-games↑ or MCTS化を検討）")
    else:
        print("  信号弱い。sp-games/eval-games を増やして再判定")


if __name__ == "__main__":
    main()
