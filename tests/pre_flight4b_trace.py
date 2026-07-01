"""pre-flight ④b: held-out(黄) 0/40 崩壊の機序トレース（バグ vs 分布シフトの切り分け）。

docs/reports/cpu_rl_preflight4_mcts_results_20260701.md の 0/40 を「憶測せず実測で」切り分ける。
3診断:
  A) value キャリブレーション: net0.value が L1評価とどれだけ相関するか・異常値/NaN が無いか・
     life 等スカラーが OOD レンジに出るか（黄 vs 非黄）。
     → 黄で相関が崩壊/値が張り付き/NaN なら「value が OOD で壊れている（分布シフト or バグ）」。
     → 黄でも相関が保たれるなら「value は正気＝0/40 の主因は別（手の粗さ/探索/報酬）」。
  B) 敗戦ゲームの value 軌跡: 黄の負け試合で value が turn0 から -1 か／途中(トリガー等)で崩れるか。
  C) 非対称性コントロール: 黄で greedy-L1(p1) vs greedy-L1(p2)。p1勝率≈0.5 でなければ盤面の
     先手/エンジン由来の交絡（＝学習value のせいでない）。
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
from pre_flight4_mcts import make_value_fn, mcts_move
from cpu_selfplay import _load_db
from opcg_sim.src.core import cpu_ai
from opcg_sim.src.learned.adapter import OPCGGame


def sample_states(leaders, db, vocab, fps, n_games, ply_cap, every, rng):
    """ランダムプレイで (encode_v2, L1評価, me_life, opp_life) を収集。"""
    rows = []
    for _g in range(n_games):
        m = _random_game(leaders, db, rng)
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
                    me = m.p1 if m.p1.name == nm else m.p2
                    opp = m.p2 if m.p1.name == nm else m.p1
                    rows.append((encode_v2(m, nm, vocab, fps),
                                 float(cpu_ai.evaluate(m, nm)),
                                 len(me.life), len(opp.life)))
                except Exception:
                    pass
            try:
                cpu_ai._apply_move_inplace(m, nm, legal[rng.randrange(len(legal))])
            except Exception:
                break
            ply += 1
    return rows


def diag_A(net, rows_tr, rows_h):
    def stats(net, rows, label):
        X = np.array([r[0] for r in rows], np.float32)
        L1 = np.array([r[1] for r in rows], np.float32)
        life = np.array([r[2] for r in rows])
        v = np.array([net.value(x) for x in X])   # 標準化 y 空間
        nan = int(np.isnan(v).sum() + np.isinf(v).sum())
        # net.value(標準化L1) vs 標準化L1 の相関
        L1n = (L1 - net.ymu) / net.ysd
        corr = float(np.corrcoef(v, L1n)[0, 1]) if len(v) > 2 else float("nan")
        print(f"  [{label}] n={len(rows)}  corr(net,L1)={corr:+.3f}  "
              f"net_v: mean={v.mean():+.2f} std={v.std():.2f} min={v.min():+.2f} max={v.max():+.2f}  "
              f"NaN/inf={nan}  life: max={life.max()} (>5={int((life>5).sum())})")
        return corr
    print("=== 診断A: value キャリブレーション（標準化 y 空間） ===")
    c_tr = stats(net, rows_tr, "非黄(train分布)")
    c_h = stats(net, rows_h, "黄(held-out)")
    print(f"  → 相関 非黄={c_tr:+.3f} / 黄={c_h:+.3f}")


def diag_B(net, leaders, db, vocab, fps, n_games, sims, ply_cap, rng, nrng):
    print("\n=== 診断B: 黄・敗戦ゲームの value 軌跡（MCTS手番で記録） ===")
    game = OPCGGame(); vf = make_value_fn(net, vocab, fps)
    shown = 0
    for _g in range(n_games):
        m = _random_game(leaders, db, rng)
        traj = []
        ply = 0
        while ply < ply_cap and m.winner is None:
            pa = m.pending_actor_action()
            if pa is None:
                break
            nm = pa[0]
            if nm == "p1":
                me = m.p1; opp = m.p2
                raw = net.value(encode_v2(m, "p1", vocab, fps))
                traj.append((m.turn_count, float(np.tanh(raw)), len(me.life), len(opp.life)))
                mv = mcts_move(game, vf, m, "p1", sims, 1.5, nrng)
            else:
                mv = greedy_by(lambda mm, me: _score_l1(mm, me, vocab, fps), m, nm, rng)
            if mv is None:
                break
            try:
                cpu_ai._apply_move_inplace(m, nm, mv)
            except Exception:
                break
            ply += 1
        if m.winner == "p2" and shown < 3 and traj:   # p1(学習) の敗戦を表示
            shown += 1
            pts = traj[::max(1, len(traj) // 8)]
            s = "  ".join(f"T{t}:v={v:+.2f}(L{ml}/{ol})" for t, v, ml, ol in pts)
            print(f"  敗戦{shown}: {s}")
    if shown == 0:
        print("  （表示対象の敗戦が無し）")


def diag_C(leaders, db, vocab, fps, n_games, ply_cap, rng):
    print("\n=== 診断C: 黄で greedy-L1(p1) vs greedy-L1(p2) の先手勝率（交絡チェック） ===")
    wins = done = 0
    for _g in range(n_games):
        m = _random_game(leaders, db, rng)
        ply = 0
        while ply < ply_cap and m.winner is None:
            pa = m.pending_actor_action()
            if pa is None:
                break
            nm = pa[0]
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
    wr = wins / done if done else float("nan")
    print(f"  p1(先手)勝率={wr:.3f} (n={done})  → 0.5付近なら交絡なし・0/40は学習valueの責")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot-games", type=int, default=140)
    ap.add_argument("--sample-games", type=int, default=40)
    ap.add_argument("--traj-games", type=int, default=20)
    ap.add_argument("--ctrl-games", type=int, default=20)
    ap.add_argument("--sims", type=int, default=40)
    ap.add_argument("--ply-cap", type=int, default=550)
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

    print(f"boot: L1評価ラベル {args.boot_games} games ...")
    Xb, Yb = gen_dataset(train_leaders, db, vocab, fps, args.boot_games, args.ply_cap, args.every, rng)
    net0 = MLP(DIM, seed=args.seed); net0.fit_norm(Xb, Yb)
    net0.train(Xb, Yb, epochs=args.epochs, rng=nrng)
    print(f"  boot states={len(Xb)}\n")

    rows_tr = sample_states(train_leaders, db, vocab, fps, args.sample_games, args.ply_cap, args.every, rng)
    rows_h = sample_states(held_leaders, db, vocab, fps, args.sample_games, args.ply_cap, args.every, rng)
    diag_A(net0, rows_tr, rows_h)
    diag_B(net0, held_leaders, db, vocab, fps, args.traj_games, args.sims, args.ply_cap, rng, nrng)
    diag_C(held_leaders, db, vocab, fps, args.ctrl_games, args.ply_cap, rng)


if __name__ == "__main__":
    main()
