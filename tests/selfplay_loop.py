"""self-play ループ（1セット本走）＝凍結設計 v4c/補遺の実装。

docs/reports/cpu_rl_frozen_design_v4c_20260702.md（＋補遺）。計算資源ゼロ(numpy)前提。

- 探索: PCR（各手を一様Bernoulli p_full で振り分け fast=25sims / full=160sims）。
  Dirichlet ノイズは **full 手のルートのみ**。着手は Temperature: **ターン≤T_END は T=1（訪問数比例
  サンプル）、以降 argmax**（fast/full 問わず＝軌跡の多様性確保・mode collapse 回避）。
- 教師: policy = **full 手の生の訪問分布のみ**。value = **全局面同重み**・ターゲット z∈{-1,+1}。
- 世代: gen0 は value を L1評価で warm-start・prior=uniform。以降 self-play → value(z)/policy(訪問)学習。
- 評価: held-out 実デッキ × **席バランス（先手/後手 等数）** × Wilson95%CI。合格 = 平均≥0.60 かつ
  各デッキ CI 下端≥0.40（SPRT の i.i.d. 問題を避け固定N+CIで判定）。checkpoint/rollback。
- 決定化は透視禁止 `_determinize_hidden`（adapter 経由）。デッキは生成器（実リスト不参照）。
"""
import argparse
import math
import os
import pickle
import random

import numpy as np

import rl_fingerprint as FP
import rl_encoder as E
from rl_encoder_v2 import DIM as _DIM_V2  # noqa: F401 (dim 参照の一貫性確認用)
from rl_effective_state import encode_v3, DIM_V3, make_value_fn_for
from az_policy import PolicyScorer, state_context, train_policy
from opcg_action import legal_action_matrix
from mini_set_trial import MLP
from pre_flight4_mcts import mask_fps, COLOR
from pre_flight4_outcome import _score_l1, greedy_by
from policy_bootstrap import make_priors_fn
from deck_generator import DeckGenerator, build_instances
from heldout_gate import gen_dataset_parametric
import heldout_decks as HD
from cpu_selfplay import _load_db
from opcg_sim.src.core.gamestate import GameManager, Player
from opcg_sim.src.core import cpu_ai
from opcg_sim.src.learned.mcts import TreeMCTS
from opcg_sim.src.learned.adapter import OPCGGame

FAST_SIMS, FULL_SIMS = 25, 160
P_FULL = 0.18
T_END = 10           # ターン≤T_END は T=1（比例サンプル）、以降 argmax
DIR_EPS = 0.25


def save_ckpt(path, state):
    """世代チェックポイントを原子的に保存（再起動耐性・scratchpad は再起動を跨いで残る）。"""
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def load_ckpt(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def wilson(w, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 1.0
    p = w / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def _run_mcts(game, value_fn, priors_fn, m, me, sims, dir_eps, nrng):
    mcts = TreeMCTS(game, value_fn=value_fn, priors_fn=priors_fn, c_puct=1.5, n_sims=sims,
                    determinize_fn=lambda s, r: game.determinize(s, me, r), rng=nrng,
                    dirichlet_alpha=0.3, dirichlet_eps=dir_eps)
    return mcts.run(m)


def selfplay_game(game, value_fn, priors_fn, gen, db, vocab, fps, rng, nrng,
                  fast_sims, full_sims):
    """PCR 自己対戦1局。value_samples[(enc_v3, to_move)], policy_samples[(ctx, am, visit)] を返す。"""
    lid1, d1 = gen.generate(rng); lid2, d2 = gen.generate(rng)
    l1, c1 = build_instances(db, lid1, d1, "p1"); l2, c2 = build_instances(db, lid2, d2, "p2")
    m = GameManager(Player("p1", c1, l1), Player("p2", c2, l2)); m.start_game()
    v_samples, p_samples = [], []
    ply = 0
    while ply < 600 and m.winner is None:
        pa = m.pending_actor_action()
        if pa is None:
            break
        nm = pa[0]
        actor = m.p1 if m.p1.name == nm else m.p2
        legal = m.get_legal_actions(actor)
        if not legal:
            break
        if len(legal) == 1:
            mv = legal[0]
        else:
            is_full = rng.random() < P_FULL
            sims = full_sims if is_full else fast_sims
            de = DIR_EPS if is_full else 0.0
            best, N, rlegal = _run_mcts(game, value_fn, priors_fn, m, nm, sims, de, nrng)
            if best is None or N is None:
                mv = legal[0]
            else:
                # 着手選択: 序盤 T=1（訪問数比例）→ 以降 argmax
                turn = getattr(m, "turn_count", 0)
                if turn <= T_END and N.sum() > 0:
                    p = N / N.sum()
                    idx = rng.choices(range(len(rlegal)), weights=p)[0]
                else:
                    idx = int(np.argmax(N))
                mv = rlegal[idx]
                # policy 教師は full 手の生訪問分布のみ
                if is_full and N.sum() > 0:
                    try:
                        ctx = state_context(m, nm, vocab)
                        am = legal_action_matrix(m, rlegal, nm)
                        if am.shape[0] == len(rlegal):
                            p_samples.append((ctx, am, (N / N.sum()).astype(np.float64)))
                    except Exception:
                        pass
        # value 教師は全局面（fast/full 問わず）
        try:
            v_samples.append((encode_v3(m, nm, vocab, fps), nm))
        except Exception:
            pass
        try:
            cpu_ai._apply_move_inplace(m, nm, mv)
        except Exception:
            break
        ply += 1
    w = m.winner
    value_rows = []
    if w is not None:
        for enc, nm in v_samples:
            value_rows.append((enc, 1.0 if w == nm else -1.0))
    return value_rows, p_samples


def train_value(net, rows, epochs, nrng):
    X = np.stack([r[0] for r in rows]); Y = np.array([r[1] for r in rows], np.float32)
    net.fit_norm(X, Y)
    net.train(X, Y, epochs=epochs, rng=nrng)


def pair_gate(learned_move, db, vocab, fps, n_pairs, ply_cap, rng):
    """held-out 実デッキ × 席バランス（先手/後手 等数）。デッキ別 (勝率, CI下端, n) を返す。"""
    l1_score = lambda mm, me: _score_l1(mm, me, vocab, fps)
    out = {}
    for did in HD.deck_ids():
        w = n = 0
        for pair in range(n_pairs):
            for learned_seat in ("p1", "p2"):
                _l1, c1 = HD.build(db, did, "p1"); _l2, c2 = HD.build(db, did, "p2")
                m = GameManager(Player("p1", c1, _l1), Player("p2", c2, _l2)); m.start_game()
                ply = 0
                while ply < ply_cap and m.winner is None:
                    pa = m.pending_actor_action()
                    if pa is None:
                        break
                    nm = pa[0]
                    mv = learned_move(m, nm) if nm == learned_seat else greedy_by(l1_score, m, nm, rng)
                    if mv is None:
                        break
                    try:
                        cpu_ai._apply_move_inplace(m, nm, mv)
                    except Exception:
                        break
                    ply += 1
                if m.winner is not None:
                    n += 1
                    if m.winner == learned_seat:
                        w += 1
        p, lo, hi = wilson(w, n)
        out[did] = (p, lo, n)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=int, default=3)
    ap.add_argument("--sp-games", type=int, default=40)
    ap.add_argument("--boot-games", type=int, default=200)
    ap.add_argument("--gate-pairs", type=int, default=15)
    ap.add_argument("--fast-sims", type=int, default=FAST_SIMS)
    ap.add_argument("--full-sims", type=int, default=FULL_SIMS)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--buffer-gens", type=int, default=2)
    ap.add_argument("--ply-cap", type=int, default=550)
    ap.add_argument("--every", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", type=str,
                    default="/tmp/claude-0/-home-user/3bd3067e-48b2-52b6-99c2-33478ab4ab32/scratchpad/selfplay_ckpt.pkl")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    nrng = np.random.default_rng(args.seed)
    db = _load_db()
    vocab = E.build_vocab(db)
    fps = mask_fps(FP.build_fingerprints(db), [COLOR])
    gen = DeckGenerator(db, seed=args.seed)
    game = OPCGGame(fair_determinize=True)

    ck = load_ckpt(args.ckpt)
    if ck is not None:
        vnet, pnet = ck["vnet"], ck["pnet"]
        v_buf, p_buf = ck["v_buf"], ck["p_buf"]
        start_gen, best_avg, best = ck["next_gen"], ck["best_avg"], ck["best"]
        rng.setstate(ck["py_rng"]); nrng.bit_generator.state = ck["np_rng"]
        print(f"[resume] ckpt から再開: 次 gen={start_gen} best_avg={best_avg:.3f}")
    else:
        # gen0: value を L1評価で warm-start・prior=uniform
        print(f"gen0 warm-start: value on L1評価 {args.boot_games} games ...")
        Xb, Yb = gen_dataset_parametric(gen, db, vocab, fps, args.boot_games, args.ply_cap,
                                        args.every, rng, encode_fn=encode_v3)
        vnet = MLP(DIM_V3, seed=args.seed); vnet.fit_norm(Xb, Yb)
        vnet.train(Xb, Yb, epochs=args.epochs, rng=nrng)
        pnet = None
        v_buf, p_buf = [], []
        start_gen, best_avg, best = 0, -1.0, None
        save_ckpt(args.ckpt, dict(vnet=vnet, pnet=pnet, v_buf=v_buf, p_buf=p_buf,
                                  next_gen=0, best_avg=best_avg, best=best,
                                  py_rng=rng.getstate(), np_rng=nrng.bit_generator.state))

    for g in range(start_gen, args.gens):
        value_fn = make_value_fn_for(vnet, vocab, fps, encode_v3)
        priors_fn = make_priors_fn(pnet, vocab) if pnet is not None else None
        # self-play（PCR）
        vr_gen, pr_gen = [], []
        for _ in range(args.sp_games):
            vr, pr = selfplay_game(game, value_fn, priors_fn, gen, db, vocab, fps, rng, nrng,
                                   args.fast_sims, args.full_sims)
            vr_gen += vr; pr_gen += pr
        v_buf.append(vr_gen); p_buf.append(pr_gen)
        v_buf, p_buf = v_buf[-args.buffer_gens:], p_buf[-args.buffer_gens:]
        vflat = [x for gg in v_buf for x in gg]; pflat = [x for gg in p_buf for x in gg]
        print(f"gen{g}: self-play value samples={len(vflat)} policy samples={len(pflat)}")
        # 学習: value(z 全局面) ＋ policy(full 訪問分布)
        if vflat:
            train_value(vnet, vflat, args.epochs, nrng)
        if pflat:
            pnet = PolicyScorer(seed=args.seed); train_policy(pnet, pflat, epochs=6, seed=args.seed)
        # ゲート（次世代の nets で）
        vf2 = make_value_fn_for(vnet, vocab, fps, encode_v3)
        pf2 = make_priors_fn(pnet, vocab) if pnet is not None else None
        lm = lambda m, me: _run_mcts(game, vf2, pf2, m, me, args.full_sims, 0.0, nrng)[0]
        res = pair_gate(lm, db, vocab, fps, args.gate_pairs, args.ply_cap, rng)
        avg = float(np.mean([p for p, lo, n in res.values()]))
        minlo = min(lo for p, lo, n in res.values())
        print(f"  gate gen{g+1}: " + "  ".join(f"{d}={p:.2f}(lo{lo:.2f})" for d, (p, lo, n) in res.items())
              + f"  | avg={avg:.3f} minCIlo={minlo:.3f}")
        passed = avg >= 0.60 and minlo >= 0.40
        print(f"  → {'PASS' if passed else 'not yet'} (合格: avg≥0.60 かつ 全デッキCI下端≥0.40)")
        if avg > best_avg:
            best_avg, best = avg, g + 1
        # 世代チェックポイント（再起動耐性）: この gen 完了後の状態を保存
        save_ckpt(args.ckpt, dict(vnet=vnet, pnet=pnet, v_buf=v_buf, p_buf=p_buf,
                                  next_gen=g + 1, best_avg=best_avg, best=best,
                                  py_rng=rng.getstate(), np_rng=nrng.bit_generator.state))
    print(f"\n完了: best avg={best_avg:.3f} @ gen{best}")


if __name__ == "__main__":
    main()
