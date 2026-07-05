"""make/unmake版MCTS の①パリティ（clone版と同手・同訪問数）②速度③clone削減 を実測（dev・使い捨て可）。

背景: cProfile で自己対戦の79%が `GameManager.clone`（deepcopy）。エンジンは make/unmake（journal巻き戻し）
を持ち製品α-βは既に使用（`test_cpu_make_unmake.py` で clone 同値）。本プローブは同機構をAZ MCTSへ移した
`az_mcts_tree_mu.TreeMCTSMakeUnmake` が clone版 `az_mcts_tree.TreeMCTS` と同結果か＋どれだけ速くcloneを消すか。

実行: OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/mu_mcts_probe.py --sims 30 --positions 6 --bench-games 2
"""
import argparse
import time

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap  # noqa: F401
import numpy as np

from opcg_sim.src.core import cpu_ai
from opcg_sim.src.core.gamestate import GameManager
import rl_encoder as E
import rl_net as RN
from az_policy import PolicyScorer
from opcg_game import OPCGGame
from cpu_selfplay import _load_db
import p3_loop as P
from az_mcts_tree import TreeMCTS
from az_mcts_tree_mu import TreeMCTSMakeUnmake

EV = 2

# ---- clone 呼び出しカウンタ（GameManager.clone をラップ）----
_CLONE = {"n": 0}
_orig_clone = GameManager.clone
def _counting_clone(self):
    _CLONE["n"] += 1
    return _orig_clone(self)
GameManager.clone = _counting_clone


def _mkpos(game, db, vocab, seed, plies):
    """seed から局面を作り、ランダム合法手で plies 手進めた mid-game manager を返す（CPU手番で停止）。"""
    m = game.new_game(db, seed, leaders=None)
    rng = np.random.default_rng(seed)
    for _ in range(plies):
        if game.winner(m) is not None or game.is_terminal(m):
            break
        name = game.current_player(m)
        if name is None:
            break
        legal = game.legal_actions(m)
        if not legal:
            break
        mv = legal[int(rng.integers(len(legal)))]
        try:
            cpu_ai._apply_move_inplace(m, name, mv)
        except Exception:
            break
    return m


def parity(game, db, vocab, vf, pf, sims, positions):
    print(f"=== ①パリティ（clone版 vs mu版・sims={sims}・eps=0・{positions}局面）===")
    ok = 0
    for i in range(positions):
        pos = _mkpos(game, db, vocab, seed=100 + i * 7, plies=12 + i)
        if game.is_terminal(pos) or game.current_player(pos) is None:
            print(f"  pos{i}: 終局/手番無し→skip"); continue
        # 同一の determinize 済み局面を両者へ（determinize_fn=None＝与えた局面をそのまま探索）
        name = game.current_player(pos)
        det = game.determinize(pos, name, np.random.default_rng(9))
        mc = TreeMCTS(game, value_fn=vf, priors_fn=pf, c_puct=1.5, n_sims=sims,
                      determinize_fn=None, rng=np.random.default_rng(0), dirichlet_eps=0.0)
        mm = TreeMCTSMakeUnmake(game, value_fn=vf, priors_fn=pf, c_puct=1.5, n_sims=sims,
                                determinize_fn=None, rng=np.random.default_rng(0), dirichlet_eps=0.0)
        mv_c, N_c, legal_c = mc.run(det)
        mv_m, N_m, legal_m = mm.run(det)
        same_move = (mv_c is not None and mv_m is not None and
                     cpu_ai_key(mv_c) == cpu_ai_key(mv_m))
        maxdiff = (float(np.max(np.abs(N_c - N_m))) if (N_c is not None and N_m is not None
                   and len(N_c) == len(N_m)) else float("nan"))
        good = same_move and (maxdiff == 0.0)
        ok += good
        print(f"  pos{i}: K={len(legal_c)} 手一致={same_move} max|ΔN|={maxdiff:.0f} "
              f"argmaxN c={int(np.argmax(N_c))} m={int(np.argmax(N_m))} {'OK' if good else 'DIFF'}")
    print(f"  → 完全一致 {ok}/{positions}\n")
    return ok


def cpu_ai_key(mv):
    return (mv.get("kind"), mv.get("action_type"), mv.get("card_uuid"), repr(mv.get("payload")))


def bench(game, db, vocab, vf, pf, sims, games, cls, label, max_steps=60):
    """自己対戦を cls（MCTS実装）で回し、壁時計と clone 回数を測る。"""
    _CLONE["n"] = 0
    rng = np.random.default_rng(2024)
    t0 = time.perf_counter()
    total_steps = 0
    for g in range(games):
        m = game.new_game(db, 5000 + g, leaders=None)
        steps = 0
        while game.winner(m) is None and not game.is_terminal(m) and steps < max_steps:
            name = game.current_player(m)
            if name is None:
                break
            mcts = cls(game, value_fn=vf, priors_fn=pf, c_puct=1.5, n_sims=sims,
                       determinize_fn=lambda s, r: game.determinize(s, name, r),
                       rng=rng, dirichlet_eps=0.25)
            move, N, legal = mcts.run(m)
            if move is None or N is None or N.sum() == 0:
                break
            a = int(np.argmax(N))
            try:
                cpu_ai._apply_move_inplace(m, name, legal[a])
            except Exception:
                break
            steps += 1
        total_steps += steps
    dt = time.perf_counter() - t0
    print(f"  {label:10s}: {dt:6.2f}s  {total_steps:4d}steps  {dt/max(total_steps,1)*1000:6.1f}ms/step  "
          f"clone={_CLONE['n']:6d} ({_CLONE['n']/max(total_steps,1):.1f}/step)")
    return dt, total_steps, _CLONE["n"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=30)
    ap.add_argument("--positions", type=int, default=6)
    ap.add_argument("--bench-games", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=60)
    args = ap.parse_args()

    db = _load_db()
    P._DB = db
    vocab = E.build_vocab(db)
    game = OPCGGame()
    feat = E.feature_dim(EV)
    vnet = RN.ValueNet(vocab_size=len(vocab), d_emb=24, hidden=128, feat_dim=feat, seed=0)
    pnet = PolicyScorer(ctx_dim=feat, hidden=128, seed=0)
    vf = P.value_fn_of(vnet, vocab, EV)
    pf = P.priors_fn_of(pnet, vocab, EV)

    parity(game, db, vocab, vf, pf, args.sims, args.positions)

    print(f"=== ②③ 速度・clone数（自己対戦 {args.bench_games}局・sims={args.sims}・max_steps={args.max_steps}）===")
    dc, sc, cc = bench(game, db, vocab, vf, pf, args.sims, args.bench_games, TreeMCTS, "clone版")
    dm, sm, cm = bench(game, db, vocab, vf, pf, args.sims, args.bench_games, TreeMCTSMakeUnmake, "mu版")
    print(f"\n  速度比 clone/mu = {dc/dm:.2f}×高速化   clone削減 {cc}→{cm} "
          f"({(1 - cm/max(cc,1))*100:.1f}%減)")


if __name__ == "__main__":
    main()
