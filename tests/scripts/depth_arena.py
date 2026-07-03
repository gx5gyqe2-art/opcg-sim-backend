"""探索深さの伸びしろ（L1 の外）を測る A/B ハーネス（dev専用）。

L1 葉評価（v2/J値）は自己対戦SPSAで互角が天井と判明した。**強さの伸びしろは葉でなく探索側**にあるはず＝
本ハーネスは「**より深く読む方（challenger）**」vs「**現行 hard（baseline）**」を**対照ペア（antithetic）＋コア並列**で
戦わせ、深さ1ターンぶんの Elo を測る（過去 horizon 3→4 が +58 Elo だった続き＝4→5 以降の伸びを定量）。

両者とも評価は出荷 J値（eval_v2 OFF）＝**葉は固定し探索深さだけを振る**＝純粋に「深さの価値」を分離する。
challenger は horizon を 1 ターン伸ばし、それを実際に読み切れるよう max_ply と clone 予算も比例して引き上げる
（予算が律速だと horizon を上げても葉に届かず無意味になるため）。latency も併記＝深さのコストを可視化。

実行例:
    OPCG_LOG_SILENT=1 python tests/depth_arena.py --pairs 20 --horizon 5 --max-ply 65 --budget 450
    OPCG_LOG_SILENT=1 python tests/depth_arena.py --pairs 20 --horizon 5 --max-ply 65 --budget 450 --time
"""
import argparse
import sys
import time

import os as _os, sys as _sys  # noqa: E402  test bootstrap (sys.path + google stub)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap  # noqa: E402,F401
import cpu_arena
from cpu_arena import elo_ci
from arena_parallel import _pair_level_ci
from opcg_sim.src.core import cpu_ai


def run(pairs, seed0, max_steps, horizon, max_ply, budget,
        base_horizon, base_max_ply, base_budget):
    from arena_parallel import paired_play
    chal = (horizon, max_ply)
    base = (base_horizon, base_max_ply) if (base_horizon or base_max_ply) else None
    t0 = time.time()
    res = paired_play(pairs, seed0=seed0, max_steps=max_steps,
                      challenger_search=chal, baseline_search=base,
                      challenger_budget=budget, baseline_budget=base_budget)
    dt = time.time() - t0
    ci = _pair_level_ci(res["pair_scores"])
    bh = base_horizon if base else cpu_ai.HARD_HORIZON
    print(f"\n=== 探索深さ A/B（hard・{pairs}ペア={2*pairs}局・対照ペア・葉=J値固定） ===")
    print(f"challenger horizon={horizon} max_ply={max_ply} budget={budget}"
          f"  vs  baseline horizon={bh} max_ply={base_max_ply or cpu_ai.HARD_MAX_PLY} "
          f"budget={base_budget or cpu_ai.HARD_PER_MOVE_BUDGET}")
    print(f"深い方の勝率 = {ci['win_rate']:.3f}  |  Elo = {ci['elo']:+.0f}")
    print(f"  ペア単位CI（分散低減・正）  : Elo95% [{ci['elo_lo']:+.0f}, {ci['elo_hi']:+.0f}]")
    naive = elo_ci(res["win_rate"] * res["games"], res["games"])
    print(f"  素朴Bernoulli CI（参考・広い）: Elo95% [{naive['elo_lo']:+.0f}, {naive['elo_hi']:+.0f}]")
    print("判定: " + ("深さが有意に強い" if ci["elo_lo"] > 0 else
                     "深さが有意に弱い" if ci["elo_hi"] < 0 else "互角（有意差なし）"))
    print(f"壁時計: {dt:.1f}s / {res['games']}局 / {res['workers']}並列 ({dt/res['games']:.1f}s/局)")
    return ci


def main(argv=None):
    ap = argparse.ArgumentParser(description="探索深さの伸びしろ A/B（深い方 vs 現行 hard・対照ペア）")
    ap.add_argument("--pairs", type=int, default=20, help="対照ペア数（総局数=2×pairs）")
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=cpu_arena.DEFAULT_MAX_STEPS)
    ap.add_argument("--horizon", type=int, default=cpu_ai.HARD_HORIZON + 1, help="challenger の探索ホライズン")
    ap.add_argument("--max-ply", type=int, default=65, help="challenger の総ply上限（horizon比例で）")
    ap.add_argument("--budget", type=int, default=450, help="challenger の深掘り1手あたり clone 予算")
    ap.add_argument("--base-horizon", type=int, default=0, help="baseline ホライズン（0=既定 HARD_HORIZON）")
    ap.add_argument("--base-max-ply", type=int, default=0, help="baseline ply上限（0=既定）")
    ap.add_argument("--base-budget", type=int, default=0, help="baseline 予算（0=既定）")
    args = ap.parse_args(argv)
    run(args.pairs, args.seed0, args.max_steps, args.horizon, args.max_ply, args.budget,
        args.base_horizon or None, args.base_max_ply or None, args.base_budget or None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
