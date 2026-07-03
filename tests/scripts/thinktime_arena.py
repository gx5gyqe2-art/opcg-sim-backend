"""思考時間（探索予算）を伸ばすと hard(α-β) が強くなるか測る A/B（dev専用）。

本番 hard ＝ **PIMC K=4・予算75/世界・horizon4**（Dockerfile 相当）。これを baseline に、challenger は
**予算を mult 倍**（＝1手の思考時間↑）し、任意で **horizon も +1**（より深く）して対照ペアで Elo を測る。
「思考時間を伸ばす価値」を定量する（伸びが大きければポンダリングで体感を隠しつつ採用、逓減なら据え置き）。

両側とも葉は出荷 J値（eval_v2 OFF）＝**探索量だけを振る**。壁時計も併記＝思考時間コストの目安。

実行例:
    OPCG_LOG_SILENT=1 python tests/thinktime_arena.py --pairs 24 --mult 4 --horizon 4
    OPCG_LOG_SILENT=1 python tests/thinktime_arena.py --pairs 24 --mult 4 --horizon 5
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

PROD_BUDGET = 75   # 本番 hard の予算/世界（OPCG_HARD_PER_MOVE_BUDGET）
PROD_PIMC = 4      # 本番 hard の PIMC 世界数（OPCG_PIMC_WORLDS）


def run(pairs, seed0, max_steps, mult, horizon):
    from arena_parallel import paired_play
    chal_budget = int(PROD_BUDGET * mult)
    chal_search = (horizon, 13 * horizon + 4) if horizon and horizon != 4 else None
    t0 = time.time()
    res = paired_play(pairs, seed0=seed0, max_steps=max_steps,
                      challenger_difficulty="hard", baseline_difficulty="hard",
                      challenger_pimc=PROD_PIMC, baseline_pimc=PROD_PIMC,
                      challenger_budget=chal_budget, baseline_budget=PROD_BUDGET,
                      challenger_search=chal_search, baseline_search=None)
    dt = time.time() - t0
    failed = res.get("failed_games", 0)
    print(f"\n=== 思考時間↑ A/B（hard・{res['pairs']}ペア={res['games']}局・対照ペア・葉=J値固定） ===")
    print(f"challenger: PIMC{PROD_PIMC}・予算{chal_budget}/世界（{mult}x）・horizon{horizon}"
          f"   vs   baseline: 本番hard（PIMC{PROD_PIMC}・予算{PROD_BUDGET}・horizon4）")
    if failed:
        print(f"⚠ 失敗局 {failed}（除外）。例: " + " | ".join(res.get("errors", [])[:2]))
    if not res["pair_scores"]:
        print("有効ペア0。"); return None
    ci = _pair_level_ci(res["pair_scores"])
    print(f"思考時間↑側 勝率 = {ci['win_rate']:.3f}  |  Elo = {ci['elo']:+.0f}")
    print(f"  ペア単位CI（分散低減・正）  : Elo95% [{ci['elo_lo']:+.0f}, {ci['elo_hi']:+.0f}]")
    naive = elo_ci(res["win_rate"] * res["games"], res["games"])
    print(f"  素朴Bernoulli CI（参考・広い）: Elo95% [{naive['elo_lo']:+.0f}, {naive['elo_hi']:+.0f}]")
    print("判定: " + ("思考時間↑が有意に強い" if ci["elo_lo"] > 0 else
                     "思考時間↑が有意に弱い" if ci["elo_hi"] < 0 else "互角（有意差なし）"))
    print(f"壁時計: {dt:.1f}s / {res['games']}局 / {res['workers']}並列 ({dt/res['games']:.1f}s/局)")
    return ci


def main(argv=None):
    ap = argparse.ArgumentParser(description="思考時間（探索予算）を伸ばす価値の A/B（本番hard基準）")
    ap.add_argument("--pairs", type=int, default=24)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=cpu_arena.DEFAULT_MAX_STEPS)
    ap.add_argument("--mult", type=float, default=4.0, help="予算倍率（思考時間の倍率の目安）")
    ap.add_argument("--horizon", type=int, default=4, help="challenger の horizon（4=同深さ・5=深く）")
    args = ap.parse_args(argv)
    run(args.pairs, args.seed0, args.max_steps, args.mult, args.horizon)
    return 0


if __name__ == "__main__":
    sys.exit(main())
