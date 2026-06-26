"""学習価値の**葉ブレンド α** の強さ A/B（NNUE step4・dev専用）。

`value_model.json`（候補モデル）を **α>0（ブレンドON）vs α=0（OFF＝現行 hard）** で対照ペア対戦させ、
Elo を測る。両側 hard・同一 PIMC/予算＝**α だけを振る**＝「学習評価葉が hard を強くするか」を切り分ける。
過去の −70Elo（弱モデル val_acc0.645）を、強データ再学習モデル（val_acc~0.78）が更新できるかの決定打。

実行例:
    OPCG_LOG_SILENT=1 python tests/blend_arena.py --pairs 24 --alphas 0.2 0.4 0.6 --pimc 1 --budget 75
"""
import argparse
import sys

import conftest  # noqa: F401
import cpu_arena
from cpu_arena import elo_ci
from eval_v2_arena import _pair_level_ci


def run(pairs, seed0, max_steps, alphas, pimc, budget):
    from arena_parallel import paired_play
    print(f"=== 葉ブレンド Elo A/B（hard・PIMC{pimc}/予算{budget}・α>0 vs α=0・{pairs}ペア） ===")
    for a in alphas:
        res = paired_play(pairs, seed0=seed0, max_steps=max_steps,
                          challenger_eval_v2=False, baseline_eval_v2=False,
                          challenger_difficulty="hard", baseline_difficulty="hard",
                          challenger_alpha=a, baseline_alpha=0.0,
                          challenger_pimc=pimc, baseline_pimc=pimc,
                          challenger_budget=budget, baseline_budget=budget)
        if not res["pair_scores"]:
            print(f"α={a}: 有効ペア0"); continue
        ci = _pair_level_ci(res["pair_scores"])
        verdict = ("ブレンドが有意に強い" if ci["elo_lo"] > 0 else
                   "ブレンドが有意に弱い" if ci["elo_hi"] < 0 else "互角（有意差なし）")
        fail = f" ⚠失敗{res['failed_games']}" if res.get("failed_games") else ""
        print(f"α={a:.2f}: 勝率 {ci['win_rate']:.3f} | Elo {ci['elo']:+.0f} "
              f"[ペアCI {ci['elo_lo']:+.0f},{ci['elo_hi']:+.0f}] → {verdict}{fail}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="学習価値の葉ブレンド α の Elo A/B（hard）")
    ap.add_argument("--pairs", type=int, default=24)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=cpu_arena.DEFAULT_MAX_STEPS)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.2, 0.4, 0.6])
    ap.add_argument("--pimc", type=int, default=1, help="PIMC世界数（1=高速・4=本番同等）")
    ap.add_argument("--budget", type=int, default=75)
    args = ap.parse_args(argv)
    run(args.pairs, args.seed0, args.max_steps, args.alphas, args.pimc, args.budget)
    return 0


if __name__ == "__main__":
    sys.exit(main())
