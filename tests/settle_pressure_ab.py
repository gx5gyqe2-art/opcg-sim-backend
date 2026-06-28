"""#4 settle pressure の Elo 検証（dev）: challenger=settle ON（既定 0.15）vs baseline=settle OFF（0.0）。

席別係数（challenger_coeffs/baseline_coeffs）で「唯一の差＝V2_W_SETTLE_THREAT」に限定して測る。
win_rate は challenger（settle ON）視点。>0.5(Elo>0)=settle pressure が Elo を回収、≈0.5=中立、<0.5=有害。

実行: OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/settle_pressure_ab.py --pairs 100 --pimc 1
"""
import argparse
import time

import conftest  # noqa: F401
from arena_parallel import paired_play


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=100)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--pimc", type=int, default=1)
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    res = paired_play(
        args.pairs, seed0=args.seed0, workers=(args.workers or None),
        challenger_pimc=args.pimc, baseline_pimc=args.pimc,
        challenger_coeffs={"V2_W_SETTLE_THREAT": 0.15},   # settle ON（既定）
        baseline_coeffs={"V2_W_SETTLE_THREAT": 0.0},      # settle OFF
    )
    dt = time.time() - t0
    print("\n=== #4 settle pressure A/B（challenger=ON 0.15 / baseline=OFF 0.0） ===")
    print(f"settle ON 勝率 = {res['win_rate']:.3f}  Elo {res['elo']:+.0f}")
    print(f"  {res['pairs']}ペア / {res['games']}局 / {res['workers']}並列 / {dt:.0f}s / 失敗{res['failed_games']}局")
    print("  解釈: >0.5(Elo>0)=Elo回収 / ≈0.5=中立 / <0.5=有害（重み下げ or SPSAで較正）")


if __name__ == "__main__":
    main()
