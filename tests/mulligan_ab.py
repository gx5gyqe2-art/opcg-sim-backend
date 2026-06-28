"""① マリガン方策の Elo 検証（dev）: challenger=方策ON vs baseline=方策OFF（従来＝汎用eval≒ほぼKEEP）。

席別オーバーライド（challenger_mulligan/baseline_mulligan）で「唯一の差＝マリガン方策の有無」に限定して
測る。win_rate は challenger（方策ON）視点。>0.5(Elo>0)=方策が Elo を回収、≈0.5=中立、<0.5=有害。

CRN（配り乱数固定）＋席交互ペアで分散低減。マリガンは初手を引き直す＝そこから先の rng は分岐するが、
配りレベルの初期分散はペアで相殺される（§3.3）。

実行: OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/mulligan_ab.py --pairs 200 --pimc 1
"""
import argparse
import time

import conftest  # noqa: F401
from arena_parallel import paired_play


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=200)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--pimc", type=int, default=1)
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    res = paired_play(
        args.pairs, seed0=args.seed0, workers=(args.workers or None),
        challenger_pimc=args.pimc, baseline_pimc=args.pimc,
        challenger_mulligan=True,    # 方策 ON
        baseline_mulligan=False,     # 方策 OFF（従来＝汎用eval）
    )
    dt = time.time() - t0
    print("\n=== ① マリガン方策 A/B（challenger=ON / baseline=OFF） ===")
    print(f"方策 ON 勝率 = {res['win_rate']:.3f}  Elo {res['elo']:+.0f}")
    print(f"  {res['pairs']}ペア / {res['games']}局 / {res['workers']}並列 / {dt:.0f}s / 失敗{res['failed_games']}局")
    print("  解釈: >0.5(Elo>0)=Elo回収 / ≈0.5=中立 / <0.5=有害（閾値調整 or 撤回）")


if __name__ == "__main__":
    main()
