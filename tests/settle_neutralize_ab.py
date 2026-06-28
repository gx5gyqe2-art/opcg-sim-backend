"""④ settle-PASS 過大検出の Elo 影響 A/B（dev・§H/§I）。

challenger = neutralize ON（settle ループが PASS 整流で生成した勝者を ±W_WIN で信用せず静的評価へ）
baseline   = 現状（生成勝者を ±W_WIN で信用）

席別オーバーライドで「唯一の差＝settle-PASS生成勝者の扱い」に限定。win_rate は challenger(neutralize) 視点。
- >0.5（Elo>0）: 信用は**有害**（過大検出が実害）→ 的を絞った quiescence/discount を実装する価値あり
- ≈0.5        : 幽霊（mis-valuation は root 選択を変えない）→ ④打ち切り
- <0.5        : 生成勝者は**真のリーサルを当てている**ことが多い→現状維持（中立化は詰めろを逃す）

実行: OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/settle_neutralize_ab.py --pairs 80 --pimc 1 --real-decks
"""
import argparse
import time

import conftest  # noqa: F401
from arena_parallel import paired_play
import deckgen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=80)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--pimc", type=int, default=1)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--real-decks", action="store_true")
    args = ap.parse_args()
    rl = None
    if args.real_decks:
        rl = [(lid, lid) for lid in deckgen.VERIFIED_LEADERS.values()]
    t0 = time.time()
    res = paired_play(
        args.pairs, seed0=args.seed0, workers=(args.workers or None),
        challenger_pimc=args.pimc, baseline_pimc=args.pimc,
        challenger_settle_neutralize=True,   # 生成勝者を中立化
        baseline_settle_neutralize=False,    # 現状（信用）
        realistic_leaders=rl,
    )
    dt = time.time() - t0
    deck_tag = "実構築デッキ" if args.real_decks else "合成同色50枚"
    print(f"\n=== ④ settle-PASS neutralize A/B（challenger=中立化 / baseline=現状・{deck_tag}） ===")
    print(f"中立化 勝率 = {res['win_rate']:.3f}  Elo {res['elo']:+.0f}")
    print(f"  {res['pairs']}ペア / {res['games']}局 / {res['workers']}並列 / {dt:.0f}s / 失敗{res['failed_games']}局")
    print("  解釈: >0.5=信用が有害(quiescence実装の価値) / ≈0.5=幽霊(打ち切り) / <0.5=真リーサル検出(現状維持)")


if __name__ == "__main__":
    main()
