"""ピーク自動アラート（v5 §4-4a・忘却の早期検知）: 本走中の checkpoint 評価系列を読み、
mark_gate 改善数と対ベースライン（v4）勝率が**同時に**ピークから後退したら「ピーク通過」を報せる。

背景（v4採用報告 §3-2）: 混合ラベルでも忘却が残り、round 後半でネットが後退する。ピークを一過性で
逃すと運用で損（v4 は手動でピーク round40 を凍結した）。本器は「単一指標のノイズ的上下」ではなく
**2指標の同時後退が patience 回続く**ことを条件にして誤報を抑え、凍結すべき round（＝これまでの
best 複合スコア）を提示する。監視サイクルで各評価後に追記・再実行する運用（`clock_error_by_leader`
等と同じ読み取り専用の診断計器）。

複合スコア = mark_improved（主・整数）→ arena_wr（従・タイブレーク）。best を更新した round が
凍結候補。alert = 最新が best_mark−mark_drop 以下 **かつ** best_wr−wr_drop 以下 の状態が patience 回連続。
"""
import argparse
import json


def detect_peak(records, patience=2, mark_drop=1, wr_drop=0.03):
    """records: [{round, mark_improved, arena_wr}, ...]（round 昇順想定・順不同でも round でソート）。

    返り値 dict: peak_round / peak_mark / peak_wr / alert(bool) / regressing_streak / latest_round。
    best は「mark_improved 優先・同数なら arena_wr」で選ぶ。alert は 2指標の同時後退が patience 回連続。
    """
    recs = sorted(records, key=lambda r: r["round"])
    if not recs:
        return {"peak_round": None, "peak_mark": None, "peak_wr": None,
                "alert": False, "regressing_streak": 0, "latest_round": None}
    best = recs[0]
    best_mark = best["mark_improved"]
    best_wr = best["arena_wr"]
    streak = 0
    for r in recs:
        m, w = r["mark_improved"], r["arena_wr"]
        # best（複合）更新
        if (m > best_mark) or (m == best_mark and w > best_wr):
            best, best_mark, best_wr = r, m, w
        # ピーク由来の running-max（各指標の最大＝後退の基準）
        best_mark = max(best_mark, m)
        best_wr = max(best_wr, w)
        # 同時後退か
        if m <= best_mark - mark_drop and w <= best_wr - wr_drop:
            streak += 1
        else:
            streak = 0
    return {"peak_round": best["round"], "peak_mark": best["mark_improved"],
            "peak_wr": best["arena_wr"], "alert": streak >= patience,
            "regressing_streak": streak, "latest_round": recs[-1]["round"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", required=True,
                    help="JSONL（1行 = {round, mark_improved, arena_wr}）＝評価系列")
    ap.add_argument("--patience", type=int, default=2, help="同時後退が何回連続でアラートするか")
    ap.add_argument("--mark-drop", type=int, default=1, help="mark_improved の後退許容（これ超で後退扱い）")
    ap.add_argument("--wr-drop", type=float, default=0.03, help="arena_wr の後退許容")
    args = ap.parse_args()
    records = [json.loads(l) for l in open(args.evals) if l.strip()]
    res = detect_peak(records, args.patience, args.mark_drop, args.wr_drop)
    print(json.dumps(res, ensure_ascii=False))
    if res["alert"]:
        print(f"⚠️ ピーク通過の疑い: round {res['peak_round']}（mark={res['peak_mark']} "
              f"wr={res['peak_wr']:.3f}）を凍結候補に。最新 round {res['latest_round']} まで"
              f"{res['regressing_streak']}回連続で2指標同時後退。", flush=True)
    else:
        print(f"継続中: 凍結候補 round {res['peak_round']}（後退 streak={res['regressing_streak']}）", flush=True)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
