"""効果オラクル・コンパレータ（カード効果検証イテレーション1）。

「期待する動き（tests/expected_effects.json）」とカードテキスト/AST を突き合わせ、
既存の品質ゲート（text_execution_audit / quality_gates）では拾えない**実装の忠実さ**の
乖離候補を静的に検出し、トリアージのカテゴリ別に集計する。

注意（多層防御・重複回避）:
  - パースの正しさ（テキスト→AST）と方向不一致（NO_IMPL/DIRECTION）は既存ゲートが担保（現状クリーン）。
  - 本ツールはそれらが拾わない高シグナルの text↔AST 整合性のみを追加検出する:
      PER_TURN_LIMIT_GAP : 「ターン1回」表記なのに TURN_LIMIT 条件が AST に無い
      UP_TO_GAP          : 「〜までを」表記なのに is_up_to の対象が一つも無い（要レビュー）
      HAS_OTHER          : 未実装句(ActionType.OTHER)が残る能力（現状 0 のはず・回帰監視）
  - 実戦シーケンス起因の乖離は cpu_selfplay.py --oracle が担当（本ツールは単体静的）。

実行:
  OPCG_LOG_SILENT=1 python tests/effect_oracle.py                 # 全カード集計
  OPCG_LOG_SILENT=1 python tests/effect_oracle.py --category PER_TURN_LIMIT_GAP
  OPCG_LOG_SILENT=1 python tests/effect_oracle.py --json /tmp/oracle.json
"""
import argparse
import json
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

import conftest  # noqa: F401  (google スタブ注入 & sys.path 設定)

from opcg_sim.src.models.enums import ConditionType
from opcg_sim.src.utils.loader import CardLoader
from expected_effects import build_manifest

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "opcg_sim", "data")

# 既知の誤検知（text_execution_audit FLAG_MISSING_ACTION）: 「公開」を engine が LOOK_LIFE で実装。
KNOWN_FALSE_POSITIVES = {"OP10-022", "ST13-007", "ST13-010", "ST13-014"}

_TURN_LIMIT_RE = re.compile(r"ターン1回|ターンに1回|1ターンに1回")
_UP_TO_RE = re.compile(r"までを|まで、|枚まで")


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s or "")


def _condition_has_turn_limit(cond) -> bool:
    """条件ツリーに TURN_LIMIT が含まれるか（AND/OR の入れ子も探索）。"""
    if cond is None:
        return False
    if getattr(cond, "type", None) == ConditionType.TURN_LIMIT:
        return True
    if getattr(cond, "type", None) in (ConditionType.AND, ConditionType.OR):
        return any(_condition_has_turn_limit(a) for a in (cond.args or []))
    return False


def detect(db: CardLoader) -> List[Dict[str, Any]]:
    """全効果カードに静的検出を走らせ、findings のリストを返す。"""
    findings: List[Dict[str, Any]] = []
    for cid in sorted(db.raw_db.keys()):
        m = db.get_card(cid)
        if not m or not m.abilities:
            continue
        text = _nfc(m.effect_text or "")
        if not text:
            continue

        # --- PER_TURN_LIMIT_GAP ---
        if _TURN_LIMIT_RE.search(text):
            ast_has = any(_condition_has_turn_limit(ab.condition) for ab in m.abilities)
            if not ast_has:
                findings.append({
                    "card_id": cid, "name": m.name, "category": "PER_TURN_LIMIT_GAP",
                    "detail": "テキストに「ターン1回」だが TURN_LIMIT 条件が AST に無い",
                    "text": text[:90],
                    "repro": f"python tests/expected_effects.py --card {cid}",
                })

        # --- HAS_OTHER（回帰監視・現状0のはず） ---
        from expected_effects import _walk_actions
        from opcg_sim.src.models.enums import ActionType
        if any(a.type == ActionType.OTHER for ab in m.abilities for a in _walk_actions(ab.effect)):
            findings.append({
                "card_id": cid, "name": m.name, "category": "HAS_OTHER",
                "detail": "未実装句(ActionType.OTHER)が残る",
                "text": text[:90], "repro": f"python tests/effect_diagnostics.py --card {cid}",
            })

        # --- UP_TO_GAP（粗い・要レビュー） ---
        if _UP_TO_RE.search(text):
            up_to_any = False
            for ab in m.abilities:
                for a in _walk_actions(ab.effect):
                    tq = getattr(a, "target", None)
                    if tq is not None and getattr(tq, "is_up_to", False):
                        up_to_any = True
                        break
                if up_to_any:
                    break
            if not up_to_any:
                findings.append({
                    "card_id": cid, "name": m.name, "category": "UP_TO_GAP",
                    "detail": "「〜までを」表記だが is_up_to の対象が無い（0枚選択可の取りこぼし疑い・要レビュー）",
                    "text": text[:90], "repro": f"python tests/expected_effects.py --card {cid}",
                })
    return findings


def main(argv=None):
    ap = argparse.ArgumentParser(description="効果オラクル・コンパレータ（静的）")
    ap.add_argument("--category", default=None, help="特定カテゴリのみ詳細表示")
    ap.add_argument("--json", default=None, help="findings を JSON 出力")
    ap.add_argument("--limit", type=int, default=30)
    args = ap.parse_args(argv)

    db = CardLoader(os.path.join(DATA, "opcg_cards.json"))
    db.load()
    for cid in list(db.raw_db.keys()):
        db.get_card(cid)

    findings = detect(db)
    by_cat: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for f in findings:
        by_cat[f["category"]].append(f)

    print("=== 効果オラクル（静的 text↔AST 整合性）===")
    counts = Counter(f["category"] for f in findings)
    for cat in ("HAS_OTHER", "PER_TURN_LIMIT_GAP", "UP_TO_GAP"):
        print(f"  {cat:20s}: {counts.get(cat, 0)}")
    print(f"  既知の誤検知(MISSING_ACTION/公開=LOOK_LIFE): {len(KNOWN_FALSE_POSITIVES)} 枚 {sorted(KNOWN_FALSE_POSITIVES)}")

    show_cats = [args.category] if args.category else ["HAS_OTHER", "PER_TURN_LIMIT_GAP", "UP_TO_GAP"]
    for cat in show_cats:
        items = by_cat.get(cat, [])
        if not items:
            continue
        print(f"\n--- {cat} ({len(items)}) ---")
        for f in items[:args.limit]:
            print(f"  {f['card_id']} {f['name']}: {f['detail']}")
            print(f"      text: {f['text']}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"counts": dict(counts), "findings": findings,
                       "known_false_positives": sorted(KNOWN_FALSE_POSITIVES)},
                      fh, ensure_ascii=False, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
