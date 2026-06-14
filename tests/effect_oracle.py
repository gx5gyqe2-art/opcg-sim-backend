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

from opcg_sim.src.models.enums import ConditionType, ActionType
from opcg_sim.src.models.effect_types import Choice, Sequence, Branch
from opcg_sim.src.utils.loader import CardLoader
from expected_effects import build_manifest, _walk_actions

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "opcg_sim", "data")

_TURN_LIMIT_RE = re.compile(r"ターン1回|ターンに1回|1ターンに1回")
_UP_TO_RE = re.compile(r"までを|まで、|枚まで")
# カード選択ではない「まで」を UP_TO_GAP の母集合から除くための前処理パターン（誤検知除去）。
# (1) 「ドン!!(合計)N枚(ずつ)まで」= DON ランプ／付与／アクティブ化の上限（報告 §2.2 の ~154 枚）。
#     「ドン !!」のように ドン と !! の間に空白が入る表記も許容する。
# (2) 「…(ターン)開始時／終了時まで」= 継続効果の期間表現。いずれもカード選択の is_up_to とは別物。
# 半角/全角数字の両方に対応。これらを取り除いてから _UP_TO_RE を当てる。
_DON_UP_TO_RE = re.compile(r"ドン[ 　]?(?:!!|‼)?(?:合計)?[\d０-９]+枚(?:ずつ)?まで")
# 「(さらに)N枚まで(を)アクティブ/レストで追加」= DON 追加（「ドン!!」トークンを伴わず
# 「さらに…」で受ける2文目も含む）。追加（追加する）は DON 専用の動詞でカード選択ではない。
_DON_ADD_UP_TO_RE = re.compile(r"[\d０-９]+枚まで[をで、　\s]*(?:アクティブ|レスト)で追加")
_DURATION_UP_TO_RE = re.compile(r"[^、。：:／/]*?(?:開始時|終了時)まで")


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s or "")


# 「N枚まで」の任意性（0枚可）を engine が is_up_to 以外の形でモデル化している場合がある。
# UP_TO_GAP は「カード選択の取りこぼし」を狙う検出器なので、以下は取りこぼしではない:
#  - HEAL / DRAW: 「デッキの上からN枚までを、ライフに加える」「カードN枚までを引く」は対象選択を
#    伴わない自動カウントで、engine は固定 N としてモデル化する（隠しゾーン上から自動取得）。
_AUTO_COUNT_ACTIONS = {ActionType.HEAL, ActionType.DRAW}


def _iter_actions_deep(node):
    """_walk_actions に加え、各 GameAction の sub_effect（置換『代わりに〜』等）も辿る。"""
    for a in _walk_actions(node):
        yield a
        sub = getattr(a, "sub_effect", None)
        if sub is not None:
            yield from _iter_actions_deep(sub)


def _has_optout_choice(node) -> bool:
    """空 Sequence を選択肢に持つ Choice（「見ない／しない」= 0枚相当の任意化）があるか。
    life_scry_top（「ライフの上から1枚までを見て…」）は「まで」を opt-out 選択肢で表現する。"""
    if isinstance(node, Choice):
        if any(isinstance(o, Sequence) and not o.actions for o in node.options):
            return True
        return any(_has_optout_choice(o) for o in node.options)
    if isinstance(node, Sequence):
        return any(_has_optout_choice(a) for a in node.actions)
    if isinstance(node, Branch):
        return _has_optout_choice(node.if_true) or _has_optout_choice(node.if_false)
    return False


def _up_to_is_modeled(abilities) -> bool:
    """「N枚まで」の任意性が engine 側で表現されているか（is_up_to / opt-out Choice / 自動カウント）。"""
    for ab in abilities:
        if _has_optout_choice(ab.effect):
            return True
        for a in _iter_actions_deep(ab.effect):
            tq = getattr(a, "target", None)
            if tq is not None and getattr(tq, "is_up_to", False):
                return True
            if a.type in _AUTO_COUNT_ACTIONS:
                return True
    return False


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
        if any(a.type == ActionType.OTHER for ab in m.abilities for a in _walk_actions(ab.effect)):
            findings.append({
                "card_id": cid, "name": m.name, "category": "HAS_OTHER",
                "detail": "未実装句(ActionType.OTHER)が残る",
                "text": text[:90], "repro": f"python tests/effect_diagnostics.py --card {cid}",
            })

        # --- UP_TO_GAP（粗い・要レビュー） ---
        # 「ドン!!N枚まで」「N枚まで…で追加」「…終了時まで」等の非カード選択の「まで」を
        # 除いてから判定し（誤検知除去）、さらに任意性が is_up_to／opt-out Choice／自動カウント
        # （HEAL/DRAW）で表現済みのものは取りこぼしではないため除外する（sub_effect も走査）。
        card_up_to_text = _DON_UP_TO_RE.sub("", text)
        card_up_to_text = _DON_ADD_UP_TO_RE.sub("", card_up_to_text)
        card_up_to_text = _DURATION_UP_TO_RE.sub("", card_up_to_text)
        if _UP_TO_RE.search(card_up_to_text) and not _up_to_is_modeled(m.abilities):
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
            json.dump({"counts": dict(counts), "findings": findings},
                      fh, ensure_ascii=False, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
