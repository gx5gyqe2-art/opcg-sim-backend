"""期待挙動マニフェスト生成器（カード効果検証イテレーション1）。

各カード×能力（Ability）について「期待する動き」を、カードテキスト→パース済み Ability から
機械生成し、人間可読な要約と機械可読サマリ（summarize_ability の指紋）を JSON 出力する。
ユーザ要望「カードごとに期待する動きを事前にまとめ、実際の出力と比較する」の“期待側”。

- 期待の出どころ（自動・breadth）: EffectParserV2 のパース結果（= テキストの機械解釈）。
  これは「engine が AST どおりに動くか」を突き合わせる基準になる（実行の忠実さ検証）。
- 人手 ground truth が必要なカードは docs/leader_specs と tests/expected_overrides.json で上書き。

出力: tests/expected_effects.json （--regen で再生成、--card/--set で限定表示）

実行:
  OPCG_LOG_SILENT=1 python tests/expected_effects.py --regen
  OPCG_LOG_SILENT=1 python tests/expected_effects.py --card OP04-002
  OPCG_LOG_SILENT=1 python tests/expected_effects.py --set EB04
"""
import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

import os as _os, sys as _sys  # noqa: E402  test bootstrap (sys.path + google stub)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap  # noqa: E402,F401

from opcg_sim.src.models.effect_types import Branch, Choice, GameAction, Sequence
from opcg_sim.src.models.enums import ActionType
from opcg_sim.src.utils.loader import CardLoader
from golden.summarize import summarize_ability, _enum_name

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "opcg_sim", "data")
OUT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fixtures", "expected_effects.json")
OVERRIDES_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fixtures", "expected_overrides.json")


def _walk_actions(node):
    """効果 AST 中の全 GameAction を順に yield する。"""
    if node is None:
        return
    if isinstance(node, GameAction):
        yield node
    elif isinstance(node, Sequence):
        for a in node.actions:
            yield from _walk_actions(a)
    elif isinstance(node, Branch):
        yield from _walk_actions(node.if_true)
        yield from _walk_actions(node.if_false)
    elif isinstance(node, Choice):
        for o in node.options:
            yield from _walk_actions(o)


def _action_expectation(act: GameAction) -> Dict[str, Any]:
    """1 アクションの「期待する動き」を機械可読 dict にする。"""
    tq = act.target
    side = _enum_name(getattr(tq, "player", None)) if tq is not None else None
    zone = _enum_name(getattr(tq, "zone", None)) if tq is not None and not isinstance(getattr(tq, "zone", None), list) else (
        [_enum_name(z) for z in tq.zone] if tq is not None and isinstance(getattr(tq, "zone", None), list) else None
    )
    return {
        "type": _enum_name(act.type),
        "target_side": side,
        "target_zone": zone,
        "destination": _enum_name(act.destination),
        "value": act.value.base if act.value else 0,
        "value_dynamic": (act.value.dynamic_source if act.value and act.value.dynamic_source else None),
        "duration": act.duration,
        "count": getattr(tq, "count", None) if tq is not None else None,
        "is_up_to": getattr(tq, "is_up_to", None) if tq is not None else None,
        "status": act.status,
    }


def build_manifest(db: CardLoader, card_filter: Optional[str] = None, set_filter: Optional[str] = None) -> Dict[str, Any]:
    """全効果カード×能力の期待マニフェストを構築する。"""
    manifest: Dict[str, Any] = {}
    for cid in sorted(db.raw_db.keys()):
        if card_filter and cid != card_filter:
            continue
        if set_filter and not cid.startswith(set_filter):
            continue
        m = db.get_card(cid)
        if not m or not m.abilities:
            continue
        entries: List[Dict[str, Any]] = []
        for idx, ab in enumerate(m.abilities):
            actions = [_action_expectation(a) for a in _walk_actions(ab.effect)
                       if a and a.type != ActionType.OTHER]
            has_other = any(a.type == ActionType.OTHER for a in _walk_actions(ab.effect))
            entries.append({
                "ability_index": idx,
                "trigger": _enum_name(ab.trigger),
                "summary": summarize_ability(ab),       # 機械可読指紋（golden と同形）
                "expected_actions": actions,            # 期待する動きの簡約リスト
                "has_other": has_other,                 # 未実装句が残るか
            })
        manifest[cid] = {
            "name": m.name,
            "type": m.type.name,
            "effect_text": m.effect_text or "",
            "trigger_text": getattr(m, "trigger_text", "") or "",
            "abilities": entries,
        }
    return manifest


def load_with_overrides(path: str = OUT_PATH) -> Dict[str, Any]:
    """生成済みマニフェストに人手 overrides をマージして読み込む。"""
    with open(path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    if os.path.exists(OVERRIDES_PATH):
        with open(OVERRIDES_PATH, "r", encoding="utf-8") as f:
            overrides = json.load(f)
        for cid, ov in overrides.items():
            manifest.setdefault(cid, {}).update(ov)
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description="期待挙動マニフェスト生成")
    ap.add_argument("--regen", action="store_true", help="全カードのマニフェストを再生成して JSON 保存")
    ap.add_argument("--card", default=None)
    ap.add_argument("--set", default=None)
    args = ap.parse_args(argv)

    db = CardLoader(os.path.join(DATA, "opcg_cards.json"))
    db.load()
    for cid in list(db.raw_db.keys()):
        db.get_card(cid)

    if args.regen:
        manifest = build_manifest(db)
        with open(OUT_PATH, "w", encoding="utf-8") as f:
            # summarize_condition の value 等に enum/オブジェクトが混じりうるため default=str で安全化。
            json.dump(manifest, f, ensure_ascii=False, indent=1, default=str)
        n_ab = sum(len(v["abilities"]) for v in manifest.values())
        print(f"wrote {OUT_PATH}: {len(manifest)} cards, {n_ab} abilities")
        return 0

    # 表示モード（--card / --set）
    manifest = build_manifest(db, card_filter=args.card, set_filter=args.set)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
