"""品質地図: NO_CHANGE / WARN を自動で三分類し、真のバグ候補を炙り出すハーネス。

effect_coverage の実行分類（EXECUTED/INTERACTIVE/NO_CHANGE/WARN）は「動いた/動かない」
までしか分からない。本ツールはその先を埋める:

  NO_CHANGE（450件規模）の三分類:
    - COND_FALSE  : ゲート条件付き能力。汎用盤面で条件未達＝想定どおりの no-op（LEGIT）
    - PASSIVE/RESTRICTION : PASSIVE 静的効果・制限(RULE_PROCESSING/PREVENT_*)＝
                            resolve_ability では盤面が動かない設計上の no-op（LEGIT）
    - NO_TARGET   : 効果が対象ゾーンを要求するが汎用盤面で候補ゼロ＝テスト足場の都合
    - NO_IMPL     : 上記以外。具体的アクション型なのに変化なし＝★エンジン実装漏れの疑い

  WARN（方向不一致, 313件規模）の二分類:
    - MODAL       : Branch/Choice を含む（別パス実行の誤検知が大半）
    - DIRECTION   : 線形単一効果で方向が逆＝★真のバグ候補

使い方:
    OPCG_LOG_SILENT=1 python tests/quality_map.py                # 全体サマリ + 上位
    OPCG_LOG_SILENT=1 python tests/quality_map.py --show NO_IMPL  # 実装漏れ候補一覧
    OPCG_LOG_SILENT=1 python tests/quality_map.py --show DIRECTION # 方向バグ候補一覧
    OPCG_LOG_SILENT=1 python tests/quality_map.py --card OP01-001
"""
import argparse
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional

import conftest  # noqa: F401

from opcg_sim.src.models.enums import ActionType, ConditionType, TriggerType, Zone, Player
from opcg_sim.src.models.effect_types import Branch, Choice, GameAction, Sequence
from opcg_sim.src.utils.loader import CardLoader

import effect_coverage as cov  # 既存の分類機構を再利用

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "opcg_sim", "data")

# 設計上 resolve_ability で盤面が動かない（=NO_CHANGE が正常な）アクション型
_STATIC_OR_RESTRICTION = {
    ActionType.RULE_PROCESSING, ActionType.PREVENT_LEAVE, ActionType.REPLACE_EFFECT,
    ActionType.PREVENT_REST, ActionType.ATTACK_DISABLE, ActionType.RESTRICTION,
    ActionType.PASSIVE_EFFECT, ActionType.VICTORY, ActionType.NEGATE_EFFECT,
    ActionType.GRANT_KEYWORD, ActionType.BUFF,  # PASSIVE/継続系は静的適用で resolve では動かない
}

# 対象ゾーンを要求するアクション（候補ゼロなら NO_TARGET の可能性）
_TARGETED = {
    ActionType.KO, ActionType.REST, ActionType.BOUNCE, ActionType.TRASH,
    ActionType.MOVE_CARD, ActionType.DECK_BOTTOM, ActionType.MOVE, ActionType.ACTIVE,
}


def _abilities_for(master, trig: str):
    return [ab for ab in master.abilities
            if (ab.trigger.name if hasattr(ab.trigger, "name") else str(ab.trigger)) == trig]


def _has_node(node, kinds) -> bool:
    if node is None:
        return False
    if isinstance(node, kinds):
        return True
    if isinstance(node, Sequence):
        return any(_has_node(a, kinds) for a in node.actions)
    if isinstance(node, Branch):
        return _has_node(node.if_true, kinds) or _has_node(node.if_false, kinds)
    if isinstance(node, Choice):
        return any(_has_node(o, kinds) for o in node.options)
    return False


def _has_real_condition(ability) -> bool:
    """ゲート条件（NONE/None 以外）を持つか。AND/OR は中身を見る。"""
    c = getattr(ability, "condition", None)
    if c is None:
        return False
    t = getattr(c, "type", None)
    if t in (None, ConditionType.NONE):
        return False
    return True


def _effect_action_types(ability):
    return [a.type for a in cov._walk(ability.effect) if a and a.type != ActionType.OTHER]


def classify_no_change(master, trig: str) -> str:
    """NO_CHANGE 能力の細分類。trig のいずれかの能力に当てはまる最も軽い理由を返す。"""
    abs_ = _abilities_for(master, trig)
    # PASSIVE/OPPONENT_TURN/YOUR_TURN は静的・継続系で resolve では動かない設計
    if trig in ("PASSIVE", "OPPONENT_TURN", "YOUR_TURN"):
        return "PASSIVE"
    saw_targeted = False
    for ab in abs_:
        types = _effect_action_types(ab)
        if not types:
            continue
        if _has_real_condition(ab):
            return "COND_FALSE"
        if all(t in _STATIC_OR_RESTRICTION for t in types):
            return "RESTRICTION"
        if any(t in _TARGETED for t in types):
            saw_targeted = True
    if saw_targeted:
        return "NO_TARGET"
    return "NO_IMPL"


def classify_warn(master, trig: str) -> str:
    """WARN（方向不一致）能力の細分類。"""
    abs_ = _abilities_for(master, trig)
    for ab in abs_:
        if _has_node(ab.effect, (Branch, Choice)):
            return "MODAL"
    return "DIRECTION"


def run(show: Optional[str] = None, card_filter: Optional[str] = None) -> None:
    db = CardLoader(os.path.join(DATA, "opcg_cards.json"))
    db.load()
    card_ids = sorted(db.raw_db.keys())
    if card_filter:
        card_ids = [c for c in card_ids if c == card_filter]

    buckets: Dict[str, list] = defaultdict(list)
    total = len(card_ids)

    for i, cid in enumerate(card_ids, 1):
        if i % 200 == 0:
            sys.stderr.write(f"\r進行中: {i}/{total}...")
            sys.stderr.flush()
        master = db.get_card(cid)
        if master is None:
            continue
        for r in cov.classify(master):
            if r.status == "NO_CHANGE":
                sub = classify_no_change(master, r.trigger)
                buckets[sub].append(r)
            elif r.status == "EXECUTED" and "WARN" in (r.detail or ""):
                sub = classify_warn(master, r.trigger)
                buckets[f"WARN_{sub}"].append(r)
    sys.stderr.write(f"\r完了: {total} カード処理済み\n")

    order = ["NO_IMPL", "NO_TARGET", "COND_FALSE", "RESTRICTION", "PASSIVE",
             "WARN_DIRECTION", "WARN_MODAL"]
    print("=== 品質地図: NO_CHANGE / WARN 三分類 ===")
    print("  [NO_CHANGE]")
    for k in ("NO_IMPL", "NO_TARGET", "COND_FALSE", "RESTRICTION", "PASSIVE"):
        mark = "  ★真のバグ候補" if k == "NO_IMPL" else ""
        print(f"    {k:<12}: {len(buckets[k]):4d}{mark}")
    print("  [WARN 方向不一致]")
    print(f"    DIRECTION   : {len(buckets['WARN_DIRECTION']):4d}  ★真のバグ候補")
    print(f"    MODAL       : {len(buckets['WARN_MODAL']):4d}  (別パス実行の誤検知が大半)")
    print()

    targets = [show] if show else ["NO_IMPL", "WARN_DIRECTION"]
    for t in targets:
        key = t if t in buckets else (f"WARN_{t}" if f"WARN_{t}" in buckets else t)
        items = buckets.get(key, [])
        print(f"--- {key} ({len(items)} 件) ---")
        for r in items[:80]:
            print(f"  {r.card_id:<12} {r.trigger:<14} {r.name}  | {r.detail[:50]}")
        if len(items) > 80:
            print(f"  ... 他 {len(items) - 80} 件")
        print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", default=None,
                    help="NO_IMPL/NO_TARGET/COND_FALSE/RESTRICTION/PASSIVE/WARN_DIRECTION/WARN_MODAL")
    ap.add_argument("--card", default=None)
    args = ap.parse_args()
    run(show=args.show, card_filter=args.card)
