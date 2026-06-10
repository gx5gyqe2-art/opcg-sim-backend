"""Ability/EffectNode を「指紋(summary dict)」に正規化するユーティリティ。

ゴールデンテストで AST 全体を完全一致比較するのは脆い（内部表現の細部変更で
壊れる）。代わりに「効果の意味として重要な属性」だけを抽出した dict に落とし、
期待値との *部分一致* を判定する。これにより:
  - テストが意味のある回帰だけを検出する
  - 期待値は気にする項目だけ書けばよい（過少指定が許される）
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from opcg_sim.src.models.effect_types import (
    Ability,
    Branch,
    Choice,
    Condition,
    GameAction,
    Sequence,
)


def _enum_name(v: Any) -> Optional[str]:
    return getattr(v, "name", None) if v is not None else None


def summarize_target(tq) -> Optional[Dict[str, Any]]:
    if tq is None:
        return None
    out: Dict[str, Any] = {}
    zone = tq.zone
    out["zone"] = _enum_name(zone) if not isinstance(zone, list) else [
        _enum_name(z) for z in zone
    ]
    out["player"] = _enum_name(tq.player)
    if tq.card_type:
        out["card_type"] = list(tq.card_type)
    if tq.traits:
        out["traits"] = list(tq.traits)
    if tq.names:
        out["names"] = list(tq.names)
    if tq.cost_max is not None:
        out["cost_max"] = tq.cost_max
    if getattr(tq, "cost_max_dynamic", None) is not None:
        out["cost_max_dynamic"] = tq.cost_max_dynamic
    if tq.cost_min is not None:
        out["cost_min"] = tq.cost_min
    if tq.power_max is not None:
        out["power_max"] = tq.power_max
    if tq.power_min is not None:
        out["power_min"] = tq.power_min
    out["count"] = tq.count
    out["is_up_to"] = tq.is_up_to
    if tq.ref_id:
        out["ref_id"] = tq.ref_id
    return out


def summarize_condition(cond: Optional[Condition]) -> Optional[Dict[str, Any]]:
    if cond is None:
        return None
    out: Dict[str, Any] = {"type": _enum_name(cond.type)}
    out["operator"] = _enum_name(cond.operator)
    out["value"] = cond.value if not hasattr(cond.value, "name") else _enum_name(cond.value)
    out["player"] = _enum_name(cond.player)
    if cond.args:
        out["args"] = [summarize_condition(a) for a in cond.args]
    return out


def summarize_node(node) -> Optional[Dict[str, Any]]:
    if node is None:
        return None
    if isinstance(node, GameAction):
        out = {
            "kind": "action",
            "type": _enum_name(node.type),
            "target": summarize_target(node.target),
            "value": node.value.base if node.value else 0,
            "status": node.status,
            "duration": node.duration,
            "destination": _enum_name(node.destination),
            "dest_position": getattr(node, "dest_position", None),
        }
        if getattr(node, "sub_effect", None) is not None:
            out["sub_effect"] = summarize_node(node.sub_effect)
        return out
    if isinstance(node, Sequence):
        return {"kind": "seq", "actions": [summarize_node(a) for a in node.actions]}
    if isinstance(node, Branch):
        return {
            "kind": "branch",
            "condition": summarize_condition(node.condition),
            "if_true": summarize_node(node.if_true),
            "if_false": summarize_node(node.if_false),
        }
    if isinstance(node, Choice):
        return {
            "kind": "choice",
            "options": [summarize_node(o) for o in node.options],
        }
    return {"kind": type(node).__name__}


def summarize_ability(ab: Ability) -> Dict[str, Any]:
    return {
        "trigger": _enum_name(ab.trigger),
        "condition": summarize_condition(ab.condition),
        "cost": summarize_node(ab.cost),
        "effect": summarize_node(ab.effect),
    }


def matches_expected(actual: Any, expected: Any, path: str = "") -> List[str]:
    """expected を「部分仕様」とみなして actual と再帰比較する。

    - expected が dict: 各キーだけを actual と比較（actual の余分なキーは無視）
    - expected が list: 同じ長さで各要素を比較
    - それ以外: 等値比較
    戻り値は不一致の説明リスト（空なら一致）。
    """
    mismatches: List[str] = []
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [f"{path or '<root>'}: expected dict, got {type(actual).__name__}"]
        for k, v in expected.items():
            mismatches += matches_expected(actual.get(k), v, f"{path}.{k}" if path else k)
    elif isinstance(expected, list):
        if not isinstance(actual, list):
            return [f"{path}: expected list, got {type(actual).__name__}"]
        if len(actual) != len(expected):
            mismatches.append(f"{path}: list length {len(actual)} != expected {len(expected)}")
        else:
            for i, (a, e) in enumerate(zip(actual, expected)):
                mismatches += matches_expected(a, e, f"{path}[{i}]")
    else:
        if actual != expected:
            mismatches.append(f"{path}: {actual!r} != expected {expected!r}")
    return mismatches
