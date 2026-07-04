"""動的値の解決（GameManager からの移管・ステートレス。第1引数 gm）。"""
from __future__ import annotations

from ..effects.matcher import get_target_cards


def get_dynamic_value(gm, player: Player, val_source: ValueSource, targets: List[CardInstance], context: Dict) -> int:
    if not val_source: return 0
    if val_source.dynamic_source == "COUNT_REFERENCE":
        return len(player.trash)
    # 文脈依存「直前アクションで捨てた/戻した/KOした…カードN枚につき」（§7-5）。
    # 生の枚数を返す（divisor/multiplier は _calculate_value が適用する）。
    if val_source.dynamic_source == "PREV_ACTION_COUNT":
        return int((context or {}).get("_last_action_count", 0) or 0)
    # 「<範囲>N枚につき」の汎用カウント（RC-4）。範囲クエリを毎回実体化して数える
    # （PASSIVE 再計算で盤面に追随する）。
    if val_source.dynamic_source == "COUNT_QUERY" and getattr(val_source, "count_query", None) is not None:
        src = None
        src_uuid = (context or {}).get("_source_card_uuid")
        if src_uuid:
            src = gm._find_card_by_uuid(src_uuid)
        if src is None:
            src = player.leader
        n = len(get_target_cards(gm, val_source.count_query, src))
        return n
    # C9「（相手のリーダー／選んだキャラ／アタックしているキャラ）と同じパワーになる」。
    # 発動時スナップショット: 参照カードの現在パワーを固定値として返す（以後の変動に追随しない）。
    if val_source.dynamic_source == "REFERENCE_POWER":
        ref = gm._resolve_power_reference(player, val_source.ref_id, context)
        if ref is None:
            return val_source.base
        ref_owner, _ = gm._find_card_location(ref)
        is_ref_turn = bool(ref_owner) and ref_owner.name == gm.turn_player.name
        return ref.get_power(is_ref_turn)
    # 「元々のパワーと同じ」: 参照カードの基礎値（master.power）を写す（バフ非追随）
    if val_source.dynamic_source == "REFERENCE_BASE_POWER":
        ref = gm._resolve_power_reference(player, val_source.ref_id, context)
        if ref is None:
            return val_source.base
        return ref.master.power
    return val_source.base

def _resolve_power_reference(gm, player, ref_id, context):
    """C9 の同値パワー参照カードを解決する。ref_id: selected/opp_leader/attacker。"""
    opponent = gm.p2 if player == gm.p1 else gm.p1
    if ref_id == "opp_leader":
        return opponent.leader
    if ref_id == "self_leader":
        return player.leader
    if ref_id == "attacker":
        return (gm.active_battle or {}).get("attacker")
    if ref_id == "selected":
        saved = (context or {}).get("saved_targets", {})
        sel = saved.get("selected_card") or saved.get("selected")
        if isinstance(sel, list):
            return sel[0] if sel else None
        return sel
    return None
