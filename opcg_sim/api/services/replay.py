"""リプレイ種＋CPU思考トレース（実アプリ対局・opt-in）。

すべて opt-in（create リクエストの cpu_trace=true）でのみ作動し、未指定の本番対局には
一切の追加処理・レイテンシ・挙動変化を与えない（トレースは観測専用＝進行不変）。
スキーマ識別子 `REPLAY_SCHEMA` は config を参照。
"""
from typing import Any, Dict

from opcg_sim.src.core import cpu_ai


def _replay_enabled(meta) -> bool:
    return bool(meta and meta.get("cpu_trace"))


def _replay_record_action(meta, manager, src: str, player_id: str, movelike: Dict[str, Any]):
    """traced CPU 対局のアクションを card_id 基準で記録する（再現用・例外安全・適用前に呼ぶ）。"""
    if not _replay_enabled(meta):
        return
    try:
        desc = cpu_ai._describe_move(manager, movelike) or {"action_type": movelike.get("action_type")}
        meta.setdefault("actions", []).append(
            {"src": src, "turn": manager.turn_count, "player": player_id, **desc})
    except Exception:
        pass


def _capture_final_winner(meta, manager):
    """traced 対局の終局勝者を meta に保持する（WS 切断後の cleanup が manager を退避しても replay で参照可能）。

    opt-in（cpu_trace）時のみ作動＝本番対局にはオーバーヘッド・挙動変化なし。**アクション適用後**に呼ぶ。
    （旧 _capture_value_samples の価値学習データ採取は学習価値サブシステムごと撤去・2026-06-28。）
    """
    if not _replay_enabled(meta):
        return
    try:
        if manager.winner is not None:
            meta["_winner"] = manager.winner
    except Exception:
        pass


# --- 盤面フレーム（リプレイビューア用・opt-in） ------------------------------

# 1 対局のフレーム上限（TURN_ACTION_CAP で対局長は有界だが、暴走時のメモリ保険）。
# 超過分は записせず frames_truncated を立てる（ビューアは末尾欠落を表示できる）。
_FRAME_CAP = 4000

# フレームに残すカード項目。静的なマスター情報（効果テキスト・特徴・属性等）は
# card_id からフロントのカード DB で引けるため落とし、**動的な状態のみ**を持つ
# （フレーム総量を約 1/5 に抑える）。
_CARD_KEEP = ("uuid", "card_id", "name", "power", "cost", "is_rest", "is_face_up",
              "attached_don", "keywords", "ability_disabled", "is_frozen")


def _compact_card(d):
    if not isinstance(d, dict):
        return None
    return {k: d[k] for k in _CARD_KEEP if k in d}


def _frame_side(manager, player) -> Dict[str, Any]:
    """1 プレイヤー分の盤面（動的状態のみのコンパクト形）。パワーは presenter と同じ
    is_my_turn 規約（付与ドン!!は手番側のみ加算）で採る。"""
    d = player.to_dict(is_my_turn=(manager.turn_player is player))
    zones = d.get("zones") or {}
    return {
        "leader": _compact_card(d.get("leader")),
        "stage": _compact_card(d.get("stage")),
        "field": [_compact_card(c) for c in zones.get("field", [])],
        "hand": [_compact_card(c) for c in zones.get("hand", [])],
        "life": [_compact_card(c) for c in zones.get("life", [])],
        "trash": [_compact_card(c) for c in zones.get("trash", [])],
        # 付与中のドン!!はキャラ側 attached_don で表現されるため、コストエリアは非付与のみ数える。
        "don_active": sum(1 for x in d.get("don_active", []) if not x.get("attached_to")),
        "don_rested": sum(1 for x in d.get("don_rested", []) if not x.get("attached_to")),
        "don_deck": d.get("don_deck_count", 0),
        "deck_count": len(player.deck),
    }


def _replay_record_frame(meta, manager):
    """traced CPU 対局の盤面フレームを記録する（リプレイビューア用・例外安全・**適用後**に呼ぶ）。

    `action_index` は「このフレームの直前に記録された actions のインデックス」（create 直後の
    初期盤面のみ None）＝ decisions/actions との整合をインデックスで明示し、記録漏れがあっても
    ビューア側の対応付けが系統的にずれない。
    """
    if not _replay_enabled(meta):
        return
    try:
        frames = meta.setdefault("frames", [])
        if len(frames) >= _FRAME_CAP:
            meta["frames_truncated"] = True
            return
        ai = len(meta.get("actions", [])) - 1
        battle = None
        if manager.active_battle:
            battle = {"attacker_uuid": manager.active_battle["attacker"].uuid,
                      "target_uuid": manager.active_battle["target"].uuid}
        frames.append({
            "action_index": ai if ai >= 0 else None,
            "turn": manager.turn_count,
            "phase": manager.phase.name,
            "active": "p1" if manager.turn_player is manager.p1 else "p2",
            "winner": manager.winner,
            "players": {"p1": _frame_side(manager, manager.p1),
                        "p2": _frame_side(manager, manager.p2)},
            "pending": manager.get_pending_request() or None,
            "battle": battle,
        })
    except Exception:
        pass
