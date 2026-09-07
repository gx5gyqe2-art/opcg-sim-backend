"""レスポンス／WS ペイロードの整形。

対局状態→JSON（`build_game_result_hybrid`）とルーム/対局メッセージ（`build_rule_message`）を
1箇所へ集約する。API 契約の中心＝validate 失敗時の raw dict フォールバックも含めて挙動不変で移設。
"""
from typing import Any, Dict

from .config import CONST
from .engine_rs import RsGame
from .schemas import GameStateSchema, PendingRequestSchema
from .state import GAMES, RULE_ROOMS


def build_game_result_hybrid(manager: RsGame, game_id: str, success: bool = True, error_code: str = None, error_msg: str = None) -> Dict[str, Any]:
    """対局状態を API レスポンス（契約）へ整形する。

    盤面（`turn_info`／`players`／`active_battle`）は Rust エンジンが同じ形で出す
    （`RsGame.board`＝`opcg_engine.Game.board_json`。計画 §15.2）ので、ここは `game_id` を足して
    schema 検証にかけるだけ。`action_events` はその要求で積まれたイベントログ（フロントの
    `EffectToast`／eventLog）。
    """
    api_root_keys = CONST.get('API_ROOT_KEYS', {}); error_props = CONST.get('ERROR_PROPERTIES', {})
    player_keys = CONST.get('PLAYER_KEYS', {})
    p1_key = player_keys.get('P1', 'p1'); p2_key = player_keys.get('P2', 'p2')
    if manager:
        raw_game_state = manager.board()
        raw_game_state["game_id"] = game_id
    else:
        raw_game_state = {
            "game_id": game_id,
            "turn_info": {"turn_count": 0, "current_phase": "N/A", "active_player_id": "N/A", "winner": None},
            "players": {p1_key: {}, p2_key: {}},
            "active_battle": None,
        }
    validated_state = None
    if success:
        try: validated_state = GameStateSchema(**raw_game_state).model_dump(by_alias=True)
        except Exception: validated_state = raw_game_state
    pending_req_data = None
    if manager and success:
        pending_obj = manager.get_pending_request()  # request_id は engine_rs の `_rid` が付ける
        if pending_obj:
            try: pending_req_data = PendingRequestSchema(**pending_obj).model_dump(by_alias=True)
            except Exception: pending_req_data = pending_obj
    error_obj = None
    if not success: error_obj = {error_props.get('CODE', 'code'): error_code, error_props.get('MESSAGE', 'message'): error_msg}
    return {api_root_keys.get('SUCCESS', 'success'): success, "game_id": game_id, api_root_keys.get('GAME_STATE', 'game_state'): validated_state, api_root_keys.get('PENDING_REQUEST', 'pending_request'): pending_req_data, api_root_keys.get('ERROR', 'error'): error_obj, "action_events": getattr(manager, 'action_events', []) if manager else []}


def _rule_room_meta(game_id: str) -> Dict[str, Any]:
    """WS メッセージへ付与するルーム メタ情報。"""
    room = RULE_ROOMS.get(game_id, {})
    return {
        "room_name": room.get("room_name", "Rule Room"),
        "status": room.get("status", "WAITING"),
        "ready_states": room.get("ready", {"p1": False, "p2": False}),
        "deck_preview": room.get("deck_preview", {"p1": None, "p2": None}),
    }


def build_rule_message(game_id: str) -> Dict[str, Any]:
    """ルーム/対局状態を 1 つの WS/REST ペイロードへまとめる。

    WAITING 中は game_state=None（フロントはセットアップ画面を表示）。
    PLAYING/FINISHED は build_game_result_hybrid の結果を内包する。
    """
    msg: Dict[str, Any] = {"type": "STATE_UPDATE", "game_id": game_id}
    msg.update(_rule_room_meta(game_id))
    if msg["status"] in ("PLAYING", "FINISHED"):
        manager = GAMES.get(game_id)
        result = build_game_result_hybrid(manager, game_id, success=True)
        msg.update(result)
    else:
        msg.update({"success": True, "game_state": None, "pending_request": None, "action_events": []})
    return msg
