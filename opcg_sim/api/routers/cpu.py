"""API ルート: CPU 対戦（ポーリング駆動）／リプレイ（ドメイン別 APIRouter）。

`routers/__init__.py` が全ドメインを束ねて app が include する。ロジックは
config/resources/state/presenters/ws/services へ委譲する。monkeypatch 対象の
`load_deck_mixed`/`_deck_preview` はサービスモジュール属性経由で呼ぶ（`deck_svc.*`）。
"""
import os
from typing import Any, Dict

from fastapi import APIRouter, Body

try:
    from google.cloud import firestore
except Exception:
    firestore = None

from opcg_sim.src.core import action_api
from opcg_sim.api import decide_client
from ..config import CONST, REPLAY_SCHEMA
from ..state import GAMES, CPU_GAMES
from ..presenters import build_game_result_hybrid
from ..ws import broadcast_rule_state
from ..services.replay import _replay_enabled, _replay_record_action, _capture_final_winner
from ..services.cpu_driver import _cached_cpu_move

router = APIRouter()


@router.options("/api/game/cpu/step")
async def options_game_cpu_step(): return {"status": "ok"}

@router.post("/api/game/cpu/step")
async def game_cpu_step(req: Dict[str, Any] = Body(...)):
    """CPU 対戦で CPU(p2) の「次の 1 手」を適用して返す（ポーリング駆動）。"""
    game_id = req.get("game_id"); manager = GAMES.get(game_id); meta = CPU_GAMES.get(game_id)
    error_codes = CONST.get('ERROR_CODES', {})
    if not manager:
        return build_game_result_hybrid(None, game_id, success=False, error_code=error_codes.get('GAME_NOT_FOUND', 'GAME_NOT_FOUND'), error_msg="指定されたゲームが見つかりません。")
    if not meta:
        return build_game_result_hybrid(manager, game_id, success=False, error_code=error_codes.get('INVALID_ACTION', 'INVALID_ACTION'), error_msg="このゲームは CPU 対戦ではありません。")

    cpu_pid = meta["cpu_player_id"]; difficulty = meta.get("difficulty", "hard")
    cpu_player = manager.p1 if manager.p1.name == cpu_pid else manager.p2

    def _waiting_for() -> str:
        if manager.winner:
            return "game_over"
        pending = manager.get_pending_request()
        if pending and pending.get("player_id") == cpu_pid:
            return "cpu"
        if pending:
            return "human_decision"
        return "human"

    cpu_acted = False; cpu_event = None
    try:
        manager.action_events = []
        if not manager.winner:
            pending = manager.get_pending_request()
            if pending and pending.get("player_id") == cpu_pid:
                turn_mem = meta.setdefault("turn_mem", {})
                # ⑥-a: 先行計画（pondering）が走行中なら完了を待つ（warm な queue を使う・既定 OFF）。
                _ptask = meta.get("plan_cache", {}).get("task")
                if _ptask is not None:
                    try:
                        await _ptask
                    except Exception:
                        pass
                trace_on = _replay_enabled(meta)
                tr = {} if trace_on else None
                move = None
                # Phase 3 ① 計画キャッシュ（OPCG_PLAN_CACHE=1・本番体感最適化・既定 OFF）。
                if os.environ.get("OPCG_PLAN_CACHE", "0") == "1" and not trace_on:
                    move = _cached_cpu_move(manager, cpu_player, difficulty, meta, turn_mem)
                if move is None:
                    move = decide_client.decide(manager, cpu_player, difficulty, mem=turn_mem,
                                                trace=tr, trace_read_ahead=False)
                if move is not None:
                    if trace_on:
                        meta.setdefault("decisions", []).append(
                            {"turn": manager.turn_count, "player": cpu_pid, **tr})
                        _replay_record_action(meta, manager, "cpu", cpu_pid, {
                            "action_type": move["action_type"], "card_uuid": move.get("card_uuid"),
                            "payload": move.get("payload")})
                    if move["kind"] == "battle":
                        action_api.apply_battle_action(manager, cpu_player, move["action_type"], move.get("card_uuid"))
                    else:
                        action_api.apply_game_action(manager, cpu_player, move["action_type"], move.get("payload", {}))
                    _capture_final_winner(meta, manager)
                    cpu_acted = True
                    cpu_event = manager.action_events[0] if manager.action_events else {"action": move["action_type"]}
        result = build_game_result_hybrid(manager, game_id, success=True)
        result["cpu_acted"] = cpu_acted
        result["cpu_event"] = cpu_event
        result["waiting_for"] = _waiting_for()
        await broadcast_rule_state(game_id)
        return result
    except Exception as e:
        return build_game_result_hybrid(manager, game_id, success=False, error_code=error_codes.get('INVALID_ACTION', 'INVALID_ACTION'), error_msg=str(e))

@router.options("/api/game/{game_id}/replay")
async def options_game_replay(game_id: str): return {"status": "ok"}

@router.get("/api/game/{game_id}/replay")
async def game_replay(game_id: str):
    """traced CPU 対局の「リプレイ種＋CPU思考トレース」を返す（GCS 不要・メモリ常駐）。"""
    meta = CPU_GAMES.get(game_id)
    if not _replay_enabled(meta):
        return {"success": False, "error": {"code": "REPLAY_NOT_FOUND",
                "message": "この対局のリプレイ記録がありません（cpu_trace 未指定 or 不明なゲーム）。"}}
    descriptor = {
        "schema": REPLAY_SCHEMA, "seed": meta.get("seed"),
        "first_player": meta.get("first_player"), "difficulty": meta.get("difficulty"),
        "cpu_player_id": meta.get("cpu_player_id"),
        "leaders": meta.get("leaders"), "decks": meta.get("decks"),
        "actions": meta.get("actions", []),
    }
    return {"success": True, "game_id": game_id,
            "replay": descriptor, "decisions": meta.get("decisions", [])}
