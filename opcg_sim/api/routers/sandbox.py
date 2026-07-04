"""API ルート: フリーモード（サンドボックス）（ドメイン別 APIRouter）。

`routers/__init__.py` が全ドメインを束ねて app が include する。ロジックは
config/resources/state/presenters/ws/services へ委譲する。monkeypatch 対象の
`load_deck_mixed`/`_deck_preview` はサービスモジュール属性経由で呼ぶ（`deck_svc.*`）。
"""
from typing import Any, Dict

from fastapi import APIRouter, Body, WebSocket, WebSocketDisconnect

try:
    from google.cloud import firestore
except Exception:
    firestore = None

from opcg_sim.src.core.sandbox import SandboxManager
from ..state import SANDBOX_GAMES
from ..ws import ws_manager
from ..services import decks as deck_svc

router = APIRouter()


@router.get("/api/sandbox/list")
async def sandbox_list():
    games = []
    for gid, mgr in SANDBOX_GAMES.items():
        try:
            games.append({
                "game_id": gid,
                "room_name": getattr(mgr, "room_name", "Untitled Room"),
                "p1_name": mgr.state["p1"]["name"],
                "p2_name": mgr.state["p2"]["name"],
                "turn": mgr.turn_count,
                "created_at": getattr(mgr, "created_at", "N/A"),
                "active_connections": len(ws_manager.active_connections.get(gid, [])),
                "status": mgr.status,
            })
        except Exception:
            continue
    return {"success": True, "games": games}

@router.post("/api/sandbox/create")
async def sandbox_create(req: Any = Body(...)):
    try:
        p1_name = req.get("p1_name", "P1"); p2_name = req.get("p2_name", "P2")
        manager = SandboxManager(p1_name=p1_name, p2_name=p2_name, room_name=req.get("room_name", "Custom Room"))
        SANDBOX_GAMES[manager.game_id] = manager
        return {"success": True, "game_id": manager.game_id, "game_state": manager.to_dict()}
    except Exception as e:
        return {"success": False, "error": str(e)}

@router.post("/api/sandbox/action")
async def sandbox_action(req: Dict[str, Any] = Body(...)):
    game_id = req.get("game_id"); manager = SANDBOX_GAMES.get(game_id)
    if not manager: return {"success": False, "error": "Sandbox game not found"}
    act_type = req.get("action_type"); pid = req.get("player_id")
    try:
        if act_type == "SET_DECK":
            deck_id = req.get("deck_id"); owner_name = manager.state[pid]["name"]
            leader, cards = deck_svc.load_deck_mixed(deck_id, owner_name)
            manager.set_player_deck(pid, cards, leader)
            manager.ready_states[pid] = True
        else: manager.process_action(req)
        new_state = manager.to_dict()
        broadcast_msg = {"type": "STATE_UPDATE", "state": new_state}
        if act_type == "KICK_PLAYER":
            broadcast_msg["kicked_player"] = req.get("target_player_id")
        await ws_manager.broadcast(game_id, broadcast_msg)
        return {"success": True, "game_id": game_id, "game_state": new_state}
    except Exception as e:
        return {"success": False, "error": str(e)}

@router.websocket("/ws/sandbox/{game_id}")
async def websocket_endpoint(websocket: WebSocket, game_id: str):
    await ws_manager.connect(websocket, game_id)
    try:
        while True: await websocket.receive_text()
    except WebSocketDisconnect:
        await ws_manager.disconnect(websocket, game_id)
