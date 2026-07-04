"""API ルート: ルールモード・オンライン対戦（ロビー / ルーム）（ドメイン別 APIRouter）。

`routers/__init__.py` が全ドメインを束ねて app が include する。ロジックは
config/resources/state/presenters/ws/services へ委譲する。monkeypatch 対象の
`load_deck_mixed`/`_deck_preview` はサービスモジュール属性経由で呼ぶ（`deck_svc.*`）。
"""
import uuid
import random
from datetime import datetime
from typing import Any, Dict

from fastapi import APIRouter, Body, WebSocket, WebSocketDisconnect

try:
    from google.cloud import firestore
except Exception:
    firestore = None

from opcg_sim.src.core.gamestate import Player, GameManager
from ..resources import materialize_all_cards
from ..state import GAMES, RULE_ROOMS
from ..presenters import build_rule_message, _rule_room_meta
from ..ws import game_ws_manager
from ..services import decks as deck_svc

router = APIRouter()


@router.options("/api/rule/create")
async def options_rule_create(): return {"status": "ok"}

@router.post("/api/rule/create")
async def rule_create(req: Any = Body(...)):
    try:
        game_id = str(uuid.uuid4())
        room = {
            "game_id": game_id,
            "room_name": req.get("room_name", "Rule Room"),
            "created_at": datetime.now().isoformat(),
            "status": "WAITING",
            "ready": {"p1": False, "p2": False},
            "decks": {"p1": None, "p2": None},
            "deck_preview": {"p1": None, "p2": None},
        }
        RULE_ROOMS[game_id] = room
        return {"success": True, "game_id": game_id, **_rule_room_meta(game_id), "game_state": None}
    except Exception as e:
        return {"success": False, "error": str(e)}

@router.get("/api/rule/list")
async def rule_list():
    rooms = []
    for gid, room in RULE_ROOMS.items():
        try:
            rooms.append({
                "game_id": gid,
                "room_name": room.get("room_name", "Rule Room"),
                "p1_name": "P1",
                "p2_name": "P2",
                "turn": GAMES[gid].turn_count if gid in GAMES else 0,
                "created_at": room.get("created_at", "N/A"),
                "active_connections": game_ws_manager.count(gid),
                "status": room.get("status", "WAITING"),
                "ready_states": room.get("ready", {"p1": False, "p2": False}),
            })
        except Exception:
            continue
    return {"success": True, "games": rooms}

@router.options("/api/rule/action")
async def options_rule_action(): return {"status": "ok"}

@router.post("/api/rule/action")
async def rule_action(req: Dict[str, Any] = Body(...)):
    game_id = req.get("game_id"); room = RULE_ROOMS.get(game_id)
    if not room:
        return {"success": False, "error": "Rule room not found"}
    act = req.get("action_type"); pid = req.get("player_id")
    try:
        if act == "SET_DECK":
            if room["status"] != "WAITING":
                return {"success": False, "error": "Game already started"}
            if pid not in ("p1", "p2"):
                return {"success": False, "error": "Invalid player_id"}
            deck_id = req.get("deck_id")
            room["decks"][pid] = deck_id
            room["deck_preview"][pid] = deck_svc._deck_preview(deck_id, pid)
            room["ready"][pid] = bool(deck_id)
        elif act == "KICK_PLAYER":
            target = req.get("target_player_id")
            if room["status"] == "WAITING" and target in ("p1", "p2"):
                room["decks"][target] = None
                room["deck_preview"][target] = None
                room["ready"][target] = False
        elif act == "START":
            if room["status"] != "WAITING":
                return {"success": False, "error": "Game already started"}
            if not (room["ready"]["p1"] and room["ready"]["p2"]):
                return {"success": False, "error": "Both players must be ready"}
            materialize_all_cards()
            p1_leader, p1_cards = deck_svc.load_deck_mixed(room["decks"]["p1"], "p1")
            p2_leader, p2_cards = deck_svc.load_deck_mixed(room["decks"]["p2"], "p2")
            player1 = Player("p1", p1_cards, p1_leader); player2 = Player("p2", p2_cards, p2_leader)
            # 対戦モードの先行はランダム（コイントス）。結果は turn_info で両クライアントへ broadcast。
            first_player = random.choice([player1, player2])
            manager = GameManager(player1, player2); manager.start_game(first_player)
            GAMES[game_id] = manager
            room["status"] = "PLAYING"
        else:
            return {"success": False, "error": f"Unknown rule action: {act}"}

        await game_ws_manager.broadcast(game_id, build_rule_message(game_id))
        return {"success": True, "game_id": game_id, **build_rule_message(game_id)}
    except Exception as e:
        return {"success": False, "error": str(e)}

@router.websocket("/ws/game/{game_id}")
async def game_websocket_endpoint(websocket: WebSocket, game_id: str):
    await game_ws_manager.connect(websocket, game_id)
    try:
        while True: await websocket.receive_text()
    except WebSocketDisconnect:
        game_ws_manager.disconnect(websocket, game_id)
