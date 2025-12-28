import os
import uuid
import logging
import sys
import json
from typing import Any, Dict, Optional, List

from fastapi import FastAPI, Body, Request
from fastapi.middleware.cors import CORSMiddleware

current_api_dir = os.path.dirname(os.path.abspath(__file__))
if current_api_dir not in sys.path:
    sys.path.append(current_api_dir)

try:
    from schemas import GameStateSchema
except ImportError:
    from .schemas import GameStateSchema

from opcg_sim.src.utils.logger_config import session_id_ctx, log_event
from opcg_sim.src.core.gamestate import Player, GameManager
from opcg_sim.src.utils.loader import CardLoader, DeckLoader

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, force=True)

def get_const():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "shared_constants.json")
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f: return json.load(f)
    return {}
CONST = get_const()

BASE_DIR = os.path.dirname(current_api_dir)
DATA_DIR = os.path.join(BASE_DIR, "data")
CARD_DB_PATH = os.path.join(DATA_DIR, "opcg_cards.json")

app = FastAPI(title="OPCG Simulator API v1.5")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def build_game_result_hybrid(manager: GameManager, game_id: str, success: bool = True, error_code: str = None, error_msg: str = None) -> Dict[str, Any]:
    player_keys = CONST.get('PLAYER_KEYS', {})
    api_root_keys = CONST.get('API_ROOT_KEYS', {})
    error_props = CONST.get('ERROR_PROPERTIES', {})
    
    p1_key = player_keys.get('P1', 'p1')
    p2_key = player_keys.get('P2', 'p2')

    raw_game_state = {
        "game_id": game_id,
        "turn_info": {
            "turn_count": manager.turn_count if manager else 0,
            "current_phase": manager.phase.name if manager else "N/A",
            "active_player_id": manager.turn_player.name if manager else "N/A"
        },
        "players": {
            p1_key: manager.p1.to_dict() if manager else {},
            p2_key: manager.p2.to_dict() if manager else {}
        }
    }
    
    validated_state = None
    if success:
        try:
            validated_state = GameStateSchema(**raw_game_state).model_dump(by_alias=True)
        except Exception as e:
            log_event("ERROR", "api.validation", f"Validation Error: {e}")
            validated_state = raw_game_state 

    error_obj = None
    if not success:
        error_obj = {
            error_props.get('CODE', 'code'): error_code,
            error_props.get('MESSAGE', 'message'): error_msg
        }

    return {
        api_root_keys.get('SUCCESS', 'success'): success,
        "game_id": game_id,
        api_root_keys.get('GAME_STATE', 'game_state'): validated_state,
        api_root_keys.get('ERROR', 'error'): error_obj
    }

@app.middleware("http")
async def trace_logging_middleware(request: Request, call_next):
    s_id = request.headers.get("X-Session-ID") or request.query_params.get("sessionId")
    if not s_id:
        s_id = f"gen-{uuid.uuid4().hex[:8]}"
    token = session_id_ctx.set(s_id)
    if not request.url.path.endswith(("/health", "/favicon.ico")):
        log_event("INFO", "api.inbound", f"{request.method} {request.url.path}")
    try:
        response = await call_next(request)
        response.headers["X-Session-ID"] = s_id
        return response
    finally:
        session_id_ctx.reset(token)

@app.options("/api/log")
async def options_log():
    return {"status": "ok"}

@app.post("/api/log")
async def receive_frontend_log(data: Dict[str, Any] = Body(...)):
    s_id = data.get("sessionId") or session_id_ctx.get()
    token = session_id_ctx.set(s_id)
    try:
        log_event(
            level_key=data.get("level", "info"),
            action=data.get("action", "client.log"),
            msg=data.get("msg", ""),
            player=data.get("player", "system"),
            payload=data.get("payload"),
            source="FE"
        )
        return {"status": "ok"}
    finally:
        session_id_ctx.reset(token)

GAMES: Dict[str, GameManager] = {}
card_db = CardLoader(CARD_DB_PATH)
card_db.load()
deck_loader = DeckLoader(card_db)

@app.options("/api/game/create")
async def options_game_create():
    return {"status": "ok"}

@app.post("/api/game/create")
async def game_create(req: Any = Body(...)):
    try:
        game_id = str(uuid.uuid4())
        log_event("INFO", "game.create", f"Creating game: {game_id}", payload=req)
        p1_path = os.path.join(DATA_DIR, req.get("p1_deck", ""))
        p2_path = os.path.join(DATA_DIR, req.get("p2_deck", ""))
        p1_leader, p1_cards = deck_loader.load_deck(p1_path, req.get("p1_name", "P1"))
        p2_leader, p2_cards = deck_loader.load_deck(p2_path, req.get("p2_name", "P2"))
        player1 = Player(req.get("p1_name", "P1"), p1_cards, p1_leader)
        player2 = Player(req.get("p2_name", "P2"), p2_cards, p2_leader)
        manager = GameManager(player1, player2)
        manager.start_game()
        GAMES[game_id] = manager
        return build_game_result_hybrid(manager, game_id)
    except Exception as e:
        log_event("ERROR", "game.create_fail", str(e))
        return {"success": False, "game_id": "", "error": {"message": str(e)}}

@app.options("/api/game/action")
async def options_game_action():
    return {"status": "ok"}

@app.post("/api/game/action")
async def game_action(req: Dict[str, Any] = Body(...)):
    game_id = req.get("game_id")
    manager = GAMES.get(game_id)
    error_codes = CONST.get('ERROR_CODES', {})
    
    if not manager:
        return build_game_result_hybrid(
            None, game_id, success=False, 
            error_code=error_codes.get('GAME_NOT_FOUND', 'GAME_NOT_FOUND'),
            error_msg="指定されたゲームが見つかりません。"
        )

    action_type = req.get("action")
    player_id = req.get("player_id")
    payload = req.get("payload", {})
    card_uuid = payload.get("uuid")

    log_event("INFO", f"game.action.{action_type}", f"Player {player_id} action: {action_type}", 
              player=player_id, payload=req)

    try:
        current_player = manager.p1 if player_id == manager.p1.name else manager.p2
        
        if action_type == "PLAY":
            target_card = next((c for c in current_player.hand if c.uuid == card_uuid), None)
            if target_card:
                manager.play_card_action(current_player, target_card)
            else:
                raise ValueError("対象のカードが手札にありません。")

        elif action_type == "TURN_END":
            manager.end_turn()

        elif action_type == "ATTACK":
            target_card = None
            if current_player.leader and current_player.leader.uuid == card_uuid:
                target_card = current_player.leader
            else:
                target_card = next((c for c in current_player.field if c.uuid == card_uuid), None)
            
            if target_card:
                target_card.is_rest = True
            else:
                raise ValueError("アタック可能なカードが見つかりません。")

        elif action_type == "ATTACH_DON":
            target_card = None
            if current_player.leader and current_player.leader.uuid == card_uuid:
                target_card = current_player.leader
            else:
                target_card = next((c for c in current_player.field if c.uuid == card_uuid), None)

            if target_card and current_player.don_active:
                don = current_player.don_active.pop(0)
                don.attached_to = target_card.uuid
                current_player.don_attached_cards.append(don)
                target_card.attached_don += 1
            else:
                raise ValueError("ドン!!を付与できません。")

        elif action_type == "ACTIVATE_MAIN":
            target_card = next((c for c in current_player.field if c.uuid == card_uuid), None)
            if not target_card and current_player.leader and current_player.leader.uuid == card_uuid:
                target_card = current_player.leader
            
            if target_card:
                from opcg_sim.src.models.enums import TriggerType
                for ability in target_card.master.abilities:
                    if ability.trigger == TriggerType.ACTIVATE_MAIN:
                        manager.resolve_ability(current_player, ability, source_card=target_card)
            else:
                raise ValueError("効果を発動できるカードが見つかりません。")

        return build_game_result_hybrid(manager, game_id, success=True)
    except Exception as e:
        log_event("ERROR", "game.action_fail", str(e), player=player_id)
        return build_game_result_hybrid(
            manager, game_id, success=False, 
            error_code=error_codes.get('INVALID_ACTION', 'INVALID_ACTION'),
            error_msg=str(e)
        )

@app.get("/health")
async def health():
    return {"status": "ok", "constants_loaded": bool(CONST), "session_id": session_id_ctx.get()}
