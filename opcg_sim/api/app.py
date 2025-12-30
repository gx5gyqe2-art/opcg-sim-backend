import os
import uuid
import sys
import json
import traceback
from typing import Any, Dict, Optional, List

from fastapi import FastAPI, Body, Request
from fastapi.middleware.cors import CORSMiddleware

current_api_dir = os.path.dirname(os.path.abspath(__file__))
if current_api_dir not in sys.path:
    sys.path.append(current_api_dir)

try:
    from schemas import GameStateSchema, PendingRequestSchema, BattleActionRequest
except ImportError:
    from .schemas import GameStateSchema, PendingRequestSchema, BattleActionRequest

from opcg_sim.src.utils.logger_config import session_id_ctx, log_event
from opcg_sim.src.core.gamestate import Player, GameManager
from opcg_sim.src.utils.loader import CardLoader, DeckLoader

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

    active_pid = "N/A"
    if manager:
        active_pid = p1_key if manager.turn_player == manager.p1 else p2_key

    log_event("DEBUG", 
        "api.active_id_check", 
        f"Logic check: manager.turn_player.name={manager.turn_player.name}, p1.name={manager.p1.name}, result_id={active_pid}", 
        player="system")
    log_event(level_key="DEBUG",
        action="api.build_state",
        msg=f"Turn Info: count={manager.turn_count if manager else 0}, active_pid={active_pid}",
        player="system")


    raw_game_state = {
        "game_id": game_id,
        "turn_info": {
            "turn_count": manager.turn_count if manager else 0,
            "current_phase": manager.phase.name if manager else "N/A",
            "active_player_id": active_pid,
            "winner": manager.winner if manager else None
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
            log_event(level_key="ERROR", action="api.validation", msg=f"Validation Error: {e}", player="system")
            validated_state = raw_game_state 

    pending_req_data = None
    if manager and success:
        pending_obj = manager.get_pending_request()
        if pending_obj:
            try:
                pending_req_data = PendingRequestSchema(**pending_obj).model_dump(by_alias=True)
            except Exception as e:
                log_event(level_key="ERROR", action="api.pending_validation", msg=f"Pending Validation Error: {e}", player="system")
                pending_req_data = pending_obj

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
        api_root_keys.get('PENDING_REQUEST', 'pending_request'): pending_req_data,
        api_root_keys.get('ERROR', 'error'): error_obj
    }

@app.middleware("http")
async def trace_logging_middleware(request: Request, call_next):
    s_id = request.headers.get("X-Session-ID") or request.query_params.get("sessionId")
    if not s_id:
        s_id = f"gen-{uuid.uuid4().hex[:8]}"
    token = session_id_ctx.set(s_id)
    if not request.url.path.endswith(("/health", "/favicon.ico")):
        log_event(level_key="INFO", action="api.inbound", msg=f"{request.method} {request.url.path}", player="system")
    try:
        # 修正：必要な部分以外は修正しない制約に基づき、既存ロジックを維持
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
        log_event(level_key="INFO", action="game.create", msg=f"Creating game: {game_id}", payload=req, player="system")
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
        log_event(level_key="ERROR", action="game.create_fail", msg=traceback.format_exc(), player="system")
        return {"success": False, "game_id": "", "error": {"message": str(e)}}

@app.options("/api/game/action")
async def options_game_action():
    return {"status": "ok"}

@app.post("/api/game/action")
async def game_action(req: Dict[str, Any] = Body(...)):
    action_type = req.get("action")
    player_id = req.get("player_id", "system")
    log_event("DEBUG", "api.action_received", f"Action: {action_type}", player=player_id, payload=req)

    game_id = req.get("game_id")
    manager = GAMES.get(game_id)
    error_codes = CONST.get('ERROR_CODES', {})
    
    if not manager:
        return build_game_result_hybrid(
            None, game_id, success=False, 
            error_code=error_codes.get('GAME_NOT_FOUND', 'GAME_NOT_FOUND'),
            error_msg="指定されたゲームが見つかりません。"
        )

    payload = req.get("payload", {})
    card_uuid = payload.get("uuid")
    target_uuid = payload.get("target_uuid")

    try:
        from opcg_sim.src.models.enums import TriggerType
            current_player = manager.p1 if player_id == manager.p1.name else manager.p2
            potential_attackers = []
            if current_player.leader: potential_attackers.append(current_player.leader)
            potential_attackers.extend(current_player.field)
            target_card = next((c for c in potential_attackers if c.uuid == card_uuid), None)

            if not target_card:
                raise ValueError(f"指定されたカード {card_uuid} が盤面に見つかりません。")

            if action_type == "ATTACK":
                log_event("DEBUG", "api.attack_final_check", 
                          f"Ready to attack. Card: {target_card.master.name}, is_rest: {target_card.is_rest}", 
                          player=player_id)
                
                manager.declare_attack(target_card, attack_target)

            elif action_type == "ATTACH_DON":

                if current_player.don_active:
                    don = current_player.don_active.pop(0)
                    don.attached_to = target_card.uuid
                    current_player.don_attached_cards.append(don)
                    target_card.attached_don += 1
                else:
                    raise ValueError("アクティブなドン!!が不足しています。")
            
            elif action_type == "ACTIVATE_MAIN":
                for ability in target_card.master.abilities:
                    if ability.trigger == TriggerType.ACTIVATE_MAIN:
                        manager.resolve_ability(current_player, ability, source_card=target_card)

        return build_game_result_hybrid(manager, game_id, success=True)

    except Exception as e:
        log_event(
            level_key="ERROR",
            action="game.action_fail",
            msg=traceback.format_exc(),
            player=player_id,
            payload=req
        )
        return build_game_result_hybrid(
            manager, game_id, success=False, 
            error_code=error_codes.get('INVALID_ACTION', 'INVALID_ACTION'),
            error_msg=str(e)
        )

@app.options("/api/game/battle")

async def options_game_battle():
    return {"status": "ok"}

@app.post("/api/game/battle")
async def game_battle(req: BattleActionRequest):
    game_id = req.game_id
    player_id = req.player_id
    action_type = req.action_type
    card_uuid = req.card_uuid
    
    manager = GAMES.get(game_id)
    error_codes = CONST.get('ERROR_CODES', {})
    battle_types = CONST.get('c_to_s_interface', {}).get('BATTLE_ACTIONS', {}).get('TYPES', {})

    if not manager:
        log_event("ERROR", "api.battle_action", f"Game not found: {game_id}", player=player_id)
        return build_game_result_hybrid(None, game_id, success=False, error_code=error_codes.get('GAME_NOT_FOUND', 'GAME_NOT_FOUND'), error_msg="Game not found")

    player = manager.p1 if player_id == manager.p1.name else manager.p2

    try:
        is_valid = False
        try:
            is_valid = manager._validate_action(player, action_type)
        except Exception as ve:
            if action_type != "PASS":
                raise ve

        log_event("DEBUG", "api.battle_validation", f"Validation: {is_valid}", player=player_id)
        
        if action_type == battle_types.get('SELECT_BLOCKER'):
            blocker = next((c for c in player.field if c.uuid == card_uuid), None)
            manager.handle_block(blocker)
        
        elif action_type == battle_types.get('SELECT_COUNTER'):
            counter_card = next((c for c in player.hand if c.uuid == card_uuid), None)
            manager.apply_counter(player, counter_card)
            
        elif action_type == "PASS":
            manager.apply_counter(player, None)

        return build_game_result_hybrid(manager, game_id, success=True)

    except Exception as e:
        log_event("ERROR", "game.battle_fail", traceback.format_exc(), player=player_id)
        return build_game_result_hybrid(manager, game_id, success=False, error_code=error_codes.get('INVALID_ACTION', 'INVALID_ACTION'), error_msg=str(e))

@app.get("/health")
async def health():
    return {"status": "ok", "constants_loaded": bool(CONST), "session_id": session_id_ctx.get()}
