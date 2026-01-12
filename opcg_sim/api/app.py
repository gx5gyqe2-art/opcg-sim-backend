import os
import uuid
import sys
import json
import traceback
from typing import Any, Dict, Optional, List, Union
from fastapi import FastAPI, Body, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import firestore

current_api_dir = os.path.dirname(os.path.abspath(__file__))
if current_api_dir not in sys.path:
    sys.path.append(current_api_dir)

try:
    from schemas import GameStateSchema, PendingRequestSchema, BattleActionRequest
except ImportError:
    from .schemas import GameStateSchema, PendingRequestSchema, BattleActionRequest

try:
    from opcg_sim.src.core.sandbox import SandboxManager
except ImportError:
    pass

from opcg_sim.src.utils.logger_config import session_id_ctx, log_event, save_batch_logs
from opcg_sim.src.core.gamestate import Player, GameManager
from opcg_sim.src.utils.loader import CardLoader, DeckLoader
from opcg_sim.src.models.models import CardInstance

def get_const():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "shared_constants.json")
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f: return json.load(f)
    return {}
CONST = get_const()

BASE_DIR = os.path.dirname(current_api_dir)
DATA_DIR = os.path.join(BASE_DIR, "data")
CARD_DB_PATH = os.path.join(DATA_DIR, "opcg_cards.json")

app = FastAPI(title="OPCG Simulator API v1.7")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

db = None
try: db = firestore.Client()
except Exception: pass

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, game_id: str):
        await websocket.accept()
        if game_id not in self.active_connections:
            self.active_connections[game_id] = []
        self.active_connections[game_id].append(websocket)
        
        manager_inst = SANDBOX_GAMES.get(game_id)
        if manager_inst:
            try:
                await websocket.send_json({"type": "STATE_UPDATE", "state": manager_inst.to_dict()})
            except Exception as e:
                print(f"Failed to send initial state: {e}")

    def disconnect(self, websocket: WebSocket, game_id: str):
        if game_id in self.active_connections:
            if websocket in self.active_connections[game_id]:
                self.active_connections[game_id].remove(websocket)
            if not self.active_connections[game_id]:
                del self.active_connections[game_id]

    async def broadcast(self, game_id: str, message: dict):
        if game_id in self.active_connections:
            for connection in self.active_connections[game_id]:
                try:
                    await connection.send_json(message)
                except Exception:
                    pass

ws_manager = ConnectionManager()

def build_game_result_hybrid(manager: GameManager, game_id: str, success: bool = True, error_code: str = None, error_msg: str = None) -> Dict[str, Any]:
    player_keys = CONST.get('PLAYER_KEYS', {}); api_root_keys = CONST.get('API_ROOT_KEYS', {}); error_props = CONST.get('ERROR_PROPERTIES', {})
    p1_key = player_keys.get('P1', 'p1'); p2_key = player_keys.get('P2', 'p2')
    active_pid = "N/A"
    if manager: active_pid = p1_key if manager.turn_player == manager.p1 else p2_key
    battle_props = CONST.get('BATTLE_PROPERTIES', {})
    raw_game_state = {
        "game_id": game_id,
        "turn_info": {"turn_count": manager.turn_count if manager else 0, "current_phase": manager.phase.name if manager else "N/A", "active_player_id": active_pid, "winner": manager.winner if manager else None},
        "players": {p1_key: manager.p1.to_dict() if manager else {}, p2_key: manager.p2.to_dict() if manager else {}},
        battle_props.get('ACTIVE_BATTLE', 'active_battle'): {battle_props.get('ATTACKER_UUID', 'attacker_uuid'): manager.active_battle["attacker"].uuid, battle_props.get('TARGET_UUID', 'target_uuid'): manager.active_battle["target"].uuid, battle_props.get('COUNTER_BUFF', 'counter_buff'): manager.active_battle.get("counter_buff", 0)} if manager and manager.active_battle else None
    }
    validated_state = None
    if success:
        try: validated_state = GameStateSchema(**raw_game_state).model_dump(by_alias=True)
        except Exception as e: log_event(level_key="ERROR", action="api.validation", msg=f"Validation Error: {e}", player="system"); validated_state = raw_game_state 
    pending_req_data = None
    if manager and success:
        pending_obj = manager.get_pending_request()
        if pending_obj:
            try: pending_req_data = PendingRequestSchema(**pending_obj).model_dump(by_alias=True)
            except Exception as e: log_event(level_key="ERROR", action="api.pending_validation", msg=f"Pending Validation Error: {e}", player="system"); pending_req_data = pending_obj
    error_obj = None
    if not success: error_obj = {error_props.get('CODE', 'code'): error_code, error_props.get('MESSAGE', 'message'): error_msg}
    return {api_root_keys.get('SUCCESS', 'success'): success, "game_id": game_id, api_root_keys.get('GAME_STATE', 'game_state'): validated_state, api_root_keys.get('PENDING_REQUEST', 'pending_request'): pending_req_data, api_root_keys.get('ERROR', 'error'): error_obj}

@app.middleware("http")
async def trace_logging_middleware(request: Request, call_next):
    s_id = request.headers.get("X-Session-ID") or request.query_params.get("sessionId")
    if not s_id: s_id = f"gen-{uuid.uuid4().hex[:8]}"
    token = session_id_ctx.set(s_id)
    try:
        response = await call_next(request); response.headers["X-Session-ID"] = s_id; return response
    finally: session_id_ctx.reset(token)

@app.post("/api/log")
async def receive_frontend_log(data: Union[Dict[str, Any], List[Dict[str, Any]]] = Body(...)):
    if isinstance(data, list):
        s_id = "unknown"
        if len(data) > 0 and isinstance(data[0], dict): s_id = data[0].get("sessionId") or session_id_ctx.get()
        token = session_id_ctx.set(s_id)
        try: save_batch_logs(data, s_id); return {"status": "ok", "mode": "batch"}
        finally: session_id_ctx.reset(token)
    else:
        s_id = data.get("sessionId") or session_id_ctx.get(); token = session_id_ctx.set(s_id)
        try: log_event(level_key=data.get("level", "info"), action=data.get("action", "client.log"), msg=data.get("msg", ""), player=data.get("player", "system"), payload=data.get("payload"), source="FE"); return {"status": "ok", "mode": "single"}
        finally: session_id_ctx.reset(token)

GAMES: Dict[str, GameManager] = {}
SANDBOX_GAMES: Dict[str, 'SandboxManager'] = {}

card_db = CardLoader(CARD_DB_PATH); card_db.load(); deck_loader = DeckLoader(card_db)

def load_deck_mixed(source_str: str, owner_id: str):
    if source_str.startswith("db:"):
        if not db: raise ValueError("Firestore is not initialized.")
        deck_id = source_str[3:]; doc = db.collection("decks").document(deck_id).get()
        if not doc.exists: raise ValueError(f"Deck ID not found: {deck_id}")
        data = doc.to_dict(); leader_id = data.get("leader_id"); card_uuids = data.get("card_uuids", [])
        leader_inst = None
        if leader_id:
            master = card_db.get_card(leader_id)
            if master: leader_inst = CardInstance(master, owner_id)
        cards_inst = [CardInstance(card_db.get_card(cid), owner_id) for cid in card_uuids if card_db.get_card(cid)]
        log_event("INFO", "loader.db_load", f"Loaded deck from DB: {deck_id}", player=owner_id)
        return leader_inst, cards_inst
    else:
        path = os.path.join(DATA_DIR, source_str); return deck_loader.load_deck(path, owner_id)

@app.post("/api/game/create")
async def game_create(req: Any = Body(...)):
    try:
        game_id = str(uuid.uuid4()); log_event(level_key="INFO", action="game.create", msg=f"Creating game: {game_id}", payload=req, player="system")
        p1_source = req.get("p1_deck", ""); p2_source = req.get("p2_deck", "")
        if len(card_db.cards) < len(card_db.raw_db):
             for card_id in card_db.raw_db.keys(): card_db.get_card(card_id)
        p1_leader, p1_cards = load_deck_mixed(p1_source, req.get("p1_name", "P1")); p2_leader, p2_cards = load_deck_mixed(p2_source, req.get("p2_name", "P2"))
        player1 = Player(req.get("p1_name", "P1"), p1_cards, p1_leader); player2 = Player(req.get("p2_name", "P2"), p2_cards, p2_leader)
        manager = GameManager(player1, player2); manager.start_game(); GAMES[game_id] = manager; return build_game_result_hybrid(manager, game_id)
    except Exception as e:
        log_event(level_key="ERROR", action="game.create_fail", msg=traceback.format_exc(), player="system"); return {"success": False, "game_id": "", "error": {"message": str(e)}}

@app.post("/api/game/action")
async def game_action(req: Dict[str, Any] = Body(...)):
    action_type = req.get("action") or req.get("type"); player_id = req.get("player_id", "system")
    game_id = req.get("game_id"); manager = GAMES.get(game_id); error_codes = CONST.get('ERROR_CODES', {})
    if not manager: return build_game_result_hybrid(None, game_id, success=False, error_code=error_codes.get('GAME_NOT_FOUND', 'GAME_NOT_FOUND'), error_msg="指定されたゲームが見つかりません。")
    payload = req.get("payload") or req.get("full_payload") or {}
    card_uuid = payload.get("uuid") or payload.get("card_id"); target_ids = payload.get("target_ids", [])
    target_uuid = target_ids[0] if isinstance(target_ids, list) and len(target_ids) > 0 else payload.get("target_uuid")
    try:
        from opcg_sim.src.models.enums import TriggerType
        c_to_s = CONST.get('c_to_s_interface', {}); game_actions = c_to_s.get('GAME_ACTIONS', {}).get('TYPES', {})
        current_player = manager.p1 if player_id == manager.p1.name else manager.p2; opponent = manager.p2 if current_player == manager.p1 else manager.p1
        potential_cards = []
        if current_player.leader: potential_cards.append(current_player.leader)
        potential_cards.extend(current_player.field)
        if current_player.stage: potential_cards.append(current_player.stage)
        operating_card = next((c for c in potential_cards if c.uuid == card_uuid), None)
        if action_type == game_actions.get('PLAY', 'PLAY'):
            target_card_in_hand = next((c for c in current_player.hand if c.uuid == card_uuid), None)
            if target_card_in_hand: manager.pay_cost(current_player, target_card_in_hand.current_cost); manager.play_card_action(current_player, target_card_in_hand)
            else: raise ValueError("対象のカードが手札にありません。")
        elif action_type == game_actions.get('TURN_END', 'TURN_END'): manager.end_turn()
        elif action_type in [game_actions.get('ATTACK', 'ATTACK'), game_actions.get('ATTACK_CONFIRM', 'ATTACK_CONFIRM')]:
            if card_uuid == target_uuid: raise ValueError("自分自身を攻撃対象に選択することはできません。")
            if not operating_card: raise ValueError("アタックするカードが見つかりません。")
            opponent_units = [opponent.leader] + opponent.field
            if opponent.stage: opponent_units.append(opponent.stage)
            attack_target = next((c for c in opponent_units if c.uuid == target_uuid), None)
            if not attack_target: raise ValueError("攻撃対象が見つかりません。")
            log_event("INFO", "api.attack_execute", f"Attacking: {operating_card.master.name} -> {attack_target.master.name}", player=player_id); manager.declare_attack(operating_card, attack_target)
        elif action_type == game_actions.get('ATTACH_DON', 'ATTACH_DON'):
            if not operating_card: raise ValueError("ドン!!を付与する対象のカードが見つかりません。")
            if current_player.don_active: don = current_player.don_active.pop(0); don.attached_to = operating_card.uuid; current_player.don_attached_cards.append(don); operating_card.attached_don += 1
            else: raise ValueError("アクティブなドン!!が不足しています。")
        elif action_type == game_actions.get('ACTIVATE_MAIN', 'ACTIVATE_MAIN'):
            if not operating_card: raise ValueError("効果を発動するカードが見つかりません。")
            for ability in operating_card.master.abilities:
                if ability.trigger == TriggerType.ACTIVATE_MAIN: manager.resolve_ability(current_player, ability, source_card=operating_card)
        elif action_type == game_actions.get('RESOLVE_EFFECT_SELECTION', 'RESOLVE_EFFECT_SELECTION'): manager.resolve_interaction(current_player, payload)
        return build_game_result_hybrid(manager, game_id, success=True)
    except Exception as e:
        log_event(level_key="ERROR", action="game.action_fail", msg=traceback.format_exc(), player=player_id, payload=req); return build_game_result_hybrid(manager, game_id, success=False, error_code=error_codes.get('INVALID_ACTION', 'INVALID_ACTION'), error_msg=str(e))

@app.post("/api/game/battle")
async def game_battle(req: BattleActionRequest):
    game_id = req.game_id; player_id = req.player_id; action_type = req.action_type; card_uuid = req.card_uuid
    manager = GAMES.get(game_id); error_codes = CONST.get('ERROR_CODES', {}); battle_types = CONST.get('c_to_s_interface', {}).get('BATTLE_ACTIONS', {}).get('TYPES', {})
    if not manager: log_event("ERROR", "api.battle_action", f"Game not found: {game_id}", player=player_id); return build_game_result_hybrid(None, game_id, success=False, error_code=error_codes.get('GAME_NOT_FOUND', 'GAME_NOT_FOUND'), error_msg="Game not found")
    player = manager.p1 if player_id == manager.p1.name else manager.p2
    try:
        try: manager._validate_action(player, action_type)
        except Exception as ve:
            if action_type != battle_types.get('PASS', 'PASS'): raise ve
        if action_type == battle_types.get('SELECT_BLOCKER', 'SELECT_BLOCKER'):
            blocker = next((c for c in player.field if c.uuid == card_uuid), None); manager.handle_block(blocker)
        elif action_type == battle_types.get('SELECT_COUNTER', 'SELECT_COUNTER'):
            counter_card = next((c for c in player.hand if c.uuid == card_uuid), None); manager.apply_counter(player, counter_card)
        elif action_type == battle_types.get('PASS', 'PASS'): manager.apply_counter(player, None)
        return build_game_result_hybrid(manager, game_id, success=True)
    except Exception as e:
        log_event("ERROR", "game.battle_fail", traceback.format_exc(), player=player_id); return build_game_result_hybrid(manager, game_id, success=False, error_code=error_codes.get('INVALID_ACTION', 'INVALID_ACTION'), error_msg=str(e))

@app.get("/api/cards")
async def get_all_cards():
    try:
        if len(card_db.cards) < len(card_db.raw_db):
            for card_id in card_db.raw_db.keys(): card_db.get_card(card_id)
        cards_data = [c.to_dict() for c in card_db.cards.values()]; return {"success": True, "cards": cards_data}
    except Exception as e:
        log_event("ERROR", "api.get_cards_fail", traceback.format_exc(), player="system"); return {"success": False, "error": str(e)}

@app.post("/api/deck")
async def save_deck(deck_data: Dict[str, Any] = Body(...)):
    if not db: return {"success": False, "error": "Database not initialized"}
    try:
        doc_ref = db.collection("decks").document(deck_data["id"]) if "id" in deck_data and deck_data["id"] else db.collection("decks").document()
        save_data = {"id": doc_ref.id, "name": deck_data.get("name", "Untitled Deck"), "leader_id": deck_data.get("leader_id"), "card_uuids": deck_data.get("card_uuids", []), "don_uuids": deck_data.get("don_uuids", []), "created_at": firestore.SERVER_TIMESTAMP}
        doc_ref.set(save_data); log_event("INFO", "deck.save", f"Deck saved: {save_data['name']}", player="system", payload={"deck_id": doc_ref.id}); return {"success": True, "deck_id": doc_ref.id}
    except Exception as e:
        log_event("ERROR", "deck.save_fail", traceback.format_exc(), player="system"); return {"success": False, "error": str(e)}

@app.get("/api/deck/list")
async def list_decks():
    decks = []
    default_files = ["imu.json", "nami.json"]
    for filename in default_files:
        try:
            path = os.path.join(DATA_DIR, filename)
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                deck_data = data[0] if isinstance(data, list) and len(data) > 0 else data
                if isinstance(deck_data, dict):
                    formatted_deck = {"id": filename, "name": deck_data.get("name", filename.replace(".json", "")), "leader_id": None, "card_uuids": [], "don_uuids": [], "created_at": None}
                    if "leader" in deck_data: formatted_deck["leader_id"] = deck_data["leader"].get("number")
                    if "cards" in deck_data:
                        for card in deck_data["cards"]:
                            cid = card.get("number"); count = card.get("count", 1)
                            if cid: formatted_deck["card_uuids"].extend([cid] * count)
                    decks.append(formatted_deck)
        except Exception as e:
            log_event("WARNING", "deck.list_local_load_fail", f"Failed to load {filename}: {e}", player="system")
    if db:
        try:
            docs = db.collection("decks").order_by("created_at", direction=firestore.Query.DESCENDING).stream()
            for doc in docs:
                d = doc.to_dict()
                if "created_at" in d and d["created_at"]: d["created_at"] = str(d["created_at"])
                decks.append(d)
        except Exception as e:
            log_event("ERROR", "deck.list_db_fail", traceback.format_exc(), player="system")
    return {"success": True, "decks": decks}

@app.get("/api/sandbox/list")
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
                "created_at": getattr(mgr, "created_at", "N/A")
            })
        except Exception:
            continue
    return {"success": True, "games": games}

@app.post("/api/sandbox/create")
async def sandbox_create(req: Any = Body(...)):
    try:
        game_id = str(uuid.uuid4())
        log_event(level_key="INFO", action="sandbox.create", msg=f"Creating sandbox: {game_id}", payload=req, player="system")
        p1_source = req.get("p1_deck", ""); p2_source = req.get("p2_deck", "")
        if len(card_db.cards) < len(card_db.raw_db):
             for card_id in card_db.raw_db.keys(): card_db.get_card(card_id)
        p1_leader, p1_cards = load_deck_mixed(p1_source, req.get("p1_name", "P1")); p2_leader, p2_cards = load_deck_mixed(p2_source, req.get("p2_name", "P2"))
        if "SandboxManager" not in globals():
            raise ImportError("SandboxManager not loaded")
        manager = SandboxManager(p1_cards, p2_cards, p1_leader, p2_leader, req.get("p1_name", "P1"), req.get("p2_name", "P2"), room_name=req.get("room_name", "Custom Room"))
        SANDBOX_GAMES[manager.game_id] = manager
        return {"success": True, "game_id": manager.game_id, "game_state": manager.to_dict()}
    except Exception as e:
        log_event(level_key="ERROR", action="sandbox.create_fail", msg=traceback.format_exc(), player="system")
        return {"success": False, "error": str(e)}

@app.post("/api/sandbox/action")
async def sandbox_action(req: Dict[str, Any] = Body(...)):
    game_id = req.get("game_id"); manager = SANDBOX_GAMES.get(game_id)
    if not manager: return {"success": False, "error": "Sandbox game not found"}
    try:
        manager.process_action(req); new_state = manager.to_dict()
        await ws_manager.broadcast(game_id, {"type": "STATE_UPDATE", "state": new_state})
        return {"success": True, "game_id": game_id, "game_state": new_state}
    except Exception as e:
        log_event(level_key="ERROR", action="sandbox.action_fail", msg=traceback.format_exc(), player="system")
        return {"success": False, "error": str(e)}

@app.websocket("/ws/sandbox/{game_id}")
async def websocket_endpoint(websocket: WebSocket, game_id: str):
    await ws_manager.connect(websocket, game_id)
    try:
        while True: await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket, game_id)

@app.get("/health")
async def health(): return {"status": "ok", "constants_loaded": bool(CONST), "session_id": session_id_ctx.get()}
