import os
import uuid
import sys
import json
import random
import traceback
import asyncio
from datetime import datetime
from typing import Any, Dict, Optional, List, Union
from fastapi import FastAPI, Body, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
try:
    from google.cloud import firestore
except Exception:
    firestore = None

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
from opcg_sim.src.core import action_api
from opcg_sim.src.core import cpu_ai
from opcg_sim.src.core import cpu_opponent_model
from opcg_sim.src.core import cpu_self_plan
from opcg_sim.src.utils.loader import CardLoader
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

    async def disconnect(self, websocket: WebSocket, game_id: str):
        if game_id in self.active_connections:
            if websocket in self.active_connections[game_id]:
                self.active_connections[game_id].remove(websocket)
            if not self.active_connections[game_id]:
                del self.active_connections[game_id]
                log_event(level_key="INFO", action="sandbox.disconnect", msg=f"All connections closed for {game_id}. Starting 20-min grace period.", player="system")
                asyncio.create_task(self.delayed_cleanup(game_id, 1200))

    async def delayed_cleanup(self, game_id: str, delay: int):
        await asyncio.sleep(delay)
        if game_id not in self.active_connections or not self.active_connections[game_id]:
            if game_id in SANDBOX_GAMES:
                del SANDBOX_GAMES[game_id]
                log_event(level_key="INFO", action="sandbox.auto_delete", msg=f"Deleted sandbox {game_id} after grace period", player="system")

    async def broadcast(self, game_id: str, message: dict):
        if game_id in self.active_connections:
            for connection in self.active_connections[game_id]:
                try:
                    await connection.send_json(message)
                except Exception:
                    pass

ws_manager = ConnectionManager()


class GameConnectionManager:
    """ルールモードのオンライン対戦用 WebSocket 接続マネージャ。

    状態は「全情報を配信し、表示制御はフロント側で行う」方針のため、接続ごとの
    視点別シリアライズは行わず、同一ペイロードを全接続へブロードキャストする。
    """

    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, game_id: str):
        await websocket.accept()
        self.active_connections.setdefault(game_id, []).append(websocket)
        # 接続直後に現在のルーム/対局状態を本人へ送る
        try:
            if game_id in RULE_ROOMS:
                await websocket.send_json(build_rule_message(game_id))
        except Exception as e:
            log_event("ERROR", "game_ws.initial_state_fail", f"Failed to send initial state: {e}", player="system")

    def disconnect(self, websocket: WebSocket, game_id: str):
        conns = self.active_connections.get(game_id)
        if not conns:
            return
        if websocket in conns:
            conns.remove(websocket)
        if not conns:
            del self.active_connections[game_id]
            log_event("INFO", "game_ws.disconnect", f"All connections closed for rule game {game_id}. Grace period started.", player="system")
            asyncio.create_task(self.delayed_cleanup(game_id, 1200))

    async def delayed_cleanup(self, game_id: str, delay: int):
        await asyncio.sleep(delay)
        if not self.active_connections.get(game_id):
            RULE_ROOMS.pop(game_id, None)
            GAMES.pop(game_id, None)
            log_event("INFO", "game_ws.auto_delete", f"Deleted rule game {game_id} after grace period", player="system")

    def count(self, game_id: str) -> int:
        return len(self.active_connections.get(game_id, []))

    async def broadcast(self, game_id: str, message: dict):
        for connection in list(self.active_connections.get(game_id, [])):
            try:
                await connection.send_json(message)
            except Exception:
                pass


game_ws_manager = GameConnectionManager()

# ルールモード・オンライン対戦のルーム（ロビー）レジストリ。
# 各値: {game_id, room_name, created_at, status(WAITING/PLAYING/FINISHED),
#        ready{p1,p2}, decks{p1,p2:deck_id}, deck_preview{p1,p2:{leader_id,leader_name}}}
# 対局開始後の GameManager 本体は GAMES[game_id] に格納し、進行は既存の
# /api/game/action・/api/game/battle を共用する（差分は WS ブロードキャストのみ）。
RULE_ROOMS: Dict[str, Dict[str, Any]] = {}


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


async def broadcast_rule_state(game_id: str):
    """ルーム対局なら最新状態を全接続へ配信する（非ルーム対局では no-op）。"""
    if game_id not in RULE_ROOMS:
        return
    manager = GAMES.get(game_id)
    room = RULE_ROOMS[game_id]
    if manager and manager.winner:
        room["status"] = "FINISHED"
    await game_ws_manager.broadcast(game_id, build_rule_message(game_id))


def build_game_result_hybrid(manager: GameManager, game_id: str, success: bool = True, error_code: str = None, error_msg: str = None) -> Dict[str, Any]:
    player_keys = CONST.get('PLAYER_KEYS', {}); api_root_keys = CONST.get('API_ROOT_KEYS', {}); error_props = CONST.get('ERROR_PROPERTIES', {})
    p1_key = player_keys.get('P1', 'p1'); p2_key = player_keys.get('P2', 'p2')
    active_pid = "N/A"
    if manager: active_pid = p1_key if manager.turn_player == manager.p1 else p2_key
    battle_props = CONST.get('BATTLE_PROPERTIES', {})
    raw_game_state = {
        "game_id": game_id,
        "turn_info": {"turn_count": manager.turn_count if manager else 0, "current_phase": manager.phase.name if manager else "N/A", "active_player_id": active_pid, "winner": manager.winner if manager else None},
        "players": {p1_key: manager.p1.to_dict(is_my_turn=(manager.turn_player == manager.p1)) if manager else {}, p2_key: manager.p2.to_dict(is_my_turn=(manager.turn_player == manager.p2)) if manager else {}},
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
    return {api_root_keys.get('SUCCESS', 'success'): success, "game_id": game_id, api_root_keys.get('GAME_STATE', 'game_state'): validated_state, api_root_keys.get('PENDING_REQUEST', 'pending_request'): pending_req_data, api_root_keys.get('ERROR', 'error'): error_obj, "action_events": getattr(manager, 'action_events', []) if manager else []}

@app.middleware("http")
async def trace_logging_middleware(request: Request, call_next):
    s_id = request.headers.get("X-Session-ID") or request.query_params.get("sessionId")
    if not s_id: s_id = f"gen-{uuid.uuid4().hex[:8]}"
    token = session_id_ctx.set(s_id)
    try:
        response = await call_next(request); response.headers["X-Session-ID"] = s_id; return response
    finally: session_id_ctx.reset(token)

@app.options("/api/log")
async def options_log(): return {"status": "ok"}

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
# CPU 対戦のメタ情報レジストリ: {game_id: {"cpu_player_id": "p2", "difficulty": "hard"}}。
# GAMES[game_id] に GameManager 本体を、ここに CPU 側の識別子と難易度を保持する。
CPU_GAMES: Dict[str, Dict[str, Any]] = {}

card_db = CardLoader(CARD_DB_PATH); card_db.load()

# NOTE: 効果定義はカードテキストの自動解析（EffectParserV2）に一本化されている。

def load_deck_mixed(source_str: str, owner_id: str):
    if not source_str.startswith("db:"):
        raise ValueError(f"Unknown deck id: {source_str}")
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


def build_opp_profile_for_leader(leader_id: Optional[str]):
    """相手リーダーに紐づくテンプレートデッキ（cpu_templates）から相手プロファイルを作る（§2.5.4）。

    normal（リーダー推測）でのみ使用。テンプレ未登録・DB 不在なら None（保守モデルへフォールバック）。
    隠れ情報（相手の実デッキ・実手札）は読まず、リーダーに紐づくテンプレ構成のみを使う＝フェア。
    """
    if not leader_id or not db:
        return None
    try:
        docs = (db.collection("cpu_templates").where("leader_id", "==", leader_id)
                .order_by("created_at", direction=firestore.Query.DESCENDING).limit(1).stream())
        data = next((d.to_dict() for d in docs), None)
        if not data:
            return None
        masters = [card_db.get_card(cid) for cid in data.get("card_uuids", [])]
        masters = [m for m in masters if m is not None]
        return cpu_opponent_model.build_profile(masters)
    except Exception:
        log_event("ERROR", "cpu_template.profile_fail", traceback.format_exc(), player="system")
        return None

@app.options("/api/game/create")
async def options_game_create(): return {"status": "ok"}

def _resolve_first_player(value: Any, player1: Player, player2: Player) -> Optional[Player]:
    """リクエストの first_player 指定を先行 Player に解決する。
      "p1"/"p2" : 明示指定（ソロでプレイヤーが選択）
      "random"  : ランダム（CPU/対戦のコイントス用。結果は turn_info に反映される）
      その他/None: 従来通り既定（start_game 側で p1 先行）
    """
    if value == "random":
        return random.choice([player1, player2])
    if value == "p1":
        return player1
    if value == "p2":
        return player2
    return None

@app.post("/api/game/create")
async def game_create(req: Any = Body(...)):
    try:
        game_id = str(uuid.uuid4()); log_event(level_key="INFO", action="game.create", msg=f"Creating game: {game_id}", payload=req, player="system")
        p1_source = req.get("p1_deck", ""); p2_source = req.get("p2_deck", "")
        if len(card_db.cards) < len(card_db.raw_db):
             for card_id in card_db.raw_db.keys(): card_db.get_card(card_id)
        vs_cpu = bool(req.get("vs_cpu", False))
        # CPU 対戦時は p2 を CPU とし、デッキは cpu_deck（無指定なら p2_deck）を使う。
        if vs_cpu and req.get("cpu_deck"):
            p2_source = req.get("cpu_deck")
        p1_leader, p1_cards = load_deck_mixed(p1_source, req.get("p1_name", "P1")); p2_leader, p2_cards = load_deck_mixed(p2_source, req.get("p2_name", "P2"))
        player1 = Player(req.get("p1_name", "P1"), p1_cards, p1_leader); player2 = Player(req.get("p2_name", "P2"), p2_cards, p2_leader)
        # 先行プレイヤー: ソロは "p1"/"p2"、CPU は "random"（コイントス）。未指定は既定。
        first_player = _resolve_first_player(req.get("first_player"), player1, player2)
        manager = GameManager(player1, player2); manager.start_game(first_player); GAMES[game_id] = manager
        if vs_cpu:
            difficulty = req.get("cpu_difficulty", "normal")
            if difficulty not in ("easy", "normal", "hard"):
                difficulty = "normal"
            # リーダー推測（normal）用: 人間(p1) のリーダーからテンプレ相手プロファイルを引き当てて保持（§2.5.4）。
            opp_profile = None
            if difficulty == "normal" and p1_leader is not None:
                opp_profile = build_opp_profile_for_leader(p1_leader.master.card_id)
            # 自デッキ勝ち筋プラン（§2.5.5）: CPU(p2) の自デッキ構成から静的に分類して保持（normal/hard で使用）。
            self_plan = None
            if difficulty in ("normal", "hard"):
                try:
                    self_plan = cpu_self_plan.build_plan([ci.master for ci in p2_cards],
                                                         leader=p2_leader.master if p2_leader else None,
                                                         opp_profile=opp_profile)
                except Exception:
                    log_event("ERROR", "cpu_self_plan.fail", traceback.format_exc(), player="system")
            CPU_GAMES[game_id] = {"cpu_player_id": player2.name, "difficulty": difficulty,
                                  "opp_profile": opp_profile, "self_plan": self_plan}
            log_event("INFO", "game.cpu_create", f"CPU game {game_id} (difficulty={difficulty}, cpu={player2.name}, profile={'yes' if opp_profile else 'no'}, plan={self_plan.archetype if self_plan else 'none'})", player="system")
        return build_game_result_hybrid(manager, game_id)
    except Exception as e:
        log_event(level_key="ERROR", action="game.create_fail", msg=traceback.format_exc(), player="system"); return {"success": False, "game_id": "", "error": {"message": str(e)}}

@app.options("/api/game/action")
async def options_game_action(): return {"status": "ok"}

@app.post("/api/game/action")
async def game_action(req: Dict[str, Any] = Body(...)):
    action_type = req.get("action") or req.get("type"); player_id = req.get("player_id", "system")
    game_id = req.get("game_id"); manager = GAMES.get(game_id); error_codes = CONST.get('ERROR_CODES', {})
    if not manager: return build_game_result_hybrid(None, game_id, success=False, error_code=error_codes.get('GAME_NOT_FOUND', 'GAME_NOT_FOUND'), error_msg="指定されたゲームが見つかりません。")
    payload = req.get("payload") or req.get("full_payload") or {}
    try:
        manager.action_events = []
        # ディスパッチは action_api（CPU ドライバ・自己対戦ランナーと同一コアパス）へ委譲する。
        current_player = manager.p1 if player_id == manager.p1.name else manager.p2
        action_api.apply_game_action(manager, current_player, action_type, payload)
        result = build_game_result_hybrid(manager, game_id, success=True)
        await broadcast_rule_state(game_id)
        return result
    except Exception as e:
        log_event(level_key="ERROR", action="game.action_fail", msg=traceback.format_exc(), player=player_id, payload=req); return build_game_result_hybrid(manager, game_id, success=False, error_code=error_codes.get('INVALID_ACTION', 'INVALID_ACTION'), error_msg=str(e))

@app.options("/api/game/state")
async def options_game_state(): return {"status": "ok"}

@app.get("/api/game/state")
async def game_state_fetch(game_id: str):
    """現在の対局状態を読み取り専用で返す（盤面は一切変更しない）。

    オンライン対戦は対局の進行を WS ブロードキャストのみで相手へ伝えるため、
    片側が（モバイルのバックグラウンド化・通信瞬断などで）ブロードキャストを
    取りこぼすと、古い「相手待ち」状態のまま自力復帰できず停止して見える。
    待機側がこのエンドポイントを軽量ポーリングして最新状態へ再同期するための
    フォールバック経路（冪等・副作用なし）。ルーム対局は WS と同形の
    build_rule_message を返す。"""
    error_codes = CONST.get('ERROR_CODES', {})
    if game_id in RULE_ROOMS:
        return build_rule_message(game_id)
    manager = GAMES.get(game_id)
    if not manager:
        return build_game_result_hybrid(None, game_id, success=False, error_code=error_codes.get('GAME_NOT_FOUND', 'GAME_NOT_FOUND'), error_msg="指定されたゲームが見つかりません。")
    return build_game_result_hybrid(manager, game_id, success=True)

@app.options("/api/game/battle")
async def options_game_battle(): return {"status": "ok"}

@app.post("/api/game/battle")
async def game_battle(req: BattleActionRequest):
    game_id = req.game_id; player_id = req.player_id; action_type = req.action_type; card_uuid = req.card_uuid
    manager = GAMES.get(game_id); error_codes = CONST.get('ERROR_CODES', {}); battle_types = CONST.get('c_to_s_interface', {}).get('BATTLE_ACTIONS', {}).get('TYPES', {})
    if not manager: log_event("ERROR", "api.battle_action", f"Game not found: {game_id}", player=player_id); return build_game_result_hybrid(None, game_id, success=False, error_code=error_codes.get('GAME_NOT_FOUND', 'GAME_NOT_FOUND'), error_msg="Game not found")
    player = manager.p1 if player_id == manager.p1.name else manager.p2
    try:
        manager.action_events = []
        # ディスパッチは action_api（CPU ドライバ・自己対戦ランナーと同一コアパス）へ委譲する。
        action_api.apply_battle_action(manager, player, action_type, card_uuid)
        result = build_game_result_hybrid(manager, game_id, success=True)
        await broadcast_rule_state(game_id)
        return result
    except Exception as e:
        log_event("ERROR", "game.battle_fail", traceback.format_exc(), player=player_id); return build_game_result_hybrid(manager, game_id, success=False, error_code=error_codes.get('INVALID_ACTION', 'INVALID_ACTION'), error_msg=str(e))

@app.options("/api/game/cpu/step")
async def options_game_cpu_step(): return {"status": "ok"}

@app.post("/api/game/cpu/step")
async def game_cpu_step(req: Dict[str, Any] = Body(...)):
    """CPU 対戦で CPU(p2) の「次の 1 手」を適用して返す（ポーリング駆動）。

    レスポンスは通常の build_game_result_hybrid に加え:
      - cpu_acted: この呼び出しで CPU が行動したか
      - cpu_event: CPU が行った手の概要（action_events の先頭）
      - waiting_for: 'cpu'(続けて step を呼べ)/'human'/'human_decision'/'game_over'
    CPU が行動すべき状況でなければ cpu_acted=False で即返す（フロントはポーリング停止）。
    """
    game_id = req.get("game_id"); manager = GAMES.get(game_id); meta = CPU_GAMES.get(game_id)
    error_codes = CONST.get('ERROR_CODES', {})
    if not manager:
        return build_game_result_hybrid(None, game_id, success=False, error_code=error_codes.get('GAME_NOT_FOUND', 'GAME_NOT_FOUND'), error_msg="指定されたゲームが見つかりません。")
    if not meta:
        return build_game_result_hybrid(manager, game_id, success=False, error_code=error_codes.get('INVALID_ACTION', 'INVALID_ACTION'), error_msg="このゲームは CPU 対戦ではありません。")

    cpu_pid = meta["cpu_player_id"]; difficulty = meta.get("difficulty", "normal")
    cpu_player = manager.p1 if manager.p1.name == cpu_pid else manager.p2

    def _waiting_for() -> str:
        if manager.winner:
            return "game_over"
        pending = manager.get_pending_request()
        if pending and pending.get("player_id") == cpu_pid:
            return "cpu"
        if pending:
            # 人間(p1)宛の選択要求（メイン操作含む）。フロントは人間の入力を待つ。
            return "human_decision"
        return "human"

    cpu_acted = False; cpu_event = None
    try:
        manager.action_events = []
        if not manager.winner:
            pending = manager.get_pending_request()
            if pending and pending.get("player_id") == cpu_pid:
                turn_mem = meta.setdefault("turn_mem", {})
                move = cpu_ai.decide_guarded(manager, cpu_player, difficulty, mem=turn_mem,
                                             profile=meta.get("opp_profile"), plan=meta.get("self_plan"))
                if move is not None:
                    if move["kind"] == "battle":
                        action_api.apply_battle_action(manager, cpu_player, move["action_type"], move.get("card_uuid"))
                    else:
                        action_api.apply_game_action(manager, cpu_player, move["action_type"], move.get("payload", {}))
                    cpu_acted = True
                    cpu_event = manager.action_events[0] if manager.action_events else {"action": move["action_type"]}
        result = build_game_result_hybrid(manager, game_id, success=True)
        result["cpu_acted"] = cpu_acted
        result["cpu_event"] = cpu_event
        result["waiting_for"] = _waiting_for()
        await broadcast_rule_state(game_id)
        return result
    except Exception as e:
        log_event("ERROR", "game.cpu_step_fail", traceback.format_exc(), player=cpu_pid)
        return build_game_result_hybrid(manager, game_id, success=False, error_code=error_codes.get('INVALID_ACTION', 'INVALID_ACTION'), error_msg=str(e))

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

@app.delete("/api/deck/{deck_id}")
async def delete_deck(deck_id: str):
    if not db:
        return {"success": False, "error": "Database not initialized"}
    try:
        db.collection("decks").document(deck_id).delete()
        log_event("INFO", "deck.delete", f"Deck deleted: {deck_id}", player="system")
        return {"success": True, "deck_id": deck_id}
    except Exception as e:
        log_event("ERROR", "deck.delete_fail", traceback.format_exc(), player="system")
        return {"success": False, "error": str(e)}

@app.get("/api/deck/get")
async def get_deck(id: str):
    try:
        leader, cards = load_deck_mixed(id, "system")
        return {
            "success": True,
            "deck": {
                "leader": [leader.master.to_dict()] if leader else [],
                "cards": [c.master.to_dict() for c in cards]
            }
        }
    except Exception as e:
        log_event(level_key="ERROR", action="api.get_deck_fail", msg=str(e), player="system")
        return {"success": False, "error": str(e)}

@app.get("/api/deck/list")
async def list_decks():
    decks = []
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

# --- CPU 相手モデル（テンプレートデッキ）: deck と同形・leader_id 引き当て（docs/SPEC.md §2.5.4） ---

@app.post("/api/cpu_template")
async def save_cpu_template(tpl_data: Dict[str, Any] = Body(...)):
    if not db: return {"success": False, "error": "Database not initialized"}
    try:
        doc_ref = (db.collection("cpu_templates").document(tpl_data["id"])
                   if tpl_data.get("id") else db.collection("cpu_templates").document())
        save_data = {"id": doc_ref.id, "name": tpl_data.get("name", "Untitled Template"),
                     "leader_id": tpl_data.get("leader_id"), "card_uuids": tpl_data.get("card_uuids", []),
                     "don_uuids": tpl_data.get("don_uuids", []), "created_at": firestore.SERVER_TIMESTAMP}
        doc_ref.set(save_data)
        log_event("INFO", "cpu_template.save", f"Template saved: {save_data['name']}", player="system", payload={"template_id": doc_ref.id})
        return {"success": True, "template_id": doc_ref.id}
    except Exception as e:
        log_event("ERROR", "cpu_template.save_fail", traceback.format_exc(), player="system")
        return {"success": False, "error": str(e)}

@app.delete("/api/cpu_template/{template_id}")
async def delete_cpu_template(template_id: str):
    if not db: return {"success": False, "error": "Database not initialized"}
    try:
        db.collection("cpu_templates").document(template_id).delete()
        log_event("INFO", "cpu_template.delete", f"Template deleted: {template_id}", player="system")
        return {"success": True, "template_id": template_id}
    except Exception as e:
        log_event("ERROR", "cpu_template.delete_fail", traceback.format_exc(), player="system")
        return {"success": False, "error": str(e)}

@app.get("/api/cpu_template/get")
async def get_cpu_template(id: str):
    if not db: return {"success": False, "error": "Database not initialized"}
    try:
        doc = db.collection("cpu_templates").document(id).get()
        if not doc.exists:
            return {"success": False, "error": f"Template not found: {id}"}
        d = doc.to_dict()
        if d.get("created_at"): d["created_at"] = str(d["created_at"])
        return {"success": True, "template": d}
    except Exception as e:
        log_event("ERROR", "cpu_template.get_fail", traceback.format_exc(), player="system")
        return {"success": False, "error": str(e)}

@app.get("/api/cpu_template/list")
async def list_cpu_templates():
    templates = []
    if db:
        try:
            docs = db.collection("cpu_templates").order_by("created_at", direction=firestore.Query.DESCENDING).stream()
            for doc in docs:
                d = doc.to_dict()
                if d.get("created_at"): d["created_at"] = str(d["created_at"])
                templates.append(d)
        except Exception as e:
            log_event("ERROR", "cpu_template.list_db_fail", traceback.format_exc(), player="system")
    return {"success": True, "templates": templates}

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
                "created_at": getattr(mgr, "created_at", "N/A"),
                "active_connections": len(ws_manager.active_connections.get(gid, [])),
                "status": mgr.status,
            })
        except Exception:
            continue
    return {"success": True, "games": games}

@app.post("/api/sandbox/create")
async def sandbox_create(req: Any = Body(...)):
    try:
        game_id = str(uuid.uuid4()); log_event(level_key="INFO", action="sandbox.create", msg=f"Creating sandbox: {game_id}", payload=req, player="system")
        p1_name = req.get("p1_name", "P1"); p2_name = req.get("p2_name", "P2")
        if "SandboxManager" not in globals(): raise ImportError("SandboxManager not loaded")
        manager = SandboxManager(p1_name=p1_name, p2_name=p2_name, room_name=req.get("room_name", "Custom Room"))
        SANDBOX_GAMES[manager.game_id] = manager
        return {"success": True, "game_id": manager.game_id, "game_state": manager.to_dict()}
    except Exception as e:
        log_event(level_key="ERROR", action="sandbox.create_fail", msg=traceback.format_exc(), player="system"); return {"success": False, "error": str(e)}

@app.post("/api/sandbox/action")
async def sandbox_action(req: Dict[str, Any] = Body(...)):
    game_id = req.get("game_id"); manager = SANDBOX_GAMES.get(game_id)
    if not manager: return {"success": False, "error": "Sandbox game not found"}
    act_type = req.get("action_type"); pid = req.get("player_id")
    try:
        if act_type == "SET_DECK":
            deck_id = req.get("deck_id"); owner_name = manager.state[pid]["name"]
            leader, cards = load_deck_mixed(deck_id, owner_name)
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
        log_event(level_key="ERROR", action="sandbox.action_fail", msg=traceback.format_exc(), player="system"); return {"success": False, "error": str(e)}

@app.websocket("/ws/sandbox/{game_id}")
async def websocket_endpoint(websocket: WebSocket, game_id: str):
    await ws_manager.connect(websocket, game_id)
    try:
        while True: await websocket.receive_text()
    except WebSocketDisconnect:
        await ws_manager.disconnect(websocket, game_id)


# ============================================================================
# ルールモード・オンライン対戦（ロビー / ルーム）
#   フリーモード(sandbox)のルーム制を踏襲しつつ、対局進行は本物のルールエンジン
#   (GameManager) を使う。状態同期は WebSocket(/ws/game/{id}) で全情報を配信し、
#   相手手札の非表示などの表示制御はフロント側で行う方針。
# ============================================================================

@app.options("/api/rule/create")
async def options_rule_create(): return {"status": "ok"}

@app.post("/api/rule/create")
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
        log_event("INFO", "rule.create", f"Creating rule room: {game_id}", payload=req, player="system")
        return {"success": True, "game_id": game_id, **_rule_room_meta(game_id), "game_state": None}
    except Exception as e:
        log_event("ERROR", "rule.create_fail", traceback.format_exc(), player="system")
        return {"success": False, "error": str(e)}

@app.get("/api/rule/list")
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

def _deck_preview(deck_id: str, owner_id: str) -> Dict[str, Any]:
    """ロビー表示用にデッキのリーダー情報のみ抽出する。"""
    try:
        leader, _cards = load_deck_mixed(deck_id, owner_id)
        if leader:
            return {"leader_id": leader.master.card_id, "leader_name": leader.master.name}
    except Exception as e:
        log_event("WARNING", "rule.deck_preview_fail", f"{deck_id}: {e}", player="system")
    return None

@app.options("/api/rule/action")
async def options_rule_action(): return {"status": "ok"}

@app.post("/api/rule/action")
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
            room["deck_preview"][pid] = _deck_preview(deck_id, pid)
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
            if len(card_db.cards) < len(card_db.raw_db):
                for card_id in card_db.raw_db.keys(): card_db.get_card(card_id)
            p1_leader, p1_cards = load_deck_mixed(room["decks"]["p1"], "p1")
            p2_leader, p2_cards = load_deck_mixed(room["decks"]["p2"], "p2")
            player1 = Player("p1", p1_cards, p1_leader); player2 = Player("p2", p2_cards, p2_leader)
            # 対戦モードの先行はランダム（コイントス）。結果は turn_info で両クライアントへ broadcast。
            first_player = random.choice([player1, player2])
            manager = GameManager(player1, player2); manager.start_game(first_player)
            GAMES[game_id] = manager
            room["status"] = "PLAYING"
            log_event("INFO", "rule.coin", f"First player: {first_player.name}", player="system")
            log_event("INFO", "rule.start", f"Rule game started: {game_id}", player="system")
        else:
            return {"success": False, "error": f"Unknown rule action: {act}"}

        await game_ws_manager.broadcast(game_id, build_rule_message(game_id))
        return {"success": True, "game_id": game_id, **build_rule_message(game_id)}
    except Exception as e:
        log_event("ERROR", "rule.action_fail", traceback.format_exc(), player="system")
        return {"success": False, "error": str(e)}

@app.websocket("/ws/game/{game_id}")
async def game_websocket_endpoint(websocket: WebSocket, game_id: str):
    await game_ws_manager.connect(websocket, game_id)
    try:
        while True: await websocket.receive_text()
    except WebSocketDisconnect:
        game_ws_manager.disconnect(websocket, game_id)

@app.get("/health")
async def health(): return {"status": "ok", "constants_loaded": bool(CONST), "session_id": session_id_ctx.get()}
