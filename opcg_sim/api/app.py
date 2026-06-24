import os
import uuid
import sys
import json
import random
import traceback
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, Optional, List, Union
from fastapi import FastAPI, Body, Request, Response, WebSocket, WebSocketDisconnect
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
from opcg_sim.src.core.gamestate import Player, GameManager
from opcg_sim.src.core import action_api
from opcg_sim.src.core import cpu_ai
from opcg_sim.src.core import cpu_mcts
from opcg_sim.src.core import cpu_self_plan
from opcg_sim.src.core import cpu_value_data
from opcg_sim.api import decide_client
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

@asynccontextmanager
async def _lifespan(_app):
    # 方式B: PyPy 探索ワーカーを常駐起動（OPCG_PYPY_WORKER=1 のときのみ）。JIT を常にウォームに保つ。
    # 未起動・失敗でも decide_client がインプロセス実行へフォールバックするので可用性は不変。
    try:
        decide_client.spawn_worker()
    except Exception:
        pass
    yield


app = FastAPI(title="OPCG Simulator API v1.7", lifespan=_lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], expose_headers=["ETag"])

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
                asyncio.create_task(self.delayed_cleanup(game_id, 1200))

    async def delayed_cleanup(self, game_id: str, delay: int):
        await asyncio.sleep(delay)
        if game_id not in self.active_connections or not self.active_connections[game_id]:
            if game_id in SANDBOX_GAMES:
                del SANDBOX_GAMES[game_id]

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
            pass

    def disconnect(self, websocket: WebSocket, game_id: str):
        conns = self.active_connections.get(game_id)
        if not conns:
            return
        if websocket in conns:
            conns.remove(websocket)
        if not conns:
            del self.active_connections[game_id]
            asyncio.create_task(self.delayed_cleanup(game_id, 1200))

    async def delayed_cleanup(self, game_id: str, delay: int):
        await asyncio.sleep(delay)
        if not self.active_connections.get(game_id):
            RULE_ROOMS.pop(game_id, None)
            GAMES.pop(game_id, None)

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
        except Exception: validated_state = raw_game_state
    pending_req_data = None
    if manager and success:
        pending_obj = manager.get_pending_request()
        if pending_obj:
            try: pending_req_data = PendingRequestSchema(**pending_obj).model_dump(by_alias=True)
            except Exception: pending_req_data = pending_obj
    error_obj = None
    if not success: error_obj = {error_props.get('CODE', 'code'): error_code, error_props.get('MESSAGE', 'message'): error_msg}
    return {api_root_keys.get('SUCCESS', 'success'): success, "game_id": game_id, api_root_keys.get('GAME_STATE', 'game_state'): validated_state, api_root_keys.get('PENDING_REQUEST', 'pending_request'): pending_req_data, api_root_keys.get('ERROR', 'error'): error_obj, "action_events": getattr(manager, 'action_events', []) if manager else []}

GAMES: Dict[str, GameManager] = {}
SANDBOX_GAMES: Dict[str, 'SandboxManager'] = {}
# CPU 対戦のメタ情報レジストリ: {game_id: {"cpu_player_id": "p2", "difficulty": "hard"}}。
# GAMES[game_id] に GameManager 本体を、ここに CPU 側の識別子と難易度を保持する。
CPU_GAMES: Dict[str, Dict[str, Any]] = {}

card_db = CardLoader(CARD_DB_PATH); card_db.load()
# ビルド時に生成したパース済みキャッシュがあれば採用し、コールドスタート時の
# 全件パース(~1.8s)を回避する。整合しない/無い場合は従来どおり遅延パースに劣化する。
card_db.load_cache()

# /api/cards の条件付きGET(ETag)用。カードDBの内容が変わると変化する。
CARDS_ETAG = f'"{card_db.db_hash()}"'

def _compute_image_version() -> str:
    """カード画像のキャッシュ版数。

    カードDB(opcg_cards.json)の内容ハッシュから自動導出する。新弾追加など
    画像をまとめて更新する場面ではカードDBも更新されるため、人手で版数を
    上げなくても版数が自動で切り替わる（＝古い画像キャッシュが確実に無効化される）。
    カードデータを変えず画像のみ差し替える稀なケース用に IMAGE_VERSION_SALT で
    手動上書きできる余地も残す。
    """
    import hashlib
    h = hashlib.md5()
    try:
        with open(CARD_DB_PATH, "rb") as f:
            h.update(f.read())
    except OSError:
        pass
    h.update(os.getenv("IMAGE_VERSION_SALT", "").encode())
    return h.hexdigest()[:8]

IMAGE_VERSION = _compute_image_version()

# NOTE: 効果定義はカードテキストの自動解析（EffectParserV2）に一本化されている。

def _load_deck_doc(source_str: str) -> Dict[str, Any]:
    """`db:<id>` 形式のデッキIDから Firestore のデッキドキュメント(dict)を取得する。"""
    if not source_str.startswith("db:"):
        raise ValueError(f"Unknown deck id: {source_str}")
    if not db: raise ValueError("Firestore is not initialized.")
    deck_id = source_str[3:]; doc = db.collection("decks").document(deck_id).get()
    if not doc.exists: raise ValueError(f"Deck ID not found: {deck_id}")
    return doc.to_dict()

def load_deck_mixed(source_str: str, owner_id: str):
    deck_id = source_str[3:] if source_str.startswith("db:") else source_str
    data = _load_deck_doc(source_str); leader_id = data.get("leader_id"); card_uuids = data.get("card_uuids", [])
    leader_inst = None
    if leader_id:
        master = card_db.get_card(leader_id)
        if master: leader_inst = CardInstance(master, owner_id)
    cards_inst = [CardInstance(m, owner_id) for cid in card_uuids if (m := card_db.get_card(cid))]
    return leader_inst, cards_inst


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

# === リプレイ種＋CPU思考トレース（実アプリ対局・Phase 2） ============================
# すべて opt-in（create リクエストの cpu_trace=true）でのみ作動し、未指定の本番対局には
# 一切の追加処理・レイテンシ・挙動変化を与えない（トレースは観測専用＝進行不変）。
REPLAY_SCHEMA = "opcg-replay/v1"


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


def _capture_value_samples(meta, manager):
    """traced 対局のターン境界で両者視点の特徴を貯める（価値学習データ・人間ログ活用(a)/(A)）。

    opt-in（cpu_trace）時のみ作動＝本番対局には一切のオーバーヘッド・挙動変化なし。**アクション適用後**に
    呼ぶ。ラベルは未確定のまま `{"f","p"}` で貯め、終局時に replay エンドポイントで勝者から確定する。
    境界検出（turn_count 変化）はオフライン自己対戦（collect_value_data）と同一規約。例外安全。
    """
    if not _replay_enabled(meta):
        return
    try:
        prev = meta.get("_value_prev_turn")
        if prev is None:                       # 初回＝基準ターンを記録するだけ（まだ境界でない）
            meta["_value_prev_turn"] = manager.turn_count
            return
        if manager.turn_count != prev:
            meta["_value_prev_turn"] = manager.turn_count
            meta.setdefault("value_samples", []).extend(
                cpu_value_data.turn_boundary_samples(manager))
    except Exception:
        pass


@app.post("/api/game/create")
async def game_create(req: Any = Body(...)):
    try:
        game_id = str(uuid.uuid4()); 
        p1_source = req.get("p1_deck", ""); p2_source = req.get("p2_deck", "")
        if len(card_db.cards) < len(card_db.raw_db):
             for card_id in card_db.raw_db.keys(): card_db.get_card(card_id)
        vs_cpu = bool(req.get("vs_cpu", False))
        # CPU 対戦時は p2 を CPU とし、デッキは cpu_deck（無指定なら p2_deck）を使う。
        if vs_cpu and req.get("cpu_deck"):
            p2_source = req.get("cpu_deck")
        p1_leader, p1_cards = load_deck_mixed(p1_source, req.get("p1_name", "P1")); p2_leader, p2_cards = load_deck_mixed(p2_source, req.get("p2_name", "P2"))
        player1 = Player(req.get("p1_name", "P1"), p1_cards, p1_leader); player2 = Player(req.get("p2_name", "P2"), p2_cards, p2_leader)
        # リプレイ種（opt-in）: cpu_trace 指定時のみ seed を固定し、コイントス＋シャッフルを再現可能にする。
        # 未指定の本番対局は seed を触らない＝従来の乱数挙動を完全に維持する。
        cpu_trace = bool(req.get("cpu_trace", False)) and vs_cpu
        replay_seed = None
        if cpu_trace:
            replay_seed = int(req.get("seed")) if req.get("seed") is not None else random.randrange(2**63)
            random.seed(replay_seed)
        # 先行プレイヤー: ソロは "p1"/"p2"、CPU は "random"（コイントス）。未指定は既定。
        first_player = _resolve_first_player(req.get("first_player"), player1, player2)
        manager = GameManager(player1, player2); manager.start_game(first_player); GAMES[game_id] = manager
        if vs_cpu:
            # CPU は **hard（α-β＋ビーム）** と **expert（MCTS・§2.5.7）** の2系統のみ（easy/normal は廃止）。
            difficulty = req.get("cpu_difficulty", "expert")
            if difficulty not in ("hard", "expert"):
                difficulty = "expert"
            # 自デッキ勝ち筋プラン（§2.5.5）: CPU(p2) の自デッキ構成から静的に分類して保持（hard で使用）。
            self_plan = None
            if difficulty == "hard":
                try:
                    self_plan = cpu_self_plan.build_plan([ci.master for ci in p2_cards],
                                                         leader=p2_leader.master if p2_leader else None)
                except Exception:
                    pass
            CPU_GAMES[game_id] = {"cpu_player_id": player2.name, "difficulty": difficulty,
                                  "opp_profile": None, "self_plan": self_plan}
            if cpu_trace:
                # リプレイ種＋思考ログの器を用意する（opt-in 時のみ）。
                CPU_GAMES[game_id].update({
                    "cpu_trace": True, "seed": replay_seed,
                    "first_player": first_player.name if first_player else None,
                    "leaders": {"p1": p1_leader.master.card_id if p1_leader else None,
                                "p2": p2_leader.master.card_id if p2_leader else None},
                    "decks": {"p1": [ci.master.card_id for ci in p1_cards],
                              "p2": [ci.master.card_id for ci in p2_cards]},
                    "actions": [], "decisions": [], "value_samples": [],
                })
        return build_game_result_hybrid(manager, game_id)
    except Exception as e:
        return {"success": False, "game_id": "", "error": {"message": str(e)}}

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
        _meta = CPU_GAMES.get(game_id)
        _src = "cpu" if (_meta and player_id == _meta.get("cpu_player_id")) else "human"
        _replay_record_action(_meta, manager, _src, player_id, {"action_type": action_type, "payload": payload})
        action_api.apply_game_action(manager, current_player, action_type, payload)
        _capture_value_samples(_meta, manager)
        result = build_game_result_hybrid(manager, game_id, success=True)
        await broadcast_rule_state(game_id)
        _kick_ponder(game_id)     # ⑥-a: 制御が CPU へ移ったら次手番の計画を前倒し（既定 OFF）
        _kick_speculate(game_id)  # ⑥-b: 人間 MAIN 継続中なら「今エンドしたら」を投機（既定 OFF）
        return result
    except Exception as e:
        return build_game_result_hybrid(manager, game_id, success=False, error_code=error_codes.get('INVALID_ACTION', 'INVALID_ACTION'), error_msg=str(e))

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
    if not manager: return build_game_result_hybrid(None, game_id, success=False, error_code=error_codes.get('GAME_NOT_FOUND', 'GAME_NOT_FOUND'), error_msg="Game not found")
    player = manager.p1 if player_id == manager.p1.name else manager.p2
    try:
        manager.action_events = []
        # ディスパッチは action_api（CPU ドライバ・自己対戦ランナーと同一コアパス）へ委譲する。
        _meta = CPU_GAMES.get(game_id)
        _src = "cpu" if (_meta and player_id == _meta.get("cpu_player_id")) else "human"
        _replay_record_action(_meta, manager, _src, player_id, {"action_type": action_type, "card_uuid": card_uuid})
        action_api.apply_battle_action(manager, player, action_type, card_uuid)
        _capture_value_samples(_meta, manager)
        result = build_game_result_hybrid(manager, game_id, success=True)
        await broadcast_rule_state(game_id)
        _kick_ponder(game_id)     # ⑥-a: 制御が CPU へ移ったら次手番の計画を前倒し（既定 OFF）
        _kick_speculate(game_id)  # ⑥-b: 人間 MAIN 継続中なら「今エンドしたら」を投機（既定 OFF）
        return result
    except Exception as e:
        return build_game_result_hybrid(manager, game_id, success=False, error_code=error_codes.get('INVALID_ACTION', 'INVALID_ACTION'), error_msg=str(e))

def _ponder_enabled() -> bool:
    """Phase 3 ⑥-a 先行計画（pondering）の作動条件。①計画キャッシュ配下のオプトイン（既定 OFF・
    本番体感最適化のみ）。OPCG_PLAN_CACHE=1（①の replay 経路）かつ OPCG_PONDER=1 のとき作動。"""
    return (os.environ.get("OPCG_PLAN_CACHE", "0") == "1"
            and os.environ.get("OPCG_PONDER", "0") == "1")


def _plan_segment(manager, cpu_player, difficulty, mem=None, profile=None, plan=None):
    """難易度に応じた「このターンの計画手列」を返す共通ディスパッチ（ポンダリング/計画キャッシュの単一の真実源）。

    - `expert`: **MCTS（ターン粒度マクロ木・公平モード）をインプロセスで実行**（`cpu_mcts.mcts_plan_turn`）。
      相手手札を覗かない（determinize=True）＝人間相手として公平。PyPy ワーカーは α-β 用なので使わない。
    - それ以外（easy/normal/hard）: 従来どおり α-β を `decide_client`（PyPy ワーカー）でセグメント計画。
    どちらも「手 dict のリスト（=このターンの連続手番）」を返す＝後段のキャッシュ/replay/ポンダリングは共通。
    """
    if difficulty == "expert":
        # レイテンシ上限を壁時計デッドラインで保証（大盤面で MCTS が数十秒に膨らむのを防ぐ）。
        # 既定 2000ms・OPCG_MCTS_DEADLINE_MS で調整可。horizon は OPCG_MCTS_HORIZON（既定2＝大盤面の発散抑制）。
        deadline_ms = int(os.environ.get("OPCG_MCTS_DEADLINE_MS", "2000"))
        horizon = int(os.environ.get("OPCG_MCTS_HORIZON", "2"))
        return cpu_mcts.mcts_plan_turn(manager, cpu_player, "hard", random,
                                       horizon=horizon, profile=profile, plan=plan,
                                       determinize=True, deadline_ms=deadline_ms)
    return decide_client.plan_segment(manager, cpu_player, difficulty,
                                      mem=mem, profile=profile, plan=plan)


async def _ponder_plan(game_id: str) -> None:
    """Phase 3 ⑥-a: 人間の手番処理で制御が CPU へ移った瞬間、CPU セグメントの計画を**前倒し**で計算して
    `meta["plan_cache"]["queue"]` を温める（次の /cpu/step で即時 replay＝CPU 初手の待ちを消す）。

    計算は `plan_segment`（PyPy ワーカー＝別プロセス）へ `asyncio.to_thread` でオフロードし、イベント
    ループを塞がない。①と同じ「decide の決定的結果の前倒し」に留め、合法性ゲート（`_cached_cpu_move`）が
    stale を安全に弾く＝挙動不変。例外時は queue を空にして通常 decide へフォールバックさせる（安全側）。
    """
    manager = GAMES.get(game_id); meta = CPU_GAMES.get(game_id)
    if not manager or not meta:
        return
    cache = meta.setdefault("plan_cache", {})
    try:
        cpu_pid = meta["cpu_player_id"]; difficulty = meta.get("difficulty", "expert")
        turn_mem = meta.setdefault("turn_mem", {})
        # live 盤面をそのまま OS スレッドへ渡すと、スレッド側の deepcopy（plan_turn 内 clone / ワーカーへの
        # pickle）がメインスレッドの盤面変更と競合する（読み取り中の書き換え）。**メインスレッドで原子的に
        # clone**してから渡し、スレッドは隔離されたスナップショットだけに触れる（_kick_speculate と同方針）。
        snap = manager.clone()
        cpu_player = snap.p1 if snap.p1.name == cpu_pid else snap.p2
        actions = await asyncio.to_thread(
            _plan_segment, snap, cpu_player, difficulty,
            mem=turn_mem, profile=meta.get("opp_profile"), plan=meta.get("self_plan"))
        cache["queue"] = actions or None
    except Exception:
        cache["queue"] = None
    finally:
        cache["task"] = None


def _kick_ponder(game_id: str) -> None:
    """人間アクション適用後、pending が CPU 手番なら先行計画タスクを起動する（二重起動防止・既定 OFF）。

    旧 queue は前提が変わったので破棄してから焼き直す。イベントループ外（同期テスト等）では `create_task`
    が起動できないため no-op（pondering は本番のみ＝決定性・既存テストへ影響なし）。"""
    if not _ponder_enabled():
        return
    manager = GAMES.get(game_id); meta = CPU_GAMES.get(game_id)
    if not manager or not meta or manager.winner is not None:
        return
    pending = manager.get_pending_request()
    cpu_pid = meta.get("cpu_player_id")
    if not pending or pending.get("player_id") != cpu_pid:
        return
    cache = meta.setdefault("plan_cache", {})
    # ⑥-b: 「人間が今エンドしたら」を投機済み（spec_queue）で、実盤面でも先頭が合法なら昇格＝投機ヒット
    # （CPU 初手の待ちすら消える）。外れ/未完なら下の ⑥-a（実盤面の先行計画）へ。合法性ゲートが採否を担保。
    spec = cache.pop("spec_queue", None)
    if spec:
        cpu_player = manager.p1 if manager.p1.name == cpu_pid else manager.p2
        legal_sigs = {cpu_ai._move_sig(m) for m in manager.get_legal_actions(cpu_player)}
        if cpu_ai._move_sig(spec[0]) in legal_sigs:
            cache["queue"] = spec
            cache["spec_hits"] = cache.get("spec_hits", 0) + 1
            return  # 投機が当たった＝再計画不要
        cache["spec_misses"] = cache.get("spec_misses", 0) + 1
    if cache.get("task") is not None:
        return  # 既に先行計画が走行中
    cache["queue"] = None
    try:
        cache["task"] = asyncio.create_task(_ponder_plan(game_id))
    except RuntimeError:
        cache["task"] = None  # 実行中のイベントループが無い（テスト等）＝起動しない


def _speculate_enabled() -> bool:
    """Phase 3 ⑥-b 投機ポンダリングの作動条件。⑥-a（OPCG_PONDER）配下のさらなるオプトイン
    （OPCG_PONDER_SPEC=1）。既定 OFF＝従来挙動完全同値。当たり率を計測してから本採用を判断する。"""
    return _ponder_enabled() and os.environ.get("OPCG_PONDER_SPEC", "0") == "1"


def _speculate_compute(clone, human_pid, cpu_pid, difficulty, profile, plan):
    """⑥-b 投機の本体（`to_thread` で別スレッド実行）。**クローン上で**人間の TURN_END を仮適用し、
    pending が素直に CPU 手番へ移ったら CPU セグメントを計画して返す（介在する人間決定があれば None）。
    live 盤面には一切触れない（呼び出し側がメインスレッドで原子的に clone 済み）。"""
    human = clone.p1 if clone.p1.name == human_pid else clone.p2
    clone.action_events = []
    action_api.apply_game_action(clone, human, "TURN_END", {})
    pa = clone.pending_actor_action()
    if not pa or pa[0] != cpu_pid:
        return None
    cpu_player = clone.p1 if clone.p1.name == cpu_pid else clone.p2
    return _plan_segment(clone, cpu_player, difficulty,
                         mem={}, profile=profile, plan=plan)


async def _speculate_plan(game_id: str, clone, human_pid: str, gen: int) -> None:
    """⑥-b: 「人間が今エンドしたら」の CPU 計画を投機して `spec_queue` に保持（次の TURN_END で昇格判定）。

    計算は `_speculate_compute` を `to_thread` でオフロード（plan は別プロセスのワーカー）。世代 `gen` が
    最新でなければ（人間がさらに動いて盤面が変わった＝supersede）結果は捨てる。使い捨て clone・使い捨て mem
    ＝live 盤面/turn_mem 不変。採否は最終的に `_kick_ponder` の合法性ゲートが担保する。"""
    meta = CPU_GAMES.get(game_id)
    if not meta:
        return
    cache = meta.setdefault("plan_cache", {})
    try:
        cpu_pid = meta["cpu_player_id"]; difficulty = meta.get("difficulty", "expert")
        result = await asyncio.to_thread(
            _speculate_compute, clone, human_pid, cpu_pid, difficulty,
            meta.get("opp_profile"), meta.get("self_plan"))
        if cache.get("spec_gen") == gen:        # まだ最新の投機なら採用
            cache["spec_queue"] = result or None
    except Exception:
        if cache.get("spec_gen") == gen:
            cache["spec_queue"] = None
    finally:
        if cache.get("spec_gen") == gen:
            cache["spec_task"] = None


def _kick_speculate(game_id: str) -> None:
    """人間の MAIN 手番（TURN_END が合法）の最中に「今エンドしたら」を投機する（⑥-b・既定 OFF）。

    新しい人間アクションのたびに世代 `spec_gen` を進めて旧投機を supersede（1 ゲーム 1 タスク＝本数ゲート）。
    clone は**メインスレッドで原子的**に取り（読み書き競合なし）、重い計算だけを task へ逃がす。"""
    if not _speculate_enabled():
        return
    manager = GAMES.get(game_id); meta = CPU_GAMES.get(game_id)
    if not manager or not meta or manager.winner is not None:
        return
    pending = manager.get_pending_request()
    cpu_pid = meta.get("cpu_player_id")
    # 人間（=CPU でない側）の MAIN_ACTION 決定点のときだけ投機（TURN_END が合法な静止点）。
    if not pending or pending.get("player_id") == cpu_pid or pending.get("action") != "MAIN_ACTION":
        return
    cache = meta.setdefault("plan_cache", {})
    try:
        clone = manager.clone()  # メインスレッドで原子的に隔離＝以降 task が触れても競合しない
    except Exception:
        return
    gen = cache.get("spec_gen", 0) + 1
    cache["spec_gen"] = gen
    human_pid = manager.p1.name if manager.p1.name != cpu_pid else manager.p2.name
    try:
        cache["spec_task"] = asyncio.create_task(_speculate_plan(game_id, clone, human_pid, gen))
    except RuntimeError:
        cache["spec_task"] = None  # 実行中のイベントループが無い（テスト等）＝起動しない


def _cached_cpu_move(manager, cpu_player, difficulty, meta, turn_mem):
    """Phase 3 ① 計画キャッシュ（本番体感最適化）: 対局ごとの `meta["plan_cache"]` を用い、
    次の計画手が現局面で**合法なら即返す**（探索なし＝即時 replay・ワーカー往復なし）。ミス/前提崩れ
    （相手の介入で前提が変わった等）なら `plan_segment` でセグメントを再計画してキャッシュ。先頭手が
    現局面で不正なら None を返し、呼び出し側が通常 `decide` にフォールバック（**合法性検証で常に安全**）。
    """
    cache = meta.setdefault("plan_cache", {})
    legal = manager.get_legal_actions(cpu_player)
    legal_by_sig = {cpu_ai._move_sig(m): m for m in legal}
    q = cache.get("queue")
    if q:
        sig = cpu_ai._move_sig(q[0])
        if sig in legal_by_sig:
            cache["queue"] = q[1:]
            return legal_by_sig[sig]
        cache["queue"] = None  # 前提崩れ＝破棄して再計画
    actions = _plan_segment(manager, cpu_player, difficulty, mem=turn_mem,
                            profile=meta.get("opp_profile"), plan=meta.get("self_plan"))
    if actions:
        sig = cpu_ai._move_sig(actions[0])
        if sig in legal_by_sig:
            cache["queue"] = actions[1:]
            return legal_by_sig[sig]
    cache["queue"] = None
    return None


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

    cpu_pid = meta["cpu_player_id"]; difficulty = meta.get("difficulty", "expert")
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
                # ⑥-a: 先行計画（pondering）が走行中なら完了を待つ（warm な queue を使う＝二重計算/競合回避・既定 OFF）。
                _ptask = meta.get("plan_cache", {}).get("task")
                if _ptask is not None:
                    try:
                        await _ptask
                    except Exception:
                        pass
                trace_on = _replay_enabled(meta)
                tr = {} if trace_on else None
                # ライブ採取は軽量トレース（read_ahead=読み筋は省く）＝CPU 思考の遅延を抑える。
                # 読み筋はオフライン（cpu_replay.py／リプレイ種）でのみ採る。
                # 探索（decide）は方式B: OPCG_PYPY_WORKER=1 のとき PyPy ワーカーへ委譲（~2.1x）。
                # 無効/失敗時はブリッジ内でインプロセス cpu_ai.decide_guarded にフォールバック（現行挙動）。
                move = None
                if difficulty == "expert":
                    # expert = MCTS（§2.5.7）。常に計画キャッシュ経路（_plan_segment→MCTS）で 1 ターンを計画し
                    # queue から即時 replay。ポンダリング（⑥-a/-b）も _plan_segment 経由で MCTS を前倒し計算する
                    # ので、人間の番に計画済みなら CPU 手番は即時。α-β/ワーカー/trace 経路は通らない。
                    move = _cached_cpu_move(manager, cpu_player, difficulty, meta, turn_mem)
                    if move is None:  # 先頭手不正等＝MCTS で再計画して合法な先頭手を採る（稀）
                        seg = _plan_segment(manager, cpu_player, difficulty, turn_mem,
                                            meta.get("opp_profile"), meta.get("self_plan"))
                        legal_by_sig = {cpu_ai._move_sig(m): m for m in manager.get_legal_actions(cpu_player)}
                        for mv in (seg or []):
                            hit = legal_by_sig.get(cpu_ai._move_sig(mv))
                            if hit is not None:
                                move = hit
                                break
                else:
                    # Phase 3 ① 計画キャッシュ（OPCG_PLAN_CACHE=1・本番体感最適化）: セグメント内の手番を
                    # 即時 replay する（待ちを1回に集約）。トレース採取時は per-action を維持（読み筋記録のため）。
                    # 既定 OFF ＝従来挙動と完全同値。
                    if os.environ.get("OPCG_PLAN_CACHE", "0") == "1" and not trace_on:
                        move = _cached_cpu_move(manager, cpu_player, difficulty, meta, turn_mem)
                    if move is None:
                        move = decide_client.decide(manager, cpu_player, difficulty, mem=turn_mem,
                                                    profile=meta.get("opp_profile"), plan=meta.get("self_plan"),
                                                    trace=tr, trace_read_ahead=False)
                if move is not None:
                    if trace_on:
                        # 思考トレース＋アクションを適用前に記録（card_id 基準・進行には不参加）。
                        meta.setdefault("decisions", []).append(
                            {"turn": manager.turn_count, "player": cpu_pid, **tr})
                        _replay_record_action(meta, manager, "cpu", cpu_pid, {
                            "action_type": move["action_type"], "card_uuid": move.get("card_uuid"),
                            "payload": move.get("payload")})
                    if move["kind"] == "battle":
                        action_api.apply_battle_action(manager, cpu_player, move["action_type"], move.get("card_uuid"))
                    else:
                        action_api.apply_game_action(manager, cpu_player, move["action_type"], move.get("payload", {}))
                    _capture_value_samples(meta, manager)
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

@app.options("/api/game/{game_id}/replay")
async def options_game_replay(game_id: str): return {"status": "ok"}


@app.get("/api/game/{game_id}/replay")
async def game_replay(game_id: str):
    """traced CPU 対局の「リプレイ種＋CPU思考トレース」を返す（GCS 不要・メモリ常駐）。

    create 時に `cpu_trace=true` を指定した対局のみ記録される。返す内容:
      - 種（descriptor）: schema/seed/first_player/leaders/decks/difficulty/actions（card_id 基準）
      - decisions: 各 CPU 意思決定の思考トレース（chosen/candidates/regret/j_components/read_ahead）
    対局はメモリ常駐（Cloud Run は揮発）なので、対局中〜終了直後に取得して保存/共有する想定。
    """
    meta = CPU_GAMES.get(game_id)
    if not _replay_enabled(meta):
        return {"success": False, "error": {"code": "REPLAY_NOT_FOUND",
                "message": "この対局のリプレイ記録がありません（cpu_trace 未指定 or 不明なゲーム）。"}}
    # 価値学習データ（人間ログ活用）: 終局していればターン境界サンプルを勝者でラベル確定して同梱する
    # （`{"f":[...],"y":0/1}`）。未決着なら空（ラベル付け不能）。フロント采取は replay 全体を運ぶので追加配線不要。
    manager = GAMES.get(game_id)
    raw_samples = meta.get("value_samples", [])
    value_samples = (cpu_value_data.label_samples(raw_samples, manager.winner)
                     if manager is not None and manager.winner is not None and raw_samples else [])
    descriptor = {
        "schema": REPLAY_SCHEMA, "seed": meta.get("seed"),
        "first_player": meta.get("first_player"), "difficulty": meta.get("difficulty"),
        "cpu_player_id": meta.get("cpu_player_id"),
        "leaders": meta.get("leaders"), "decks": meta.get("decks"),
        "actions": meta.get("actions", []), "value_samples": value_samples,
    }
    return {"success": True, "game_id": game_id,
            "replay": descriptor, "decisions": meta.get("decisions", [])}


@app.get("/api/assets/version")
async def get_assets_version():
    """カード画像のキャッシュ版数を返す（フロントが ?v= に付与してキャッシュ無効化に使う）。"""
    return {"success": True, "v": IMAGE_VERSION}

@app.get("/api/cards")
async def get_all_cards(request: Request, response: Response):
    try:
        if len(card_db.cards) < len(card_db.raw_db):
            for card_id in card_db.raw_db.keys(): card_db.get_card(card_id)
        # 内容に変化が無ければ本体を返さず 304（1.2MBの転送・再パースをスキップ）
        if request.headers.get("if-none-match") == CARDS_ETAG:
            return Response(status_code=304, headers={"ETag": CARDS_ETAG, "Cache-Control": "no-cache"})
        cards_data = [c.to_dict() for c in card_db.cards.values()]
        response.headers["ETag"] = CARDS_ETAG
        response.headers["Cache-Control"] = "no-cache"
        return {"success": True, "cards": cards_data}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/api/deck")
async def save_deck(deck_data: Dict[str, Any] = Body(...)):
    if not db: return {"success": False, "error": "Database not initialized"}
    try:
        doc_ref = db.collection("decks").document(deck_data["id"]) if "id" in deck_data and deck_data["id"] else db.collection("decks").document()
        save_data = {"id": doc_ref.id, "name": deck_data.get("name", "Untitled Deck"), "leader_id": deck_data.get("leader_id"), "card_uuids": deck_data.get("card_uuids", []), "don_uuids": deck_data.get("don_uuids", []), "created_at": firestore.SERVER_TIMESTAMP}
        doc_ref.set(save_data); return {"success": True, "deck_id": doc_ref.id}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.delete("/api/deck/{deck_id}")
async def delete_deck(deck_id: str):
    if not db:
        return {"success": False, "error": "Database not initialized"}
    try:
        db.collection("decks").document(deck_id).delete()
        return {"success": True, "deck_id": deck_id}
    except Exception as e:
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
            pass
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
        game_id = str(uuid.uuid4()); 
        p1_name = req.get("p1_name", "P1"); p2_name = req.get("p2_name", "P2")
        if "SandboxManager" not in globals(): raise ImportError("SandboxManager not loaded")
        manager = SandboxManager(p1_name=p1_name, p2_name=p2_name, room_name=req.get("room_name", "Custom Room"))
        SANDBOX_GAMES[manager.game_id] = manager
        return {"success": True, "game_id": manager.game_id, "game_state": manager.to_dict()}
    except Exception as e:
        return {"success": False, "error": str(e)}

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
        return {"success": False, "error": str(e)}

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
        return {"success": True, "game_id": game_id, **_rule_room_meta(game_id), "game_state": None}
    except Exception as e:
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
    """ロビー表示用にデッキのリーダー情報のみ抽出する。

    デッキ全カードはロードせず、Firestore のメタデータ(leader_id)から
    リーダー1枚だけを解決する（一覧表示のたびに50枚パースしない）。
    """
    try:
        leader_id = _load_deck_doc(deck_id).get("leader_id")
        master = card_db.get_card(leader_id) if leader_id else None
        if master:
            return {"leader_id": master.card_id, "leader_name": master.name}
    except Exception as e:
        pass
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
        else:
            return {"success": False, "error": f"Unknown rule action: {act}"}

        await game_ws_manager.broadcast(game_id, build_rule_message(game_id))
        return {"success": True, "game_id": game_id, **build_rule_message(game_id)}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.websocket("/ws/game/{game_id}")
async def game_websocket_endpoint(websocket: WebSocket, game_id: str):
    await game_ws_manager.connect(websocket, game_id)
    try:
        while True: await websocket.receive_text()
    except WebSocketDisconnect:
        game_ws_manager.disconnect(websocket, game_id)

@app.get("/health")
async def health(): return {"status": "ok", "constants_loaded": bool(CONST)}
