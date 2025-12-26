import os
import uuid
import logging
import sys
import json
from typing import Any, Dict, Optional, List

from fastapi import FastAPI, HTTPException, Body, Request
from pydantic import BaseModel, Field, ConfigDict
from fastapi.middleware.cors import CORSMiddleware

# srcを明示したインポート
from opcg_sim.src.models import CardInstance
from opcg_sim.src.gamestate import Player, GameManager
from opcg_sim.src.loader import CardLoader, DeckLoader
from opcg_sim.src.enums import Phase

# --- 1. ロギング設定 (Cloud Run ログビューア統合) ---
logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    force=True
)
logger = logging.getLogger("opcg_sim_api")

# --- 2. 共通定数ファイルのロード ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONST_PATH = os.path.join(BASE_DIR, "..", "shared_constants.json")

try:
    with open(CONST_PATH, "r", encoding="utf-8") as f:
        CONST = json.load(f)
    logger.info(f"Successfully loaded shared_constants.json from {CONST_PATH}")
except Exception as e:
    logger.error(f"Failed to load shared_constants.json: {e}")
    # フォールバック
    CONST = {
        "PLAYER_KEYS": {"P1": "p1", "P2": "p2"},
        "API_ROOT_KEYS": {"GAME_STATE": "game_state", "SUCCESS": "success"},
        "PLAYER_PROPERTIES": {"LIFE_COUNT": "life_count", "DON_DECK_COUNT": "don_deck_count", "DON_ACTIVE": "don_active", "DON_RESTED": "don_rested"},
        "CARD_PROPERTIES": {"UUID": "uuid", "NAME": "name", "TYPE": "type", "TYPE_LEADER": "LEADER"}
    }

# --- 3. Pydantic スキーマ定義 (API v1.4 準拠) ---

class CardSchema(BaseModel):
    """盤面上のカード1枚の情報"""
    uuid: str = Field(..., description="カード固有のUUID")
    card_id: str = Field(..., description="カード型番")
    name: str = Field(..., description="カード名称")
    power: int = Field(..., description="パワー")
    cost: int = Field(..., description="コスト")
    attribute: str = Field(..., description="属性")
    traits: List[str] = Field(..., description="特徴リスト")
    text: str = Field(..., description="効果テキスト")
    type: str = Field(..., description="カード種類")
    is_rest: bool = Field(..., description="レスト状態")
    is_face_up: bool = Field(..., description="表向き状態")
    attached_don: int = Field(0, description="付与ドン数")
    owner_id: str = Field(..., description="所有プレイヤーID")
    keywords: List[str] = Field(default_factory=list)

class ZoneSchema(BaseModel):
    """プレイヤーの各ゾーン状態"""
    field: List[CardSchema] = Field(default_factory=list)
    hand: List[CardSchema] = Field(default_factory=list)
    life: List[CardSchema] = Field(default_factory=list)
    trash: List[CardSchema] = Field(default_factory=list)
    stage: Optional[CardSchema] = None

class DonSchema(BaseModel):
    """ドン実体の構造"""
    uuid: str
    owner_id: str
    is_rest: bool
    attached_to: Optional[str] = None

class PlayerSchema(BaseModel):
    """プレイヤー全データ (定数エイリアス適用)"""
    player_id: str
    name: str
    # 定数ファイルから動的にキーを割り当て
    life_count: int = Field(..., alias=CONST['PLAYER_PROPERTIES']['LIFE_COUNT'])
    don_deck_count: int = Field(..., alias=CONST['PLAYER_PROPERTIES']['DON_DECK_COUNT'])
    don_active: List[DonSchema] = Field(default_factory=list, alias=CONST['PLAYER_PROPERTIES']['DON_ACTIVE'])
    don_rested: List[DonSchema] = Field(default_factory=list, alias=CONST['PLAYER_PROPERTIES']['DON_RESTED'])
    leader: Optional[CardSchema]
    zones: ZoneSchema

    model_config = ConfigDict(populate_by_name=True)

class TurnInfoSchema(BaseModel):
    """ターン進行状況"""
    turn_count: int = Field(..., ge=1)
    current_phase: str
    active_player_id: str
    winner: Optional[str] = None

class GameStateSchema(BaseModel):
    """全量データ構造"""
    game_id: str
    turn_info: TurnInfoSchema
    players: Dict[str, PlayerSchema]

class GameActionResult(BaseModel):
    """API レスポンスのルート構造 (動的キー対応)"""
    model_config = ConfigDict(extra="allow", populate_by_name=True)
    # success は定数 API_ROOT_KEYS['SUCCESS'] の alias としたかったが、
    # 実際には build_game_result_raw で動的に生成するため extra="allow" で対応

# --- 4. FastAPI アプリケーション設定 ---

app = FastAPI(title="OPCG Simulator API v1.4", description="shared_constants.json 同期モデル")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 冪等性キャッシュ
REQUEST_CACHE: Dict[str, Dict[str, Any]] = {}

# パス解決
DATA_DIR = os.path.join(BASE_DIR, "data")
CARD_DB_PATH = os.path.join(DATA_DIR, "opcg_cards.json")

# 起動時診断ログ
@app.on_event("startup")
async def startup_event():
    logger.info("!!! STARTUP DIAGNOSTICS !!!")
    logger.info(f"Successfully loaded shared_constants.json. P1 Key: {CONST['PLAYER_KEYS']['P1']}")
    logger.info(f"DATA_DIR: {DATA_DIR}")
    if os.path.exists(DATA_DIR):
        logger.info(f"Files in data: {os.listdir(DATA_DIR)}")
    logger.info("--- REGISTERED ROUTES ---")
    for route in app.routes:
        if hasattr(route, "path"):
            logger.info(f"Route: {route.path}")

# カードDBロード
try:
    card_db = CardLoader(CARD_DB_PATH)
    card_db.load()
    deck_loader = DeckLoader(card_db)
    logger.info("Card database loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load DB: {e}")

GAMES: Dict[str, GameManager] = {}

# --- 5. リクエストモデル ---

class CreateReq(BaseModel):
    p1_deck: str
    p2_deck: str
    p1_name: str = "Player 1"
    p2_name: str = "Player 2"

class ActionDetail(BaseModel):
    action_type: str
    player_id: str
    card_uuid: Optional[str] = None
    target_uuid: Optional[str] = None
    don_count: Optional[int] = None

class ActionReq(BaseModel):
    request_id: str
    action: ActionDetail

# --- 6. 内部ロジック & エンドポイント ---

def build_game_result_raw(manager: GameManager, game_id: str, success: bool = True, error_msg: str = None) -> Dict[str, Any]:
    """定数ファイルに基づき、キー名を同期させて辞書を生成"""
    p_props = CONST['PLAYER_PROPERTIES']
    root_keys = CONST['API_ROOT_KEYS']
    p_keys = CONST['PLAYER_KEYS']

    def sync_player(p: Player):
        """Playerオブジェクトを定数キー準拠の辞書に変換"""
        d = p.to_dict()
        return {
            "player_id": d["player_id"],
            "name": d["name"],
            p_props['LIFE_COUNT']: len(p.life),
            p_props['DON_DECK_COUNT']: len(p.don_deck),
            p_props['DON_ACTIVE']: d.get("don_active", []),
            p_props['DON_RESTED']: d.get("don_rested", []),
            "leader": d.get("leader"),
            "zones": d.get("zones")
        }

    game_state = {
        "game_id": game_id,
        "turn_info": {
            "turn_count": manager.turn_count if manager else 0,
            "current_phase": manager.phase.name if manager else "N/A",
            "active_player_id": manager.turn_player.name if manager else "N/A",
            "winner": getattr(manager, 'winner', None)
        },
        "players": {
            p_keys['P1']: sync_player(manager.p1),
            p_keys['P2']: sync_player(manager.p2)
        }
    }
    
    return {
        root_keys['SUCCESS']: success,
        "game_id": game_id,
        root_keys['GAME_STATE']: game_state,
        "error": {"message": error_msg} if error_msg else None
    }

@app.post("/api/game/create", response_model=GameActionResult)
def game_create(req: CreateReq = Body(...)):
    try:
        game_id = str(uuid.uuid4())
        p1_path = os.path.join(DATA_DIR, req.p1_deck)
        p2_path = os.path.join(DATA_DIR, req.p2_deck)
        
        p1_leader, p1_cards = deck_loader.load_deck(p1_path, req.p1_name)
        p2_leader, p2_cards = deck_loader.load_deck(p2_path, req.p2_name)
        
        player1 = Player(req.p1_name, p1_cards, p1_leader)
        player2 = Player(req.p2_name, p2_cards, p2_leader)
        
        manager = GameManager(player1, player2)
        manager.start_game()
        GAMES[game_id] = manager
        
        return build_game_result_raw(manager, game_id)
    except Exception as e:
        logger.error(f"Creation Error: {e}", exc_info=True)
        return {CONST['API_ROOT_KEYS']['SUCCESS']: False, "game_id": "", "error": {"message": str(e)}}

@app.post("/api/game/{gameId}/action", response_model=GameActionResult)
def post_game_action(gameId: str, req: ActionReq):
    if req.request_id in REQUEST_CACHE:
        return REQUEST_CACHE[req.request_id]

    manager = GAMES.get(gameId)
    if not manager:
        return {CONST['API_ROOT_KEYS']['SUCCESS']: False, "game_id": gameId, "error": {"message": "Game not found"}}

    action = req.action
    player = manager.p1 if action.player_id == manager.p1.name else manager.p2
    
    try:
        if action.action_type == "ATTACK":
            attacker = next((c for c in [player.leader] + player.field if c and c.uuid == action.card_uuid), None)
            if attacker: attacker.is_rest = True
        elif action.action_type == "ATTACH_DON":
            target = next((c for c in [player.leader] + player.field if c and c.uuid == action.target_uuid), None)
            if target: target.attached_don += (action.don_count or 1)
        elif action.action_type == "PLAY_CARD":
            card = next((c for c in player.hand if c.uuid == action.card_uuid), None)
            if card: manager.play_card_action(player, card)
        elif action.action_type == "END_TURN":
            manager.end_turn()

        result_raw = build_game_result_raw(manager, gameId)
        REQUEST_CACHE[req.request_id] = result_raw
        return result_raw

    except Exception as e:
        logger.critical(f"Action Error: {e}", exc_info=True)
        return {CONST['API_ROOT_KEYS']['SUCCESS']: False, "game_id": gameId, "error": {"message": str(e)}}

@app.get("/health")
def health():
    return {"ok": True, "version": "1.4", "shared_constants_synced": True}
