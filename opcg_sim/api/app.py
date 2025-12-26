import os, uuid, logging, sys
from typing import Any, Dict, Optional, List
from fastapi import FastAPI, HTTPException, Body, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# srcを明示してインポート
from opcg_sim.src.models import CardInstance
from opcg_sim.src.gamestate import Player, GameManager
from opcg_sim.src.loader import CardLoader, DeckLoader
from opcg_sim.src.enums import Phase

# --- 1. 詳細ログ設定 ---
logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    force=True
)
logger = logging.getLogger("opcg_sim_api")

app = FastAPI(title="OPCG Simulator API v1.4")

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. パス解決とデータロード ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CARD_DB_PATH = os.path.join(DATA_DIR, "opcg_cards.json")

# インメモリ管理
REQUEST_CACHE: Dict[str, Dict[str, Any]] = {}
GAMES: Dict[str, GameManager] = {}

# 起動時の診断
@app.on_event("startup")
async def startup_event():
    logger.info("!!! STARTUP DIAGNOSTICS !!!")
    logger.info(f"DATA_DIR: {DATA_DIR}")
    if os.path.exists(DATA_DIR):
        logger.info(f"Files in data: {os.listdir(DATA_DIR)}")
    
    logger.info("--- REGISTERED ROUTES ---")
    for route in app.routes:
        if hasattr(route, "path"):
            logger.info(f"Route: {route.path}")
    logger.info("!!!!!!!!!!!!!!!!!!!!!!!!!!")

try:
    card_db = CardLoader(CARD_DB_PATH)
    card_db.load()
    deck_loader = DeckLoader(card_db)
    logger.info("Card database loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load DB: {e}")

# --- 3. モデル定義 ---
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

# --- 4. 共通レスポンスビルド ---
def build_game_result(manager: GameManager, game_id: str, success: bool = True, error_msg: str = None) -> Dict[str, Any]:
    return {
        "success": success,
        "game_id": game_id,
        "state": {
            "turn_info": {
                "turn_count": manager.turn_count if manager else 0,
                "current_phase": manager.phase.name if manager else "N/A",
                "active_player_id": manager.turn_player.name if manager else "N/A",
                "winner": None
            },
            "players": {
                manager.p1.name: manager.p1.to_dict() if manager else {},
                manager.p2.name: manager.p2.to_dict() if manager else {}
            }
        },
        "error": {"message": error_msg} if error_msg else None
    }

# --- 5. エンドポイント実装 ---

@app.get("/health")
def health():
    return {"ok": True, "version": "1.4"}

@app.post("/api/game/create")
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
        return build_game_result(manager, game_id)
    except Exception as e:
        logger.error(f"Creation Error: {e}", exc_info=True)
        return {"success": False, "error": {"message": str(e)}}

@app.get("/api/game/{gameId}/state")
def get_game_state(gameId: str):
    manager = GAMES.get(gameId)
    if not manager:
        raise HTTPException(status_code=404, detail="Game not found")
    return build_game_result(manager, gameId)

@app.post("/api/game/{gameId}/action")
def post_game_action(gameId: str, req: ActionReq):
    if req.request_id in REQUEST_CACHE: return REQUEST_CACHE[req.request_id]

    manager = GAMES.get(gameId)
    if not manager: return build_game_result(None, gameId, success=False, error_msg="Game not found")

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

        result = build_game_result(manager, gameId)
        REQUEST_CACHE[req.request_id] = result
        return result
    except Exception as e:
        logger.error(f"Action Error: {e}", exc_info=True)
        return build_game_result(manager, gameId, success=False, error_msg=str(e))
