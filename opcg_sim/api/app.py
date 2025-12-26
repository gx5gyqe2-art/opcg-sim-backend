import os, uuid, logging, sys
from typing import Any, Dict, Optional, List
from fastapi import FastAPI, HTTPException, Body, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from opcg_sim.src.models import CardInstance
from opcg_sim.src.gamestate import Player, GameManager
from opcg_sim.src.loader import CardLoader, DeckLoader
from opcg_sim.src.enums import Phase

# --- 1. 詳細ログ設定 (標準出力へ流し Cloud Run ログビューアと統合) ---
logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("opcg_sim_api")

app = FastAPI(title="OPCG Simulator API v1.4")

# --- 2. リクエスト履歴の可視化ミドルウェア ---
@app.middleware("http")
async def log_requests(request: Request, call_next):
    # すべてのリクエスト（URL、メソッド、ヘッダーの一部）を記録
    logger.debug(f"Incoming Request: {request.method} {request.url}")
    logger.debug(f"Headers: {dict(request.headers)}")
    try:
        response = await call_next(request)
        logger.debug(f"Response status: {response.status_code}")
        return response
    except Exception as e:
        # エラー発生時に詳細なトレースバックを出力
        logger.error(f"Request Error: {str(e)}", exc_info=True)
        raise

# --- 3. 起動時のパス一覧出力 (404原因特定用) ---
@app.on_event("startup")
async def startup_event():
    logger.info("=== Valid API Routes ===")
    for route in app.routes:
        methods = ", ".join(route.methods) if hasattr(route, "methods") else "N/A"
        logger.info(f"Route: {route.path} [{methods}]")
    logger.info("==========================")
    
    # データディレクトリの状態確認
    logger.info(f"Checking DATA_DIR: {DATA_DIR}")
    if os.path.exists(DATA_DIR):
        logger.info(f"Files in data: {os.listdir(DATA_DIR)}")
    else:
        logger.warning(f"DATA_DIR NOT FOUND: {DATA_DIR}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 冪等性のためのキャッシュ
REQUEST_CACHE: Dict[str, Dict[str, Any]] = {}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CARD_DB_PATH = os.path.join(DATA_DIR, "opcg_cards.json")

# 初期ロード
try:
    card_db = CardLoader(CARD_DB_PATH)
    card_db.load()
    deck_loader = DeckLoader(card_db)
except Exception as e:
    logger.error(f"Failed to load card database: {e}")

GAMES: Dict[str, GameManager] = {}

class ActionDetail(BaseModel):
    action_type: str
    player_id: str
    card_uuid: Optional[str] = None
    target_uuid: Optional[str] = None
    ability_idx: Optional[int] = None
    don_count: Optional[int] = None

class ActionReq(BaseModel):
    request_id: str
    action: ActionDetail

def build_game_result(manager: GameManager, game_id: str, success: bool = True, error_msg: str = None) -> Dict[str, Any]:
    res = {
        "success": success,
        "game_id": game_id,
        "state": {
            "turn_info": {
                "turn_count": manager.turn_count,
                "current_phase": manager.phase.name,
                "active_player_id": manager.turn_player.name,
                "winner": None
            },
            "players": {
                manager.p1.name: manager.p1.to_dict(),
                manager.p2.name: manager.p2.to_dict()
            }
        }
    }
    if error_msg: res["error"] = {"message": error_msg}
    return res

@app.post("/api/game/{gameId}/action")
def post_game_action(gameId: str, req: ActionReq):
    if req.request_id in REQUEST_CACHE:
        return REQUEST_CACHE[req.request_id]

    manager = GAMES.get(gameId)
    if not manager: return {"success": False, "error": {"message": "Game not found"}}

    action = req.action
    player = manager.p1 if action.player_id == manager.p1.name else manager.p2
    
    try:
        if action.action_type == "ATTACK":
            attacker = next((c for c in [player.leader] + player.field if c and c.uuid == action.card_uuid), None)
            if attacker: attacker.is_rest = True

        elif action.action_type == "ATTACH_DON":
            target = next((c for c in [player.leader] + player.field if c and c.uuid == action.target_uuid), None)
            count = action.don_count or 1
            if target and len(player.don_active) >= count:
                for _ in range(count):
                    player.don_active.pop(0)
                    target.attached_don += 1

        elif action.action_type == "PLAY_CARD":
            card = next((c for c in player.hand if c.uuid == action.card_uuid), None)
            if card:
                manager.play_card_action(player, card)
                card.attached_don = 0

        elif action.action_type == "END_TURN":
            manager.end_turn()

        result = build_game_result(manager, gameId)
        REQUEST_CACHE[req.request_id] = result
        if len(REQUEST_CACHE) > 500: REQUEST_CACHE.pop(next(iter(REQUEST_CACHE)))
        return result

    except Exception as e:
        logger.error(f"Action processing error: {e}", exc_info=True)
        return build_game_result(manager, gameId, success=False, error_msg=str(e))
