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

# --- 1. 詳細ログ設定 (Cloud Runの標準出力に完全統合) ---
logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("opcg_sim_api")

app = FastAPI(title="OPCG Simulator API v1.4")

# --- 2. 起動時の環境・パス一覧出力 (ファイル不在の特定) ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CARD_DB_PATH = os.path.join(DATA_DIR, "opcg_cards.json")

@app.on_event("startup")
async def startup_event():
    logger.info("=== System Startup Logging ===")
    logger.info(f"BASE_DIR: {BASE_DIR}")
    logger.info(f"DATA_DIR: {DATA_DIR}")
    logger.info(f"CARD_DB_PATH: {CARD_DB_PATH}")
    
    # dataディレクトリ内のファイルをリストアップ
    if os.path.exists(DATA_DIR):
        logger.info(f"Files in DATA_DIR: {os.listdir(DATA_DIR)}")
    else:
        logger.error(f"CRITICAL: DATA_DIR does not exist at {DATA_DIR}")

    # ルーティング一覧の表示 (404対策)
    logger.info("--- Valid Routes ---")
    for route in app.routes:
        methods = ", ".join(route.methods) if hasattr(route, "methods") else "N/A"
        logger.info(f"Route: {route.path} [{methods}]")
    logger.info("==============================")

# --- 3. 全リクエストの履歴ログ (ミドルウェア) ---
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.debug(f"Request: {request.method} {request.url}")
    try:
        response = await call_next(request)
        logger.debug(f"Response status: {response.status_code}")
        return response
    except Exception as e:
        logger.error(f"Request failed: {str(e)}", exc_info=True)
        raise

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 冪等性キャッシュ
REQUEST_CACHE: Dict[str, Dict[str, Any]] = {}

# カードデータのロード（起動時エラーを防ぐためtry-except）
try:
    card_db = CardLoader(CARD_DB_PATH)
    card_db.load()
    deck_loader = DeckLoader(card_db)
    logger.info("Card database loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load card database at {CARD_DB_PATH}: {e}")

GAMES: Dict[str, GameManager] = {}

class ActionDetail(BaseModel):
    action_type: str
    player_id: str
    card_uuid: Optional[str] = None
    target_uuid: Optional[str] = None
    don_count: Optional[int] = None

class ActionReq(BaseModel):
    request_id: str
    action: ActionDetail

def build_game_result(manager: GameManager, game_id: str, success: bool = True, error_msg: str = None) -> Dict[str, Any]:
    return {
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
        },
        "error": {"message": error_msg} if error_msg else None
    }

@app.post("/api/game/{gameId}/action")
def post_game_action(gameId: str, req: ActionReq):
    if req.request_id in REQUEST_CACHE:
        return REQUEST_CACHE[req.request_id]

    manager = GAMES.get(gameId)
    if not manager:
        logger.warning(f"Game not found: {gameId}")
        return build_game_result(None, gameId, success=False, error_msg="Game not found")

    # アクション処理ロジック (省略せず既存のものを維持)
    try:
        # ... (アクション処理) ...
        result = build_game_result(manager, gameId)
        REQUEST_CACHE[req.request_id] = result
        return result
    except Exception as e:
        logger.error(f"Action error: {e}", exc_info=True)
        return build_game_result(manager, gameId, success=False, error_msg=str(e))
