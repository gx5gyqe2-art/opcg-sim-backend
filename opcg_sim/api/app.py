import os
import uuid
import logging
import sys
import json
from typing import Any, Dict, Optional, List

from fastapi import FastAPI, Body, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# --- 1. インポートパスの安定化 ---
current_api_dir = os.path.dirname(os.path.abspath(__file__))
if current_api_dir not in sys.path:
    sys.path.append(current_api_dir)

try:
    from schemas import GameStateSchema, GameActionResultSchema
except ImportError:
    from .schemas import GameStateSchema, GameActionResultSchema

from opcg_sim.src.gamestate import Player, GameManager
from opcg_sim.src.loader import CardLoader, DeckLoader

# --- 2. 設定と定数のロード ---
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, force=True)
logger = logging.getLogger("opcg_sim_api")

def get_const():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "shared_constants.json")
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f: return json.load(f)
    return {}
CONST = get_const()

BASE_DIR = os.path.dirname(current_api_dir)
DATA_DIR = os.path.join(BASE_DIR, "data")
CARD_DB_PATH = os.path.join(DATA_DIR, "opcg_cards.json")

# --- 3. 内部ロジック (エンドポイントより前に定義して NameError 回避) ---

def build_game_result_hybrid(manager: GameManager, game_id: str, success: bool = True, error_msg: str = None) -> Dict[str, Any]:
    """盤面データは検証し、それ以外は柔軟に構築して返す"""
    raw_game_state = {
        "game_id": game_id,
        "turn_info": {
            "turn_count": manager.turn_count if manager else 0,
            "current_phase": manager.phase.name if manager else "N/A",
            "active_player_id": manager.turn_player.name if manager else "N/A"
        },
        "players": {
            "p1": manager.p1.to_dict() if manager else {},
            "p2": manager.p2.to_dict() if manager else {}
        }
    }

    try:
        # GameState部分のみ schemas.py で型チェック
        validated_state = GameStateSchema(**raw_game_state).model_dump(by_alias=True)
    except Exception as e:
        logger.error(f"Validation Error: {e}")
        validated_state = raw_game_state 

    return {
        "success": success,
        "game_id": game_id,
        "game_state": validated_state,
        "error": {"message": error_msg} if error_msg else None
    }

# --- 4. API定義 ---

app = FastAPI(title="OPCG Simulator API v1.4")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

GAMES: Dict[str, GameManager] = {}
card_db = CardLoader(CARD_DB_PATH)
card_db.load()
deck_loader = DeckLoader(card_db)

@app.post("/api/game/create", response_model=Dict[str, Any])
async def game_create(req: Any = Body(...)):
    """response_model を Dict に緩和し、内部で部分バリデーションを行う"""
    try:
        game_id = str(uuid.uuid4())
        p1_path = os.path.join(DATA_DIR, req.get("p1_deck", ""))
        p2_path = os.path.join(DATA_DIR, req.get("p2_deck", ""))
        
        p1_leader, p1_cards = deck_loader.load_deck(p1_path, req.get("p1_name", "P1"))
        p2_leader, p2_cards = deck_loader.load_deck(p2_path, req.get("p2_name", "P2"))
        
        player1 = Player(req.get("p1_name", "P1"), p1_cards, p1_leader)
        player2 = Player(req.get("p2_name", "P2"), p2_cards, p2_leader)
        
        manager = GameManager(player1, player2)
        manager.start_game()
        GAMES[game_id] = manager
        
        # 【解決策】生成した結果を確実に return する
        return build_game_result_hybrid(manager, game_id)

    except Exception as e:
        logger.error(f"Create Error: {e}", exc_info=True)
        return {"success": False, "game_id": "", "error": {"message": str(e)}}

@app.get("/health")
async def health():
    return {"status": "ok", "constants_loaded": bool(CONST)}
