import os
import uuid
import logging
import sys
import json
from typing import Any, Dict, Optional, List

from fastapi import FastAPI, Body, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# パス設定の安定化
current_api_dir = os.path.dirname(os.path.abspath(__file__))
if current_api_dir not in sys.path:
    sys.path.append(current_api_dir)

# 自作モジュールのインポート
try:
    from schemas import GameActionResultSchema
except ImportError:
    from .schemas import GameActionResultSchema

from opcg_sim.src.gamestate import Player, GameManager
from opcg_sim.src.loader import CardLoader, DeckLoader

# ロギング
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, force=True)
logger = logging.getLogger("opcg_sim_api")

# 定数のロード (ロジック用)
def get_const():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "shared_constants.json")
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f: return json.load(f)
    return {}
CONST = get_const()

app = FastAPI(title="OPCG Simulator API v1.4")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- 内部関数 (エンドポイントより前に定義して NameError を防止) ---

def build_game_result_raw(manager: GameManager, game_id: str, success: bool = True, error_msg: str = None) -> Dict[str, Any]:
    p_props = CONST.get('PLAYER_PROPERTIES', {})
    root_keys = CONST.get('API_ROOT_KEYS', {})
    p_keys = CONST.get('PLAYER_KEYS', {"P1": "p1", "P2": "p2"})

    def sync_player(p: Player):
        d = p.to_dict()
        return {
            "player_id": d["player_id"],
            "name": d["name"],
            p_props.get('LIFE_COUNT', 'life_count'): len(p.life),
            p_props.get('DON_DECK_COUNT', 'don_deck_count'): len(p.don_deck),
            "don_active": d.get("don_active", []),
            "don_rested": d.get("don_rested", []),
            "leader": d.get("leader"),
            "zones": d.get("zones")
        }

    game_state = {
        "game_id": game_id,
        "turn_info": {
            "turn_count": manager.turn_count if manager else 0,
            "current_phase": manager.phase.name if manager else "N/A",
            "active_player_id": manager.turn_player.name if manager else "N/A"
        },
        "players": {
            p_keys.get('P1', 'p1'): sync_player(manager.p1),
            p_keys.get('P2', 'p2'): sync_player(manager.p2)
        }
    }
    return {
        root_keys.get('SUCCESS', 'success'): success,
        "game_id": game_id,
        root_keys.get('GAME_STATE', 'game_state'): game_state,
        "error": {"message": error_msg} if error_msg else None
    }

# --- エンドポイント ---

GAMES: Dict[str, GameManager] = {}

@app.post("/api/game/create", response_model=GameActionResultSchema)
async def game_create(req: Any = Body(...)):
    try:
        # ゲーム生成ロジック (省略部分は既存のものを維持)
        # manager = ... 
        # game_id = ...
        # return build_game_result_raw(manager, game_id)
        pass 
    except Exception as e:
        logger.error(f"Error: {e}")
        return {"success": False, "game_id": "", "error": {"message": str(e)}}

@app.get("/health")
async def health():
    return {"status": "ok", "constants": "loaded" if CONST else "not_found"}
