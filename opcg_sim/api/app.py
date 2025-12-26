import os, uuid, logging
from typing import Any, Dict, Optional, List
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from opcg_sim.src.models import CardInstance
from opcg_sim.src.gamestate import Player, GameManager
from opcg_sim.src.loader import CardLoader, DeckLoader
from opcg_sim.src.enums import Phase

logger = logging.getLogger("opcg_sim_api")
app = FastAPI(title="OPCG Simulator API v1.4")

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

card_db = CardLoader(CARD_DB_PATH)
card_db.load()
deck_loader = DeckLoader(card_db)
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
    """レスポンス構造の完全同期"""
    res = {
        "success": success,
        "game_id": game_id,
        "state": {
            # turn_infoの階層化
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
    # 1. 重複排除チェック
    if req.request_id in REQUEST_CACHE:
        return REQUEST_CACHE[req.request_id]

    manager = GAMES.get(gameId)
    if not manager: return {"success": False, "error": {"message": "Game not found"}}

    action = req.action
    player = manager.p1 if action.player_id == manager.p1.name else manager.p2
    opp = manager.p2 if action.player_id == manager.p1.name else manager.p1
    
    try:
        # 3. アクション処理の統合
        if action.action_type == "ATTACK":
            attacker = next((c for c in [player.leader] + player.field if c and c.uuid == action.card_uuid), None)
            if attacker: attacker.is_rest = True # レスト更新

        elif action.action_type == "ATTACH_DON":
            target = next((c for c in [player.leader] + player.field if c and c.uuid == action.target_uuid), None)
            count = action.don_count or 1
            if target and len(player.don_active) >= count:
                for _ in range(count):
                    player.don_active.pop(0)
                    target.attached_don += 1 # ドン加算

        elif action.action_type == "PLAY_CARD":
            card = next((c for c in player.hand if c.uuid == action.card_uuid), None)
            if card:
                manager.play_card_action(player, card) # zonesへの移動
                card.attached_don = 0 # 初期化

        elif action.action_type == "END_TURN":
            manager.end_turn()

        result = build_game_result(manager, gameId)
        # 4. キャッシュ保存
        REQUEST_CACHE[req.request_id] = result
        if len(REQUEST_CACHE) > 500: REQUEST_CACHE.pop(next(iter(REQUEST_CACHE)))
        return result

    except Exception as e:
        logger.error(f"Action error: {e}")
        return build_game_result(manager, gameId, success=False, error_msg=str(e))

# (create_game 等は既存の絶対パス解決を維持)
