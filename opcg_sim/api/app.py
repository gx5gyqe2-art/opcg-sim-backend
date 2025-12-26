import os
import uuid
import logging
from typing import Any, Dict, Optional, List

from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# --- 修正依頼:インポートパスの修正 (srcを明示) ---
from opcg_sim.src.models import CardInstance
from opcg_sim.src.gamestate import Player, GameManager
from opcg_sim.src.loader import CardLoader, DeckLoader
from opcg_sim.src.enums import Phase

# ロガー設定
logger = logging.getLogger("opcg_sim_api")
app = FastAPI(title="OPCG Simulator API v1.4")

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # テストのため一時的に全許可
    allow_methods=["*"],
    allow_headers=["*"],
)
# --- パス解決の自動化 ---
# app.py (opcg_sim/api/app.py) から見て、プロジェクトのルートディレクトリ(opcg_sim/)を取得
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# カードデータベースのロード (opcg_cards.json を参照)
CARD_DB_PATH = os.path.join(DATA_DIR, "opcg_cards.json")

if not os.path.exists(CARD_DB_PATH):
    raise RuntimeError(f"Critical Error: Card database not found at {CARD_DB_PATH}")

card_db = CardLoader(CARD_DB_PATH)
card_db.load()
deck_loader = DeckLoader(card_db)

# --- インメモリ管理 ---
GAMES: Dict[str, GameManager] = {}

# --- リクエスト/レスポンスモデル ---

class CreateReq(BaseModel):
    p1_deck: str  # 例: "imu.json"
    p2_deck: str
    p1_name: str = "Player 1"
    p2_name: str = "Player 2"

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

# --- ユーティリティ ---

def build_game_result(manager: GameManager, success: bool = True, error_msg: str = None) -> Dict[str, Any]:
    """GameActionResult (v1.4) 形式のレスポンスを構築"""
    res = {
        "success": success,
        "game_id": "", 
        "state": {
            "turn_count": manager.turn_count,
            "phase": manager.phase.name,
            "turn_player_id": manager.turn_player.name,
            "players": {
                manager.p1.name: manager.p1.to_dict(),
                manager.p2.name: manager.p2.to_dict()
            }
        }
    }
    if error_msg:
        res["error"] = {"message": error_msg}
    return res

# --- エンドポイント ---

@app.get("/health")
def health():
    return {"ok": True, "version": "1.4", "db_status": "loaded"}

@app.post("/api/game/create")
def game_create(req: CreateReq = Body(...)):
    try:
        game_id = str(uuid.uuid4())
        
        # デッキファイルへの絶対パス構築
        p1_path = os.path.join(DATA_DIR, req.p1_deck)
        p2_path = os.path.join(DATA_DIR, req.p2_deck)
        
        # ファイル存在チェック
        for path in [p1_path, p2_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Deck file not found: {path}")

        # 絶対パスを使用してロード
        p1_leader, p1_cards = deck_loader.load_deck(p1_path, req.p1_name)
        p2_leader, p2_cards = deck_loader.load_deck(p2_path, req.p2_name)
        
        player1 = Player(req.p1_name, p1_cards, p1_leader)
        player2 = Player(req.p2_name, p2_cards, p2_leader)
        
        manager = GameManager(player1, player2)
        manager.start_game()
        
        GAMES[game_id] = manager
        
        result = build_game_result(manager)
        result["game_id"] = game_id
        return result
    except Exception as e:
        logger.error(f"Game Creation Failed: {e}")
        return {"success": False, "error": {"message": str(e)}}

@app.get("/api/game/{gameId}/state")
def get_game_state(gameId: str):
    manager = GAMES.get(gameId)
    if not manager:
        raise HTTPException(status_code=404, detail="Game session not found")
    
    result = build_game_result(manager)
    result["game_id"] = gameId
    return result

@app.post("/api/game/{gameId}/action")
def post_game_action(gameId: str, req: ActionReq):
    manager = GAMES.get(gameId)
    if not manager:
        return {"success": False, "error": {"message": "Game not found"}}

    action = req.action
    player = manager.p1 if action.player_id == manager.p1.name else manager.p2
    
    try:
        if action.action_type == "PLAY_CARD":
            card = next((c for c in player.hand if c.uuid == action.card_uuid), None)
            if card:
                manager.play_card_action(player, card)
            else:
                raise ValueError("Card not found in hand")

        elif action.action_type == "ATTACH_DON":
            target_card = next((c for c in [player.leader] + player.field if c and c.uuid == action.target_uuid), None)
            count = action.don_count or 1
            if target_card and len(player.don_active) >= count:
                for _ in range(count):
                    don = player.don_active.pop(0)
                    don.attached_to = target_card.uuid
                    player.don_attached_cards.append(don)
                    target_card.attached_don += 1

        elif action.action_type == "END_TURN":
            manager.end_turn()

        result = build_game_result(manager)
        result["game_id"] = gameId
        return result

    except Exception as e:
        logger.error(f"Action execution error: {e}")
        return build_game_result(manager, success=False, error_msg=str(e))
