from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from typing import Any, Dict, Optional, List
import uuid
import logging
from fastapi.middleware.cors import CORSMiddleware

# コアロジックのインポート
from opcg_sim.models import CardInstance
from opcg_sim.gamestate import Player, GameManager
from opcg_sim.loader import CardLoader, DeckLoader
from opcg_sim.enums import Phase

# ロガー設定
logger = logging.getLogger("opcg_sim_api")
app = FastAPI(title="OPCG Simulator API v1.4")

# CORS設定 (既存のものを維持)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://opcg-sim-frontend.pages.dev"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- インメモリ管理 ---
# 実際の運用ではDBやRedisが必要ですが、指示通り辞書で管理します
GAMES: Dict[str, GameManager] = {}

# カードデータの事前ロード (パスは環境に合わせて調整してください)
card_db = CardLoader("data/card_db.json")
card_db.load()
deck_loader = DeckLoader(card_db)

# --- リクエスト/レスポンスモデル (v1.4準拠) ---

class CreateReq(BaseModel):
    p1_deck: str  # 例: "st01_deck.json"
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
    """GameActionResult (v1.4) 形式のレスポンスを構築"""
    res = {
        "success": success,
        "game_id": "", # 呼び出し側で付与
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

# --- エンドポイント ---

@app.get("/health")
def health():
    return {"ok": True, "version": "1.4"}

@app.post("/api/game/create")
def game_create(req: CreateReq):
    try:
        game_id = str(uuid.uuid4())
        
        # デッキのロード
        p1_leader, p1_cards = deck_loader.load_deck(f"data/decks/{req.p1_deck}", req.p1_name)
        p2_leader, p2_cards = deck_loader.load_deck(f"data/decks/{req.p2_deck}", req.p2_name)
        
        # プレイヤーとマネージャーの初期化
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
        # v1.4 アクションマッピング
        if action.action_type == "PLAY_CARD":
            # UUIDからカードオブジェクトを特定する補助ロジックが必要
            card = next((c for c in player.hand if c.uuid == action.card_uuid), None)
            if card:
                manager.play_card_action(player, card)
            else:
                raise ValueError("Card not found in hand")

        elif action.action_type == "ATTACK":
            # コアロジック側の attack_action(attacker, target) を呼び出し
            # ※ 既存ロジックにUUIDベースの検索がある前提、なければ適宜追加
            attacker = next((c for c in [player.leader] + player.field if c and c.uuid == action.card_uuid), None)
            # ターゲットは相手の全カードから検索
            opp = manager.opponent if player == manager.turn_player else manager.turn_player
            target = next((c for c in [opp.leader] + opp.field if c and c.uuid == action.target_uuid), None)
            
            if attacker and target:
                # GameManagerに実装されている攻撃メソッドを呼ぶ
                # ※ 実装状況により manager.execute_attack(attacker, target) 等
                pass 

        elif action.action_type == "ACTIVATE":
            card = next((c for c in [player.leader] + player.field + ([player.stage] if player.stage else []) if c and c.uuid == action.card_uuid), None)
            if card and action.ability_idx is not None:
                ability = card.master.abilities[action.ability_idx]
                manager.resolve_ability(player, ability, source_card=card)

        elif action.action_type == "ATTACH_DON":
            # アクティブなドンを指定枚数、指定UUIDのカードへ
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

        else:
            return build_game_result(manager, success=False, error_msg=f"Unknown action: {action.action_type}")

        # 正常終了時
        result = build_game_result(manager)
        result["game_id"] = gameId
        return result

    except Exception as e:
        logger.error(f"Action execution error: {e}")
        return build_game_result(manager, success=False, error_msg=str(e))
