import os
import uuid
import logging
import sys
import json
from typing import Any, Dict, Optional, List

from fastapi import FastAPI, Body, Request
from fastapi.middleware.cors import CORSMiddleware

# パス設定とインポート
current_api_dir = os.path.dirname(os.path.abspath(__file__))
if current_api_dir not in sys.path:
    sys.path.append(current_api_dir)

try:
    # 内部バリデーション用にスキーマをインポート
    from schemas import GameStateSchema, CardSchema
except ImportError:
    from .schemas import GameStateSchema, CardSchema

from opcg_sim.src.gamestate import Player, GameManager
from opcg_sim.src.loader import CardLoader, DeckLoader

# --- ロギングとパスの設定は維持 ---

# --- 2. 内部ロジック (電文の振り分け) ---

def build_game_result_hybrid(manager: GameManager, game_id: str, success: bool = True, error_msg: str = None) -> Dict[str, Any]:
    """
    盤面データ(GameState)は厳格にチェックし、それ以外は柔軟に構築する
    """
    # 1. まずは生の辞書を構築
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

    # 2. 【厳格電文】GameState 部分だけを schemas.py でセルフチェック
    try:
        # Pydanticモデルに流し込み、型エラーがあればここで検知する
        validated_state = GameStateSchema(**raw_game_state).model_dump(by_alias=True)
    except Exception as e:
        logger.error(f"GameState Validation Failed: {e}")
        # バリデーションに失敗しても、最低限の情報を返してアプリのフリーズを防ぐ
        validated_state = raw_game_state 

    # 3. 【柔軟電文】メタ情報と合体させて最終的な電文を作成
    return {
        "success": success,
        "game_id": game_id,
        "game_state": validated_state,
        "error": {"message": error_msg} if error_msg else None,
        "server_time": str(uuid.uuid4()) # 自由なフィールドもここなら追加可能
    }

# --- 3. エンドポイント ---

@app.post("/api/game/create", response_model=Dict[str, Any]) # 全体の縛りを緩和
async def game_create(req: Any = Body(...)):
    try:
        # ... ゲーム生成ロジック ...
        # return build_game_result_hybrid(manager, game_id)
        pass 
    except Exception as e:
        return {"success": False, "error": {"message": str(e)}}

@app.post("/api/game/{gameId}/action", response_model=Dict[str, Any]) # 全体の縛りを緩和
async def post_game_action(gameId: str, req: Any):
    # ... アクション処理 ...
    # return build_game_result_hybrid(manager, gameId)
    pass
