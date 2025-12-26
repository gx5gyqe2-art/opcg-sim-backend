import os
import uuid
import logging
import sys
import json
from typing import Any, Dict, Optional, List

from fastapi import FastAPI, Body, Request
from fastapi.middleware.cors import CORSMiddleware

# 外部化されたスキーマをインポート
from .schemas import GameActionResultSchema
from opcg_sim.src.gamestate import Player, GameManager
from opcg_sim.src.loader import CardLoader, DeckLoader

# --- ロギングと定数の設定 ---
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, force=True)
logger = logging.getLogger("opcg_sim_api")

app = FastAPI(title="OPCG Simulator API v1.4")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# (共通定数、パス、データのロード処理は以前のコードを維持)

# --- エンドポイント ---

@app.post("/api/game/create", response_model=GameActionResultSchema)
def game_create(req: Any = Body(...)):
    """response_model を通じて schemas.py の型チェックを自動適用"""
    # ... ゲーム生成ロジック ...
    result_raw = build_game_result_raw(manager, game_id)
    return result_raw

@app.post("/api/game/{gameId}/action", response_model=GameActionResultSchema)
def post_game_action(gameId: str, req: Any):
    """アクション実行後も厳格な型チェックを経てレスポンスを返却"""
    # ... アクション処理 ...
    result_raw = build_game_result_raw(manager, game_id)
    return result_raw
