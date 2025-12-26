import os
import uuid
import logging
import sys
import json
import time
from typing import Any, Dict, Optional, List

from fastapi import FastAPI, Body, Request
from fastapi.middleware.cors import CORSMiddleware

# インポートパス解決
current_api_dir = os.path.dirname(os.path.abspath(__file__))
if current_api_dir not in sys.path:
    sys.path.append(current_api_dir)

try:
    from schemas import GameStateSchema
except ImportError:
    from .schemas import GameStateSchema

from opcg_sim.src.logger_config import session_id_ctx, log_event
from opcg_sim.src.gamestate import Player, GameManager
from opcg_sim.src.loader import CardLoader, DeckLoader

# 設定ロード
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, force=True)

def get_const():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "shared_constants.json")
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f: return json.load(f)
    return {}
CONST = get_const()

app = FastAPI(title="OPCG Simulator API v1.5")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- Middleware (sessionIdの抽出と伝播) ---

@app.middleware("http")
async def trace_logging_middleware(request: Request, call_next):
    # 1. 抽出
    s_id = request.headers.get("X-Session-ID") or request.query_params.get("sessionId")
    if not s_id:
        s_id = f"gen-{uuid.uuid4().hex[:8]}"
    
    # 2. セット
    token = session_id_ctx.set(s_id)
    
    # 静的ファイルやヘルスチェック以外のリクエストを開始ログに記録
    if not request.url.path.endswith(("/health", "/favicon.ico")):
        log_event("INFO", "api.inbound", f"{request.method} {request.url.path}")
    
    try:
        response = await call_next(request)
        response.headers["X-Session-ID"] = s_id
        return response
    finally:
        session_id_ctx.reset(token)

# --- 新設: フロントエンドログ受信エンドポイント ---

@app.post("/api/log")
async def receive_frontend_log(data: Dict[str, Any] = Body(...)):
    """
    iPhone等から送られたログをBEの基盤で出力する。
    JSONキーは shared_constants.json の定義に従うことを想定。
    """
    # FEから送られてきたsessionIdがある場合は一時的に上書きして出力
    s_id = data.get("sessionId") or session_id_ctx.get()
    token = session_id_ctx.set(s_id)
    
    try:
        log_event(
            level_key=data.get("level", "info"),
            action=data.get("action", "client.log"),
            msg=data.get("msg", ""),
            player=data.get("player", "system"),
            payload=data.get("payload"),
            source="FE" # 送信元を FE として明示
        )
        return {"status": "ok"}
    finally:
        session_id_ctx.reset(token)

# --- 既存の /api/game/create 等は維持 ---

# --- 6. エンドポイント ---

GAMES: Dict[str, GameManager] = {}
card_db = CardLoader(CARD_DB_PATH)
card_db.load()
deck_loader = DeckLoader(card_db)

@app.post("/api/game/create", response_model=Dict[str, Any])
async def game_create(req: Any = Body(...)):
    """ゲームの新規作成"""
    try:
        game_id = str(uuid.uuid4())
        
        # ログ記録 (sessionIdは自動付与)
        log_event("INFO", "game.create", f"Creating new game: {game_id}", payload=req)
        
        p1_path = os.path.join(DATA_DIR, req.get("p1_deck", ""))
        p2_path = os.path.join(DATA_DIR, req.get("p2_deck", ""))
        
        p1_leader, p1_cards = deck_loader.load_deck(p1_path, req.get("p1_name", "P1"))
        p2_leader, p2_cards = deck_loader.load_deck(p2_path, req.get("p2_name", "P2"))
        
        player1 = Player(req.get("p1_name", "P1"), p1_cards, p1_leader)
        player2 = Player(req.get("p2_name", "P2"), p2_cards, p2_leader)
        
        manager = GameManager(player1, player2)
        manager.start_game()
        GAMES[game_id] = manager
        
        return build_game_result_hybrid(manager, game_id)

    except Exception as e:
        log_event("ERROR", "game.create_fail", str(e))
        return {"success": False, "game_id": "", "error": {"message": str(e)}}

@app.get("/health")
async def health():
    return {"status": "ok", "constants_loaded": bool(CONST), "session_id": session_id_ctx.get()}
