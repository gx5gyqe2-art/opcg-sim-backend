from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict
import uuid

app = FastAPI()

# 超簡易インメモリ（本番では永続化や設計見直し）
GAMES: Dict[str, Dict[str, Any]] = {}


class CreateReq(BaseModel):
    # 仕様に合わせて後で増やす（deck等）
    pass


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/api/game/create")
def game_create(_: CreateReq):
    game_id = str(uuid.uuid4())
    observer_id = str(uuid.uuid4())
    state = {
        "gameId": game_id,
        "turn": 1,
        "phase": "main",
        "you": {"life": 5, "handCount": 5},
        "opponent": {"life": 5, "handCount": 5},
    }
    GAMES[game_id] = {"state": state}
    return {"success": True, "gameId": game_id, "observerId": observer_id, "state": state}


@app.post("/api/game/{gameId}/reset")
def game_reset(gameId: str):
    if gameId not in GAMES:
        # success=falseでも state を返す、の形に寄せる
        return {"success": False, "state": None}
    state = GAMES[gameId]["state"]
    state["turn"] = 1
    return {"success": True, "state": state}


@app.get("/api/game/{gameId}/state")
def game_state(gameId: str, observerId: str):
    state = GAMES.get(gameId, {}).get("state")
    return {"success": state is not None, "state": state}


class ActionReq(BaseModel):
    requestId: str
    action: Dict[str, Any]


@app.post("/api/game/{gameId}/action")
def game_action(gameId: str, observerId: str, req: ActionReq):
    state = GAMES.get(gameId, {}).get("state")
    if not state:
        return {"success": False, "state": None}
    # ダミーでターンだけ進める
    state["turn"] += 1
    return {"success": True, "state": state}