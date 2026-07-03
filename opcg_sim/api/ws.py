"""WebSocket 接続マネージャ（フリーモード／ルールモード）とルーム状態のブロードキャスト。

状態は「全情報を配信し、表示制御はフロント側で行う」方針のため、視点別シリアライズは行わず
同一ペイロードを全接続へ配信する。app.py の WS エンドポイントとルート群がこれらを共用する。
"""
import asyncio
import logging
from typing import Dict, List

from fastapi import WebSocket

from .presenters import build_rule_message
from .state import GAMES, RULE_ROOMS, SANDBOX_GAMES

_logger = logging.getLogger("opcg.api")


class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, game_id: str):
        await websocket.accept()
        if game_id not in self.active_connections:
            self.active_connections[game_id] = []
        self.active_connections[game_id].append(websocket)

        manager_inst = SANDBOX_GAMES.get(game_id)
        if manager_inst:
            try:
                await websocket.send_json({"type": "STATE_UPDATE", "state": manager_inst.to_dict()})
            except Exception:
                # 接続直後の初期状態送信失敗（切断直後等）。正常系寄りのため debug で痕跡のみ残す。
                _logger.debug("Failed to send initial sandbox state", exc_info=True)

    async def disconnect(self, websocket: WebSocket, game_id: str):
        if game_id in self.active_connections:
            if websocket in self.active_connections[game_id]:
                self.active_connections[game_id].remove(websocket)
            if not self.active_connections[game_id]:
                del self.active_connections[game_id]
                asyncio.create_task(self.delayed_cleanup(game_id, 1200))

    async def delayed_cleanup(self, game_id: str, delay: int):
        await asyncio.sleep(delay)
        if game_id not in self.active_connections or not self.active_connections[game_id]:
            if game_id in SANDBOX_GAMES:
                del SANDBOX_GAMES[game_id]

    async def broadcast(self, game_id: str, message: dict):
        if game_id in self.active_connections:
            for connection in self.active_connections[game_id]:
                try:
                    await connection.send_json(message)
                except Exception:
                    pass


ws_manager = ConnectionManager()


class GameConnectionManager:
    """ルールモードのオンライン対戦用 WebSocket 接続マネージャ。

    状態は「全情報を配信し、表示制御はフロント側で行う」方針のため、接続ごとの
    視点別シリアライズは行わず、同一ペイロードを全接続へブロードキャストする。
    """

    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, game_id: str):
        await websocket.accept()
        self.active_connections.setdefault(game_id, []).append(websocket)
        # 接続直後に現在のルーム/対局状態を本人へ送る
        try:
            if game_id in RULE_ROOMS:
                await websocket.send_json(build_rule_message(game_id))
        except Exception as e:
            pass

    def disconnect(self, websocket: WebSocket, game_id: str):
        conns = self.active_connections.get(game_id)
        if not conns:
            return
        if websocket in conns:
            conns.remove(websocket)
        if not conns:
            del self.active_connections[game_id]
            asyncio.create_task(self.delayed_cleanup(game_id, 1200))

    async def delayed_cleanup(self, game_id: str, delay: int):
        await asyncio.sleep(delay)
        if not self.active_connections.get(game_id):
            RULE_ROOMS.pop(game_id, None)
            GAMES.pop(game_id, None)

    def count(self, game_id: str) -> int:
        return len(self.active_connections.get(game_id, []))

    async def broadcast(self, game_id: str, message: dict):
        for connection in list(self.active_connections.get(game_id, [])):
            try:
                await connection.send_json(message)
            except Exception:
                pass


game_ws_manager = GameConnectionManager()


async def broadcast_rule_state(game_id: str):
    """ルーム対局なら最新状態を全接続へ配信する（非ルーム対局では no-op）。"""
    if game_id not in RULE_ROOMS:
        return
    manager = GAMES.get(game_id)
    room = RULE_ROOMS[game_id]
    if manager and manager.winner:
        room["status"] = "FINISHED"
    await game_ws_manager.broadcast(game_id, build_rule_message(game_id))
