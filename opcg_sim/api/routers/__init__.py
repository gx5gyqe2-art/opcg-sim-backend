"""ドメイン別ルータ（game/cpu/cards/decks/sandbox/rule）を束ねる集約点。

`app.py` は従来どおり `from .routers import router` で単一ルータを include する。
新ドメイン追加時は本ファイルに1行加える。
"""
from fastapi import APIRouter

from . import game, cpu, cards, decks, sandbox, rule

ALL_ROUTERS = [game.router, cpu.router, cards.router, decks.router, sandbox.router, rule.router]

router = APIRouter()
for _r in ALL_ROUTERS:
    router.include_router(_r)
