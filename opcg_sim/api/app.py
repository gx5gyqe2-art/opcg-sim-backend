"""FastAPI アプリのエントリ（create_app シェル）。

ルート定義は `routers.py`（単一 APIRouter）へ分離済み。本モジュールは
`create_app()` でアプリを組み立て（CORS＋lifespan＋include_router）、
モジュール属性 `app` として公開する。

分離済みモジュール（config/resources/state/presenters/ws/services）の主要名は
**後方互換のためここで再公開**する（既存の import・テストの参照点を維持）。
ただしルートは routers.py 側でサービスモジュール属性経由（`deck_svc.load_deck_mixed`）
で呼ぶため、`load_deck_mixed` の monkeypatch は `services.decks` を対象にする。
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from opcg_sim.src.core.sandbox import SandboxManager
from opcg_sim.api import engine_rs
# 設定・定数／常駐リソース／対局レジストリ／サービスは分離済みモジュールから取り込む（後方互換の名前で再公開）。
from .schemas import GameStateSchema, PendingRequestSchema, BattleActionRequest
from .config import CONST, constants_hash, IMAGE_VERSION, REPLAY_SCHEMA, SCHEMA_HASH
from .resources import db, card_db, CARDS_ETAG, materialize_all_cards
from .state import GAMES, SANDBOX_GAMES, CPU_GAMES, RULE_ROOMS
from .presenters import build_game_result_hybrid, build_rule_message, _rule_room_meta
from .ws import ws_manager, game_ws_manager, broadcast_rule_state
# 注意: デッキ読込（load_deck_mixed / _deck_preview / _load_deck_doc）はここで**再エクスポートしない**。
# ルート（routers.py）はサービスモジュール属性経由（deck_svc.load_deck_mixed）で解決するため、
# app モジュール属性を差し替えても効かない。テスト・スタブは `opcg_sim.api.services.decks` を patch する。
from .services.replay import _replay_enabled, _replay_record_action, _capture_final_winner
from .services.games import _resolve_first_player
from .routers import router as _api_router
from .flagship.router import router as _flagship_router

_logger = logging.getLogger("opcg.api")


@asynccontextmanager
async def _lifespan(_app):
    # Rust エンジン（`opcg_engine`）のカード定義表を**プロセスで 1 度**読む（約 8MB の
    # 効果構造 JSON）。起動時に済ませておけば最初の対局生成が待たされない。失敗しても
    # ここでは落とさず（診断だけ残す）、対局生成時に同じ例外で明示的に失敗させる。
    try:
        engine_rs.load_engine()
    except Exception:
        _logger.warning("Rust エンジンのカード定義読込に失敗（対局生成時に再試行）", exc_info=True)
    yield


def create_app() -> FastAPI:
    """FastAPI アプリを構築して返す。ルートは routers.py の単一 APIRouter を include する。"""
    _app = FastAPI(title="OPCG Simulator API v1.7", lifespan=_lifespan)
    _app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], expose_headers=["ETag"],
    )
    # NOTE: 効果定義はカードテキストの自動解析（EffectParserV2）に一本化されている。
    _app.include_router(_api_router)
    # フラッグシップ結果集計（設計: flagship リポジトリ docs/design.md §12）。SQLite は遅延初期化のため
    # このドメインを使わない限りファイルは作られない（既存デプロイへの影響なし）。
    _app.include_router(_flagship_router)
    return _app


app = create_app()
