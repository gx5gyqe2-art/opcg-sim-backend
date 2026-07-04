"""API ルート: カード／アセット／ヘルス（ドメイン別 APIRouter）。

`routers/__init__.py` が全ドメインを束ねて app が include する。ロジックは
config/resources/state/presenters/ws/services へ委譲する。monkeypatch 対象の
`load_deck_mixed`/`_deck_preview` はサービスモジュール属性経由で呼ぶ（`deck_svc.*`）。
"""

from fastapi import APIRouter, Request, Response

try:
    from google.cloud import firestore
except Exception:
    firestore = None

from ..config import CONST, IMAGE_VERSION, constants_hash, SCHEMA_HASH
from ..resources import card_db, CARDS_ETAG, materialize_all_cards

router = APIRouter()


@router.get("/api/assets/version")
async def get_assets_version():
    """カード画像のキャッシュ版数を返す（フロントが ?v= に付与してキャッシュ無効化に使う）。"""
    return {"success": True, "v": IMAGE_VERSION}

@router.get("/api/cards")
async def get_all_cards(request: Request, response: Response):
    try:
        materialize_all_cards()
        # 内容に変化が無ければ本体を返さず 304（1.2MBの転送・再パースをスキップ）
        if request.headers.get("if-none-match") == CARDS_ETAG:
            return Response(status_code=304, headers={"ETag": CARDS_ETAG, "Cache-Control": "no-cache"})
        cards_data = [c.to_dict() for c in card_db.cards.values()]
        response.headers["ETag"] = CARDS_ETAG
        response.headers["Cache-Control"] = "no-cache"
        return {"success": True, "cards": cards_data}
    except Exception as e:
        return {"success": False, "error": str(e)}

@router.get("/health")
async def health():
    # constants_hash / schema_hash: フロントが埋め込みハッシュと照合して定数・APIスキーマの乖離
    # （同期漏れ deploy）を検出する契約照合用。schema_hash は contract/manifest.json 由来。
    return {"status": "ok", "constants_loaded": bool(CONST),
            "constants_hash": constants_hash(), "schema_hash": SCHEMA_HASH}
