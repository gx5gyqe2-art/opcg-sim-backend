"""API ルート: デッキ CRUD（ドメイン別 APIRouter）。

`routers/__init__.py` が全ドメインを束ねて app が include する。ロジックは
config/resources/state/presenters/ws/services へ委譲する。monkeypatch 対象の
`load_deck_mixed`/`_deck_preview` はサービスモジュール属性経由で呼ぶ（`deck_svc.*`）。
"""
from typing import Any, Dict

from fastapi import APIRouter, Body

try:
    from google.cloud import firestore
except Exception:
    firestore = None

from ..resources import db
from ..services import decks as deck_svc

import logging

_logger = logging.getLogger("opcg.api")

router = APIRouter()


@router.post("/api/deck")
async def save_deck(deck_data: Dict[str, Any] = Body(...)):
    if not db: return {"success": False, "error": "Database not initialized"}
    try:
        doc_ref = db.collection("decks").document(deck_data["id"]) if "id" in deck_data and deck_data["id"] else db.collection("decks").document()
        save_data = {"id": doc_ref.id, "name": deck_data.get("name", "Untitled Deck"), "leader_id": deck_data.get("leader_id"), "card_uuids": deck_data.get("card_uuids", []), "don_uuids": deck_data.get("don_uuids", []), "created_at": firestore.SERVER_TIMESTAMP}
        doc_ref.set(save_data); return {"success": True, "deck_id": doc_ref.id}
    except Exception as e:
        return {"success": False, "error": str(e)}

@router.delete("/api/deck/{deck_id}")
async def delete_deck(deck_id: str):
    if not db:
        return {"success": False, "error": "Database not initialized"}
    try:
        db.collection("decks").document(deck_id).delete()
        return {"success": True, "deck_id": deck_id}
    except Exception as e:
        return {"success": False, "error": str(e)}

@router.get("/api/deck/get")
async def get_deck(id: str):
    try:
        leader, cards = deck_svc.load_deck_mixed(id, "system")
        return {
            "success": True,
            "deck": {
                "leader": [leader.master.to_dict()] if leader else [],
                "cards": [c.master.to_dict() for c in cards]
            }
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@router.get("/api/deck/list")
async def list_decks():
    decks = []
    if db:
        try:
            docs = db.collection("decks").order_by("created_at", direction=firestore.Query.DESCENDING).stream()
            for doc in docs:
                d = doc.to_dict()
                if "created_at" in d and d["created_at"]: d["created_at"] = str(d["created_at"])
                decks.append(d)
        except Exception:
            # 一覧取得失敗（資格情報・インデックス変更など）は空一覧で応答しつつ診断を残す。
            _logger.warning("デッキ一覧の取得に失敗（空一覧で応答）", exc_info=True)
    return {"success": True, "decks": decks}
