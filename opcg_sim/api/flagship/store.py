"""flagship 結果の永続化ストア抽象（設計 §17）。

結果データ（開催スナップショット・ポスト・placement）を **デッキ管理と同じ Firestore** に置く
のを第一候補にし、Firestore 未設定（ローカル/CI）では従来の **SQLite にフォールバック**する
（graceful degrade。デッキ CRUD の `db=None` 継続と同方針）。router はこの `Store` 経由でのみ
永続化に触れ、SQLite/Firestore を意識しない。

Firestore はコレクション `flagship_events`・ドキュメント ID=`str(event_id)` に、開催スナップショット
＋ポスト＋結果を **1ドキュメントに非正規化**して持つ（常に開催単位で全置換・読み出すため）。
Cloud Run のコンテナは非永続で SQLite は再デプロイで消えるため、本番は Firestore を使う。
"""
from contextlib import closing
from typing import Any, Dict, List, Optional

from .. import resources
from . import db as sqlite_db

COLLECTION = "flagship_events"


def _post_fields(post: Optional[Dict[str, Any]]) -> Dict[str, Optional[str]]:
    """post を保存用フィールドへ。url も body_text も無ければ両方 None（＝ポスト無し）。"""
    if post and (post.get("url") or post.get("body_text")):
        return {"post_url": post.get("url"), "post_body_text": post.get("body_text")}
    return {"post_url": None, "post_body_text": None}


def _result_fields(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {"placement": r["placement"],
         "leader_card_number": r.get("leader_card_number"),
         "leader_raw": r.get("leader_raw")}
        for r in results
    ]


class SqliteStore:
    """従来の SQLite 実装（`db.py`）をリクエスト毎の接続で包む。ローカル/CI の既定。"""

    def find_url_owner(self, url: str) -> Optional[int]:
        with closing(sqlite_db.connect()) as c:
            return sqlite_db.find_url_owner(c, url)

    def replace_event_results(self, event, post, results) -> None:
        with closing(sqlite_db.connect()) as c:
            sqlite_db.replace_event_results(c, event, post, results)

    def delete_event_results(self, event_id: int) -> int:
        with closing(sqlite_db.connect()) as c:
            return sqlite_db.delete_event_results(c, event_id)

    def get_event_results(self, event_id: int) -> Optional[Dict[str, Any]]:
        with closing(sqlite_db.connect()) as c:
            return sqlite_db.get_event_results(c, event_id)

    def get_series_summary(self, series_id: int) -> List[Dict[str, Any]]:
        with closing(sqlite_db.connect()) as c:
            return sqlite_db.get_series_summary(c, series_id)


class FirestoreStore:
    """Firestore 実装（デッキ管理と同一クライアント `resources.db`）。本番の既定。"""

    def __init__(self, client):
        self._client = client

    def _col(self):
        return self._client.collection(COLLECTION)

    def find_url_owner(self, url: str) -> Optional[int]:
        for snap in self._col().where("post_url", "==", url).limit(1).stream():
            return int(snap.id)
        return None

    def replace_event_results(self, event, post, results) -> None:
        doc = {
            "series_id": event["series_id"],
            "event": event,
            "results": _result_fields(results),
            "updated_at": sqlite_db._now(),
            **_post_fields(post),
        }
        self._col().document(str(event["id"])).set(doc)  # 全置換

    def delete_event_results(self, event_id: int) -> int:
        ref = self._col().document(str(event_id))
        snap = ref.get()
        if not getattr(snap, "exists", False):
            return 0
        data = snap.to_dict() or {}
        count = len(data.get("results") or [])
        # 開催スナップショットは残し、結果・ポストだけ消す（SQLite と同じ）。
        ref.update({"results": [], "post_url": None, "post_body_text": None,
                    "updated_at": sqlite_db._now()})
        return count

    def get_event_results(self, event_id: int) -> Optional[Dict[str, Any]]:
        snap = self._col().document(str(event_id)).get()
        if not getattr(snap, "exists", False):
            return None
        d = snap.to_dict() or {}
        results = sorted(d.get("results") or [], key=lambda r: r["placement"])
        return {
            "event_id": int(event_id),
            "event": d.get("event"),
            "updated_at": d.get("updated_at"),
            "post_url": d.get("post_url"),
            "body_text": d.get("post_body_text"),
            "results": results,
        }

    def get_series_summary(self, series_id: int) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for snap in self._col().where("series_id", "==", series_id).stream():
            d = snap.to_dict() or {}
            results = d.get("results") or []
            if not results:  # 結果を持つ開催のみ（SQLite の JOIN results と同じ）
                continue
            winner = next((r for r in results if r.get("placement") == 1), None)
            out.append({
                "event_id": int(snap.id),
                "result_count": len(results),
                "winner_card_number": (winner or {}).get("leader_card_number"),
                "winner_raw": (winner or {}).get("leader_raw"),
                "post_url": d.get("post_url"),
            })
        return out


def get_store():
    """Firestore が使えれば FirestoreStore、無ければ SqliteStore（graceful degrade）。"""
    client = getattr(resources, "db", None)
    if client is not None:
        return FirestoreStore(client)
    return SqliteStore()
