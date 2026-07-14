"""店舗X（Twitter/X アカウント）の手動ディレクトリ（設計 §16.9）。

TCG+ の開催マスターは店舗X（`sns_url`）が未登録のことが多い。ここでは **店名 → X アカウント** を
人が手動登録し、開催マスターに**上書き優先**でかぶせる（TCG+ 再同期でも消えない）。店舗Xが分かると
紐付け（handle 照合・§16.7）と発見（`from:`・§16.5）の精度が上がる。

Firestore（コレクション `flagship_store_sns`・doc=店名）第一・未設定なら SQLite（`store_sns`）へ
フォールバック（`store.py` と同方針）。手動運用・定期ジョブ無し。
"""
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from typing import Dict, Optional

from .. import resources
from . import db as fdb

COLLECTION = "flagship_store_sns"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS store_sns (
  store      TEXT PRIMARY KEY,
  sns_url    TEXT,
  updated_at TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class SqliteStoreSns:
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(fdb.db_path())
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(_SCHEMA)
        return conn

    def set(self, store: str, sns_url: Optional[str]) -> None:
        """店名 → X を登録/更新。`sns_url` が空/None なら登録解除（削除）。"""
        with closing(self._connect()) as c, c:
            if sns_url:
                c.execute(
                    """INSERT INTO store_sns (store, sns_url, updated_at) VALUES (?, ?, ?)
                       ON CONFLICT(store) DO UPDATE SET sns_url=excluded.sns_url, updated_at=excluded.updated_at""",
                    (store, sns_url, _now()),
                )
            else:
                c.execute("DELETE FROM store_sns WHERE store=?", (store,))

    def get_all(self) -> Dict[str, str]:
        with closing(self._connect()) as c:
            return {r["store"]: r["sns_url"] for r in c.execute("SELECT store, sns_url FROM store_sns").fetchall()
                    if r["sns_url"]}

    def get(self, store: str) -> Optional[str]:
        return self.get_all().get(store)


class FirestoreStoreSns:
    def __init__(self, client):
        self._client = client

    def _col(self):
        return self._client.collection(COLLECTION)

    def set(self, store: str, sns_url: Optional[str]) -> None:
        ref = self._col().document(store)
        if sns_url:
            ref.set({"store": store, "sns_url": sns_url, "updated_at": _now()})
        elif getattr(ref.get(), "exists", False):
            ref.delete()

    def get_all(self) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for snap in self._col().stream():
            d = snap.to_dict() or {}
            if d.get("sns_url"):
                out[snap.id] = d["sns_url"]
        return out

    def get(self, store: str) -> Optional[str]:
        return self.get_all().get(store)


def get_store_sns():
    """Firestore が使えれば FirestoreStoreSns、無ければ SqliteStoreSns。"""
    client = getattr(resources, "db", None)
    return FirestoreStoreSns(client) if client is not None else SqliteStoreSns()


def overlay(rows: list) -> list:
    """開催マスター行（dict）に手動店舗Xを上書き優先でかぶせる（TCG+ 値より優先）。"""
    directory = get_store_sns().get_all()
    if directory:
        for r in rows:
            manual = directory.get(r.get("store"))
            if manual:
                r["sns_url"] = manual
    return rows
