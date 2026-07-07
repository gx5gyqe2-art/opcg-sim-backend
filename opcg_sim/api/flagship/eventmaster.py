"""開催マスターの永続化（設計 §16.8）。

TCG+ は開催日を過ぎた大会を一覧 API から順次削除する。当アプリの一覧はフロントが TCG+ を
ライブ取得するだけだったため、過去開催が消えると一覧・紐付け候補からも消えていた。ここでは
**取得した開催をバックエンドにスナップショット保存**し、TCG+ から消えても保持できるようにする。

`GET /events?series_id=` が TCG+ の最新を upsert してから master を返すので、閲覧するたびに
現行開催が蓄積され、過去開催も残る（＝TCG+ 消去に耐える）。Firestore（`flagship_event_master`）
第一・未設定なら SQLite フォールバック（`store.py` と同方針）。**既に TCG+ から消えた開催は
遡って復元できない**（一度スナップショットした時点以降を保持）。
"""
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from typing import Any, Dict, List

from .. import resources
from . import db as fdb

COLLECTION = "flagship_event_master"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS event_master (
  id             INTEGER PRIMARY KEY,
  series_id      INTEGER NOT NULL,
  start_datetime TEXT,
  store          TEXT,
  pref           TEXT,
  capacity       INTEGER,
  sns_url        TEXT,
  updated_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_event_master_series ON event_master(series_id);
"""

_FIELDS = ("series_id", "start_datetime", "store", "pref", "capacity", "sns_url")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _out(event_id, d: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": int(event_id),
        "series_id": d.get("series_id"),
        "start_datetime": d.get("start_datetime") or "",
        "store": d.get("store") or "",
        "pref": d.get("pref") or "",
        "capacity": d.get("capacity"),
        "sns_url": d.get("sns_url"),
    }


class SqliteEventMaster:
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(fdb.db_path())
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(_SCHEMA)
        return conn

    def upsert(self, events: List[Dict[str, Any]]) -> int:
        now = _now()
        with closing(self._connect()) as c, c:
            for e in events:
                c.execute(
                    """INSERT INTO event_master (id, series_id, start_datetime, store, pref, capacity, sns_url, updated_at)
                       VALUES (:id, :series_id, :start_datetime, :store, :pref, :capacity, :sns_url, :now)
                       ON CONFLICT(id) DO UPDATE SET
                         series_id=excluded.series_id, start_datetime=excluded.start_datetime,
                         store=excluded.store, pref=excluded.pref, capacity=excluded.capacity,
                         sns_url=excluded.sns_url, updated_at=excluded.updated_at""",
                    {"id": int(e["id"]), "now": now, **{k: e.get(k) for k in _FIELDS}},
                )
        return len(events)

    def list(self, series_id: int) -> List[Dict[str, Any]]:
        with closing(self._connect()) as c:
            rows = c.execute(
                "SELECT * FROM event_master WHERE series_id = ? ORDER BY start_datetime, store",
                (series_id,),
            ).fetchall()
        return [_out(r["id"], dict(r)) for r in rows]


class FirestoreEventMaster:
    def __init__(self, client):
        self._client = client

    def _col(self):
        return self._client.collection(COLLECTION)

    def upsert(self, events: List[Dict[str, Any]]) -> int:
        now = _now()
        for e in events:
            self._col().document(str(e["id"])).set(
                {**{k: e.get(k) for k in _FIELDS}, "updated_at": now}
            )
        return len(events)

    def list(self, series_id: int) -> List[Dict[str, Any]]:
        out = [_out(snap.id, snap.to_dict() or {})
               for snap in self._col().where("series_id", "==", series_id).stream()]
        out.sort(key=lambda e: (e["start_datetime"], e["store"]))
        return out


def get_event_master():
    """Firestore が使えれば FirestoreEventMaster、無ければ SqliteEventMaster。"""
    client = getattr(resources, "db", None)
    return FirestoreEventMaster(client) if client is not None else SqliteEventMaster()
