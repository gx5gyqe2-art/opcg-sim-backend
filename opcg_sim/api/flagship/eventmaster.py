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
  capacity         INTEGER,
  sns_url          TEXT,
  apply_end        TEXT,
  count_applicants INTEGER,
  updated_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_event_master_series ON event_master(series_id);
"""

# TCG+ 同期で upsert するフィールド。count_applicants は含めない（フロントが別途 sync するため、
# 開催同期で上書き（NULL 化）しない・§16.14）。
_FIELDS = ("series_id", "start_datetime", "store", "pref", "capacity", "sns_url", "apply_end")


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
        "apply_end": d.get("apply_end") or "",
        "count_applicants": d.get("count_applicants"),
    }


class SqliteEventMaster:
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(fdb.db_path())
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(_SCHEMA)
        # 既存DBへの後方互換マイグレーション（§16.13 apply_end / §16.14 count_applicants）。
        for col, decl in (("apply_end", "TEXT"), ("count_applicants", "INTEGER")):
            try:
                conn.execute(f"ALTER TABLE event_master ADD COLUMN {col} {decl}")
            except sqlite3.OperationalError:
                pass  # 既に存在
        return conn

    def upsert(self, events: List[Dict[str, Any]]) -> int:
        now = _now()
        with closing(self._connect()) as c, c:
            for e in events:
                c.execute(
                    """INSERT INTO event_master (id, series_id, start_datetime, store, pref, capacity, sns_url, apply_end, updated_at)
                       VALUES (:id, :series_id, :start_datetime, :store, :pref, :capacity, :sns_url, :apply_end, :now)
                       ON CONFLICT(id) DO UPDATE SET
                         series_id=excluded.series_id, start_datetime=excluded.start_datetime,
                         store=excluded.store, pref=excluded.pref, capacity=excluded.capacity,
                         sns_url=excluded.sns_url, apply_end=excluded.apply_end, updated_at=excluded.updated_at""",
                    {"id": int(e["id"]), "now": now, **{k: e.get(k) for k in _FIELDS}},
                )
        return len(events)

    def update_applicants(self, counts: Dict[int, Any]) -> int:
        """フロントが取得した申込人数を既存の開催行へ反映する（§16.14）。開催同期とは独立。"""
        now = _now()
        n = 0
        with closing(self._connect()) as c, c:
            for eid, cnt in counts.items():
                cur = c.execute(
                    "UPDATE event_master SET count_applicants = ?, updated_at = ? WHERE id = ?",
                    (cnt, now, int(eid)),
                )
                n += cur.rowcount
        return n

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
            # merge=True で既存の count_applicants（フロント sync 分・§16.14）を消さない。
            self._col().document(str(e["id"])).set(
                {**{k: e.get(k) for k in _FIELDS}, "updated_at": now}, merge=True
            )
        return len(events)

    def update_applicants(self, counts: Dict[int, Any]) -> int:
        """申込人数だけを既存ドキュメントへマージ更新する（§16.14）。"""
        now = _now()
        for eid, cnt in counts.items():
            self._col().document(str(eid)).set(
                {"count_applicants": cnt, "updated_at": now}, merge=True
            )
        return len(counts)

    def list(self, series_id: int) -> List[Dict[str, Any]]:
        out = [_out(snap.id, snap.to_dict() or {})
               for snap in self._col().where("series_id", "==", series_id).stream()]
        out.sort(key=lambda e: (e["start_datetime"], e["store"]))
        return out


def get_event_master():
    """Firestore が使えれば FirestoreEventMaster、無ければ SqliteEventMaster。"""
    client = getattr(resources, "db", None)
    return FirestoreEventMaster(client) if client is not None else SqliteEventMaster()
