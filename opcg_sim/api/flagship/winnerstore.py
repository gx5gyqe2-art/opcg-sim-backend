"""収集した優勝ポストの永続化（設計 §16.7・案1）。

X から収集した優勝ポスト（tweet 単位）と、その TCG+ 開催への紐付け（`event_id`）を貯める。
Firestore（コレクション `flagship_winner_posts`・doc=tweet_id）を第一候補に、未設定なら SQLite
へフォールバック（`store.py` と同方針）。**tweet_id で重複除去**し、**再収集で既存の `event_id`
（人の承認結果）は上書きしない**。定期ジョブは無く、手動収集をここに貯めて月次トレンド/紐付けを伸ばす。
"""
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .. import resources
from . import db as fdb

COLLECTION = "flagship_winner_posts"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS winner_posts (
  tweet_id      TEXT PRIMARY KEY,
  author        TEXT,
  author_name   TEXT,
  date          TEXT,
  char_name     TEXT,
  card_number   TEXT,
  leader_raw    TEXT,
  tweet_url     TEXT,
  event_id      INTEGER,
  created_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_winner_posts_event ON winner_posts(event_id);
"""

_FIELDS = ("author", "author_name", "date", "char_name", "card_number", "leader_raw", "tweet_url")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _row(p: Dict[str, Any]) -> Dict[str, Any]:
    """入力 dict を保存フィールドへ（欠けはNone）。"""
    return {k: p.get(k) for k in _FIELDS}


class SqliteWinnerStore:
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(fdb.db_path())
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(_SCHEMA)
        return conn

    def upsert(self, posts: List[Dict[str, Any]]) -> int:
        now = _now()
        with closing(self._connect()) as c, c:
            for p in posts:
                r = _row(p)
                # 既存行は本文系のみ更新し event_id は保持（承認を消さない）。新規は event_id=NULL。
                c.execute(
                    """INSERT INTO winner_posts
                         (tweet_id, author, author_name, date, char_name, card_number, leader_raw, tweet_url, event_id, created_at)
                       VALUES (:tid, :author, :author_name, :date, :char_name, :card_number, :leader_raw, :tweet_url, NULL, :now)
                       ON CONFLICT(tweet_id) DO UPDATE SET
                         author=excluded.author, author_name=excluded.author_name, date=excluded.date,
                         char_name=excluded.char_name, card_number=excluded.card_number,
                         leader_raw=excluded.leader_raw, tweet_url=excluded.tweet_url""",
                    {"tid": str(p["tweet_id"]), "now": now, **r},
                )
        return len(posts)

    def list(self, only_unlinked: bool = False) -> List[Dict[str, Any]]:
        q = "SELECT * FROM winner_posts"
        if only_unlinked:
            q += " WHERE event_id IS NULL"
        with closing(self._connect()) as c:
            return [dict(r) for r in c.execute(q + " ORDER BY date DESC, tweet_id").fetchall()]

    def set_event(self, tweet_id: str, event_id: Optional[int]) -> int:
        with closing(self._connect()) as c, c:
            cur = c.execute("UPDATE winner_posts SET event_id=? WHERE tweet_id=?", (event_id, str(tweet_id)))
            return cur.rowcount


class FirestoreWinnerStore:
    def __init__(self, client):
        self._client = client

    def _col(self):
        return self._client.collection(COLLECTION)

    def upsert(self, posts: List[Dict[str, Any]]) -> int:
        now = _now()
        for p in posts:
            ref = self._col().document(str(p["tweet_id"]))
            data = _row(p)
            if getattr(ref.get(), "exists", False):
                ref.set(data, merge=True)                 # event_id は触らない
            else:
                ref.set({**data, "event_id": None, "created_at": now})
        return len(posts)

    def list(self, only_unlinked: bool = False) -> List[Dict[str, Any]]:
        out = []
        for snap in self._col().stream():
            d = snap.to_dict() or {}
            if only_unlinked and d.get("event_id") is not None:
                continue
            out.append({"tweet_id": snap.id, **d})
        out.sort(key=lambda r: (r.get("date") or "", r["tweet_id"]), reverse=True)
        return out

    def set_event(self, tweet_id: str, event_id: Optional[int]) -> int:
        ref = self._col().document(str(tweet_id))
        if not getattr(ref.get(), "exists", False):
            return 0
        ref.update({"event_id": event_id})
        return 1


def get_winner_store():
    """Firestore が使えれば FirestoreWinnerStore、無ければ SqliteWinnerStore。"""
    client = getattr(resources, "db", None)
    return FirestoreWinnerStore(client) if client is not None else SqliteWinnerStore()
