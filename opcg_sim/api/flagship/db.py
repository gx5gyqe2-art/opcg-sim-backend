"""flagship の SQLite 永続化（遅延初期化・接続はリクエスト毎に開閉）。

DB ファイルは既定 `opcg_sim/data/flagship.db`。環境変数 `OPCG_FLAGSHIP_DB` で
差し替えられる（テストは tmp パスを指す）。スキーマは接続時に CREATE IF NOT EXISTS
で保証するため、flagship を使わないプロセスにはファイルすら作られない。

スキーマ（設計 §12.3）:
- events: 結果が付いた開催のみ蓄積するスナップショット（TCG+ の event id が PK）
- result_posts: 結果の出どころ（URL は非 NULL のみ UNIQUE）
- results: placement 単位・(event_id, placement) UNIQUE。登録は開催単位の全置換
"""
import os
import json
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..config import DATA_DIR

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
  id             INTEGER PRIMARY KEY,
  series_id      INTEGER NOT NULL,
  start_datetime TEXT NOT NULL,
  store          TEXT NOT NULL,
  pref           TEXT NOT NULL DEFAULT '',
  capacity       INTEGER,
  sns_url        TEXT,
  snapshot_json  TEXT NOT NULL,
  updated_at     TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS result_posts (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  event_id   INTEGER NOT NULL REFERENCES events(id),
  url        TEXT,
  body_text  TEXT,
  created_at TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_result_posts_url
  ON result_posts(url) WHERE url IS NOT NULL;
CREATE TABLE IF NOT EXISTS results (
  id                 INTEGER PRIMARY KEY AUTOINCREMENT,
  event_id           INTEGER NOT NULL REFERENCES events(id),
  post_id            INTEGER REFERENCES result_posts(id),
  placement          INTEGER NOT NULL,
  leader_card_number TEXT,
  leader_raw         TEXT,
  created_at         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_results_event ON results(event_id);
-- 優勝(placement=1)は定員64の2ブロック開催で最大2件（§16.11）。入賞(≥2)は開催内で一意。
CREATE UNIQUE INDEX IF NOT EXISTS idx_results_place ON results(event_id, placement) WHERE placement > 1;
"""


def db_path() -> str:
    return os.environ.get("OPCG_FLAGSHIP_DB") or os.path.join(DATA_DIR, "flagship.db")


def connect() -> sqlite3.Connection:
    """接続を開いてスキーマを保証する。呼び出し側が close する（with 推奨）。"""
    conn = sqlite3.connect(db_path())
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(_SCHEMA)
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def find_url_owner(conn: sqlite3.Connection, url: str) -> Optional[int]:
    """URL が既に別開催へ紐付いていればその event_id を返す（重複 409 判定用）。"""
    row = conn.execute("SELECT event_id FROM result_posts WHERE url = ?", (url,)).fetchone()
    return row["event_id"] if row else None


def replace_event_results(
    conn: sqlite3.Connection,
    event: Dict[str, Any],
    post: Optional[Dict[str, Any]],
    results: List[Dict[str, Any]],
) -> None:
    """開催スナップショットを UPSERT し、その開催の結果・ポストを全置換する（1トランザクション）。"""
    now = _now()
    with conn:  # BEGIN..COMMIT（例外時 ROLLBACK）
        conn.execute(
            """INSERT INTO events (id, series_id, start_datetime, store, pref, capacity, sns_url, snapshot_json, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                 series_id=excluded.series_id, start_datetime=excluded.start_datetime,
                 store=excluded.store, pref=excluded.pref, capacity=excluded.capacity,
                 sns_url=excluded.sns_url, snapshot_json=excluded.snapshot_json,
                 updated_at=excluded.updated_at""",
            (
                event["id"], event["series_id"], event["start_datetime"], event["store"],
                event.get("pref") or "", event.get("capacity"), event.get("sns_url"),
                json.dumps(event, ensure_ascii=False, sort_keys=True), now,
            ),
        )
        conn.execute("DELETE FROM results WHERE event_id = ?", (event["id"],))
        conn.execute("DELETE FROM result_posts WHERE event_id = ?", (event["id"],))
        post_id = None
        if post and (post.get("url") or post.get("body_text")):
            cur = conn.execute(
                "INSERT INTO result_posts (event_id, url, body_text, created_at) VALUES (?, ?, ?, ?)",
                (event["id"], post.get("url"), post.get("body_text"), now),
            )
            post_id = cur.lastrowid
        for r in results:
            conn.execute(
                """INSERT INTO results (event_id, post_id, placement, leader_card_number, leader_raw, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (event["id"], post_id, r["placement"], r.get("leader_card_number"), r.get("leader_raw"), now),
            )


def delete_event_results(conn: sqlite3.Connection, event_id: int) -> int:
    """開催の結果・ポストを削除する（開催スナップショット行は残す）。削除した結果行数を返す。"""
    with conn:
        cur = conn.execute("DELETE FROM results WHERE event_id = ?", (event_id,))
        conn.execute("DELETE FROM result_posts WHERE event_id = ?", (event_id,))
    return cur.rowcount


def get_event_results(conn: sqlite3.Connection, event_id: int) -> Optional[Dict[str, Any]]:
    """開催詳細用: スナップショット + ポスト + 全 placement。開催行が無ければ None。"""
    ev = conn.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
    if ev is None:
        return None
    post = conn.execute(
        "SELECT url, body_text FROM result_posts WHERE event_id = ? ORDER BY id DESC LIMIT 1", (event_id,)
    ).fetchone()
    rows = conn.execute(
        "SELECT placement, leader_card_number, leader_raw FROM results WHERE event_id = ? ORDER BY placement",
        (event_id,),
    ).fetchall()
    return {
        "event_id": ev["id"],
        "event": json.loads(ev["snapshot_json"]),
        "updated_at": ev["updated_at"],
        "post_url": post["url"] if post else None,
        "body_text": post["body_text"] if post else None,
        "results": [dict(r) for r in rows],
    }


def get_series_summary(conn: sqlite3.Connection, series_id: int) -> List[Dict[str, Any]]:
    """一覧オーバーレイ用: シリーズ内で結果を持つ開催のサマリ（優勝・件数・ポストURL）。

    優勝（placement=1）は定員64の2ブロック開催で2件になり得るため `winners` はリストで返す（§16.11）。
    """
    rows = conn.execute(
        """SELECT e.id AS event_id, r.placement, r.leader_card_number, r.leader_raw,
                  (SELECT url FROM result_posts p WHERE p.event_id = e.id ORDER BY p.id DESC LIMIT 1) AS post_url
           FROM events e JOIN results r ON r.event_id = e.id
           WHERE e.series_id = ?
           ORDER BY e.id, r.placement, r.id""",
        (series_id,),
    ).fetchall()
    agg: Dict[int, Dict[str, Any]] = {}
    order: List[int] = []
    for r in rows:
        eid = r["event_id"]
        a = agg.get(eid)
        if a is None:
            a = agg[eid] = {"event_id": eid, "result_count": 0, "winners": [], "post_url": r["post_url"]}
            order.append(eid)
        a["result_count"] += 1
        if r["placement"] == 1:
            a["winners"].append({
                "leader_card_number": r["leader_card_number"], "leader_raw": r["leader_raw"],
            })
    return [agg[eid] for eid in order]
