"""収集の蓄積と開催紐付け（`/collect`・`/link/review`・`/link/approve`、設計 §16.7）のテスト。

収集→DB蓄積、未紐付けポストの TCG+開催への照合レビュー（handle自動候補）、一括承認保存を、
検索と TCG+ を monkeypatch し SQLite(tmp) 永続で検証する。

実行: OPCG_LOG_SILENT=1 python -m pytest tests/test_flagship_link.py -q -s -p no:cacheprovider
"""
import conftest  # noqa: F401

import pytest
from fastapi.testclient import TestClient

from opcg_sim.api import app as A
from opcg_sim.api import resources
from opcg_sim.api.flagship import match as M
from opcg_sim.api.flagship import router as R
from opcg_sim.api.flagship import winnerstore as W
from opcg_sim.api.flagship import xsearch as S


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("OPCG_FLAGSHIP_DB", str(tmp_path / "f.db"))
    monkeypatch.setenv("X_BEARER_TOKEN", "tok")
    monkeypatch.setattr(resources, "db", None)   # winner_posts は SQLite(tmp)
    with TestClient(A.app) as c:
        yield c


def _seed_collect(client, monkeypatch):
    hits = [
        S.SearchHit("111", "https://x.com/shopA/status/111", "本日フラッグシップ 優勝：赤ゾロ",
                    "shopa", "ゲームスペース鶴岡", "2026-07-05T10:00:00.000Z"),
        S.SearchHit("222", "https://x.com/gokuu/status/222", "フラッグシップ優勝しました！青クロコダイル",
                    "gokuu_08", "孫悟空", "2026-07-05T11:00:00.000Z"),  # 個人
    ]
    monkeypatch.setattr(R.xsearch, "search_recent", lambda *a, **k: hits)
    return client.post("/api/flagship/collect", json={"pages": 1})


def test_collect_stores_posts(client, monkeypatch):
    res = _seed_collect(client, monkeypatch)
    assert res.status_code == 200 and res.json()["collected"] == 2


def test_review_matches_by_handle_and_lists_unlinked(client, monkeypatch):
    _seed_collect(client, monkeypatch)
    # shopa は snsUrl 一致（handle 自動候補）、孫悟空は候補ゼロ（個人）。
    events = [M.StoreEvent(1, "ゲームスペース鶴岡", "2026-07-05", "https://x.com/shopA")]
    monkeypatch.setattr(R.tcgplus, "fetch_events", lambda sid: events)
    body = client.get("/api/flagship/link/review", params={"series_id": 7395}).json()
    by = {p["tweet_id"]: p for p in body["posts"]}
    assert body["events"] == 1
    assert by["111"]["candidates"][0]["event_id"] == 1 and by["111"]["candidates"][0]["auto"] is True
    assert by["222"]["candidates"] == []          # 個人ポストは未紐付けのまま


def test_approve_deletes_collected_post(client, monkeypatch):
    # 結果はフロントが別途 PUT 済みの想定。承認＝紐付け確定で収集ポストを掃除する（§16.7）。
    _seed_collect(client, monkeypatch)
    events = [M.StoreEvent(1, "ゲームスペース鶴岡", "2026-07-05", "https://x.com/shopA")]
    monkeypatch.setattr(R.tcgplus, "fetch_events", lambda sid: events)
    res = client.post("/api/flagship/link/approve", json={"links": [{"tweet_id": "111", "event_id": 1}]})
    assert res.status_code == 200 and res.json()["updated"] == 1
    # 承認した収集ポストは winner_posts から完全に削除（ポスト内容は恒久保持しない）
    all_ids = {p["tweet_id"] for p in W.get_winner_store().list()}
    assert "111" not in all_ids and "222" in all_ids
    # レビュー（未紐付け）にも当然出ない
    body = client.get("/api/flagship/link/review", params={"series_id": 7395}).json()
    assert "111" not in {p["tweet_id"] for p in body["posts"]}


def test_approve_null_unlinks_and_keeps_row(client, monkeypatch):
    # event_id=null は紐付け解除＝行は残す（未紐付けへ戻すだけ・削除しない）。
    _seed_collect(client, monkeypatch)
    res = client.post("/api/flagship/link/approve", json={"links": [{"tweet_id": "222", "event_id": None}]})
    assert res.status_code == 200
    assert "222" in {p["tweet_id"] for p in W.get_winner_store().list()}


def test_review_tcgplus_error_falls_back_to_master(client, monkeypatch):
    _seed_collect(client, monkeypatch)

    def boom(sid):
        raise R.tcgplus.TcgPlusError("TCG+ down")

    monkeypatch.setattr(R.tcgplus, "fetch_events", boom)
    res = client.get("/api/flagship/link/review", params={"series_id": 7395})
    # TCG+ 不達でもマスターを使う（今回はマスター空→候補なし・200）。
    assert res.status_code == 200
    assert all(not p["candidates"] for p in res.json()["posts"])


def test_collect_disabled_is_503(client, monkeypatch):
    monkeypatch.delenv("X_BEARER_TOKEN", raising=False)
    assert client.post("/api/flagship/collect", json={}).status_code == 503
