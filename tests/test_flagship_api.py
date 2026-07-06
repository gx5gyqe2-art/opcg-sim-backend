"""フラッグシップ結果集計 API のテスト（`opcg_sim/api/flagship/`、設計 §12）。

リーダー辞書（カードDB `種類=リーダー` 137件）の配信と、結果の登録（開催単位の
全置換・冪等）→ サマリ → 詳細 → 削除の一連、URL 重複 409、入力バリデーション、
SQLite の遅延作成（`OPCG_FLAGSHIP_DB`）を検証する。

実行: OPCG_LOG_SILENT=1 python -m pytest tests/test_flagship_api.py -q -s -p no:cacheprovider
"""
import conftest  # noqa: F401  (google スタブ注入 & sys.path 設定)

import os

import pytest
from fastapi.testclient import TestClient

from opcg_sim.api import app as A


@pytest.fixture
def client(tmp_path, monkeypatch):
    """flagship DB を tmp に向けた TestClient。テスト毎に空の DB から始まる。"""
    monkeypatch.setenv("OPCG_FLAGSHIP_DB", str(tmp_path / "flagship.db"))
    with TestClient(A.app) as c:
        yield c


def _snapshot(event_id=7516027, series_id=7664, **over):
    ev = {
        "id": event_id, "series_id": series_id,
        "start_datetime": "2026-08-01 10:00:00",
        "store": "タンヨ玩具店", "pref": "宮城県", "capacity": 32,
        "sns_url": "https://twitter.com/Tanyo_Shiogama",
    }
    ev.update(over)
    return ev


def _put(client, event_id=7516027, url="https://x.com/foo/status/1", results=None, **snap_over):
    return client.put(
        f"/api/flagship/events/{event_id}/results",
        json={
            "event": _snapshot(event_id, **snap_over),
            "post": {"url": url},
            "results": results or [{"placement": 1, "leader_card_number": "OP01-001"}],
        },
    )


# --- リーダー辞書 -----------------------------------------------------------

def test_leaders_dictionary(client):
    res = client.get("/api/flagship/leaders")
    assert res.status_code == 200
    leaders = res.json()
    assert len(leaders) == 137  # docs/leader_specs/ の全リーダー数と一致
    by_number = {l["card_number"]: l for l in leaders}
    zoro = by_number["OP01-001"]
    assert zoro["name"] == "ロロノア・ゾロ" and zoro["color"] == "赤" and zoro["life"] == "5"
    # 全件が名前を持つ（辞書として使える）
    assert all(l["name"] for l in leaders)


# --- 登録 → サマリ → 詳細 → 全置換 → 削除 -----------------------------------

def test_register_summary_detail_roundtrip(client):
    res = _put(client, results=[
        {"placement": 1, "leader_card_number": "OP01-001"},
        {"placement": 2, "leader_raw": "赤シャンクス"},
    ])
    assert res.status_code == 200
    body = res.json()
    assert body["event_id"] == 7516027 and len(body["results"]) == 2
    assert body["results"][0]["leader"]["name"] == "ロロノア・ゾロ"  # 辞書解決
    assert body["results"][1]["leader"] is None  # raw のみは未解決

    # サマリ（一覧オーバーレイ）
    res = client.get("/api/flagship/results", params={"series_id": 7664})
    assert res.status_code == 200
    items = res.json()["items"]
    assert len(items) == 1
    assert items[0]["event_id"] == 7516027 and items[0]["result_count"] == 2
    assert items[0]["winner"]["leader"]["name"] == "ロロノア・ゾロ"
    assert items[0]["post_url"] == "https://x.com/foo/status/1"

    # 別シリーズのサマリには出ない
    assert client.get("/api/flagship/results", params={"series_id": 7395}).json()["items"] == []

    # 詳細
    res = client.get("/api/flagship/events/7516027/results")
    assert res.status_code == 200
    detail = res.json()
    assert detail["event"]["store"] == "タンヨ玩具店"
    assert [r["placement"] for r in detail["results"]] == [1, 2]


def test_put_is_full_replacement(client):
    _put(client, results=[
        {"placement": 1, "leader_card_number": "OP01-001"},
        {"placement": 2, "leader_card_number": "OP01-002"},
    ])
    # 同じ開催へ優勝のみで再登録 → 追記ではなく置換される（訂正フロー）
    res = _put(client, results=[{"placement": 1, "leader_card_number": "OP01-060"}])
    assert res.status_code == 200
    results = res.json()["results"]
    assert len(results) == 1 and results[0]["leader_card_number"] == "OP01-060"


def test_delete_results(client):
    _put(client)
    res = client.delete("/api/flagship/events/7516027/results")
    assert res.status_code == 200 and res.json()["deleted"] == 1
    # サマリから消える（回収状況は results の有無で導出）
    assert client.get("/api/flagship/results", params={"series_id": 7664}).json()["items"] == []
    # スナップショット行は残り、詳細は空結果で返る
    detail = client.get("/api/flagship/events/7516027/results").json()
    assert detail["results"] == []


def test_unknown_event_detail_404(client):
    assert client.get("/api/flagship/events/999999/results").status_code == 404


# --- URL 重複・バリデーション ------------------------------------------------

def test_url_conflict_409(client):
    _put(client, event_id=7516027, url="https://x.com/foo/status/1")
    # 同じ URL を別開催に投入 → 409（同一開催への再投入は許容 = 冪等）
    res = _put(client, event_id=7516028, url="https://x.com/foo/status/1")
    assert res.status_code == 409
    assert "7516027" in res.json()["detail"]
    res = _put(client, event_id=7516027, url="https://x.com/foo/status/1")
    assert res.status_code == 200


def test_validation_errors(client):
    # placement 重複
    assert _put(client, results=[
        {"placement": 1, "leader_card_number": "OP01-001"},
        {"placement": 1, "leader_card_number": "OP01-002"},
    ]).status_code == 422
    # 優勝(1)なし
    assert _put(client, results=[{"placement": 2, "leader_card_number": "OP01-001"}]).status_code == 422
    # リーダー情報なし
    assert _put(client, results=[{"placement": 1}]).status_code == 422
    # 未知のリーダー番号
    assert _put(client, results=[{"placement": 1, "leader_card_number": "ZZ99-999"}]).status_code == 422
    # 範囲外 placement
    assert _put(client, results=[{"placement": 9, "leader_card_number": "OP01-001"}]).status_code == 422
    # event.id とパスの不一致
    res = client.put(
        "/api/flagship/events/1/results",
        json={"event": _snapshot(2), "results": [{"placement": 1, "leader_card_number": "OP01-001"}]},
    )
    assert res.status_code == 400


# --- SQLite 遅延初期化 --------------------------------------------------------

def test_db_created_lazily(client, tmp_path):
    path = tmp_path / "flagship.db"
    # アプリ起動・他エンドポイントの疎通では DB ファイルは作られない
    client.get("/api/flagship/leaders")
    assert not path.exists()
    # 結果系エンドポイントに触れた時点で作成される
    client.get("/api/flagship/results", params={"series_id": 7664})
    assert path.exists()
