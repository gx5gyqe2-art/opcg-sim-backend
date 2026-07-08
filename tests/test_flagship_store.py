"""flagship 結果永続化ストア（`opcg_sim/api/flagship/store.py`、設計 §17）のテスト。

本番は Firestore（デッキ管理と同一クライアント）、未設定なら SQLite にフォールバックする。
`get_store()` の選択、FirestoreStore の全置換/取得/削除/サマリ/URL 重複判定を**インメモリの
Fake Firestore** で検証し、さらに `resources.db` を差し替えて **API 全経路（PUT→サマリ→詳細→
DELETE）が Firestore バックエンドでも SQLite と同じ挙動**になることを確認する。

実行: OPCG_LOG_SILENT=1 python -m pytest tests/test_flagship_store.py -q -s -p no:cacheprovider
"""
import conftest  # noqa: F401  (google スタブ注入 & sys.path 設定)

import pytest
from fastapi.testclient import TestClient

from opcg_sim.api import app as A
from opcg_sim.api import resources
from opcg_sim.api.flagship import store as fstore


# --- インメモリ Fake Firestore（== フィルタのみ対応） -----------------------

class _Snap:
    def __init__(self, id, data):
        self.id = id
        self._data = data

    @property
    def exists(self):
        return self._data is not None

    def to_dict(self):
        return dict(self._data) if self._data is not None else None


class _Query:
    def __init__(self, col, preds, limit=None):
        self._col, self._preds, self._limit = col, preds, limit

    def where(self, field, op, value):
        return _Query(self._col, self._preds + [(field, value)], self._limit)

    def limit(self, n):
        return _Query(self._col, self._preds, n)

    def stream(self):
        out = [_Snap(i, d) for i, d in self._col._docs.items()
               if all(d.get(f) == v for f, v in self._preds)]
        return iter(out[:self._limit] if self._limit else out)


class _DocRef:
    def __init__(self, col, id):
        self._col, self.id = col, id

    def set(self, data):
        self._col._docs[self.id] = dict(data)

    def update(self, patch):
        cur = self._col._docs.get(self.id) or {}
        cur.update(patch)
        self._col._docs[self.id] = cur

    def get(self):
        return _Snap(self.id, self._col._docs.get(self.id))


class _Collection:
    def __init__(self):
        self._docs = {}

    def document(self, id):
        return _DocRef(self, id)

    def where(self, field, op, value):
        return _Query(self, [(field, value)])

    def stream(self):
        return _Query(self, []).stream()


class FakeFirestore:
    def __init__(self):
        self._cols = {}

    def collection(self, name):
        return self._cols.setdefault(name, _Collection())


# --- get_store 選択 ---------------------------------------------------------

def test_get_store_selects_firestore_when_available(monkeypatch):
    monkeypatch.setattr(resources, "db", FakeFirestore())
    assert isinstance(fstore.get_store(), fstore.FirestoreStore)


def test_get_store_falls_back_to_sqlite(monkeypatch):
    monkeypatch.setattr(resources, "db", None)
    assert isinstance(fstore.get_store(), fstore.SqliteStore)


# --- FirestoreStore 単体 ----------------------------------------------------

def _event(event_id=7516027, series_id=7664):
    return {"id": event_id, "series_id": series_id, "start_datetime": "2026-08-01 10:00:00",
            "store": "テスト店", "pref": "宮城県", "capacity": 32, "sns_url": None}


def test_firestore_roundtrip_and_summary():
    st = fstore.FirestoreStore(FakeFirestore())
    st.replace_event_results(
        event=_event(),
        post={"url": "https://x.com/a/status/1", "body_text": "優勝：赤ゾロ"},
        results=[{"placement": 1, "leader_card_number": "OP01-001", "leader_raw": None},
                 {"placement": 2, "leader_card_number": None, "leader_raw": "青クロコダイル"}],
    )
    doc = st.get_event_results(7516027)
    assert doc["event_id"] == 7516027
    assert doc["post_url"] == "https://x.com/a/status/1"
    assert doc["body_text"] == "優勝：赤ゾロ"
    assert [r["placement"] for r in doc["results"]] == [1, 2]

    # URL 所有者判定
    assert st.find_url_owner("https://x.com/a/status/1") == 7516027
    assert st.find_url_owner("https://x.com/none") is None

    # シリーズサマリ（優勝リーダー・件数）
    summary = st.get_series_summary(7664)
    assert len(summary) == 1
    assert summary[0]["result_count"] == 2
    assert [w["leader_card_number"] for w in summary[0]["winners"]] == ["OP01-001"]


def test_firestore_full_replace_overwrites():
    st = fstore.FirestoreStore(FakeFirestore())
    st.replace_event_results(_event(), {"url": "https://x.com/a/status/1"},
                             [{"placement": 1, "leader_card_number": "OP01-001"},
                              {"placement": 2, "leader_raw": "X"}])
    # 全置換: 1件だけの結果で上書き
    st.replace_event_results(_event(), {"url": "https://x.com/a/status/2"},
                             [{"placement": 1, "leader_card_number": "ST03-001"}])
    doc = st.get_event_results(7516027)
    assert len(doc["results"]) == 1
    assert doc["results"][0]["leader_card_number"] == "ST03-001"
    assert doc["post_url"] == "https://x.com/a/status/2"
    assert st.find_url_owner("https://x.com/a/status/1") is None  # 旧URLは消える


def test_firestore_delete_keeps_snapshot():
    st = fstore.FirestoreStore(FakeFirestore())
    st.replace_event_results(_event(), {"url": "https://x.com/a/status/1"},
                             [{"placement": 1, "leader_card_number": "OP01-001"}])
    assert st.delete_event_results(7516027) == 1
    doc = st.get_event_results(7516027)
    assert doc is not None                # 開催スナップショットは残る
    assert doc["results"] == [] and doc["post_url"] is None
    assert st.get_series_summary(7664) == []   # 結果無しはサマリに出ない
    assert st.delete_event_results(999999) == 0  # 無い開催は 0


# --- API 全経路（Firestore バックエンド） -----------------------------------

@pytest.fixture
def fs_client(monkeypatch):
    monkeypatch.setattr(resources, "db", FakeFirestore())
    with TestClient(A.app) as c:
        yield c


def _put(client, event_id=7516027, url="https://x.com/foo/status/1", results=None):
    return client.put(f"/api/flagship/events/{event_id}/results", json={
        "event": {"id": event_id, "series_id": 7664, "start_datetime": "2026-08-01 10:00:00",
                  "store": "テスト店", "pref": "宮城県", "capacity": 32, "sns_url": None},
        "post": {"url": url},
        "results": results or [{"placement": 1, "leader_card_number": "OP01-001"}],
    })


def test_api_full_flow_on_firestore(fs_client):
    # 登録
    assert _put(fs_client).status_code == 200
    # サマリ（オーバーレイ）
    summ = fs_client.get("/api/flagship/results", params={"series_id": 7664}).json()
    assert summ["items"][0]["event_id"] == 7516027
    assert [w["leader"]["name"] for w in summ["items"][0]["winners"]] == ["ロロノア・ゾロ"]
    # 詳細
    detail = fs_client.get("/api/flagship/events/7516027/results").json()
    assert detail["post_url"] == "https://x.com/foo/status/1"
    # URL 重複は 409
    assert _put(fs_client, event_id=7516028).status_code == 409
    # 削除
    assert fs_client.delete("/api/flagship/events/7516027/results").json()["deleted"] == 1
    assert fs_client.get("/api/flagship/events/7516027/results").json()["results"] == []
