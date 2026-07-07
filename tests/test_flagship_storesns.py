"""店舗X の手動ディレクトリ（`opcg_sim/api/flagship/storesns.py`、設計 §16.7/§16.9）のテスト。

店名 → 店舗X の登録/更新/解除、開催マスターへの上書き優先オーバーレイ、`POST /stores/sns`
（@handle→URL 正規化・空で解除）、`/events` が手動店舗X を TCG+ 値より優先して返すことを、
SQLite（tmp）と Fake Firestore の両実装で検証する。

実行: OPCG_LOG_SILENT=1 python -m pytest tests/test_flagship_storesns.py -q -s -p no:cacheprovider
"""
import conftest  # noqa: F401

import pytest
from fastapi.testclient import TestClient

from opcg_sim.api import app as A
from opcg_sim.api import resources
from opcg_sim.api.flagship import router as R
from opcg_sim.api.flagship import storesns as SS


# --- Fake Firestore（set/get/delete/stream） --------------------------------

class _Snap:
    def __init__(self, i, d): self.id = i; self._d = d
    @property
    def exists(self): return self._d is not None
    def to_dict(self): return dict(self._d) if self._d is not None else None

class _Ref:
    def __init__(self, col, i): self._col = col; self.id = i
    def set(self, data): self._col._docs[self.id] = dict(data)
    def get(self): return _Snap(self.id, self._col._docs.get(self.id))
    def delete(self): self._col._docs.pop(self.id, None)

class _Col:
    def __init__(self): self._docs = {}
    def document(self, i): return _Ref(self, str(i))
    def stream(self): return (_Snap(i, d) for i, d in self._docs.items())

class FakeFS:
    def __init__(self): self._c = {}
    def collection(self, n): return self._c.setdefault(n, _Col())


# --- 選択・両実装の振る舞い -------------------------------------------------

def test_get_store_sns_selection(monkeypatch):
    monkeypatch.setattr(resources, "db", FakeFS())
    assert isinstance(SS.get_store_sns(), SS.FirestoreStoreSns)
    monkeypatch.setattr(resources, "db", None)
    assert isinstance(SS.get_store_sns(), SS.SqliteStoreSns)


@pytest.fixture(params=["sqlite", "firestore"])
def dir_store(request, tmp_path, monkeypatch):
    if request.param == "sqlite":
        monkeypatch.setenv("OPCG_FLAGSHIP_DB", str(tmp_path / "f.db"))
        monkeypatch.setattr(resources, "db", None)
    else:
        monkeypatch.setattr(resources, "db", FakeFS())
    return SS.get_store_sns()


def test_set_get_and_clear(dir_store):
    dir_store.set("カメレオンクラブ福久店", "https://x.com/kame_fukyu")
    dir_store.set("別店", "https://x.com/betten")
    assert dir_store.get("カメレオンクラブ福久店") == "https://x.com/kame_fukyu"
    assert dir_store.get_all() == {
        "カメレオンクラブ福久店": "https://x.com/kame_fukyu", "別店": "https://x.com/betten"}
    # 空/None で解除（削除）
    dir_store.set("別店", None)
    assert dir_store.get("別店") is None
    assert "別店" not in dir_store.get_all()


def test_overlay_prefers_manual(monkeypatch, tmp_path):
    monkeypatch.setenv("OPCG_FLAGSHIP_DB", str(tmp_path / "f.db"))
    monkeypatch.setattr(resources, "db", None)
    SS.get_store_sns().set("店A", "https://x.com/manual_a")
    rows = [
        {"id": 1, "store": "店A", "sns_url": None},           # TCG+ 未登録 → 手動で埋まる
        {"id": 2, "store": "店A", "sns_url": "https://x.com/tcg_a"},  # TCG+ 値も手動が優先
        {"id": 3, "store": "店B", "sns_url": "https://x.com/tcg_b"},  # 手動なし → そのまま
    ]
    out = {r["id"]: r["sns_url"] for r in SS.overlay(rows)}
    assert out[1] == "https://x.com/manual_a"
    assert out[2] == "https://x.com/manual_a"
    assert out[3] == "https://x.com/tcg_b"


# --- API 契約 ---------------------------------------------------------------

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("OPCG_FLAGSHIP_DB", str(tmp_path / "f.db"))
    monkeypatch.setattr(resources, "db", None)
    with TestClient(A.app) as c:
        yield c


def test_stores_sns_normalizes_handle(client):
    res = client.post("/api/flagship/stores/sns", json={"store": "店A", "sns_url": "@kame_fukyu"})
    assert res.status_code == 200
    assert res.json() == {"store": "店A", "sns_url": "https://x.com/kame_fukyu"}


def test_stores_sns_empty_clears(client):
    client.post("/api/flagship/stores/sns", json={"store": "店A", "sns_url": "@x"})
    res = client.post("/api/flagship/stores/sns", json={"store": "店A", "sns_url": ""})
    assert res.status_code == 200 and res.json()["sns_url"] is None
    assert SS.get_store_sns().get("店A") is None


def test_events_returns_manual_sns_over_tcgplus(client, monkeypatch):
    # TCG+ は sns_url 未登録で返す。手動登録した店舗X が /events に反映される。
    monkeypatch.setattr(R.tcgplus, "fetch_events", lambda sid: [
        _FakeEvent(7236374, "カメレオンクラブ福久店", "石川県"),
    ])
    client.post("/api/flagship/stores/sns",
                json={"store": "カメレオンクラブ福久店", "sns_url": "https://x.com/kame"})
    body = client.get("/api/flagship/events", params={"series_id": 7396}).json()
    ev0 = next(e for e in body["events"] if e["id"] == 7236374)
    assert ev0["sns_url"] == "https://x.com/kame"


class _FakeEvent:
    def __init__(self, event_id, store, pref):
        self.event_id = event_id
        self.store = store
        self.pref = pref
        self.date = "2026-07-07"
        self.start_datetime = "2026-07-07T12:30:00"
        self.capacity = 32
        self.sns_url = None
