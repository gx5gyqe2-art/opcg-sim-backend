"""開催マスターの永続化（`opcg_sim/api/flagship/eventmaster.py`・`GET /events`、設計 §16.8）のテスト。

TCG+ が過去開催を消しても、once スナップショットした開催がマスターに残ること（＝`/events` が
過去+現行を返すこと）を、検索/ TCG+ を monkeypatch し SQLite(tmp)・Fake Firestore で検証する。

実行: OPCG_LOG_SILENT=1 python -m pytest tests/test_flagship_eventmaster.py -q -s -p no:cacheprovider
"""
import conftest  # noqa: F401

import pytest
from fastapi.testclient import TestClient

from opcg_sim.api import app as A
from opcg_sim.api import resources
from opcg_sim.api.flagship import eventmaster as EM
from opcg_sim.api.flagship import match as M
from opcg_sim.api.flagship import router as R


# --- Fake Firestore ---------------------------------------------------------

class _Snap:
    def __init__(self, i, d): self.id = i; self._d = d
    def to_dict(self): return dict(self._d)

class _Q:
    def __init__(self, col, preds): self._col = col; self._preds = preds
    def where(self, f, op, v): return _Q(self._col, self._preds + [(f, v)])
    def stream(self):
        return (_Snap(i, d) for i, d in self._col._docs.items()
                if all(d.get(f) == v for f, v in self._preds))

class _Ref:
    def __init__(self, col, i): self._col = col; self.id = i
    def set(self, data): self._col._docs[self.id] = dict(data)

class _Col:
    def __init__(self): self._docs = {}
    def document(self, i): return _Ref(self, str(i))
    def where(self, f, op, v): return _Q(self, [(f, v)])

class FakeFS:
    def __init__(self): self._c = {}
    def collection(self, n): return self._c.setdefault(n, _Col())


def _ev(**o):
    d = {"id": 1, "series_id": 7395, "start_datetime": "2026-07-05T13:00:00",
         "store": "店A", "pref": "東京都", "capacity": 32, "sns_url": "https://x.com/a"}
    d.update(o)
    return d


def test_get_event_master_selection(monkeypatch):
    monkeypatch.setattr(resources, "db", FakeFS())
    assert isinstance(EM.get_event_master(), EM.FirestoreEventMaster)
    monkeypatch.setattr(resources, "db", None)
    assert isinstance(EM.get_event_master(), EM.SqliteEventMaster)


@pytest.fixture(params=["sqlite", "firestore"])
def store(request, tmp_path, monkeypatch):
    if request.param == "sqlite":
        monkeypatch.setenv("OPCG_FLAGSHIP_DB", str(tmp_path / "f.db"))
        monkeypatch.setattr(resources, "db", None)
    else:
        monkeypatch.setattr(resources, "db", FakeFS())
    return EM.get_event_master()


def test_upsert_and_list_by_series(store):
    store.upsert([_ev(id=1, series_id=7395), _ev(id=2, series_id=7395, store="店B"),
                  _ev(id=3, series_id=7664, store="店C")])
    ids = [e["id"] for e in store.list(7395)]
    assert ids == [1, 2] or set(ids) == {1, 2}
    assert [e["id"] for e in store.list(7664)] == [3]
    got = store.list(7395)[0]
    assert got["store"] and got["capacity"] == 32 and got["sns_url"] == "https://x.com/a"


# --- /events：過去開催が TCG+ から消えても残る ------------------------------

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("OPCG_FLAGSHIP_DB", str(tmp_path / "f.db"))
    monkeypatch.setattr(resources, "db", None)
    with TestClient(A.app) as c:
        yield c


def _se(eid, store, date):
    return M.StoreEvent(event_id=eid, store=store, date=date, sns_url=None, pref="東京都",
                        start_datetime=f"{date}T13:00:00", capacity=32)


def test_events_persists_past_after_tcgplus_drops(client, monkeypatch):
    # 1回目: TCG+ が A(7/05)・B(7/06) を返す → マスターに両方入る
    monkeypatch.setattr(R.tcgplus, "fetch_events", lambda sid: [_se(1, "店A", "2026-07-05"), _se(2, "店B", "2026-07-06")])
    r1 = client.get("/api/flagship/events", params={"series_id": 7395}).json()
    assert {e["id"] for e in r1["events"]} == {1, 2}

    # 2回目: TCG+ が A を消し B のみ返す → /events は A も残して返す（過去保持）
    monkeypatch.setattr(R.tcgplus, "fetch_events", lambda sid: [_se(2, "店B", "2026-07-06")])
    r2 = client.get("/api/flagship/events", params={"series_id": 7395}).json()
    assert {e["id"] for e in r2["events"]} == {1, 2}          # A(1) は消えない
    a = next(e for e in r2["events"] if e["id"] == 1)
    assert a["store"] == "店A" and a["start_datetime"].startswith("2026-07-05")


def test_events_tcgplus_error_returns_master(client, monkeypatch):
    monkeypatch.setattr(R.tcgplus, "fetch_events", lambda sid: [_se(1, "店A", "2026-07-05")])
    client.get("/api/flagship/events", params={"series_id": 7395})   # マスターに1件
    monkeypatch.setattr(R.tcgplus, "fetch_events",
                        lambda sid: (_ for _ in ()).throw(R.tcgplus.TcgPlusError("down")))
    r = client.get("/api/flagship/events", params={"series_id": 7395})
    assert r.status_code == 200 and {e["id"] for e in r.json()["events"]} == {1}
