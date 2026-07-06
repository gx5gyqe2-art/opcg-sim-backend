"""収集優勝ポストの永続化（`opcg_sim/api/flagship/winnerstore.py`、設計 §16.7）のテスト。

tweet_id での重複除去、再収集で event_id（承認結果）を保持、未紐付け抽出、開催割り当てを、
SQLite（tmp DB）と Fake Firestore の両実装で検証し、`get_winner_store()` の選択も確認する。

実行: OPCG_LOG_SILENT=1 python -m pytest tests/test_flagship_winnerstore.py -q -s -p no:cacheprovider
"""
import conftest  # noqa: F401

import pytest

from opcg_sim.api import resources
from opcg_sim.api.flagship import winnerstore as W


def _post(tid, author="shop", name="店", date="2026-07-05", char="ロロノア・ゾロ", cn="OP01-001"):
    return {"tweet_id": tid, "author": author, "author_name": name, "date": date,
            "char_name": char, "card_number": cn, "leader_raw": None,
            "tweet_url": f"https://x.com/{author}/status/{tid}"}


# --- Fake Firestore（== フィルタのみ・store テストと同型） --------------------

class _Snap:
    def __init__(self, i, d): self.id = i; self._d = d
    @property
    def exists(self): return self._d is not None
    def to_dict(self): return dict(self._d) if self._d is not None else None

class _Ref:
    def __init__(self, col, i): self._col = col; self.id = i
    def set(self, data, merge=False):
        if merge and self.id in self._col._docs:
            self._col._docs[self.id].update(data)
        else:
            self._col._docs[self.id] = dict(data)
    def update(self, patch): self._col._docs.setdefault(self.id, {}).update(patch)
    def get(self): return _Snap(self.id, self._col._docs.get(self.id))

class _Col:
    def __init__(self): self._docs = {}
    def document(self, i): return _Ref(self, str(i))
    def stream(self): return (_Snap(i, d) for i, d in self._docs.items())

class FakeFS:
    def __init__(self): self._c = {}
    def collection(self, n): return self._c.setdefault(n, _Col())


# --- 選択 -------------------------------------------------------------------

def test_get_winner_store_selection(monkeypatch):
    monkeypatch.setattr(resources, "db", FakeFS())
    assert isinstance(W.get_winner_store(), W.FirestoreWinnerStore)
    monkeypatch.setattr(resources, "db", None)
    assert isinstance(W.get_winner_store(), W.SqliteWinnerStore)


# --- 両実装で共通の振る舞い -------------------------------------------------

@pytest.fixture(params=["sqlite", "firestore"])
def store(request, tmp_path, monkeypatch):
    if request.param == "sqlite":
        monkeypatch.setenv("OPCG_FLAGSHIP_DB", str(tmp_path / "f.db"))
        monkeypatch.setattr(resources, "db", None)
    else:
        monkeypatch.setattr(resources, "db", FakeFS())
    return W.get_winner_store()


def test_upsert_dedup_and_list(store):
    store.upsert([_post("1"), _post("2", author="b"), _post("1")])  # tid=1 重複
    rows = store.list()
    assert len(rows) == 2
    assert {r["tweet_id"] for r in rows} == {"1", "2"}
    assert all(r.get("event_id") is None for r in rows)   # 初期は未紐付け


def test_set_event_and_unlinked_filter(store):
    store.upsert([_post("1"), _post("2", author="b")])
    assert store.set_event("1", 7516027) == 1
    unlinked = store.list(only_unlinked=True)
    assert [r["tweet_id"] for r in unlinked] == ["2"]     # 1 は紐付いたので出ない


def test_reupsert_preserves_event_id(store):
    store.upsert([_post("1")])
    store.set_event("1", 999)
    # 再収集（同 tweet_id・本文更新）しても紐付けは保持。
    store.upsert([_post("1", char="更新後")])
    rows = {r["tweet_id"]: r for r in store.list()}
    assert rows["1"]["event_id"] == 999
    assert rows["1"]["char_name"] == "更新後"             # 本文系は更新される


def test_set_event_missing_returns_zero(store):
    assert store.set_event("nope", 1) == 0
