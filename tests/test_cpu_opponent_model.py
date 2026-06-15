"""リーダー推測の相手モデル（cpu_opponent_model）＋ cpu_templates エンドポイント/配線のテスト。

docs/SPEC.md §2.5.4。Firestore はテストでは未初期化（conftest スタブ）なので、エンドポイントと
プロファイル引き当ては最小の in-memory fake Firestore を appmod.db に差し込んで検証する。
"""
import random
import types

import conftest  # noqa: F401  (google スタブ注入 & sys.path 設定)
import pytest
from fastapi.testclient import TestClient

from opcg_sim.api import app as appmod
from opcg_sim.src.core import cpu_ai, cpu_opponent_model
from opcg_sim.src.core.gamestate import GameManager, Player
from cpu_selfplay import build_deck, _load_db


# ---------------------------------------------------------------------------
# build_profile（純関数）
# ---------------------------------------------------------------------------

def _master(counter=0, cost=3, keywords=None, text=""):
    return types.SimpleNamespace(counter=counter, cost=cost,
                                 keywords=set(keywords or []), effect_text=text)


def test_build_profile_empty_is_neutral():
    p = cpu_opponent_model.build_profile([])
    assert p.n_cards == 0
    assert p.defense_factor == 1.0
    assert p.aggro_lean == 0.5


def test_build_profile_counter_heavy_control_is_defensive():
    """高カウンター・ブロッカー多・除去ありの構築 → defense_factor 高・aggro_lean 低。"""
    cards = ([_master(counter=2000, cost=5, keywords=["ブロッカー"], text="このキャラをKOする")] * 8
             + [_master(counter=1000, cost=4)] * 12)
    p = cpu_opponent_model.build_profile(cards)
    assert p.counter_card_ratio == 1.0
    assert p.blocker_ratio > 0
    assert p.defense_factor > 1.2
    assert p.aggro_lean < 0.4


def test_build_profile_low_cost_no_counter_is_aggro():
    """低コスト・カウンター無し・除去無しの構築 → aggro_lean 高・defense_factor 低。"""
    cards = [_master(counter=0, cost=1)] * 10 + [_master(counter=0, cost=2)] * 10
    p = cpu_opponent_model.build_profile(cards)
    assert p.counter_card_ratio == 0.0
    assert p.aggro_lean > 0.7
    assert p.defense_factor < 1.0


# ---------------------------------------------------------------------------
# evaluate へのプロファイル影響
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def db():
    return _load_db()


def _profile(defense_factor=1.0, aggro_lean=0.0):
    return cpu_opponent_model.OpponentProfile(
        n_cards=50, counter_avg=0.0, counter_card_ratio=0.0, blocker_ratio=0.0,
        removal_ratio=0.0, avg_cost=3.0, defense_factor=defense_factor, aggro_lean=aggro_lean)


def test_profile_defense_factor_lowers_my_eval(db):
    """defense_factor>1（相手の守りが厚い推測）→ 相手手札の価値が増し、自分視点の評価が下がる。"""
    random.seed(0)
    l1, c1 = build_deck(db, "p1")
    l2, c2 = build_deck(db, "p2")
    gm = GameManager(Player("p1", c1, l1), Player("p2", c2, l2))
    gm.start_game()
    if not gm.p2.hand:
        pytest.skip("相手手札が空")
    base = cpu_ai.evaluate(gm, "p1", see_opp_hand=False, profile=None)
    defensive = cpu_ai.evaluate(gm, "p1", see_opp_hand=False, profile=_profile(defense_factor=1.6))
    assert defensive < base


def test_profile_aggro_lean_raises_own_life_value(db):
    """aggro_lean 高（相手が攻め寄り）→ 自分のライフ重視で、ライフを持つ自分視点の評価が上がる。"""
    random.seed(0)
    l1, c1 = build_deck(db, "p1")
    l2, c2 = build_deck(db, "p2")
    gm = GameManager(Player("p1", c1, l1), Player("p2", c2, l2))
    gm.start_game()
    base = cpu_ai.evaluate(gm, "p1", see_opp_hand=False, profile=_profile(aggro_lean=0.0))
    aggro = cpu_ai.evaluate(gm, "p1", see_opp_hand=False, profile=_profile(aggro_lean=1.0))
    assert aggro > base  # 自分のライフ価値が増す


# ---------------------------------------------------------------------------
# cpu_templates エンドポイント + プロファイル引き当て（in-memory fake Firestore）
# ---------------------------------------------------------------------------

class _FakeSnap:
    def __init__(self, data):
        self._data = data
    @property
    def exists(self):
        return self._data is not None
    def to_dict(self):
        return dict(self._data) if self._data is not None else None


class _FakeDoc:
    def __init__(self, coll, doc_id):
        self._coll = coll
        self.id = doc_id
    def set(self, data):
        self._coll.store[self.id] = dict(data)
    def get(self):
        return _FakeSnap(self._coll.store.get(self.id))
    def delete(self):
        self._coll.store.pop(self.id, None)


class _FakeQuery:
    def __init__(self, rows):
        self._rows = rows
    def where(self, field, op, value):
        return _FakeQuery([r for r in self._rows if r.get(field) == value])
    def order_by(self, field, direction=None):
        return _FakeQuery(list(reversed(self._rows)))  # 挿入順の逆＝新しい順の近似
    def limit(self, n):
        return _FakeQuery(self._rows[:n])
    def stream(self):
        return [_FakeSnap(r) for r in self._rows]


class _FakeColl:
    def __init__(self):
        self.store = {}
        self._auto = 0
    def document(self, doc_id=None):
        if not doc_id:
            self._auto += 1
            doc_id = f"auto{self._auto}"
        return _FakeDoc(self, doc_id)
    def _q(self):
        return _FakeQuery(list(self.store.values()))
    def where(self, *a, **k):
        return self._q().where(*a, **k)
    def order_by(self, *a, **k):
        return self._q().order_by(*a, **k)
    def stream(self):
        return self._q().stream()


class _FakeClient:
    def __init__(self):
        self._colls = {}
    def collection(self, name):
        return self._colls.setdefault(name, _FakeColl())


@pytest.fixture
def fake_db(monkeypatch):
    client = _FakeClient()
    monkeypatch.setattr(appmod, "db", client)
    return client


def test_cpu_template_crud_and_profile(fake_db):
    """テンプレ保存→一覧→取得→プロファイル引き当て→削除が一通り通る。"""
    client = TestClient(appmod.app)
    # 実カード DB のリーダー1枚＋キャラ数枚で簡易テンプレを作る。
    appmod.card_db.load()
    leader_id = next(cid for cid in appmod.card_db.raw_db
                     if appmod.card_db.get_card(cid) and appmod.card_db.get_card(cid).type.name == "LEADER")
    char_ids = [cid for cid in appmod.card_db.raw_db
                if appmod.card_db.get_card(cid) and appmod.card_db.get_card(cid).type.name == "CHARACTER"][:20]

    res = client.post("/api/cpu_template", json={
        "name": "テストテンプレ", "leader_id": leader_id, "card_uuids": char_ids,
    }).json()
    assert res["success"] and res["template_id"]
    tid = res["template_id"]

    lst = client.get("/api/cpu_template/list").json()
    assert lst["success"] and any(t["id"] == tid for t in lst["templates"])

    got = client.get(f"/api/cpu_template/get?id={tid}").json()
    assert got["success"] and got["template"]["leader_id"] == leader_id

    # リーダーからプロファイルを引き当て（隠れ情報を読まずテンプレ集計のみ）。
    prof = appmod.build_opp_profile_for_leader(leader_id)
    assert prof is not None and prof.n_cards == len(char_ids)

    dele = client.delete(f"/api/cpu_template/{tid}").json()
    assert dele["success"]
    assert appmod.build_opp_profile_for_leader(leader_id) is None  # 削除後は引き当て無し


def test_cpu_template_save_without_db_is_graceful(monkeypatch):
    """db 未初期化でもエンドポイントは例外を投げず success=False を返す。"""
    monkeypatch.setattr(appmod, "db", None)
    client = TestClient(appmod.app)
    res = client.post("/api/cpu_template", json={"name": "x", "leader_id": "L", "card_uuids": []}).json()
    assert res["success"] is False
