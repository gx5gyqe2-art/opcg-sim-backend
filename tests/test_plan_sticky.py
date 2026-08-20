"""プラン読み出しの sticky 鍵（W4・2026-08-20）。

`_plan_step` のターン内 sticky はかつて `id(manager)` を鍵にしており、apply が新クローンを
返す進め方（リプレイ/ロールアウト）では毎手 id が変わって**毎手立案**になっていた（挙動が
呼び出し環境で変わる隠れ結合）。`_plan_sticky_id`（初見時にトークンを付与・clone=deepcopy が
継承）でどちらの進め方でも「ターンに1回立案・以後は列を1手ずつ」になることを検査する。
"""
import copy
import types

import pytest

import _bootstrap  # noqa: F401

from opcg_sim.src.core import cpu_learned
from opcg_sim.src.learned import plan as PL

pytestmark = pytest.mark.cpu_infra


def _mv(uuid):
    return {"action_type": "PLAY", "payload": {"uuid": uuid}}


MV_A, MV_B, MV_END = _mv("uA"), _mv("uB"), {"action_type": "TURN_END", "payload": {}}


class _Game:
    """legal_actions だけの最小ゲーム（プランの手が常に合法な世界）。"""

    def legal_actions(self, mgr):
        return [MV_A, MV_B, MV_END]


def _engine():
    e = object.__new__(cpu_learned.LearnedEngine)
    e._turn_plans = {}
    e.vnet = None
    e.pnet = None
    e.vocab = None
    e.enc_version = 1
    e.aux_tiebreak = None
    e.game = _Game()
    return e


def _mgr(turn=3):
    return types.SimpleNamespace(turn_count=turn)


@pytest.fixture()
def counted_select(monkeypatch):
    calls = []

    def fake_select(game, manager, name, vf, pf, rng, **kw):
        calls.append(1)
        return (PL.move_sig(MV_A), PL.move_sig(MV_B)), {}

    monkeypatch.setattr(PL, "select_plan", fake_select)
    return calls


def test_inplace_progress_plans_once(counted_select):
    """実対局型（同一オブジェクトを in-place 進行）: 立案1回・列を順に返す。"""
    e, m = _engine(), _mgr()
    assert e._plan_step(m, "p1", None, 0) == MV_A
    assert e._plan_step(m, "p1", None, 0) == MV_B
    assert sum(counted_select) == 1


def test_clone_progress_plans_once(counted_select):
    """リプレイ/ロールアウト型（毎手クローン）: トークンが deepcopy で継承され立案は1回。

    旧実装（id(manager) 鍵）ではここが2回立案になり、2手目も MV_A に戻っていた。"""
    e, m1 = _engine(), _mgr()
    assert e._plan_step(m1, "p1", None, 0) == MV_A
    m2 = copy.deepcopy(m1)          # GameManager.clone() と同じ継承経路
    assert e._plan_step(m2, "p1", None, 0) == MV_B
    assert sum(counted_select) == 1


def test_new_turn_replans(counted_select):
    """ターンが替わったら鍵が変わり再立案する（sticky はターン内のみ）。"""
    e, m = _engine(), _mgr(turn=3)
    assert e._plan_step(m, "p1", None, 0) == MV_A
    m.turn_count = 5
    assert e._plan_step(m, "p1", None, 0) == MV_A
    assert sum(counted_select) == 2


def test_distinct_games_do_not_share(counted_select):
    """別対局（別オブジェクト・トークン未付与）はプランを共有しない。"""
    e = _engine()
    m1, m2 = _mgr(), _mgr()
    assert e._plan_step(m1, "p1", None, 0) == MV_A
    assert e._plan_step(m2, "p1", None, 0) == MV_A   # 別対局は自分の1手目から
    assert sum(counted_select) == 2


def test_plan_exhausted_closes_turn(counted_select):
    """プランが尽きたら TURN_END を返してターンを閉じる（既存規約の回帰）。"""
    e, m = _engine(), _mgr()
    assert e._plan_step(m, "p1", None, 0) == MV_A
    assert e._plan_step(m, "p1", None, 0) == MV_B
    assert e._plan_step(m, "p1", None, 0) == MV_END
    assert sum(counted_select) == 1
