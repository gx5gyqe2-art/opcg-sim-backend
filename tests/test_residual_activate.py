"""残り起動 seam（`LearnedEngine(residual_activate="low"|"high")`・対照生成の腕A2・2026-09-02）の契約。

基盤健全性（`cpu_infra`）: 自己対戦の対照生成の機構であり、ゲームプレイの正しさには触れない。

守る性質:
  1. serve 既定（None）は挙動不変（同一 seed で同一手・events 空・フラグ立たず）。
  2. `_pick_attach_target` の方針: low＝攻撃できるキャラの最低パワー（無ければ全体の最低）／
     high＝最高パワー／候補なし＝None／同点は uuid 順で決定論。
  3. `_leader_has_don_ramp` はテキスト構造語で判定（エネル真・シャンクス偽・None 偽）。
  4. h1@2 の掘り分岐 turn2（エネル・場にサトリ・ドンデッキ残4）で、起動手が返り、続く付与対話で
     サトリが選ばれ、適用するとサトリに付与ドンが乗る（起動→付与の1周がエンジン上で通る）。
"""
import conftest  # noqa: F401
import pytest

import _bootstrap  # noqa: F401

import don_refund_audit as A
from cpu_selfplay import _load_db
from opcg_sim.src.core import action_api
from opcg_sim.src.core.cpu_learned import (LearnedEngine, _leader_has_don_ramp,
                                            _pick_attach_target)

pytestmark = pytest.mark.cpu_infra


@pytest.fixture(scope="module")
def db():
    d = _load_db()
    A._load_h1()
    return d


def test_pick_attach_target_policies():
    c = [("u-big", 8000, True), ("u-mid", 5000, False), ("u-low", 2000, True), ("u-low2", 2000, False)]
    assert _pick_attach_target(c, "low") == "u-low"          # 攻撃できる最低パワー
    assert _pick_attach_target(c, "high") == "u-big"
    assert _pick_attach_target([("a", 3000, False), ("b", 2000, False)], "low") == "b"  # 攻撃不可のみ→全体最低
    assert _pick_attach_target([("z", 2000, True), ("y", 2000, True)], "low") == "y"    # 同点は uuid 順
    assert _pick_attach_target([], "low") is None


def test_leader_has_don_ramp_is_textual(db):
    assert _leader_has_don_ramp(db.get_card("OP15-058"))       # エネル
    assert not _leader_has_don_ramp(db.get_card("OP09-001"))   # シャンクス
    assert not _leader_has_don_ramp(None)


def test_serve_default_unchanged(db):
    eng = LearnedEngine()
    assert eng.residual_activate is None and eng.residual_activate_events == []
    assert eng._resact_pending is False


def test_activate_then_attach_roundtrip_on_h1(db):
    """掘り分岐 turn2: 起動手→（適用）→付与対話で low 方針がサトリを選ぶ→（適用）→付与が乗る。"""
    m, me = A.to_turn2(db, True, log=lambda *_: None)
    pl = A.P(m, me)
    assert len(pl.don_deck) >= 1 and any(c.master.card_id == A.DIG_CARD for c in pl.field)
    eng = LearnedEngine(residual_activate="low")
    mv = eng._residual_activate_move(m, pl)
    assert mv is not None and mv.get("action_type") == "ACTIVATE_MAIN"
    assert eng.residual_activate_events and eng.residual_activate_events[-1]["kind"] == "activate"
    A.apply(m, me, "ACTIVATE_MAIN", mv["payload"])
    assert m.pending_actor_action() and m.pending_actor_action()[0] == me
    att = eng._residual_attach_move(m, pl)
    assert att is not None and att.get("action_type") == "RESOLVE_EFFECT_SELECTION"
    assert eng.residual_activate_events[-1]["kind"] == "attach"
    assert eng.residual_activate_events[-1]["card"] == A.DIG_CARD
    A.apply(m, me, "RESOLVE_EFFECT_SELECTION", att["payload"])
    A.drain(m)
    sat = [c for c in pl.field if c.master.card_id == A.DIG_CARD][0]
    assert A._ad(sat) >= 1
    assert len(pl.don_deck) == 0                       # ドンデッキから引き切った
