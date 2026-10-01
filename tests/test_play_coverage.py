"""`play_coverage.py`（エンジンはその手を打てるのか・能力の検査）を固める。

**頻度と能力を混ぜないための器**なので、**判定の母数の決め方**が命。
ここが緩むと、**正しい挙動を欠陥として報告する**（2026-09-14 に実際やった）。
"""
import os
import sys

import pytest

pytestmark = pytest.mark.cpu_infra

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bootstrap  # noqa: F401,E402

_SCRIPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import play_coverage as P  # noqa: E402


def test_a_counter_only_event_is_not_expected_in_the_main_phase():
    """**`COUNTER`／`TRIGGER` だけのイベントはメインで打てないのが正解**。

    2026-09-14 に母数から外し忘れ、**正しい挙動を「打てない」と誤報した**。
    """
    counter_only = [{"trigger": "COUNTER"}, {"trigger": "TRIGGER"}]
    assert P.main_playable_expected("event", counter_only) is False
    with_main = [{"trigger": "ACTIVATE_MAIN"}, {"trigger": "COUNTER"}]
    assert P.main_playable_expected("event", with_main) is True
    assert P.main_playable_expected("event", [{"trigger": "ON_PLAY"}]) is True


def test_permanents_are_always_expected_to_be_playable():
    """キャラとステージは場に残る常設なので、契機に関係なくメインで打てるはず。"""
    assert P.main_playable_expected("char", [{"trigger": "COUNTER"}]) is True
    assert P.main_playable_expected("stage", [{"trigger": "COUNTER"}]) is True


def test_an_event_with_no_parsed_ability_is_still_expected():
    """**契機が読めていないカードを母数から外すと、穴が見えなくなる**ので通す。"""
    assert P.main_playable_expected("event", []) is True
    assert P.main_playable_expected("event", None) is True


def test_the_kind_is_read_from_the_card_flags():
    assert P.kind_of({"leader": True}) == "leader"
    assert P.kind_of({"stage": True}) == "stage"
    assert P.kind_of({"event": True}) == "event"
    assert P.kind_of({}) == "char"
    assert P.kind_of(None) is None


def test_a_kind_that_never_plays_is_named_in_the_verdict():
    """**①合法かの層で × が出たら名指しする**——そこは評価では直せない（§6）。"""
    bad = P.verdict({"char": {"playable_share": 1.0},
                     "stage": {"playable_share": 0.0}})
    assert bad["verdict"] == "not_legal_for_some_kinds" and bad["kinds"] == ["stage"]
    ok = P.verdict({"char": {"playable_share": 1.0}, "stage": {"playable_share": 1.0}})
    assert ok["verdict"] == "all_kinds_legal"


def test_a_kind_with_nothing_in_scope_is_not_called_broken():
    """**母数が空なら判定しない**（`None`）——0 と混ぜると偽の欠陥になる。"""
    assert P.verdict({"event": {"playable_share": None}})["verdict"] == "all_kinds_legal"


def test_the_engine_offers_a_cheap_stage_in_the_main_phase():
    """**ステージは打てる**——棋譜で 0 だったのは頻度の話で、能力の話ではなかった。

    ここが落ちたらエンジン側の退行（ステージが出せなくなった）である。
    """
    got = P.probe_play("EB01-011")          # コスト 1 のステージ
    assert got["in_hand"] > 0, got
    assert got["playable"] is True, got
    assert "PLAY" in got["action_types"], got
