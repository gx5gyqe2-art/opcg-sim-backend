"""中断（対話）スタックの基礎ユニットテスト（Phase 2a・純粋リファクタの土台）。

`active_interaction` は互換プロパティ＝スタック先頭。単一スロット時代の読み書き
（getter=先頭、setter(None)=pop、setter(dict)=先頭置換／空なら push）が保たれること、
および `push_interaction` でネスト中断（深さ>1）を表現できることを固定する。
"""
import conftest  # noqa: F401

from opcg_sim.src.core.gamestate import GameManager, Player


def _gm():
    p1 = Player(name="P1", deck=[], leader=None)
    p2 = Player(name="P2", deck=[], leader=None)
    return GameManager(player1=p1, player2=p2)


def test_empty_is_none():
    gm = _gm()
    assert gm.active_interaction is None
    assert gm._interaction_stack == []


def test_set_then_clear_single_slot():
    gm = _gm()
    a = {"action_type": "A"}
    gm.active_interaction = a
    assert gm.active_interaction is a
    assert len(gm._interaction_stack) == 1
    # 既存スロットへの再代入は「先頭置換」（単一スロット相当・深さは増えない）
    b = {"action_type": "B"}
    gm.active_interaction = b
    assert gm.active_interaction is b
    assert len(gm._interaction_stack) == 1
    # None で先頭を pop
    gm.active_interaction = None
    assert gm.active_interaction is None
    assert gm._interaction_stack == []


def test_set_none_on_empty_is_noop():
    gm = _gm()
    gm.active_interaction = None
    assert gm.active_interaction is None


def test_push_creates_nesting_and_resolves_top_first():
    gm = _gm()
    outer = {"action_type": "OUTER"}
    inner = {"action_type": "INNER"}
    gm.active_interaction = outer          # 外側（先頭）
    gm.push_interaction(inner)             # 内側を上に積む（外側は残す）
    assert gm.active_interaction is inner   # 先頭＝内側を提示
    assert len(gm._interaction_stack) == 2
    # 内側を解決（pop）すると外側が再び先頭になる
    gm.active_interaction = None
    assert gm.active_interaction is outer
    gm.active_interaction = None
    assert gm.active_interaction is None
