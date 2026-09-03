"""V7 リーサル族生成器の判定規約（系統3・2026-08-21）。

`tests/scripts/plan_lethal_gen.py` の契約を固定する:
  - lethal_exit は「今ターンに name が勝ち切る」時だけ勝利済み manager を返す
    （相手勝ち・手詰まり・ターン跨ぎは None＝対を立てない）
  - cut_at_group はシャード境界で組（group）を割らない
"""
import types

import pytest

import _bootstrap  # noqa: F401

import plan_lethal_gen as PLG

pytestmark = pytest.mark.cpu_infra


class _Mgr:
    def __init__(self, winner=None):
        self.winner = winner
        self.action_events = ["x"]

    def clone(self):
        c = _Mgr(self.winner)
        return c


def test_lethal_exit_immediate_win_returns_state():
    gs = types.SimpleNamespace(current_player=lambda m: "p1")
    out = PLG.lethal_exit(gs, _Mgr(winner="p1"), "p1")
    assert out is not None and out.winner == "p1"
    assert out.action_events == []          # 監視イベントはクリアされる


def test_lethal_exit_opponent_win_returns_none():
    gs = types.SimpleNamespace(current_player=lambda m: "p1")
    assert PLG.lethal_exit(gs, _Mgr(winner="p2"), "p1") is None


def test_lethal_exit_dead_position_returns_none():
    gs = types.SimpleNamespace(current_player=lambda m: None)   # 手番不在＝測定不能
    assert PLG.lethal_exit(gs, _Mgr(), "p1") is None


def test_lethal_exit_turn_end_crossing_returns_none(monkeypatch):
    # 台本が TURN_END を返す＝今ターンの詰みではない → None（次ターンへ進まない）
    gs = types.SimpleNamespace(current_player=lambda m: "p1")
    monkeypatch.setattr(PLG.LT, "_script_move", lambda gs, m, name, defend: "mv")
    monkeypatch.setattr(PLG.LT, "_desc", lambda m, mv: {"action_type": "TURN_END"})
    assert PLG.lethal_exit(gs, _Mgr(), "p1") is None


def test_lethal_exit_script_dries_up_returns_none(monkeypatch):
    gs = types.SimpleNamespace(current_player=lambda m: "p1")
    monkeypatch.setattr(PLG.LT, "_script_move", lambda gs, m, name, defend: None)
    assert PLG.lethal_exit(gs, _Mgr(), "p1") is None


def _row(g):
    return (None, 0.5, g)


def test_cut_at_group_extends_to_boundary():
    buf = [_row(1), _row(1), _row(2), _row(2), _row(2), _row(3)]
    chunk, rest = PLG.cut_at_group(buf, 3)      # 3行目は group2 の途中 → 5行まで伸ばす
    assert [r[2] for r in chunk] == [1, 1, 2, 2, 2]
    assert [r[2] for r in rest] == [3]


def test_cut_at_group_exact_boundary_unchanged():
    buf = [_row(1), _row(1), _row(2)]
    chunk, rest = PLG.cut_at_group(buf, 2)
    assert [r[2] for r in chunk] == [1, 1]
    assert [r[2] for r in rest] == [2]
