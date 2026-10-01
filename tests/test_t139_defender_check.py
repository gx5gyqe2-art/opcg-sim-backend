"""`t139_defender_check.py`（P8-7(b)・T139 の残りを `aux_def` で読む）の算術を固める。

押さえるのは 2 つ:

1. **`defender_actuals_by_turn` は「守り手の直前の自席ターン開始の行」で `aux_def`／`aux_def_row` を読む**
   （その窓が攻め手の手番 `(w, t)` にちょうど一致する・`aux_def` が無い波は出さない）。
2. **`collect` は `full_swing`（`attacks_made >= xs`）だけを `means` に集計し、`actual` が無い行は外す**
   （理論の `stops`／`through` と実際の `blocked`／`counters_used` を混ぜない）。

**基盤健全性ではない**——器の誤りは T139 の「守り手側に定義に無い何かがある」という診断そのものを
誤らせる（`settlement_split.py`／`lethal_rule.py` と同じ扱い）。
"""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bootstrap  # noqa: F401,E402

_SCRIPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import t139_defender_check as TC  # noqa: E402
from opcg_sim.loop.record_gen import LEFT_DEST_TRASH_BATTLE, LEFT_DEST_TRASH_EFFECT  # noqa: E402


class _FakeGames:
    """`PL.iter_games` を模す——`aux_def`／`aux_def_row` は行 index をそのまま埋め込む
    （`_extra_def` が読む形〔(D,6,4) と (D,2)〕・`aux_def` を渡さない局は `None` を返す）。"""

    def __init__(self, games):
        self.games = games  # [(seed, rows, aux_def_or_None, aux_def_row_or_None)]

    def __iter__(self):
        for seed, rows, ad, adr in self.games:
            n = len(rows)
            r = {"seed": [seed] * n, "who": [row[0] for row in rows], "turn": [row[1] for row in rows],
                 "kind": [row[2] for row in rows], "pol_chosen": [0] * n}
            L = [1] * n
            ptr = list(range(n))
            pol = {"pol_sig": ["[]"] * n}
            ex = {"aux_def": ad, "aux_def_row": adr}
            idx = list(range(n))
            yield r, pol, ex, L, ptr, idx


# --- 1. defender_actuals_by_turn ---------------------------------------------------------------

def test_reads_the_defenders_previous_own_turn_start_row():
    """守り手 (d=1) のターン 2 の開始行（index 1）で `aux_def` を読む——窓は攻め手 (w=0) のターン 3 に一致。"""
    # 行: (who, turn, kind)。ターン 1=w0・2=w1・3=w0・4=w1（main 行だけ）。
    rows = [(0, 1, 0), (1, 2, 0), (0, 3, 0), (1, 4, 0)]
    ad = np.zeros((4, 6, 4), np.float32)
    ad[1, 0] = [1.0, 0.0, 2.0, 0.0]        # 行 1（守り手 d=1・ターン 2 開始）のリーダー枠: 狙われた1・失ったライフ2
    ad[1, 1] = [1.0, 0.0, 0.0, LEFT_DEST_TRASH_BATTLE]
    ad[1, 2] = [0.0, 0.0, 0.0, LEFT_DEST_TRASH_EFFECT]
    adr = np.zeros((4, 2), np.float32)
    adr[1] = [3.0, 1.0]                    # counters_used=3・blocks=1
    games = [(100, rows, ad, adr)]
    result = _run_defender_actuals(games)
    assert (100, 0, 3) in result                  # 攻め手 w=0・ターン 3 の窓
    got = result[(100, 0, 3)]
    assert got["targeted"] == pytest.approx(2.0)    # リーダー枠1 + 場枠1
    assert got["life_lost"] == pytest.approx(2.0)
    assert got["left_battle"] == 1
    assert got["left_effect"] == 1
    assert got["counters_used"] == pytest.approx(3.0)
    assert got["blocks"] == pytest.approx(1.0)
    # 攻め手 w=1・ターン 4 の窓は守り手 w=0 の「直前の自席ターン」= ターン 3（index 2）
    assert (100, 1, 4) in result


def _run_defender_actuals(games):
    """`defender_actuals_by_turn` の本体ロジックを `PL.iter_games` を経由せず直接検証するための
    薄いラッパー——`_FakeGames` を `PL.iter_games` の代わりに渡す。"""
    import unittest.mock as mock
    with mock.patch.object(TC.PL, "iter_games", return_value=iter(_FakeGames(games))):
        return TC.defender_actuals_by_turn(["dummy"])


def test_games_without_aux_def_are_skipped():
    rows = [(0, 1, 0), (1, 2, 0)]
    games = [(200, rows, None, None)]
    result = _run_defender_actuals(games)
    assert result == {}


def test_no_previous_own_turn_yields_no_entry():
    """守り手がまだ自席ターンを打っていない（先手 1 ターン目の相手）行は出さない。"""
    rows = [(0, 1, 0)]                     # w1 の自席ターンが 1 つも無い
    ad = np.zeros((1, 6, 4), np.float32)
    adr = np.zeros((1, 2), np.float32)
    games = [(300, rows, ad, adr)]
    result = _run_defender_actuals(games)
    assert result == {}


# --- 2. collect の full_swing 集計 --------------------------------------------------------------

def test_collect_only_averages_full_swing_rows_with_actual(monkeypatch):
    dump = [
        {"seed": 1, "w": 0, "t": 3, "declared": True, "winner": 1, "xs": 2, "blockers": 0,
         "through": 2, "stops": 0, "hits": 2, "life": 1.0, "counters": []},
        {"seed": 2, "w": 0, "t": 3, "declared": True, "winner": 1, "xs": 3, "blockers": 0,
         "through": 1, "stops": 0, "hits": 1, "life": 0.0, "counters": []},     # partial_swing（attacks_made < xs）
        {"seed": 3, "w": 0, "t": 3, "declared": True, "winner": 0, "xs": 1, "blockers": 0,
         "through": 1, "stops": 0, "hits": 1, "life": 0.0, "counters": []},     # winner==w（false declared から除外）
    ]

    def fake_lr_collect(dirs, limit_games, with_don, dump=None):
        dump.extend([{"seed": r["seed"], "w": r["w"], "t": r["t"], "declared": r["declared"],
                     "winner": r["winner"], "xs": r["xs"], "blockers": r["blockers"],
                     "through": r["through"], "stops": r["stops"], "hits": r["hits"],
                     "life": r["life"], "counters": r["counters"]} for r in dump_src])
        return {}

    dump_src = dump
    monkeypatch.setattr(TC.LR, "collect", fake_lr_collect)
    monkeypatch.setattr(TC.SS, "attacks_made_by_turn",
                        lambda dirs, limit_games: {(1, 0, 3): 2, (2, 0, 3): 1})
    monkeypatch.setattr(TC, "defender_actuals_by_turn",
                        lambda dirs, limit_games: {(1, 0, 3): {"targeted": 2.0, "blocked": 0.0,
                                                                "life_lost": 1.0, "left_battle": 0,
                                                                "left_effect": 0, "counters_used": 1.0,
                                                                "blocks": 0.0}})
    out = TC.collect(["dummy"])
    assert out["n_false_declared"] == 2                # seed 3 は winner==w なので除外
    assert out["n_full_swing"] == 1                     # seed 1 だけ attacks_made(2) >= xs(2)
    assert out["n_full_swing_with_actual"] == 1
    assert out["means"]["stops"] == pytest.approx(0.0)
    assert out["means"]["blocked_actual"] == pytest.approx(0.0)
    assert out["means"]["counters_used_actual"] == pytest.approx(1.0)
