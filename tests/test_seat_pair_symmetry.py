"""`seat_pair_symmetry.py`（T151-1・両席の行を対にして `p` の反対称を測る）の算術を固める。

押さえるのは 4 つ:

1. **対の規則**: 行 `(seed, w, t)` の相手は `(seed, 1−w, t+1)`（相手の次の自席ターン）。相手が無い行は `unpaired`。
2. **反対称なら** `mean(sum_minus_1) = 0`・`both_fav = 0`。**自席びいきなら** 正・`both_fav > 0`。
3. **恒等式**: 現行の読み方では `τ_me(a) == τ_opp(b)`（同じ `per_seat` 項の写し）＝`shared_term_share = 1.0`・
   `d_a + d_b = τ_opp(a) − τ_me(b)`。相手の時計を別の瞬間から読めば `shared_term_share` が落ちる（切替の署名）。
4. **段・席の割り方**は行 `a` の値で決める（取りこぼし無し）。

**基盤健全性ではない**（対の取り違えは「自席びいき」の大きさを誤って読ませる）。必須側。
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bootstrap  # noqa: F401,E402

_SCRIPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import seat_pair_symmetry as SP  # noqa: E402


def _row(seed, who, t, p, z, tau_me=None, tau_opp=None, stage="mid"):
    d = (tau_opp - tau_me) if (tau_me is not None and tau_opp is not None) else 0.0
    return {"seed": seed, "who": who, "t": t, "p": p, "z": z, "d": d, "stage": stage,
            "tau_me": tau_me, "tau_opp": tau_opp}


def test_partner_is_the_opponents_next_own_turn():
    assert SP.partner_key(5, 0, 3) == (5, 1, 4)
    assert SP.partner_key(5, 1, 4) == (5, 0, 5)


def test_pair_rows_pairs_each_row_with_the_opponents_next_row_and_counts_the_rest():
    rows = [_row(1, 1, 2, 0.4, 0), _row(1, 0, 3, 0.6, 1), _row(1, 1, 4, 0.4, 0), _row(1, 0, 5, 0.7, 1)]
    pairs, unpaired = SP.pair_rows(rows)
    assert [(q["who"], q["t"]) for q in pairs] == [(1, 2), (0, 3), (1, 4)]
    assert unpaired == 1                       # (0, 5) は局の最後の行＝相手の次の行が無い


def test_pair_rows_does_not_cross_games():
    rows = [_row(1, 0, 3, 0.6, 1), _row(2, 1, 4, 0.4, 0)]
    pairs, unpaired = SP.pair_rows(rows)
    assert pairs == [] and unpaired == 2


def test_antisymmetric_rows_give_zero_mean_and_no_both_favorite_pairs():
    # 反対称＝隣り合う行が鏡: 0.7 ↔ 0.3 ↔ 0.7 ↔ 0.3（どの対も足して 1）
    rows = [_row(1, 0, 3, 0.7, 1), _row(1, 1, 4, 0.3, 0), _row(1, 0, 5, 0.7, 1), _row(1, 1, 6, 0.3, 0)]
    pairs, _ = SP.pair_rows(rows)
    t = SP.pair_table(pairs)
    assert t["all"]["n"] == 3
    assert t["all"]["mean_sum_minus_1"] == pytest.approx(0.0)
    assert t["all"]["both_fav_share"] == 0.0 and t["all"]["both_dog_share"] == 0.0


def test_self_favouring_rows_give_a_positive_mean_and_both_favorite_pairs():
    rows = [_row(1, 0, 3, 0.7, 1), _row(1, 1, 4, 0.6, 0), _row(1, 0, 5, 0.7, 1)]
    pairs, _ = SP.pair_rows(rows)
    t = SP.pair_table(pairs)
    assert t["all"]["mean_sum_minus_1"] == pytest.approx(0.3)
    assert t["all"]["both_fav_share"] == 1.0


def test_identity_holds_when_the_shared_term_is_the_same_per_seat_value():
    """現行の読み方: 行 a の `τ_me` と行 b の `τ_opp` は同じ `per_seat[(w,t)]` の写し。"""
    rows = [_row(1, 0, 3, 0.6, 1, tau_me=4.0, tau_opp=6.0),      # a: d = 2
            _row(1, 1, 4, 0.45, 0, tau_me=5.0, tau_opp=4.0)]     # b: τ_opp == τ_me(a) → d = −1
    pairs, _ = SP.pair_rows(rows)
    idc = SP.identity_check(pairs)
    assert idc["n"] == 1 and idc["shared_term_share"] == 1.0
    assert idc["identity_resid"] == pytest.approx(0.0)           # (2 + −1) == (6 − 5)


def test_identity_breaks_when_the_opponent_clock_is_read_from_another_instant():
    rows = [_row(1, 0, 3, 0.6, 1, tau_me=4.0, tau_opp=6.0),
            _row(1, 1, 4, 0.45, 0, tau_me=5.0, tau_opp=3.5)]     # τ_opp(b) は別の瞬間の値
    pairs, _ = SP.pair_rows(rows)
    idc = SP.identity_check(pairs)
    assert idc["shared_term_share"] == 0.0
    assert idc["identity_resid"] == pytest.approx(0.5)


def test_pair_table_splits_by_the_first_rows_stage_and_seat_without_loss():
    rows = [_row(1, 0, 3, 0.6, 1, stage="early"), _row(1, 1, 4, 0.5, 0, stage="mid"),
            _row(1, 0, 5, 0.6, 1, stage="late"), _row(1, 1, 6, 0.5, 0, stage="late")]
    pairs, unpaired = SP.pair_rows(rows)
    t = SP.pair_table(pairs, unpaired)
    assert sum(t["by_stage"][s]["n"] for s in SP.PA.STAGES) == t["all"]["n"] == 3
    assert t["by_who"]["0"]["n"] + t["by_who"]["1"]["n"] == 3
    assert t["unpaired"] == 1
