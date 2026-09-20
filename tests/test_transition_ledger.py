"""**値段の付いていない遷移**の配分の代数を固める（T123・`H`・2026-09-20）。

`transition_ledger` は **`W` の差を恒等式で配るだけ**の器（当てはめゼロ）。押さえるのは 5 つ:

1. **シャープレイ値の和は厳密に `W(st1) − W(st0)`**（効率性）——配分に取りこぼしが無いこと。
2. **動かなかった軸の取り分は厳密に 0**（ヌルプレイヤー）——「動いていない物のせい」にしない。
3. **順序に依らない**（対称性）——2 つの軸を入れ替えた状態で配分も入れ替わる。
4. **席 1 の視点 → 席 0 の視点の写しが対合**（2 回かけると戻る・`Θ` と `A` が対で入れ替わる）。
5. **`curve` の読みでは速さの軸の取り分が厳密に 0**（輪郭は固定の表＝T121 の「速さの軸が無い」の帰結）。

**`priced` と `residual` の切り方**も固める——`priced` は「手を当てた状態」まで・`residual` はそこから
次の行まで。**2 つを足すと差に一致する**（これも恒等式）。
"""
import math
import os
import sys

import pytest

pytestmark = pytest.mark.cpu_infra

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bootstrap  # noqa: F401,E402

_SCRIPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import kappa_vector as KV  # noqa: E402
import relative_ledger as RL  # noqa: E402
import transition_ledger as TL  # noqa: E402

SIG = 0.2418
_FLAT_PROF = [0.2] * 14
_RAMP_PROF = [0.05, 0.1, 0.2, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]


@pytest.fixture(autouse=True)
def _restore():
    d = KV.D_MODE
    yield
    KV.set_d_mode(d)


def _w(st, prof=None):
    return RL.w_of(*RL.clocks_of(st, prof), sigma_rel=SIG)


# --------------------------------------------------------------------------- 1. 効率性
def test_shapley_sums_exactly_to_the_gap():
    """**配分に取りこぼしが無い**（シャープレイ値の効率性）＝恒等式であることの検算。"""
    KV.set_d_mode("clock")
    st0 = (0.9, 0.6, 0.12, 0.10, 2)
    st1 = (0.7, 0.45, 0.15, 0.09, 3)
    sh = TL.shapley(st0, st1, None, SIG)
    assert sum(sh.values()) == pytest.approx(_w(st1) - _w(st0), abs=1e-12)


def test_shapley_sums_to_the_gap_in_the_curve_reading_too():
    KV.set_d_mode("curve")
    st0 = (0.9, 0.6, 0.12, 0.10, 1)
    st1 = (0.8, 0.30, 0.12, 0.10, 2)
    sh = TL.shapley(st0, st1, _RAMP_PROF, SIG)
    assert sum(sh.values()) == pytest.approx(_w(st1, _RAMP_PROF) - _w(st0, _RAMP_PROF), abs=1e-12)


def test_shapley_is_zero_when_nothing_moved():
    KV.set_d_mode("clock")
    st = (0.9, 0.6, 0.12, 0.10, 2)
    assert all(v == pytest.approx(0.0) for v in TL.shapley(st, st, None, SIG).values())


def test_shapley_keys_are_exactly_the_five_axes():
    KV.set_d_mode("clock")
    sh = TL.shapley((0.9, 0.6, 0.12, 0.10, 2), (0.8, 0.5, 0.13, 0.09, 3), None, SIG)
    assert set(sh) == set(TL.AXES5) == {"th_me", "th_opp", "a_me", "a_opp", "j"}


# --------------------------------------------------------------------------- 2. ヌルプレイヤー
def test_an_axis_that_did_not_move_gets_exactly_zero():
    """**動いていない軸のせいにしない**（ヌルプレイヤー）。"""
    KV.set_d_mode("clock")
    st0 = (0.9, 0.6, 0.12, 0.10, 2)
    st1 = (0.9, 0.45, 0.12, 0.10, 2)          # `th_opp` だけ動かす
    sh = TL.shapley(st0, st1, None, SIG)
    assert sh["th_opp"] == pytest.approx(_w(st1) - _w(st0))
    for a in ("th_me", "a_me", "a_opp", "j"):
        assert sh[a] == pytest.approx(0.0)


def test_one_axis_moving_gives_that_axis_the_whole_gap():
    KV.set_d_mode("clock")
    st0 = (0.9, 0.6, 0.12, 0.10, 2)
    for i, a in enumerate(TL.AXES5[:4]):
        st1 = list(st0); st1[i] = st0[i] * 1.4
        sh = TL.shapley(st0, tuple(st1), None, SIG)
        assert sh[a] == pytest.approx(_w(tuple(st1)) - _w(st0))
        assert sum(abs(v) for k, v in sh.items() if k != a) == pytest.approx(0.0)


# --------------------------------------------------------------------------- 3. 対称性
def test_swapping_the_two_endurances_swaps_their_shares():
    """**順序に依らない**（対称性）——`Θ_me` と `Θ_opp` を入れ替えれば取り分も入れ替わる。"""
    KV.set_d_mode("clock")
    a = TL.shapley((0.9, 0.6, 0.12, 0.12, 2), (0.7, 0.45, 0.12, 0.12, 2), None, SIG)
    b = TL.shapley((0.6, 0.9, 0.12, 0.12, 2), (0.45, 0.7, 0.12, 0.12, 2), None, SIG)
    assert a["th_me"] == pytest.approx(-b["th_opp"])
    assert a["th_opp"] == pytest.approx(-b["th_me"])


def test_shapley_matches_the_average_over_both_orders_for_two_axes():
    """2 軸だけ動く場合、シャープレイ値は**2 通りの順序の平均**に厳密に一致する（定義の検算）。"""
    KV.set_d_mode("clock")
    st0 = (0.9, 0.6, 0.12, 0.10, 2)
    st1 = (0.7, 0.45, 0.12, 0.10, 2)
    sh = TL.shapley(st0, st1, None, SIG)
    mid_a = (st1[0], st0[1], st0[2], st0[3], st0[4])     # th_me を先に動かす
    mid_b = (st0[0], st1[1], st0[2], st0[3], st0[4])     # th_opp を先に動かす
    want = 0.5 * ((_w(mid_a) - _w(st0)) + (_w(st1) - _w(mid_b)))
    assert sh["th_me"] == pytest.approx(want, abs=1e-12)


# --------------------------------------------------------------------------- 4. 席の写し
def test_the_seat_swap_is_an_involution():
    st = (0.9, 0.6, 0.12, 0.10, 3)
    assert TL._swap_state(TL._swap_state(st)) == st
    dx = {"th_opp": -0.3, "a_me": 0.05}
    assert TL._swap_dx(TL._swap_dx(dx)) == dx


def test_the_seat_swap_exchanges_endurance_and_rate_in_pairs():
    assert TL._swap_state((1.0, 2.0, 3.0, 4.0, 5)) == (2.0, 1.0, 4.0, 3.0, 5)
    assert TL._swap_dx({"th_opp": -0.3}) == {"th_me": -0.3}
    assert TL._swap_dx({"a_me": 0.05}) == {"a_opp": 0.05}


def test_the_seat_swap_flips_the_sign_of_w():
    """席を入れ替えれば勝率は `1 − W`（零和）——写しが正しいことの意味づけ。"""
    KV.set_d_mode("clock")
    st = (0.9, 0.6, 0.12, 0.10, 3)
    assert _w(TL._swap_state(st)) == pytest.approx(1.0 - _w(st))


def test_the_seat_swap_keeps_the_turn_index():
    assert TL._swap_state((1.0, 2.0, 3.0, 4.0, 7))[4] == 7


# --------------------------------------------------------------------------- 5. curve に速さの軸は無い
def test_rate_axes_get_exactly_zero_in_the_curve_reading():
    """**輪郭は両席共通の固定の表**なので、速さがどれだけ動いても取り分は厳密に 0（T121 の帰結）。"""
    KV.set_d_mode("curve")
    st0 = (0.9, 0.6, 0.12, 0.10, 1)
    st1 = (0.9, 0.6, 5.00, 9.00, 1)           # 速さだけ大きく動かす
    sh = TL.shapley(st0, st1, _RAMP_PROF, SIG)
    assert sh["a_me"] == pytest.approx(0.0) and sh["a_opp"] == pytest.approx(0.0)
    assert sum(sh.values()) == pytest.approx(0.0)


def test_the_turn_index_carries_a_share_in_the_curve_reading():
    """**`j` は誰の手でもない軸**（輪郭の読み出し位置が進む）＝ターンの境目で `W` が動く経路。"""
    KV.set_d_mode("curve")
    st0 = (0.9, 0.6, 0.12, 0.10, 0)
    st1 = (0.9, 0.6, 0.12, 0.10, 3)
    sh = TL.shapley(st0, st1, _RAMP_PROF, SIG)
    assert sh["j"] == pytest.approx(_w(st1, _RAMP_PROF) - _w(st0, _RAMP_PROF))
    assert abs(sh["j"]) > 0.0


def test_the_turn_index_does_nothing_in_the_clock_reading():
    KV.set_d_mode("clock")
    st0 = (0.9, 0.6, 0.12, 0.10, 0)
    st1 = (0.9, 0.6, 0.12, 0.10, 5)
    assert TL.shapley(st0, st1, None, SIG)["j"] == pytest.approx(0.0)


# --------------------------------------------------------------------------- priced と residual
def test_priced_plus_residual_is_exactly_the_gap():
    """**切り方も恒等式**——`priced`（手を当てるまで）＋ `residual`（そこから次の行まで）＝差。"""
    KV.set_d_mode("clock")
    st0 = (0.9, 0.6, 0.12, 0.10, 2)
    dx = {"th_opp": -0.08}
    st1 = (0.88, 0.50, 0.13, 0.10, 3)
    gap = _w(st1) - _w(st0)
    priced = _w(KV.apply_dx(st0, dx)) - _w(st0)
    resid = sum(TL.shapley(KV.apply_dx(st0, dx), st1, None, SIG).values())
    assert priced + resid == pytest.approx(gap, abs=1e-12)


def test_a_row_with_no_move_puts_everything_in_the_residual():
    KV.set_d_mode("clock")
    st0 = (0.9, 0.6, 0.12, 0.10, 2)
    st1 = (0.8, 0.55, 0.12, 0.10, 3)
    assert sum(TL.shapley(st0, st1, None, SIG).values()) == pytest.approx(_w(st1) - _w(st0))


def test_mix_replaces_only_the_named_axes():
    st0 = (1.0, 2.0, 3.0, 4.0, 5)
    st1 = (9.0, 8.0, 7.0, 6.0, 0)
    assert TL._mix(st0, st1, frozenset({"th_opp", "j"})) == (1.0, 8.0, 3.0, 4.0, 0)
    assert TL._mix(st0, st1, frozenset()) == st0
    assert TL._mix(st0, st1, frozenset(TL.AXES5)) == st1


def test_causes_are_the_two_rule_origins():
    assert TL.CAUSES == ("turn_boundary", "same_turn")
