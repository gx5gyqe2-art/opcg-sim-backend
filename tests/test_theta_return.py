"""**C-5c**（2026-09-25・ユーザ決定「正しい形だけやれば良い」）: **レスト中のブロッカーは「次の自席ターンから
戻る耐久」**——`THETA_RETURN_MODE=untap` を帳簿の状態（7 つ組）・両方の歩き（輪郭・積み上がり・一定）・
攻撃の価格（`ATTACK_REST_MODE=return`＝消さずに移す）に通した形を固める。

押さえるのは 6 つ:

1. **歩きの規約**: `step` は **2 段目から**的に足す（`tau_from_profile`・`tau_of`・`tau_theory` の 3 本で同じ）。
   1 段目で届くなら `step` は一切効かない。
2. **7 つ組の代数**: `split_state` は 5 つ組を戻る分 0 として読む／`apply_dx` は `th_*_back` を 7 つ組にだけ足し、
   5 つ組に足そうとすると落ちる（黙って捨てない）。
3. **状態の組み立て**: `untap` なら `state_of_row` が末尾に戻る分を持ち、`Θ` 本体（アクティブなブロッカーだけ）は
   既定と同じ数字のまま。`off` なら 5 つ組（既定はビット一致）。
4. **攻撃の価格**: `return` はブロッカーで攻めた `ν` を **`th_me` から `th_me_back` へ移す**（総量は不変）。
5. **席の入れ替えと混ぜ方**: `_swap_state` は戻る分も入れ替え、`_mix` は耐久の軸と戻る分を一緒に動かし、
   シャープレイ配分の和は 7 つ組でも厳密に `W(st1) − W(st0)`。
6. **器の CLI に切替が在る**（帳簿・攻撃の残差の 2 器・耐久の内訳の器）。

**基盤健全性ではない**——式の形（規則）そのものを固める。
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bootstrap  # noqa: F401,E402

_SCRIPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import attack_axis_by_result as AX  # noqa: E402
import attack_theta_parts as AP  # noqa: E402
import crossing_bridge as CB  # noqa: E402
import kappa_vector as KV  # noqa: E402
import relative_ledger as RL  # noqa: E402
import theory_order as TO  # noqa: E402
import transition_ledger as TL  # noqa: E402

_FLAT = [0.2] * 14


@pytest.fixture(autouse=True)
def _restore():
    old = (CB.THETA_RETURN_MODE, KV.ATTACK_REST_MODE, KV.D_MODE)
    yield
    CB.set_theta_return_mode(old[0]); KV.set_attack_rest_mode(old[1]); KV.set_d_mode(old[2])


# --- 1. 歩きの規約 -----------------------------------------------------------------------------

def test_profile_walk_adds_the_step_from_the_second_turn_only():
    assert CB.tau_from_profile(0.3, 0, _FLAT) == pytest.approx(1.5)
    # 2 段目から的が 0.5 遠のく: 0.2 / 0.4 / 0.6 / 0.8 → 4 段目で届く
    assert CB.tau_from_profile(0.3, 0, _FLAT, step=0.5) == pytest.approx(4.0)
    # 1 段目で届くなら step は効かない
    assert CB.tau_from_profile(0.1, 0, _FLAT, step=0.5) == pytest.approx(0.5)
    assert CB.tau_from_profile(0.1, 0, _FLAT) == pytest.approx(0.5)


def test_clock_walk_uses_the_same_convention():
    assert KV.tau_of(0.3, 0.2) == pytest.approx(1.5)
    assert KV.tau_of(0.3, 0.2, step=0.5) == pytest.approx(4.0)
    assert KV.tau_of(0.1, 0.2, step=0.5) == pytest.approx(0.5)


def test_theory_walk_matches_tau_grow_with_a_step():
    KV.set_rate_shape((0.0, 1.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0))   # 一定の速さ
    try:
        a = KV.tau_theory(0.3, 0.2, KV.RATE_SHAPE["me"], 2, step=0.5)
        b = CB.tau_grow(0.3, 0.0, 0.2, 0.0, 0.0, step=0.5, j0=3)
        assert a == pytest.approx(b)
        assert a > KV.tau_theory(0.3, 0.2, KV.RATE_SHAPE["me"], 2)
    finally:
        KV.set_rate_shape(None, None)


# --- 2. 7 つ組の代数 ---------------------------------------------------------------------------

def test_split_state_reads_a_five_tuple_as_zero_returning_stock():
    assert KV.split_state((1.0, 0.5, 0.3, 0.2, 4)) == (1.0, 0.5, 0.3, 0.2, 4, 0.0, 0.0)
    assert KV.split_state((1.0, 0.5, 0.3, 0.2, 4, 0.1, 0.2)) == (1.0, 0.5, 0.3, 0.2, 4, 0.1, 0.2)
    with pytest.raises(ValueError):
        KV.split_state((1.0, 0.5, 0.3, 0.2, 4, 0.1))


def test_apply_dx_moves_the_returning_stock_only_on_a_seven_tuple():
    st = KV.apply_dx((1.0, 0.5, 0.3, 0.2, 4, 0.0, 0.0), {"th_me": -0.15, "th_me_back": 0.15})
    assert st == pytest.approx((0.85, 0.5, 0.3, 0.2, 4, 0.15, 0.0))
    assert len(KV.apply_dx((1.0, 0.5, 0.3, 0.2, 4), {"th_me": -0.1})) == 5     # 既定は 5 つ組のまま
    with pytest.raises(ValueError):
        KV.apply_dx((1.0, 0.5, 0.3, 0.2, 4), {"th_me_back": 0.15})


def test_returning_stock_only_lengthens_the_clock_of_the_seat_that_owns_it():
    KV.set_d_mode("curve")
    base = (0.9, 0.9, 1.0, 1.0, 0)
    mine = (0.9, 0.9, 1.0, 1.0, 0, 0.5, 0.0)     # 私のブロッカーが戻る＝相手が私を倒すのが遅れる
    theirs = (0.9, 0.9, 1.0, 1.0, 0, 0.0, 0.5)
    assert KV.d_of(base, _FLAT) == pytest.approx(0.0)
    assert KV.d_of(mine, _FLAT) > 0.0 and KV.d_of(theirs, _FLAT) < 0.0
    t_me, t_opp = RL.clocks_of(mine, _FLAT)
    t_me0, t_opp0 = RL.clocks_of(base, _FLAT)
    assert t_me == pytest.approx(t_me0) and t_opp > t_opp0


# --- 3. 状態の組み立て ---------------------------------------------------------------------------

class _Cards:
    def info(self, cid):
        return {"B": {"blocker": True, "power": 5000.0}}.get(cid)


def _board_with_a_rested_blocker():
    sc = np.zeros(70, float)
    sc[TO.SC_MY_LIFE] = 3.0; sc[TO.SC_OPP_LIFE] = 3.0
    sc[TO.SC_MY_LEADER_POWER] = sc[TO.SC_OPP_LEADER_POWER] = 0.5
    tok = np.zeros((22, 24), float)
    tok[0, TO.S_POWER] = tok[1, TO.S_POWER] = 0.5
    tok[2, TO.S_IS_CHAR] = 1.0; tok[2, TO.S_POWER] = 0.5
    tok[2, TO.S_IS_REST] = 1.0; tok[2, TO.S_IS_BLOCKER] = 0.0      # レストしたブロッカー（列 6 は 0）
    ci = np.zeros(22, int); ci[2] = 7
    return sc, tok, ci, {7: "B"}, _Cards()


def test_state_of_row_carries_the_rested_blocker_as_returning_stock_under_untap():
    sc, tok, ci, idx2cid, cards = _board_with_a_rested_blocker()
    CB.set_theta_return_mode("off")
    st_off = KV.state_of_row(sc, tok, 0.1, 0.1, 2, ci_row=ci, idx2cid=idx2cid, cards=cards)
    assert len(st_off) == 5
    CB.set_theta_return_mode("untap")
    st = KV.state_of_row(sc, tok, 0.1, 0.1, 2, ci_row=ci, idx2cid=idx2cid, cards=cards)
    assert len(st) == 7
    assert st[:5] == st_off                                         # Θ 本体は既定と同じ数字
    assert st[5] == pytest.approx(CB.nu_meas_of(5000.0, 5000.0))    # 私の戻る分
    assert st[6] == pytest.approx(0.0)
    assert KV.state5_of_row(sc, tok, 0.1, 0.1, 2, ci_row=ci, idx2cid=idx2cid, cards=cards) == st_off


def test_state_of_row_needs_the_card_db_to_see_the_returning_stock():
    sc, tok, ci, idx2cid, cards = _board_with_a_rested_blocker()
    CB.set_theta_return_mode("untap")
    st = KV.state_of_row(sc, tok, 0.1, 0.1, 2)                       # 札の原本が無ければ 0（落ちない）
    assert len(st) == 7 and st[5] == 0.0 and st[6] == 0.0


# --- 4. 攻撃の価格 -------------------------------------------------------------------------------

def test_return_mode_moves_the_blocker_from_active_to_returning():
    class _C:
        def info(self, cid):
            return {"power": 5000.0, "blocker": True}

    KV.set_attack_rest_mode("return")
    sc = np.zeros(70, float); sc[12] = sc[13] = 0.5
    tok = np.zeros((22, 24), float); tok[0, 0] = 0.5
    dx = KV.axis_of_move("attack", 0.37, ["ATTACK", None, "L"], "X", _C(), sc, tok, 5000.0, 4.0)
    nu = CB.nu_meas_of(5000.0, 5000.0)
    assert dx == pytest.approx({"th_opp": -0.37, "th_me": -nu, "th_me_back": nu})
    st = KV.apply_dx((1.0, 0.8, 0.3, 0.2, 3, 0.0, 0.0), dx)
    assert st[0] + st[5] == pytest.approx(1.0)                      # 総量は不変（移すだけ）


def test_attack_rest_modes_include_return():
    assert "return" in KV.ATTACK_REST_MODES
    KV.set_attack_rest_mode("return")
    assert KV.ATTACK_REST_MODE == "return"


# --- 5. 席の入れ替えと混ぜ方 ---------------------------------------------------------------------

def test_swap_state_swaps_the_returning_stock_too():
    assert TL._swap_state((1.0, 0.5, 0.3, 0.2, 4, 0.1, 0.2)) == (0.5, 1.0, 0.2, 0.3, 4, 0.2, 0.1)
    assert TL._swap_state((1.0, 0.5, 0.3, 0.2, 4)) == (0.5, 1.0, 0.2, 0.3, 4)


def test_swap_dx_moves_the_returning_stock_to_the_other_seat_too():
    """席 1 の攻撃の価格（`th_me` −ν・`th_me_back` +ν）を席 0 視点に写すと**両方**が相手側へ移る
    （戻る分だけ席 0 に残ると、守り手の耐久が増えたことになる＝踏んだ誤り）。"""
    dx = TL._swap_dx({"th_opp": -0.3, "th_me": -0.15, "th_me_back": 0.15})
    assert dx == {"th_me": -0.3, "th_opp": -0.15, "th_opp_back": 0.15}
    assert TL._swap_dx(TL._swap_dx(dx)) == dx


def test_mix_moves_the_endurance_axis_together_with_its_returning_stock():
    st0 = (1.0, 0.5, 0.3, 0.2, 4, 0.0, 0.0)
    st1 = (0.8, 0.4, 0.3, 0.2, 4, 0.2, 0.1)
    assert TL._mix(st0, st1, ("th_me",)) == (0.8, 0.5, 0.3, 0.2, 4, 0.2, 0.0)
    assert TL._mix(st0, st1, ("th_opp",)) == (1.0, 0.4, 0.3, 0.2, 4, 0.0, 0.1)
    assert TL._mix((1.0, 0.5, 0.3, 0.2, 4), (0.8, 0.4, 0.3, 0.2, 4), ("th_me",)) == (0.8, 0.5, 0.3, 0.2, 4)


def test_shapley_split_still_sums_to_the_difference_on_seven_tuples():
    KV.set_d_mode("curve")
    st0 = (1.0, 0.7, 0.3, 0.2, 2, 0.0, 0.0)
    st1 = (0.85, 0.5, 0.3, 0.2, 2, 0.15, 0.1)
    sh = TL.shapley(st0, st1, _FLAT, 0.3)
    w0 = RL.w_of(*RL.clocks_of(st0, _FLAT), sigma_rel=0.3)
    w1 = RL.w_of(*RL.clocks_of(st1, _FLAT), sigma_rel=0.3)
    assert sum(sh.values()) == pytest.approx(w1 - w0, abs=1e-12)
    assert sh["a_me"] == 0.0 and sh["a_opp"] == 0.0 and sh["j"] == 0.0


# --- 6. 器の CLI -----------------------------------------------------------------------------------

def test_the_instruments_expose_the_switch():
    for build in (RL.build_parser, AX.build_parser, AP.build_parser):
        a = build().parse_args(["--in", "x", "--theta-return", "untap", "--attack-rest", "return"])
        assert a.theta_return == "untap" and a.attack_rest == "return"
