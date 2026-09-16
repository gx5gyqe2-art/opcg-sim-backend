"""`clock_calib.py`（T51・2 本の導火線の競争の較正）の算術を固める。

**当てはめない**器なので、耐久・時計・帯・Φ の定義がそのまま出ること、4 本の方程式の左辺が
記録の事実だけから作られること（倒した側の残差・負けた側の止めた回数）を値で押さえる。
"""
import math
import os
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.cpu_infra

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bootstrap  # noqa: F401,E402

_SCRIPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import clock_calib as CC  # noqa: E402
import theory_order as T  # noqa: E402


def test_endurance_and_clock_follow_the_fuse_definition():
    assert CC.endurance(3, 4, 0) == pytest.approx(3 + 4 / T.CBAR)
    assert CC.endurance(2, 3, 1, draws_turns=2) == pytest.approx(2 + 5 / T.CBAR + 1)
    e = CC.endurance(3, 4, 0)
    assert CC.clock(e, 1.5) == pytest.approx(e / 1.5)
    assert CC.clock(e, 0) == pytest.approx(e)                              # 分母の床 1
    assert CC.clock(e, 1.5, "draws") == pytest.approx(e / (1.5 - 1 / T.CBAR))
    assert CC.clock(e, 1.0, "draws") == pytest.approx(e / max(0.25, 1.0 - 1 / T.CBAR))
    assert CC.phi_cdf(0.0) == pytest.approx(0.5) and CC.phi_cdf(3.0) > 0.99
    assert CC.d_bin(-5) == "<-3" and CC.d_bin(0.0) == "-1..1" and CC.d_bin(2.5) == "1..3" and CC.d_bin(9) == ">3"


def test_the_board_counts_only_attackers_that_clear_the_opposing_leader():
    tok = np.zeros((22, 24), np.float32)
    tok[0, T.S_POWER], tok[1, T.S_POWER] = 0.5, 0.5
    tok[2, T.S_POWER], tok[2, T.S_IS_CHAR] = 0.6, 1.0          # 6000 → 越える
    tok[3, T.S_POWER], tok[3, T.S_IS_CHAR] = 0.4, 1.0          # 4000 → 越えない
    tok[7, T.S_POWER], tok[7, T.S_IS_CHAR], tok[7, T.S_IS_BLOCKER] = 0.7, 1.0, 1.0
    sc = np.zeros(70, np.float32)
    sc[T.SC_MY_LIFE], sc[T.SC_OPP_LIFE], sc[T.SC_MY_HAND], sc[T.SC_OPP_HAND] = 3, 2, 4, 3
    sc[T.SC_MY_LEADER_POWER], sc[T.SC_OPP_LEADER_POWER] = 0.5, 0.5
    q = CC.row_inputs(sc, tok)
    assert q["A_me"] == 2 and q["A_opp"] == 2                  # 相手はリーダー＋7000
    assert q["B_me"] == 0 and q["B_opp"] == 1
    assert (q["L_me"], q["L_opp"], q["H_me"], q["H_opp"]) == (3, 2, 4, 3)


def _row(won, T_me, T_opp, t_me_act, t_opp_act):
    return {"who": 0, "won": won, "t_me_act": t_me_act, "t_opp_act": t_opp_act,
            "T_me_static": T_me, "T_opp_static": T_opp, "T_me_draws": T_me, "T_opp_draws": T_opp}


def test_the_time_residual_uses_the_clock_of_the_side_that_actually_ran_out():
    """勝った行は自分の時計 対 自分の残りターン・負けた行は相手の時計 対 相手の残りターン。"""
    rows = [_row(True, 2.0, 6.0, 2, 4)] * 30 + [_row(False, 5.0, 3.0, 4, 2)] * 30
    out = CC.summarise(rows, [], [])
    t = out["time"]["static"]
    assert t["bias"] == pytest.approx((0.0 * 30 + 1.0 * 30) / 60)            # 負けた行は 3 − 2 = 1 ずれ
    assert t["sigma_T"] == pytest.approx(0.5)
    assert t["sigma_D"] == pytest.approx(math.sqrt(2) * 0.5, abs=1e-3)
    assert t["sign_accuracy"] == pytest.approx(1.0)                          # D > 0 ⇔ 勝ち
    bins = out["win_by_D"]["static"]
    assert bins[">3"]["win_rate"] == 1.0 and bins["-3..-1"]["win_rate"] == 0.0
    assert bins[">3"]["phi_measured_sigma"] > 0.99


def test_the_threshold_and_rate_equations_only_average_record_facts():
    thr = [{"L": 2, "H": 3, "B": 1, "hits": 5, "stops": 3, "turns_left": 2,
            "formula_static": 3 / T.CBAR + 1, "formula_draws": 5 / T.CBAR + 1}] * 10
    rate = [{"A_board": 2, "A_actual": 1}] * 25 + [{"A_board": 3, "A_actual": 3}] * 25
    out = CC.summarise([], rate, thr)
    th = out["threshold"]
    assert th["stops_actual"] == 3 and th["hits_actual"] == 5
    assert th["stops_per_hand_card"] == pytest.approx(1.0)
    assert th["implied_cbar_static"] == pytest.approx(3 / (3 - 1))           # ブロッカーの 1 回を引いた札あたり
    r = out["rate"]
    assert r["A_board_mean"] == 2.5 and r["A_actual_mean"] == 2.0
    assert r["by_board"][2]["actual_mean"] == 1.0 and r["by_board"][3]["actual_mean"] == 3.0
