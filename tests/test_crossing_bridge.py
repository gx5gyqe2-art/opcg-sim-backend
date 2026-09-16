"""`crossing_bridge.py`（T52・交点の橋）の算術を固める。

**当てはめない・回帰しない**器なので、しきい値・傾き・交点・予測勝者の定義がそのまま出ること、
残差が「実際に届いた側の τ 対 その側の残りターン」で作られること、単位の検算が比そのままであることを値で押さえる。
"""
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

import crossing_bridge as CB  # noqa: E402
import price_realised as PR  # noqa: E402
import theory_order as T  # noqa: E402


def _sc(opp_life=3, opp_hand=4):
    sc = np.zeros(70, np.float32)
    sc[T.SC_OPP_LIFE], sc[T.SC_OPP_HAND] = opp_life, opp_hand
    sc[T.SC_MY_LEADER_POWER], sc[T.SC_OPP_LEADER_POWER] = 0.5, 0.5
    return sc


def test_the_threshold_is_the_opponents_endurance_in_price_units():
    tok = np.zeros((22, 24), np.float32)
    assert CB.threshold(_sc(3, 4), tok) == pytest.approx(3 * T.LAM + 4 * T.MU)
    tok[7, T.S_POWER], tok[7, T.S_IS_CHAR], tok[7, T.S_IS_BLOCKER] = 0.6, 1.0, 1.0      # アクティブなブロッカー 6000
    assert CB.threshold(_sc(3, 4), tok) == pytest.approx(3 * T.LAM + 4 * T.MU + PR.NU_MEAS["leader_to_sat"])
    tok[7, T.S_IS_REST] = 1.0                                                            # レスト中は数えない
    assert CB.threshold(_sc(3, 4), tok) == pytest.approx(3 * T.LAM + 4 * T.MU)


def test_the_theory_slope_is_the_priced_attack_flow_of_the_board():
    tok = np.zeros((22, 24), np.float32)
    tok[0, T.S_POWER], tok[1, T.S_POWER] = 0.5, 0.5
    lead_only = CB.theory_slope(tok, 5000.0)
    assert lead_only == pytest.approx(T.attack_value_don(5000.0, 5000.0, True))
    tok[2, T.S_POWER], tok[2, T.S_IS_CHAR], tok[2, T.S_CAN_ATTACK] = 0.8, 1.0, 1.0
    assert CB.theory_slope(tok, 5000.0) == pytest.approx(lead_only + T.attack_value_don(8000.0, 5000.0, True))


def test_the_crossing_picks_the_side_that_reaches_its_threshold_first():
    tau_me, tau_opp, pred = CB.predict(0.4, 0.4, 0.2, 0.1)
    assert (tau_me, tau_opp, pred) == (pytest.approx(2.0), pytest.approx(4.0), True)
    assert CB.predict(0.4, 0.4, 0.1, 0.2)[2] is False
    assert CB.predict(0.4, 0.4, 0.1, 0.1)[2] is True                          # 同数なら手番の自席
    assert CB.predict(0.4, 0.4, 0.0, 0.1)[0] == pytest.approx(0.4 / CB.SLOPE_FLOOR)   # 傾き 0 は届かない
    assert CB.harm_of({"opp_life": 0.136, "opp_hand": -0.05, "opp_body": 0.02, "my_life": -9, "my_hand": 9,
                       "my_body": 9, "don": 9}) == pytest.approx(0.106)


def _row(won, tau_me, tau_opp, t_me, t_opp):
    return {"who": 0, "won": won, "t_me_act": t_me, "t_opp_act": t_opp, "theta_me": 0.5, "theta_opp": 0.5,
            "tau_me_hist": tau_me, "tau_opp_hist": tau_opp, "pred_hist": tau_me <= tau_opp,
            "tau_me_theory": tau_me, "tau_opp_theory": tau_opp, "pred_theory": tau_me <= tau_opp}


def test_the_residual_uses_the_side_that_actually_reached_and_the_ledger_check_is_a_plain_ratio():
    rows = [_row(True, 2.0, 6.0, 2, 4)] * 30 + [_row(False, 5.0, 3.0, 4, 2)] * 30
    ledger = [{"F_end": 0.8, "theta_start": 1.0, "F_priced_end": 0.4}] * 10
    out = CB.summarise(rows, ledger)
    o = out["by_slope"]["hist"]
    assert o["sign_accuracy"] == 1.0
    assert o["bias"] == pytest.approx(0.5)                                     # 負けた行だけ 3 − 2 = 1
    assert o["sigma_T"] == pytest.approx(0.5)
    assert o["win_by_D"][">3"]["win_rate"] == 1.0 and o["win_by_D"]["-3..-1"]["win_rate"] == 0.0
    assert out["ledger"]["F_end_over_theta_start"] == pytest.approx(0.8)
    assert out["ledger"]["F_priced_over_F_real"] == pytest.approx(0.5)


def test_the_harm_profile_extends_its_last_value_and_the_profile_crossing_interpolates():
    """損害の輪郭は `j` ごとの平均・薄い先は最後の値を伸ばす。輪郭に沿った交点は端数を比例配分する。"""
    th = [{"j": 0, "harm": 0.05, "slope_theory": 0.1}] * 20 + [{"j": 1, "harm": 0.15, "slope_theory": 0.2}] * 20 + \
         [{"j": 2, "harm": 0.25, "slope_theory": 0.3}] * 5                  # j=2 は標本不足
    prof, prof_th = CB.harm_profile(th, j_max=4)
    assert prof == pytest.approx([0.05, 0.15, 0.15, 0.15, 0.15])
    assert prof_th == pytest.approx([0.1, 0.2, 0.2, 0.2, 0.2])
    assert CB.tau_from_profile(0.05, 0, prof) == pytest.approx(1.0)          # 1 ターン目でちょうど
    assert CB.tau_from_profile(0.125, 0, prof) == pytest.approx(1.5)         # 0.05 + 0.15/2
    assert CB.tau_from_profile(0.30, 1, prof) == pytest.approx(2.0)          # j=1 から 0.15 + 0.15
    assert CB.tau_from_profile(0.30, 1, prof, scale=2.0) == pytest.approx(1.0)
    rows = [_row(True, 2.0, 6.0, 2, 4) | {"j_me": 0, "j_opp": 0, "slope_theory_me": 0.2, "slope_theory_opp": 0.1}] * 30
    out = CB.summarise(rows, [], th)
    assert out["harm_profile"]["harm_by_turn"][:2] == pytest.approx([0.05, 0.15])
    assert set(out["by_slope"]) == {"hist", "theory", "curve", "curve_scaled"}
    assert out["by_slope"]["curve"]["sign_accuracy"] == 1.0                 # Θ が同じなら τ も同じ＝手番の自席
