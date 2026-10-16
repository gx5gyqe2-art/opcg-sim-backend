"""`crossing_bridge.py`（T52・交点の橋）の集計と、輪郭に沿った到達ターンの算術を固める。

**当てはめない・回帰しない**器なので、残差が「実際に届いた側の τ 対 その側の残りターン」で作られること、
単位の検算が比そのままであること、損害の輪郭（`harm_profile`）と輪郭に沿った到達ターン（Rust の `tau_from_profile`）が
手計算どおりであることを値で押さえる。段 7（2026-10-07）: 耐久・速さ・守る側の計算は Rust だけになり、その式の手計算の
試験は Python の写しと一緒に消した（Rust の側は `cargo test` の記録の再生が見る）。
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

import crossing_bridge as CB  # noqa: E402


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


def test_sigma_rel_drops_rows_whose_scale_is_zero_instead_of_dividing_by_a_floor():
    """**T154**: 両席の τ が 0 の行は相対残差が 0/0＝定義できない。母数から外し、外した行数を `n_scale0` に出す。"""
    rows = [_row(True, 2.0, 6.0, 2, 4)] * 30 + [_row(False, 5.0, 3.0, 4, 2)] * 30
    o_clean = CB.summarise(rows, [])["by_slope"]["hist"]
    assert o_clean["n_scale0"] == 0 and o_clean["sigma_rel"] is not None
    o = CB.summarise(rows + [_row(True, 0.0, 0.0, 1, 1)] * 3, [])["by_slope"]["hist"]
    assert o["n_scale0"] == 3
    assert o["sigma_rel"] == pytest.approx(o_clean["sigma_rel"])          # 尺度 0 の行は σ_rel に入らない
    assert o["sigma_rel"] < 10.0
    only0 = CB.summarise([_row(True, 0.0, 0.0, 1, 1)] * 5, [])["by_slope"]["hist"]
    assert only0["sigma_rel"] is None and only0["n_scale0"] == 5


def _tau(theta, j, prof, scale=1.0):
    return CB.tau_from_profile([(theta, j, scale, 0.0, 0.0, 0.0, 0.0, 0.0)], prof)[0]


def test_the_harm_profile_extends_its_last_value_and_the_profile_crossing_interpolates():
    """損害の輪郭は `j` ごとの平均・薄い先は最後の値を伸ばす。輪郭に沿った交点は端数を比例配分する。"""
    th = [{"j": 0, "harm": 0.05, "slope_theory": 0.1}] * 20 + [{"j": 1, "harm": 0.15, "slope_theory": 0.2}] * 20 + \
         [{"j": 2, "harm": 0.25, "slope_theory": 0.3}] * 5                  # j=2 は標本不足
    prof, prof_th = CB.harm_profile(th, j_max=4)
    assert prof == pytest.approx([0.05, 0.15, 0.15, 0.15, 0.15])
    assert prof_th == pytest.approx([0.1, 0.2, 0.2, 0.2, 0.2])
    assert _tau(0.05, 0, prof) == pytest.approx(1.0)          # 1 ターン目でちょうど
    assert _tau(0.125, 0, prof) == pytest.approx(1.5)         # 0.05 + 0.15/2
    assert _tau(0.30, 1, prof) == pytest.approx(2.0)          # j=1 から 0.15 + 0.15
    assert _tau(0.30, 1, prof, scale=2.0) == pytest.approx(1.0)
    rows = [_row(True, 2.0, 6.0, 2, 4) | {"j_me": 0, "j_opp": 0, "slope_theory_me": 0.2, "slope_theory_opp": 0.1}] * 30
    out = CB.summarise(rows, [], th)
    assert out["harm_profile"]["harm_by_turn"][:2] == pytest.approx([0.05, 0.15])
    assert set(out["by_slope"]) == {"hist", "theory", "curve", "curve_scaled"}
    assert out["by_slope"]["curve"]["sign_accuracy"] == 1.0                 # Θ が同じなら τ も同じ＝手番の自席
