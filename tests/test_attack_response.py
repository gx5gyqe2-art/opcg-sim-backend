"""`attack_response.py`（T48・攻撃 1 回を相手の応答で割る）の算術を固める。"""
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

import attack_response as AR  # noqa: E402
import price_realised as PR  # noqa: E402


def _sc(opp_life=4, opp_hand=5, my_life=4, my_hand=5):
    sc = np.zeros(70, np.float32)
    sc[0], sc[1], sc[6], sc[7] = my_life, opp_life, my_hand, opp_hand
    sc[2], sc[4] = 3, 3
    sc[12], sc[13] = 0.5, 0.5
    return sc


def _tok(opp=()):
    tok = np.zeros((22, 24), np.float32)
    for k, pw in enumerate(opp):
        tok[7 + k, PR.S_POWER] = pw / 1e4
        tok[7 + k, PR.S_IS_CHAR] = 1.0
    return tok


def test_the_response_is_read_from_the_deltas():
    """受けた＝相手ライフ−・カウンター＝相手手札−・ブロッカーが倒れた＝相手の体−・何も無し＝全部＝。"""
    a, ta = _sc(), _tok(opp=(6000,))
    assert AR.classify(a, ta, _sc(opp_life=3, opp_hand=6), ta) == "took"          # ライフの札が手に入る
    assert AR.classify(a, ta, _sc(opp_hand=4), ta) == "countered"
    assert AR.classify(a, ta, _sc(opp_life=3, opp_hand=4), ta) == "took_and_countered"
    assert AR.classify(a, ta, _sc(), _tok(opp=())) == "blocker_died"
    assert AR.classify(a, ta, _sc(), ta) == "nothing"


def test_the_realised_parts_add_up_to_the_state_difference():
    a, ta = _sc(), _tok(opp=(6000,))
    b, tb = _sc(opp_life=3, opp_hand=6), _tok(opp=())
    p = AR.parts(a, ta, b, tb)
    assert p["opp_life"] == pytest.approx(PR.LAM)
    assert p["opp_hand"] == pytest.approx(-PR.MU)                      # ライフの札が相手の手に
    assert p["opp_body"] == pytest.approx(PR.NU_MEAS["leader_to_sat"])  # 6000 の体が消えた
    assert sum(p.values()) == pytest.approx(PR.state_meas(b, tb) - PR.state_meas(a, ta))


def test_x_bands_follow_the_cost_curve_steps():
    assert AR.x_band(-500.0) == "x<0"
    assert AR.x_band(0.0) == "0..1000" and AR.x_band(1000.0) == "0..1000"
    assert AR.x_band(1500.0) == "1000..2000"
    assert AR.x_band(3000.0) == ">2000"


def test_summary_reports_shares_by_response_and_by_margin():
    rows = []
    for i in range(40):
        rows.append({"leader": True, "resp": "took" if i % 2 else "countered", "x": 1500.0 if i % 2 else 500.0,
                     "xb": AR.x_band(1500.0 if i % 2 else 500.0), "price": 0.06, "real": 0.09 if i % 2 else 0.08,
                     "parts": {"opp_life": 0.136 if i % 2 else 0.0, "opp_hand": -0.055 if i % 2 else 0.08,
                               "opp_body": 0.0, "my_life": 0.0, "my_hand": 0.0, "my_body": 0.0, "don": 0.0},
                     "theta_says_take": bool(i % 2)})
    out = AR.summarise(rows)["leader"]
    assert out["n"] == 40 and out["by_response"]["took"]["share"] == pytest.approx(0.5)
    assert out["by_response"]["took"]["parts"]["opp_life"] == pytest.approx(0.136)
    assert out["by_x"]["1000..2000"]["response_shares"]["took"] == pytest.approx(1.0)
    assert out["by_x"]["0..1000"]["theta_says_take_share"] == 0.0
    assert "char" not in AR.summarise(rows)                              # 標本が無い側は出さない
