"""`nu_stock.py`（T50・`ν` の在庫の台帳）の算術を固める。

**在庫の側は式を通さない**（体が生きている間に打った攻撃の実現の和）のが器の値打ち。
式の項の分解が `nu_of(mode="pair")` と 1 字も違わない和になること・生存と残りターンの数え方を値で押さえる。
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

import nu_stock as NS  # noqa: E402
import theory_order as T  # noqa: E402


def test_the_formula_terms_add_up_to_nu_of_pair():
    """項に分けても和は `nu_of(mode="pair")` そのもの（分解で値が動かない）。"""
    for pw, blk in ((3000.0, False), (6000.0, True), (9000.0, False)):
        f = NS.formula_terms(pw, 5000.0, 5000.0, 4.0, blk)
        assert f["nu"] == pytest.approx(T.nu_of(pw, 5000.0, 4.0, is_blocker=blk, mode="pair"))
        assert f["lead_R"] == pytest.approx(f["lead"] * 4.0)
        assert f["ko_p"] == T.ko_p_of(pw)
        assert (f["block"] > 0) == blk


def test_the_remaining_turns_follow_the_shipped_clip():
    assert NS.r_of(0) == 1.0 and NS.r_of(3) == 3.0 and NS.r_of(7) == 5.0


def _body(band="over_sat", alive=(1, 1, 1, None, None), turns_alive=3, n_atk=2, stock=0.2, opp_life=4):
    return {"band": band, "opp_life": opp_life, "r_form": NS.r_of(opp_life), "r_act": 3, "alive": list(alive),
            "turns_alive": turns_alive, "n_atk": n_atk, "stock_real": stock,
            "lead": 0.087, "lead_R": 0.087 * NS.r_of(opp_life), "option": 0.01, "block": 0.0, "shield": 0.0271,
            "ko_p": 0.2, "nu": 0.3}


def test_survival_ignores_turns_after_the_game_ended_and_the_stock_is_the_sum_of_realised_attacks():
    """`alive[k] = None`（対局が終わった）は生存率の母数に入れない。在庫は実現の和そのまま。"""
    bodies = [_body(alive=(1, 1, 0, None, None), turns_alive=2, n_atk=2, stock=0.18)] * 15 + \
             [_body(alive=(1, 1, 1, 1, 0), turns_alive=4, n_atk=4, stock=0.36)] * 15
    out = NS.summarise(bodies, {4: [3] * 30})
    o = out["by_band"]["over_sat"]
    assert o["n"] == 30
    assert o["surv"][1] == pytest.approx(1.0) and o["surv"][2] == pytest.approx(0.5)
    assert o["surv"][3] == pytest.approx(1.0)                   # None は母数から外れる（15 体だけで数える）
    assert o["surv"][4] == pytest.approx(0.0)
    assert o["stock_real"] == pytest.approx(0.27)
    assert o["real_per_attack"] == pytest.approx((0.18 * 15 + 0.36 * 15) / (2 * 15 + 4 * 15))
    assert o["attacks_per_turn_alive"] == pytest.approx(1.0)
    assert o["composed_stock"] == pytest.approx(o["real_per_attack"] * 1.0 * o["turns_alive_mean"])
    assert o["ratios"]["stock_real_over_measured"] == pytest.approx(0.27 / NS.NU_MEAS["over_sat"], abs=1e-3)
    assert out["r_by_opp_life"][4] == {"n": 30, "R_formula": 4.0, "R_actual": 3.0}
    # 標本が足りない帯は出さない
    assert "lt_leader" not in out["by_band"]


def test_the_formula_turn_weight_is_R_times_survival_once():
    """式の生存の重みは `(1 − ko_p)·R`——測った「生きて迎えたターン数」と並べる量。"""
    bodies = [_body()] * 25
    o = NS.summarise(bodies, {})["by_band"]["over_sat"]
    assert o["turn_weight_formula"] == pytest.approx(4.0 * 0.8, abs=1e-3)
    assert o["turn_weight_measured"] == pytest.approx(3.0)
