"""`hand_spend.py`（T64・守りで切った札は手札の中で安い札か）の算術を固める。

**多重集合の差**（同名 2 枚のうち 1 枚を切れば 1 枚）・**順位**（0 = 一番安い・1 = 一番高い・同値は中点・1 枚なら 0.5）・
**価値の和 対 枚数 × μ** の 3 つを値で押さえる。`use_value` は機会費用なので 0 で床を打つ。
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

import hand_spend as HS  # noqa: E402
from theory_order import MU  # noqa: E402


def test_spent_cards_is_the_multiset_difference():
    assert sorted(HS.spent_cards(["A", "A", "B", "C"], ["A", "C", "D"])) == ["A", "B"]   # 引いた D は無視
    assert HS.spent_cards(["A"], ["A", "B"]) == []
    assert HS.spent_cards([], ["A"]) == []


def test_rank_is_the_position_in_the_ascending_hand():
    vals = [0.01, 0.05, 0.10, 0.20]
    assert HS.rank_of(vals, 0.01) == 0.0 and HS.rank_of(vals, 0.20) == 1.0
    assert HS.rank_of(vals, 0.05) == pytest.approx(1 / 3)
    assert HS.rank_of([0.05, 0.05, 0.10], 0.05) == pytest.approx(0.25)     # 同値 2 枚は中点
    assert HS.rank_of([0.05], 0.05) == 0.5                                  # 1 枚なら順位に意味が無い
    assert HS.x_band(0.0) == "0..1000" and HS.x_band(2000.0) == "1000..2000" and HS.x_band(-1000.0) == "x<0"


def test_use_value_is_the_opportunity_cost_floored_at_zero():
    # 2 コスト 3000 の素の体（相手リーダー 5000・R 4）: ν は小さくコスト 2·δ を引くと負 → 0 で床
    v = HS.use_value("X", {"power": 3000, "cost": 2, "blocker": False}, 5000.0, 4.0, cards={})
    assert v == 0.0
    big = HS.use_value("Y", {"power": 8000, "cost": 0, "blocker": False}, 5000.0, 4.0, cards={})
    assert big is not None and big > 0.1                                    # 大きい体は ν の分だけ高い
    assert HS.use_value("Z", None, 5000.0, 4.0, cards={}) is None            # 知らない札は None（0 にしない）


def _win(played, xb, spent, hand_v_mean=0.05):
    return {"played": played, "x_max": 2000.0, "xb": xb, "hand_n": 4, "hand_v_mean": hand_v_mean, "hand_v_min": 0.0,
            "spent": spent, "my_life": 3, "c_x": 2.25}


def test_block_reads_the_rank_and_the_value_over_count_times_mu():
    ws = [_win("guard", "1000..2000", [{"cid": "A", "v": 0.01, "counter": 2000.0, "event": True, "rank": 0.0},
                                        {"cid": "B", "v": 0.03, "counter": 1000.0, "event": False, "rank": 0.5}]),
          _win("guard", "1000..2000", [{"cid": "C", "v": 0.02, "counter": 2000.0, "event": False, "rank": 0.2}])]
    b = HS.block(ws)
    assert b["n"] == 2 and b["spent_cards"] == 3 and b["spent_per_window"] == 1.5
    assert b["rank_mean"] == pytest.approx((0.0 + 0.5 + 0.2) / 3)
    assert b["share_cheapest_quartile"] == pytest.approx(2 / 3)
    assert b["v_spent_over_mu"] == pytest.approx((0.01 + 0.03 + 0.02) / (3 * MU))   # 枚数 × μ より安いか
    assert b["event_share"] == pytest.approx(1 / 3) and b["counter_mean"] == pytest.approx(5000.0 / 3)
    out = HS.summarise(ws + [_win("take", ">2000", [])])
    assert out["take"]["spent_cards"] == 0 and out["take"]["v_spent_over_mu"] is None
    assert out["guard|1000..2000"]["n"] == 2 and "guard_rank_hist" in out
