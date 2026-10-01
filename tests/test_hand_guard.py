"""`hand_guard.py`（T67・手札の守る備え `G_guard`）の算術を固める。

**止める組は「カウンター合計 ≥ x + 1000」（同値は命中）のうち `v` の和が最小**・**受ける方が安ければ節約 0**・
**札は地平の中で 1 回だけ**（大きい攻撃から当てる）・**2000 カウンターは `v` が低くカウンター値が高いので `ΔG` が立つ**。
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

import hand_guard as HG  # noqa: E402
from theory_order import KO_P  # noqa: E402

S = 1.0 - KO_P


def test_the_guard_cost_is_the_cheapest_subset_that_exceeds_the_excess():
    items = [(1000.0, 0.05), (2000.0, 0.01), (1000.0, 0.00)]        # (カウンター値, v)
    assert HG.guard_cost_min_v(items, -1000.0) == (0.0, ())           # 通らない攻撃は守る必要が無い
    cost, idx = HG.guard_cost_min_v(items, 0.0)                       # 合計 ≥ 1000: v 0 の 1000 カウンター 1 枚
    assert cost == 0.0 and idx == (2,)
    cost, idx = HG.guard_cost_min_v(items, 1000.0)                    # 合計 ≥ 2000: 2000 カウンター 1 枚（v 0.01）< 1000+1000（0.05）
    assert cost == pytest.approx(0.01) and idx == (1,)
    cost, idx = HG.guard_cost_min_v(items, 3000.0)                    # 合計 ≥ 4000: 全部（0.06）
    assert cost == pytest.approx(0.06) and set(idx) == {0, 1, 2}
    assert HG.guard_cost_min_v(items, 4000.0) == (None, ())           # 止められない


def test_guard_value_is_the_saving_over_taking_with_each_card_used_once():
    items = [(2000.0, 0.01), (1000.0, 0.00)]
    take = 0.087
    # 攻撃 x=1000 が 1 本: 2000 カウンター（0.01）で止める → 節約 0.077（t=0 は割引なし）
    assert HG.guard_value(items, [1000.0], take, S, turns=1) == pytest.approx(take - 0.01)
    # 2 本（x=1000 と x=0）: 大きい方に 2000、小さい方に 1000（v 0）→ 0.077 + 0.087
    assert HG.guard_value(items, [1000.0, 0.0], take, S, turns=1) == pytest.approx((take - 0.01) + take)
    # 3 本目は札が残らない＝受ける（節約 0）
    assert HG.guard_value(items, [1000.0, 0.0, 0.0], take, S, turns=1) == pytest.approx((take - 0.01) + take)
    # 2 ターン目は割引 s・札は使い切っているので 0
    assert HG.guard_value(items, [1000.0, 0.0], take, S, turns=2) == pytest.approx((take - 0.01) + take)
    assert HG.guard_value([(1000.0, 0.0)], [0.0], take, S, turns=2) == pytest.approx(take)    # 1 枚を t=0 に使う
    # 受ける方が安ければ守らない（高い札しか無い）
    assert HG.guard_value([(2000.0, 0.20)], [1000.0], take, S, turns=1) == 0.0


def test_delta_g_rewards_a_cheap_counter_card_when_attacks_are_coming():
    hand = [(0.0, 0.15), (1000.0, 0.09)]                              # 大物（カウンター 0）と v の高い 1000 カウンター
    take = 0.087
    dg = HG.delta_g(hand, (2000.0, 0.005), [1000.0], take, S, turns=1)   # 2000 カウンター（切っても惜しくない）
    assert dg == pytest.approx(take - 0.005)                          # 元は受ける方が安い（0）→ 節約 0.082
    assert HG.delta_g(hand, (0.0, 0.30), [1000.0], take, S, turns=1) == 0.0   # 大物はカウンター 0＝備えに効かない
    assert HG.delta_g(hand, (2000.0, 0.005), [], take, S, turns=1) == 0.0     # 攻撃が来なければ 0
    assert HG.take_cost_of(3) == pytest.approx(0.087, abs=2e-3) and HG.take_cost_of(0) > 0.4   # ライフ 0 は受ければ負け
