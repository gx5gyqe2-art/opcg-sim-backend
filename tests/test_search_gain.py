"""`search_gain.py`（T65・探す効果の選択の利得を `v` で実測）の算術を固める。

**選択の利得の実現 = 探した札の v − 引いた札の v（基準）**・**能力の実現 = 手に入れた − 捨てた**・
**札全体 = 体 + 能力 − 探し札のドン**（ユーザ指示「探すのに使ったコストは考慮」）・価格側も同じ形で割る。
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

import effect_value as EV  # noqa: E402
import search_gain as SG  # noqa: E402
from theory_order import DELTA, MU  # noqa: E402


def _s(k, cost, body, v_added, v_removed, price_effect, cid="A"):
    return {"cid": cid, "k": float(k), "cost": float(cost), "body": body, "found": len(v_added), "added": [], "removed": [],
            "v_added": v_added, "v_removed": v_removed, "unknown_added": 0, "price_effect": price_effect, "turn": 3}


def test_the_premium_is_the_searched_cards_value_over_the_drawn_baseline():
    draws = [{"cid": "D", "v": 0.04, "turn": 2}, {"cid": "E", "v": 0.08, "turn": 3}]         # 基準 0.06
    ss = [_s(5, 1, 0.05, [0.10], [], 0.146), _s(5, 1, 0.05, [], [], 0.146)]                    # 2 回に 1 回見つかる
    out = SG.summarise(ss, draws, min_card=1)
    b = out["all"]
    assert out["draws"]["v_mean"] == pytest.approx(0.06) and b["found"] == 0.5
    assert b["premium_real"] == pytest.approx(0.10 - 0.06)                                     # 見つかった札だけで測る
    assert b["sel_k"] == pytest.approx(EV._sel_premium(5), abs=1e-6)
    assert b["premium_over_sel"] == pytest.approx(0.04 / EV._sel_premium(5), abs=1e-4)
    assert b["effect_real_mean"] == pytest.approx(0.05)                                        # (0.10 + 0) / 2
    assert b["effect_over_price"] == pytest.approx(0.05 / 0.146, abs=1e-4)
    assert b["net_mean"] == pytest.approx(0.05 + 0.05 - 1 * DELTA)                            # 体 + 能力 − ドン
    assert b["price_play_mean"] == pytest.approx(0.05 + 0.146 - 1 * DELTA)
    assert "A" in out["by_card"] and out["k=5"]["n"] == 2 and "k=3" not in out


def test_discards_and_the_don_cost_are_subtracted_from_the_realised_side():
    draws = [{"cid": "D", "v": MU, "turn": 2}]
    ss = [_s(5, 6, 0.30, [0.12], [0.06], 0.063)]                                               # エネル型: 5 枚見て 1 枚加え 1 枚捨てる
    b = SG.summarise(ss, draws, min_card=1)["all"]
    assert b["v_removed_mean"] == pytest.approx(0.06) and b["don_cost_mean"] == pytest.approx(6 * DELTA)
    assert b["effect_real_mean"] == pytest.approx(0.12 - 0.06)
    assert b["net_mean"] == pytest.approx(0.30 + 0.06 - 6 * DELTA)
    assert b["premium_real"] == pytest.approx(0.12 - MU)
    assert SG.summarise([], draws)["all"] == {"n": 0}
