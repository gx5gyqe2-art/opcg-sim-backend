"""`price_realised.py`（T41・型ごとの「価格」対「実現した価値」）の算術を固める。

**実現の側は A 層の実測だけで作る**（式 `nu_of` を通さない）のが器の値打ちなので、
その約束と、ドンを**総在庫**で数えること（付与で動かない）を値で押さえる。
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

import price_realised as PR  # noqa: E402
import theory_order as T  # noqa: E402


def _sc(my_life=4, opp_life=4, my_hand=5, opp_hand=5, my_act=3, my_rest=0, opp_act=3, opp_rest=0,
        my_ldon=0, opp_ldon=0):
    sc = np.zeros(70, np.float32)
    sc[0], sc[1] = my_life, opp_life
    sc[2], sc[3], sc[4], sc[5] = my_act, my_rest, opp_act, opp_rest
    sc[6], sc[7] = my_hand, opp_hand
    sc[12], sc[13] = 0.5, 0.5                     # リーダー 5000/5000
    sc[14], sc[15] = my_ldon / 5.0, opp_ldon / 5.0
    return sc


def _tok(own=(), opp=()):
    """`own`/`opp` は (パワー, 付与ドン) の列。"""
    tok = np.zeros((22, 24), np.float32)
    for base, bodies in ((2, own), (7, opp)):
        for k, (pw, don) in enumerate(bodies):
            tok[base + k, PR.S_POWER] = pw / 1e4
            tok[base + k, PR.S_ATTACHED_DON] = don / 5.0
            tok[base + k, PR.S_IS_CHAR] = 1.0
    return tok


def test_the_realised_nu_is_the_measured_band_value_not_the_formula():
    """実現の側の `ν` は**帯ごとの実測値**——式 `nu_of` とは別物（式の欠陥が実現に混ざらない）。"""
    assert PR.nu_meas_of(3000.0, 5000.0) == PR.NU_MEAS["lt_leader"]
    assert PR.nu_meas_of(5000.0, 5000.0) == PR.NU_MEAS["leader_to_sat"]
    assert PR.nu_meas_of(7000.0, 5000.0) == PR.NU_MEAS["leader_to_sat"]     # 飽和点まで
    assert PR.nu_meas_of(7000.0 + 2 * PR.PWR_EPS, 5000.0) == PR.NU_MEAS["over_sat"]
    # 式の側（`base`）はリーダー未満を 0 と言う——実測は 0.069。ここが混ざっていないことの証拠
    assert T.nu_of(3000.0, 5000.0, 4.128, is_blocker=False, mode="base") == 0.0
    assert PR.nu_meas_of(3000.0, 5000.0) > 0.0


def test_the_don_stock_counts_active_rested_and_attached_alike():
    """**ドンは総在庫**——付与しても動かず、`RAMP_DON` だけが増やす。"""
    tok = _tok(own=((5000, 2),), opp=())
    sc = _sc(my_act=3, my_rest=1, my_ldon=1)
    assert PR.don_stock(sc, tok, "me") == pytest.approx(3 + 1 + 1 + 2)
    assert PR.don_stock(sc, tok, "opp") == pytest.approx(3)
    # 付与＝アクティブ 1 減・キャラ付与 1 増 → 総在庫は不変
    tok2 = _tok(own=((5000, 3),), opp=())
    sc2 = _sc(my_act=2, my_rest=1, my_ldon=1)
    assert PR.don_stock(sc2, tok2, "me") == pytest.approx(PR.don_stock(sc, tok, "me"))
    assert PR.state_meas(sc2, tok2) == pytest.approx(PR.state_meas(sc, tok))


def test_the_state_is_priced_with_measured_prices_and_is_antisymmetric():
    """`S_meas` は 4 通貨の実測価格の差の和。等しい盤面は 0・差は席を入れ替えると符号が返る。"""
    assert PR.state_meas(_sc(), _tok()) == pytest.approx(0.0)
    assert PR.state_meas(_sc(my_life=5), _tok()) == pytest.approx(PR.LAM)
    assert PR.state_meas(_sc(my_hand=6), _tok()) == pytest.approx(PR.MU)
    assert PR.state_meas(_sc(my_act=4), _tok()) == pytest.approx(PR.DELTA)
    assert PR.state_meas(_sc(), _tok(own=((3000, 0),))) == pytest.approx(PR.NU_MEAS["lt_leader"])
    assert PR.state_meas(_sc(), _tok(opp=((3000, 0),))) == pytest.approx(-PR.NU_MEAS["lt_leader"])
    a = PR.state_meas(_sc(my_life=3, opp_life=5, my_hand=7, opp_hand=4), _tok(own=((9000, 0),)))
    b = PR.state_meas(_sc(my_life=5, opp_life=3, my_hand=4, opp_hand=7), _tok(opp=((9000, 0),)))
    assert a == pytest.approx(-b)


def test_the_effect_breakdown_reads_the_first_action_of_the_activated_ability():
    """効果の型の内訳は**最初の能力の最初の動作の型**で切る（読めなければ `?`）。"""
    assert PR.primary_action("OP15-058") == "RAMP_DON"        # エネル: ドン 1＋4 を追加し 4 を付与
    assert PR.primary_action("__no_such_card__") == "?"
    assert PR.primary_action(None) == "?"


def test_the_gross_price_adds_back_the_don_opportunity_cost_the_theory_charges():
    """登場・イベントの価格が引く `費用·δ` は**機会費用**で在庫の差分に現れない——`gross` で足し戻す。"""
    assert PR.DON_COST == pytest.approx(0.66 * PR.MU)
    pv = T.play_value(5000.0, 4, 5000.0, 4.128)
    assert pv + 4 * PR.DON_COST == pytest.approx(T.play_value(5000.0, 0, 5000.0, 4.128))
