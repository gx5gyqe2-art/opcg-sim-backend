"""`price_scaling.py`（4 つの価格は 1 つの `w` を共有するか）の判定を固める。

**この器は §17.1（転換の式）の検定そのもの**なので、判定の出し方がずれると
理論の採否が変わる。**「支持」を出す条件を厳しく固定する**。
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

import price_scaling as P  # noqa: E402
import theory_order as T  # noqa: E402


def _tok(opp=(), my_leader=5000):
    tok = np.zeros((22, 24), np.float32)
    tok[0, T.S_POWER] = my_leader / 1e4
    tok[1, T.S_POWER] = 0.5
    for j, pw in enumerate(opp):
        tok[7 + j, T.S_POWER] = pw / 1e4
        tok[7 + j, T.S_IS_CHAR] = 1.0
    return tok


def test_the_split_counts_only_attacks_that_connect():
    """既定の切り方は **`A_opp`＝相手の通る攻撃数**（`T_opp` の分母・§17.1）。"""
    assert P.race_of(0.0, _tok(opp=(3000, 7000), my_leader=5000)) == "a2"
    assert P.race_of(0.0, _tok(opp=(3000, 7000), my_leader=9000)) == "a0"
    assert P.race_of(0.0, _tok(opp=(6000, 7000, 8000, 9000))) == "a3"   # 上限で潰す


def test_the_v0_split_stays_available_but_is_not_the_default():
    """**`|v0|` で切ると器が壊れる**（説明変数の下流）——再現用に残すが既定にしない。

    実測で接戦帯の `μ` が **−0.0254**（手札 1 枚の価値が負）になった。
    `ν` を測るとき自分の場で帯を切れないのと同じ誤り。
    """
    assert P.SPLITS[0] == "a_opp"
    assert P.race_of(0.05, _tok(opp=(9000,)), split="v0") in ("close", "mid", "decided")


def test_a_negative_price_is_reported_as_a_broken_instrument():
    """**価格が負なら器が壊れている**——手札もライフもドンも体も正の価値を持つ。"""
    ok = {"mu": {"value": 0.04}, "lambda": {"value": 0.11}}
    bad = {"mu": {"value": -0.03}, "lambda": {"value": 0.11}}
    assert P.price_signs_ok(ok) and not P.price_signs_ok(bad)
    assert not P.price_signs_ok({"mu": {"value": None}})


def test_a_wide_ci_cannot_support_flatness():
    """**CI が狭くて初めて「重なる＝動かない」が言える**（2026-09-14 に判定を直した）。

    初版は [−9, 16] と [−40, 42] が重なるのを見て「比は動かない＝★の式を支持」と出した
    ——**雑音を支持と読んでいた**。罠 17（分解能を先に数える）を CI に当てはめたもの。
    """
    assert P.informative([3.0, 3.4], 3.2)                 # 幅 0.4 / 3.2 = 0.125
    assert not P.informative([-9.0, 16.0], 3.2)           # 幅 25 / 3.2 = 7.8
    assert not P.informative([None, 4.0], 3.2)
    assert not P.informative([1.0, 2.0], 0.0)


def test_thin_bands_are_dropped_before_judging():
    """**薄い帯は判定に入れない**——`a0`（52 行）の負の値だけで「壊れている」と出た。"""
    recs = ([{"seed": i, "race": "a1", "sc": np.zeros(127), "z": 1.0} for i in range(500)]
            + [{"seed": i, "race": "a0", "sc": np.zeros(127), "z": 1.0} for i in range(20)])
    out = P.summarise(recs, reps=0, min_rows=200)
    assert [d["race"] for d in out["dropped_thin"]] == ["a0"]
    assert "a0" not in out["by_race"]


def test_the_ratio_set_is_anchored_on_the_hand():
    """比は**`μ` を基準**に取る——手札は 4 通貨で唯一「攻めにも守りにも使える」。

    `λ/μ` は `c̄`（守る費用）そのもので、**デッキと盤面の量だから帯で動かないはず**
    ——これが★の式の一番鋭い予測。
    """
    names = [n for n, _a, _b in P.RATIOS]
    assert names == ["lambda_over_mu", "delta_over_mu", "nu_over_mu"]
    assert all(den == "hand" for _n, _num, den in P.RATIOS)


def test_every_price_axis_drops_its_own_regressor_from_the_band():
    """**説明変数そのものは帯に入れない**（入れると帯の中で動かず傾きが定義できない）。

    軸ごとに帯の中身が入れ替わることを、値で押さえる。
    """
    from hand_value_slope import band_key, SC_MY_HAND, SC_MY_LIFE
    a = np.zeros(127, np.float32); a[SC_MY_LIFE] = 3.0; a[SC_MY_HAND] = 5.0
    b = a.copy(); b[SC_MY_LIFE] = 1.0                 # ライフだけ違う
    c = a.copy(); c[SC_MY_HAND] = 9.0                 # 手札だけ違う
    # `life` 軸はライフを帯から外す＝ライフが違っても同じ帯
    assert band_key(a, "life") == band_key(b, "life")
    assert band_key(a, "life") != band_key(c, "life")
    # `hand` 軸はその逆
    assert band_key(a, "hand") == band_key(c, "hand")
    assert band_key(a, "hand") != band_key(b, "hand")
