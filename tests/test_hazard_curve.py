"""`tests/scripts/hazard_curve.py`（`P(T=t|s)` とハザード）の算術。

基盤健全性（`cpu_infra`）。ハザードは `h(t) = P(T=t | T≥t)` なので**母数が段ごとに減る**——
ここを取り違えると「決着が近い」の読みが逆になる。
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

import hazard_curve as H  # noqa: E402
import numpy as np  # noqa: E402


def test_t_bucket_clamps_both_ends():
    assert H.t_bucket(0.4) == "1"            # 0 以下は 1 に寄せる
    assert H.t_bucket(1.0) == "1"
    assert H.t_bucket(5.0) == "5"
    assert H.t_bucket(6.0) == "6+"
    assert H.t_bucket(12.0) == "6+"


def test_band_key_narrow_and_wide():
    sc = np.zeros(14, np.float32)
    sc[H.SC_L], sc[H.SC_OPP_L], sc[H.SC_H], sc[H.SC_OPP_FIELD] = 3, 5, 6, 2
    assert H.band_key(sc) == "3|5"
    assert H.band_key(sc, wide=True) == "3|5|6|2"


def test_hazard_denominator_shrinks_each_step():
    """T=1 が 2 件・T=2 が 2 件・T=3 が 6 件 → h(1)=0.2・h(2)=2/8=0.25・h(3)=1.0。"""
    recs = ([{"T": 1.0, "z": 1.0}] * 2 + [{"T": 2.0, "z": 0.0}] * 2
            + [{"T": 3.0, "z": 1.0}] * 6)
    d = H.dist_of(recs)
    assert d["n"] == 10
    assert d["p_T"]["1"] == 0.2 and d["p_T"]["3"] == 0.6
    assert d["hazard"]["1"] == pytest.approx(0.2)
    assert d["hazard"]["2"] == pytest.approx(0.25)
    assert d["hazard"]["3"] == pytest.approx(1.0)
    assert d["hazard"]["4"] is None            # 母数が尽きた段は None
    assert d["E_T"] == pytest.approx(2.4)


def test_winrate_by_t_needs_enough_rows():
    recs = [{"T": 1.0, "z": 1.0}] * 25 + [{"T": 2.0, "z": 0.0}] * 5
    d = H.dist_of(recs)
    assert d["winrate_by_T"]["1"] == pytest.approx(1.0)
    assert d["winrate_by_T"]["2"] is None      # n<20 は出さない
    assert d["winrate"] == pytest.approx(25 / 30, abs=1e-4)


def test_flat_hazard_score_detects_a_memoryless_process():
    """ハザードが一定なら spread ≈ 0（＝盤面から時計を読む意味が薄い）。"""
    flat = {"hazard": {"1": 0.3, "2": 0.3, "3": 0.3, "4": None, "5": None, "6+": 1.0}}
    assert H.flat_hazard_score(flat) == pytest.approx(0.0)
    peaked = {"hazard": {"1": 0.8, "2": 0.1, "3": 0.1, "4": None, "5": None, "6+": 1.0}}
    assert H.flat_hazard_score(peaked) == pytest.approx(0.7)
    assert H.flat_hazard_score({"hazard": {"1": 0.5, "6+": 1.0}}) is None   # 1 点では出さない


def test_buckets_cover_one_through_six_plus():
    assert H.buckets() == ["1", "2", "3", "4", "5", "6+"]
