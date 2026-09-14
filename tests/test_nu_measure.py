"""`ν` を種類別に直接測る算術（`tests/scripts/nu_measure.py`）。

基盤健全性（`cpu_infra`）。この器は**式の検算**をするので、器自身が疑われたら意味が無い。
固めるのは 4 つ:

1. **1 変数にすれば素の within 傾きと一致する**（Phase 1a の 0.1087 と突き合わせる根拠）。
2. **帯で中心化している**（帯ごとに水準が違っても騙されない）。
3. **SE は対局でクラスタする**——全行を一律に複製しても縮まない。
4. **パワーの帯は式が値を変える境目と同じ**（`x<0`／飽和点 2000）で切る。
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

import nu_measure as M  # noqa: E402
import nu_calib as N  # noqa: E402


def _rows(spec, key="chars"):
    """`spec` = [(band, seed, x, z), …]。"""
    return [{"band": b, "seed": s, key: x, "z": z} for b, s, x, z in spec]


def test_one_variable_matches_a_plain_within_slope():
    """**器の突き合わせ**——1 本なら Phase 1a と同じ量を測っていること。"""
    rows = _rows([("A", 1, 0.0, 0.0), ("A", 1, 1.0, 0.5),
                  ("B", 2, 10.0, 10.0), ("B", 2, 11.0, 10.5)])
    out = M.within_multi(rows, ["chars"])
    assert out["beta"]["chars"] == pytest.approx(0.5)      # 帯の中はどちらも 0.5
    assert out["bands"] == 2 and out["n"] == 4 and out["games"] == 2


def test_the_band_level_cannot_leak_in():
    """帯ごとに水準が大きく違っても傾きは変わらない（pooled なら騙される形）。"""
    flat = _rows([("A", 1, 0.0, 0.0), ("A", 1, 1.0, 1.0),
                  ("B", 2, 0.0, 100.0), ("B", 2, 1.0, 101.0)])
    assert M.within_multi(flat, ["chars"])["beta"]["chars"] == pytest.approx(1.0)


def test_cluster_se_ignores_duplicated_rows():
    """**全行を一律に複製しても SE は変わらない**（局の構造が変わっていないので）。"""
    base = _rows([("A", g, v, 0.1 * g + 0.4 * v)
                  for g in (1, 2, 3, 4, 5) for v in (0.0, 1.0, 2.0)])
    one = M.within_multi(base, ["chars"])
    twice = M.within_multi(base + [dict(r) for r in base], ["chars"])
    assert one["beta"]["chars"] == pytest.approx(twice["beta"]["chars"])
    assert one["se"]["chars"] == pytest.approx(twice["se"]["chars"], rel=1e-6)
    assert one["games"] == twice["games"] == 5
    assert twice["n"] == 2 * one["n"]


def test_two_variables_separate_their_own_slopes():
    """多変量＝**種類ごとの価格**が分かれて出ること（これが本器の目的）。"""
    recs = []
    for g in range(1, 21):
        for a in (0.0, 1.0, 2.0):
            for b in (0.0, 1.0):
                recs.append({"band": "A", "seed": g, "blocker": b, "plain": a,
                             "z": 0.3 * b + 0.1 * a})
    out = M.within_multi(recs, ["blocker", "plain"])
    assert out["beta"]["blocker"] == pytest.approx(0.3, abs=1e-6)
    assert out["beta"]["plain"] == pytest.approx(0.1, abs=1e-6)


def test_a_collinear_design_is_reported_not_crashed():
    recs = [{"band": "A", "seed": g, "a": v, "b": 2.0 * v, "z": v}
            for g in (1, 2, 3) for v in (0.0, 1.0)]
    out = M.within_multi(recs, ["a", "b"])
    assert out["error"] == "rank_deficient"
    assert M.within_multi([], ["a"]) is None


def test_power_bands_cut_where_the_formula_changes_value():
    """境目は**式が値を変える点**と同じ（`x<0` と 飽和点 2000）。"""
    assert M.power_band(4000.0, 5000.0) == "lt_leader"     # 通らない＝式では ν=0
    assert M.power_band(5000.0, 5000.0) == "leader_to_sat"
    assert M.power_band(6990.0, 5000.0) == "leader_to_sat"
    assert M.power_band(7000.0, 5000.0) == "over_sat"      # 超過 2000＝飽和
    assert M.power_band(12000.0, 5000.0) == "over_sat"
    # f32 の丸めを吸う（0.7×1e4 が 6999.999… でも飽和側に入る）
    assert M.power_band(7000.0 - 1e-6, 5000.0) == "over_sat"


def _tok(slots):
    t = np.zeros((22, 22), np.float32)
    for s, (pw, blk) in slots.items():
        t[s, N.S_POWER] = pw / 1e4
        t[s, N.S_IS_CHAR] = 1.0
        t[s, N.S_BLOCKER_ACTIVE] = 1.0 if blk else 0.0
    return t


def test_categories_count_by_scheme():
    tok = _tok({2: (3000.0, False), 3: (6000.0, True), 4: (9000.0, False)})
    assert M.categories(tok, 5000.0, "all") == {"chars": 3.0}
    assert M.categories(tok, 5000.0, "blocker") == {"plain": 2.0, "blocker": 1.0}
    assert M.categories(tok, 5000.0, "power") == {
        "lt_leader": 1.0, "leader_to_sat": 1.0, "over_sat": 1.0}
    assert M.categories(tok, 5000.0, "power_blocker") == {
        "lt_leader_plain": 1.0, "leader_to_sat_blk": 1.0, "over_sat_plain": 1.0}
    assert M.categories(np.zeros((22, 22), np.float32), 5000.0, "all") == {}


def test_check_flags_whether_the_formula_lands_in_the_ci():
    fit = {"beta": {"lt_leader": 0.069, "over_sat": 0.211},
           "se": {"lt_leader": 0.019, "over_sat": 0.019},
           "ci95": {"lt_leader": [0.031, 0.107], "over_sat": [0.175, 0.247]},
           "share": {"lt_leader": 0.45, "over_sat": 0.25}}
    ch = M.check(fit, {"lt_leader": 0.0, "over_sat": 0.186})
    by = {r["kind"]: r for r in ch["rows"]}
    assert by["lt_leader"]["in_ci"] is False            # 0 は CI の外＝式の穴
    assert by["over_sat"]["in_ci"] is True
    # 差の大きい順に並ぶ（読む順を固定する）
    assert ch["rows"][0]["kind"] == "lt_leader"
    assert M.check(None, {}) is None


def test_check_reports_whether_phase1a_is_reproduced():
    """**器の健全性の判定**——1 本の傾きが Phase 1a の 0.1087 を含むか。"""
    fit = {"beta": {"chars": 0.123}, "se": {"chars": 0.0125},
           "ci95": {"chars": [0.0985, 0.1475]}, "share": {"chars": 1.1}}
    assert M.check(fit, {})["reproduces_phase1a"] is True
    off = {"beta": {"chars": 0.30}, "se": {"chars": 0.01},
           "ci95": {"chars": [0.28, 0.32]}, "share": {"chars": 1.1}}
    assert M.check(off, {})["reproduces_phase1a"] is False


def test_predict_gives_zero_for_the_band_the_formula_zeroes():
    """式の予測側の回帰——**リーダー未満は 0**（そこが最大の穴だと判った場所）。"""
    p = M.predict(["lt_leader", "leader_to_sat", "over_sat"])
    assert p["lt_leader"] == 0.0
    assert p["leader_to_sat"] < p["over_sat"]           # 飽和までは伸びる
    # ブロッカーは上乗せされる
    pb = M.predict(["leader_to_sat_blk", "leader_to_sat_plain"])
    assert pb["leader_to_sat_blk"] > pb["leader_to_sat_plain"]
