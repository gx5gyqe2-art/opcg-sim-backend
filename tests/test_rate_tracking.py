"""`rate_tracking.py`（T128）の器のテスト——**`A` が「このターンの損害」を追えているか**を測る器。

**基盤健全性ではない**——ここで守っているのは「**器が正しい量を出すか**」であり、
ゲームプレイの正しさではない。だが**計測の誤りは理論の誤判断に直結する**（T125／T126 の実害）ので
必須側に置く（マーカー無し・常時実行）。
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import _bootstrap  # noqa: F401,E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
import rate_tracking as RT  # noqa: E402


def _rows(n=200, noise=0.0, seed=0):
    """`turn_harm` と同じ形の行を作る（`g`／`who`／`j`／`harm`／`slope_theory`／`t_left`）。"""
    rng = np.random.RandomState(seed)
    out = []
    for i in range(n):
        j = i % 5
        a = 0.05 * (j + 1)
        h = a + (rng.randn() * noise)
        out.append({"g": i // 10, "who": i % 2, "j": j, "harm": float(h),
                    "slope_theory": float(a), "priced": 0.0, "t_left": 3})
    return out


# --------------------------------------------------------------------------- 1. 部品
def test_at_clamps_to_the_last_bucket():
    """輪郭の索引は**末尾で止まる**（`tau_from_profile` と同じ規約）。"""
    prof = [0.0, 0.1, 0.2]
    assert RT.at(prof, 0) == 0.0
    assert RT.at(prof, 2) == 0.2
    assert RT.at(prof, 99) == 0.2          # 超えたら末尾
    assert RT.at([], 3) == 0.0             # 輪郭が無ければ 0


def test_corr_returns_none_for_a_constant_column():
    assert RT.corr([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None
    assert RT.corr([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


# --------------------------------------------------------------------------- 2. とどめのターンの除外（T107 の規則）
def test_the_finishing_turn_is_excluded_by_default():
    rows = _rows(20)
    rows[0]["t_left"] = 1                  # とどめのターン
    rows[1]["t_left"] = 0
    assert RT.measure(rows, [0.1] * 6, [0.1] * 6)["n"] == 18
    assert RT.measure(rows, [0.1] * 6, [0.1] * 6, include_last=True)["n"] == 20


def test_no_rows_left_returns_an_empty_result_instead_of_dividing_by_zero():
    rows = _rows(4)
    for r in rows:
        r["t_left"] = 1
    assert RT.measure(rows, [0.1] * 6, [0.1] * 6) == {"n": 0}


# --------------------------------------------------------------------------- 3. 3 つの量
def test_a_perfect_rate_is_reported_as_perfect():
    """`A` がそのターンの損害と厳密に一致するなら、水準 1・相関 1・説明力 1。"""
    out = RT.measure(_rows(200, noise=0.0), [0.15] * 6, [0.15] * 6)
    assert out["level_ratio"] == pytest.approx(1.0, abs=1e-6)
    assert out["corr_a"] == pytest.approx(1.0, abs=1e-6)
    assert out["r2_a"] == pytest.approx(1.0, abs=1e-6)


def test_the_level_ratio_is_theory_over_actual():
    """**向き**を固定する（1 より大きい＝理論が多い）。"""
    rows = _rows(50, noise=0.0)
    for r in rows:
        r["harm"] = r["slope_theory"] / 2.0
    assert RT.measure(rows, [0.1] * 6, [0.1] * 6)["level_ratio"] == pytest.approx(2.0, abs=1e-6)


def test_the_residual_correlation_is_zero_when_a_only_knows_the_turn_number():
    """**本命の量**——`A` が `j` の関数でしかないなら、輪郭を引いた残りは定数 0 ＝相関は `None`。

    「`A` が輪郭を超える情報を持つか」を測る量なので、**持たないときに 0 と言えなければ意味が無い**。
    """
    rows = _rows(200, noise=0.3, seed=1)               # 損害は暴れるが `A` は `j` だけの関数
    prof_th = [0.05 * (j + 1) for j in range(6)]       # 理論の輪郭 = `A` そのもの
    prof = [0.05 * (j + 1) for j in range(6)]
    out = RT.measure(rows, prof, prof_th)
    assert out["corr_resid"] is None                   # `A − prof_th[j]` が恒等的に 0


def test_the_residual_correlation_is_positive_when_a_carries_extra_information():
    rng = np.random.RandomState(2)
    rows, prof, prof_th = [], [0.1] * 6, [0.1] * 6
    for i in range(400):
        extra = float(rng.randn()) * 0.02              # `j` では説明できない当たり外れ
        rows.append({"g": i // 10, "who": i % 2, "j": 2, "harm": 0.1 + extra,
                     "slope_theory": 0.1 + extra, "t_left": 3})
    out = RT.measure(rows, prof, prof_th)
    assert out["corr_resid"] == pytest.approx(1.0, abs=1e-6)


def test_r2_does_not_refit_the_predictor():
    """**当てはめない**——2 倍ずれた予測は、相関が 1 でも説明力は負になる。"""
    rows = _rows(200, noise=0.0)
    for r in rows:
        r["slope_theory"] = r["harm"] * 2.0
    out = RT.measure(rows, [0.15] * 6, [0.15] * 6)
    assert out["corr_a"] == pytest.approx(1.0, abs=1e-6)
    assert out["r2_a"] < 0.0


def test_by_j_skips_thin_buckets_and_keeps_the_counts():
    rows = _rows(200)
    rows.append({"g": 99, "who": 0, "j": 9, "harm": 0.5, "slope_theory": 0.5, "t_left": 3})
    out = RT.measure(rows, [0.15] * 6, [0.15] * 6)
    assert "9" not in out["by_j"]                      # n < 20 は出さない
    assert sum(v["n"] for v in out["by_j"].values()) == 200
