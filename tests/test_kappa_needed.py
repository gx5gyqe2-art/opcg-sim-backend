"""**必要な `κ` の集計**（T80）の算術——**基盤健全性**（`cpu_infra`）。

`kappa_needed.summarise` は「ターン（ラウンド）ごとの（`D`・生の価格 `g`・勝率の動き `ΔW`）」から
**`ΔW` を `g` に回した傾き＝必要な `κ`** と相関を出し、`|D|` 別・`r_turns` 別に層別する。
ここでは**与えた数字がそのまま出るか**だけを見る（記録は読まない）。
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
import _bootstrap  # noqa: F401,E402
import kappa_needed as KN  # noqa: E402

pytestmark = pytest.mark.cpu_infra


def test_the_needed_kappa_is_the_regression_slope_and_the_bands_split_by_d_and_horizon():
    # `ΔW = 0.5 × g` を仕込めば傾きは 0.5・相関は 1
    turns = [{"d0": 0.5, "g0": g, "r_turns": 4, "dW": 0.5 * g}
             for g in (0.02, 0.05, -0.03, 0.08, -0.06, 0.04, 0.07, -0.05, 0.03, 0.09, -0.02, 0.06)]
    out = KN.summarise(turns)
    assert out["n_used"] == len(turns)
    assert out["fit"]["kappa_fit"] == pytest.approx(0.5)
    assert out["fit"]["corr"] == pytest.approx(1.0)
    # `g` の小さいターンは落ちる（比が発散するため）
    assert KN.summarise(turns + [{"d0": 0.0, "g0": 0.001, "r_turns": 4, "dW": 0.4}])["n_used"] == len(turns)
    # 層別: `|D|` の帯と `r_turns` で別の傾きが出る
    mixed = turns + [{"d0": 5.0, "g0": g, "r_turns": 1, "dW": 0.1 * g}
                     for g in (0.02, 0.05, -0.03, 0.08, -0.06, 0.04, 0.07, -0.05, 0.03, 0.09, -0.02, 0.06)]
    out2 = KN.summarise(mixed)
    assert out2["by_d"]["|D|<=1"]["kappa_fit"] == pytest.approx(0.5)
    assert out2["by_d"]["|D|>3"]["kappa_fit"] == pytest.approx(0.1)
    assert out2["by_r_turns"]["4"]["kappa_fit"] == pytest.approx(0.5)
    assert out2["by_r_turns"]["1"]["kappa_fit"] == pytest.approx(0.1)
    # **相関がほぼ 0 ＝ どんな `κ` でも恒等式は立たない**、が読める（`ΔW` が `g` と無関係な符号で振れる場合）
    # 同じ `g` を `+ΔW` と `−ΔW` の両方に組ませる＝相関はちょうど 0（どんな `κ` でも合わない状態）
    noise = [{"d0": 0.0, "g0": g, "r_turns": 4, "dW": w}
             for g in (0.02, 0.05, 0.03, 0.08, 0.06, 0.04) for w in (0.03, -0.03)]
    assert KN.summarise(noise)["fit"]["corr"] == pytest.approx(0.0)
    assert KN.summarise(noise)["fit"]["kappa_fit"] == pytest.approx(0.0)
    assert KN.d_band(0.0) == "|D|<=1" and KN.d_band(1.0) == "|D|<=1"
    assert KN.d_band(1.01) == "1<|D|<=3" and KN.d_band(3.0) == "1<|D|<=3" and KN.d_band(3.01) == "|D|>3"
    assert KN.summarise([]) == {"n_turns": 0, "n_used": 0}
