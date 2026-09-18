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


def test_the_windows_and_the_three_splits_read_the_series(monkeypatch):
    """**T81**: 相関の中身を割る 3 つ——**窓の広さ**（時差）・**手の型**（時計に見えない手）・**場が動いたか**（時計の粗さ）。
    `windows` は `rounds` ラウンド（`2 × rounds` ターン）の窓を並べ、`g` はその間の生の価格の和・`ΔW` は前後の勝率の差。"""
    import theory_order as TO
    ser = [{"d0": 0.0, "g0": 0.05, "g_fam": {"attack": 0.04, "effect": 0.01}, "r_turns": 4, "chars": 2},
           {"d0": 0.5, "g0": -0.02, "g_fam": {"attack": -0.02}, "r_turns": 4, "chars": 2},
           {"d0": 1.0, "g0": 0.03, "g_fam": {"attack": 0.03}, "r_turns": 3, "chars": 4},
           {"d0": 1.5, "g0": 0.01, "g_fam": {"guard": 0.01}, "r_turns": 3, "chars": 4}]
    w1 = KN.windows(ser, 1)
    assert len(w1) == 2                                            # 4 ターン → 1 ラウンドの窓は 2 つ
    assert w1[0]["g0"] == pytest.approx(0.03)                      # 0.05 + (−0.02)
    assert w1[0]["dW"] == pytest.approx(TO.prob_of_d(1.0) - TO.prob_of_d(0.0))
    assert w1[0]["g_fam"] == {"attack": pytest.approx(0.02), "effect": pytest.approx(0.01)}
    assert w1[0]["dchars"] == 2 and w1[1]["dchars"] == 2
    w2 = KN.windows(ser, 2)
    assert len(w2) == 0                                            # 2 ラウンド（4 ターン）ぶんの前後は取れない
    assert KN.windows(ser + list(ser), 2)[0]["g0"] == pytest.approx(0.07)
    # 3 つの層別が出る（行が足りない層は None）
    out = KN.summarise_games([ser + list(ser)] * 8)
    assert set(out) == {"by_window", "by_family", "by_combo", "by_board_change"}
    assert set(out["by_combo"]) == {"攻め＋守り", "攻め＋守り＋付与", "攻めだけ", "出す＋効果を除いた残り"}
    assert "attack" in out["by_family"] and "attack_除いた残り" in out["by_family"]
    assert set(out["by_board_change"]) == {"場が動いた", "場は同じ"}
    assert set(out["by_window"]) == {"1", "2", "3"}
