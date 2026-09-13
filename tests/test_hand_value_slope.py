"""`tests/scripts/hand_value_slope.py` の純関数（帯・within 推定・枠落とし・符号判定）。

基盤健全性（`cpu_infra`）: 実プレイのゲームプレイ退行は見ない——読み取り専用の計器で、
**測り方の算術が正しいか**（帯の中だけで傾きを取れているか・符号判定が効くか）を固める。
ここが壊れると `docs/measurement.md` §12 の判定（V が手札の価値を逆に学んでいるか）が
静かに間違う。
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

import hand_value_slope as H  # noqa: E402


def test_turn_band_edges():
    assert H.turn_band(1) == "T<=4"
    assert H.turn_band(4) == "T<=4"
    assert H.turn_band(5) == "T5-8"
    assert H.turn_band(8) == "T5-8"
    assert H.turn_band(9) == "T9+"


def test_band_key_excludes_hand():
    """帯は手札枚数を含まない（含めたら説明変数が消える）。"""
    sc = np.zeros(16, np.float32)
    sc[H.SC_MY_LIFE], sc[H.SC_OPP_LIFE] = 3, 5
    sc[H.SC_MY_FIELD], sc[H.SC_TURN] = 2, 6
    sc[H.SC_MY_HAND] = 4
    k1 = H.band_key(sc)
    sc2 = np.array(sc, copy=True)
    sc2[H.SC_MY_HAND] = 7
    assert k1 == H.band_key(sc2) == (3, 5, "T5-8", 2)


def test_band_key_life_axis_swaps_the_explanatory_variable():
    """`axis="life"` は帯から**自ライフを外し手札を入れる**（`λ` の測定・§11.1）。"""
    sc = np.zeros(14, np.float32)
    sc[H.SC_MY_LIFE], sc[H.SC_OPP_LIFE], sc[H.SC_MY_HAND] = 3, 5, 6
    sc[H.SC_MY_FIELD], sc[H.SC_TURN] = 2, 7
    assert H.band_key(sc, "hand") == (3, 5, "T5-8", 2)
    assert H.band_key(sc, "life") == (6, 5, "T5-8", 2)
    sc2 = np.array(sc)
    sc2[H.SC_MY_LIFE] = 1                       # ライフを動かしても life 軸の帯は変わらない
    assert H.band_key(sc2, "life") == H.band_key(sc, "life")
    assert H.band_key(sc2, "hand") != H.band_key(sc, "hand")


def test_drop_one_hand_picks_the_cheapest_counter():
    tok = np.zeros((22, 22), np.float32)
    # 手札 3 枠（12,13,14）に埋める。counter_value が最小なのは 13。
    for slot, counter in ((12, 2.0), (13, 0.5), (14, 1.5)):
        tok[slot, 0] = 5000.0                      # power_now（枠が埋まっている印）
        tok[slot, H.S_COUNTER] = counter
    out, ok = H.drop_one_hand(tok)
    assert ok
    assert float(np.abs(out[13]).sum()) == 0.0     # 最小の枠が 0 になった
    assert float(out[12, H.S_COUNTER]) == 2.0      # 他は触っていない
    assert float(out[14, H.S_COUNTER]) == 1.5
    assert float(np.abs(tok[13]).sum()) > 0.0     # 入力は書き換えない


def test_drop_one_hand_empty_hand():
    tok = np.zeros((22, 22), np.float32)
    tok[0, 0] = 5000.0                            # リーダーだけ埋まっている
    out, ok = H.drop_one_hand(tok)
    assert not ok
    assert np.array_equal(out, tok)


def test_within_slope_removes_the_band_confound():
    """帯ごとには正の傾き・帯をまたぐと負に見える（Simpson）データで、within が正を返す。"""
    def block(band, hand, wins, n=10):
        return [{"band": band, "hand": float(hand), "z": 1.0 if k < wins else 0.0}
                for k in range(n)]

    # 帯 A（手札が少ない局面・勝率の水準は高い）: 手札 1 枚で 8/10・2 枚で 10/10 → 傾き +0.2
    rows = block("A", 1, 8) + block("A", 2, 10)
    # 帯 B（手札が多い局面・水準は低い）: 5 枚で 0/10・6 枚で 2/10 → 同じ傾き +0.2
    rows += block("B", 5, 0) + block("B", 6, 2)
    res = H.within_slope(rows, "z")
    assert res["within"] == pytest.approx(0.2), res      # 帯の中では正
    assert res["pooled"] < 0.0, res                      # 帯を混ぜると負に見える
    assert res["bands_used"] == 2
    assert res["rows_used"] == res["n"] == 40


def test_within_slope_ignores_bands_without_variation():
    """帯の中で手札枚数が動かない帯は分母に入らない（0 除算も起きない）。"""
    rows = [{"band": "flat", "hand": 3.0, "z": 1.0} for _ in range(5)]
    rows += [{"band": "v", "hand": 1.0, "z": 0.0}, {"band": "v", "hand": 2.0, "z": 1.0}]
    res = H.within_slope(rows, "z")
    assert res["bands_used"] == 1
    assert res["rows_used"] == 2
    assert res["within"] == pytest.approx(1.0)


def test_within_slope_se_is_zero_for_a_perfect_fit():
    rows = [{"band": "a", "hand": float(h), "z": 0.5 * h} for h in range(6)]
    res = H.within_slope(rows, "z")
    assert res["within"] == pytest.approx(0.5)
    assert res["within_se"] == pytest.approx(0.0, abs=1e-9)


def test_sign_agrees_flags_a_flipped_model():
    """実測が正で V が負なら「符号が違う」＝§12 の偏りが実在する、と出る。"""
    out = H._sign_agrees(0.05, -0.02, -0.01, -0.03)
    assert out["slope_v_obs"] is False
    assert out["dv_count"] is False
    assert out["all"] is False
    ok = H._sign_agrees(0.05, 0.02, 0.01, None)
    assert ok["slope_v_obs"] is True and ok["dv_slot"] is None and ok["all"] is True
    assert H._sign_agrees(None, 0.1) is None


def test_half_and_round_helpers():
    assert H._half(None) is None
    assert H._half(0.4) == pytest.approx(0.2)
    assert H._r(None) is None
    assert H._r(0.123456) == 0.12346
