"""`ν` の 3 定数を記録から数える算術（`tests/scripts/nu_calib.py`）。

基盤健全性（`cpu_infra`）。要は 3 つ:

1. **ブロックできるのはブロッカーだけ**——トークンの `is_blocker_active`（列 6）で見る。
   定数を全キャラに掛けていたのが**種類の誤り**だった（2026-09-13）。
2. **ブロック率は 2 通りに数えられ、`ν` の式が欲しいのは「ブロッカー 1 体・相手ターン 1 回あたり」**
   （窓あたりの率は「ブロッカーが居ない相手ターン」で薄まるので別の量）。
3. **`ko_p` は「居た + 出した」を分母にする**（そのターンに出したキャラも失いうる）。
"""
import os
import sys
from collections import Counter

import numpy as np
import pytest

pytestmark = pytest.mark.cpu_infra

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bootstrap  # noqa: F401,E402

_SCRIPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import nu_calib as N  # noqa: E402
import theory_order as T  # noqa: E402


def _tok(slots):
    """`slots` = {枠: (power, is_blocker_active)}。"""
    t = np.zeros((22, 22), np.float32)
    for s, (pw, blk) in slots.items():
        t[s, N.S_POWER] = pw / 1e4
        t[s, N.S_IS_CHAR] = 1.0
        t[s, N.S_BLOCKER_ACTIVE] = 1.0 if blk else 0.0
    return t


def test_own_field_counts_characters_and_active_blockers():
    tok = _tok({2: (5000.0, False), 3: (7000.0, True), 4: (0.0, False)})
    n, blk, powers = N.own_field(tok)
    assert n == 3 and blk == 1                      # 枠 4 はパワー 0 でも is_char で数える
    assert sorted(powers) == [0.0, 5000.0, 7000.0]
    empty, eb, _ = N.own_field(np.zeros((22, 22), np.float32))
    assert empty == 0 and eb == 0


def test_the_two_block_rates_are_different_quantities():
    """窓あたりの率は「ブロッカーが居ない相手ターン」で薄まる＝`ν` が欲しいのは per-blocker。"""
    guard = Counter({"PASS": 90, "SELECT_COUNTER": 0, "SELECT_BLOCKER": 10})
    field = {"chars": 100, "blockers": 10, "rows": 50}
    # 相手ターンに持っていたブロッカーの延べ数が 10 なら、ブロッカーは毎回ブロックしている
    cal = N.calibrate([], guard, field, opp_turn_blockers=10)
    assert cal["block_rate_per_window"] == pytest.approx(0.10)      # 窓あたりは 10%
    assert cal["block_per_blocker_turn"] == pytest.approx(1.00)     # ブロッカーは 100%
    assert cal["blocker_share"] == pytest.approx(0.10)


def test_ko_rate_counts_the_characters_played_that_turn_too():
    """そのターンに出したキャラも失いうる＝分母は「居た + 出した」。"""
    base = {"r_real": 3.0, "opp_life": 3.0, "chars": 1, "blockers": 0,
            "opp_leader_power": 5000.0}
    recs = [dict(base, seed=1, had=4, lost=1), dict(base, seed=2, had=2, lost=2)]
    cal = N.calibrate(recs, Counter(), {"chars": 0, "blockers": 0, "rows": 0}, 0)
    # 局ごとの平均（0.25 と 1.0）の平均 = 0.625
    assert cal["ko_p"]["mean"] == pytest.approx(0.625)
    assert cal["ko_p"]["games"] == 2


def test_the_f32_dust_is_rounded_away():
    """0.7×1e4 は 6999.999… になる——**この計画で 4 回踏んだ罠**なので枠の読みでも吸う。"""
    tok = _tok({2: (7000.0, False)})
    assert N.own_field(tok)[2] == [pytest.approx(7000.0, abs=1e-9)]
    assert N.own_field(tok)[2][0] == 7000.0          # 丸めた後は厳密に一致する


def test_empty_input_does_not_crash():
    cal = N.calibrate([], Counter(), {"chars": 0, "blockers": 0, "rows": 0}, 0)
    assert cal["blocker_share"] is None and cal["block_rate_per_window"] is None
    assert cal["ko_p"]["n"] == 0
    assert N.nu_effect([], cal) is None


def test_nu_is_zero_block_for_a_non_blocker():
    """**ブロックできるのはブロッカーだけ**＝非ブロッカーの `ν` にブロック項は入らない。"""
    blk = T.nu_of(5000.0, 5000.0, 4.0, is_blocker=True)
    plain = T.nu_of(5000.0, 5000.0, 4.0, is_blocker=False)
    assert blk > plain
    # 差はちょうどブロック項ぶん（KO 率で割り引かれた分）
    expect = T.BLOCK_P_BLOCKER * T.THETA * T.MU * (1.0 - T.KO_P)
    assert (blk - plain) == pytest.approx(expect, rel=1e-6)
    # 素性が判らないときは母集団の平均で置く（両者の間に入る）
    unknown = T.nu_of(5000.0, 5000.0, 4.0)
    assert plain < unknown < blk


def test_play_value_passes_the_blocker_flag_through():
    """登場の値付けは**そのカードがブロッカーかどうか**で変わる。"""
    a = T.play_value(5000.0, 3, 5000.0, 4.0, is_blocker=True)
    b = T.play_value(5000.0, 3, 5000.0, 4.0, is_blocker=False)
    assert a > b
    cards = type("C", (), {"info": staticmethod(lambda cid: {
        "power": 5000, "cost": 3, "leader": False, "event": False,
        "blocker": cid == "C_BLK"}.get("x", None) or {
        "power": 5000, "cost": 3, "leader": False, "event": False,
        "blocker": cid == "C_BLK"})})()
    ctx = {"theta": T.THETA, "mu": T.MU, "opp_leader_power": 5000.0,
           "my_leader_power": 5000.0, "r_turns": 4.0, "don_k": 1}
    sb = T.score_candidate(["PLAY", "u", [], [], None], "C_BLK", None, ctx, cards)
    sp = T.score_candidate(["PLAY", "u", [], [], None], "C_PLAIN", None, ctx, cards)
    assert sb > sp                                   # ブロッカーの方が高く値付けされる


def test_the_measured_constants_replaced_the_guesses():
    """旧値（0.3／0.25）が残っていないこと＝較正が効いていることの回帰。"""
    assert T.BLOCK_P_BLOCKER == pytest.approx(0.773)
    assert T.KO_P == pytest.approx(0.289)
    # 明示すれば旧値でも引ける（過去の数字を再現するため）
    old = T.nu_of(5000.0, 5000.0, 4.0, block_p=0.3, ko_p=0.25)
    new = T.nu_of(5000.0, 5000.0, 4.0, is_blocker=False)
    assert old != pytest.approx(new)
