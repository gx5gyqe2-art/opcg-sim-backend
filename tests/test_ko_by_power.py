"""`ko_by_power.py`（`ν` の生存割引・T22）の数え方を固める。

**枠の番号では追えない**（枠は入れ替わる）ので**多重集合の差**で数える。
ここがずれると `ko_p` のパワー依存の判定が静かに壊れ、**身代わり項を入れてよいか**の
結論が変わる（T21 の対）。
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

import ko_by_power as K  # noqa: E402
import theory_order as T  # noqa: E402


def _tok(powers=(), blockers=()):
    tok = np.zeros((22, 24), np.float32)
    for j, pw in enumerate(powers):
        tok[2 + j, T.S_POWER] = pw / 1e4
        tok[2 + j, T.S_IS_CHAR] = 1.0
        if j in blockers:
            tok[2 + j, T.S_IS_BLOCKER] = 1.0
    return tok


def test_the_field_is_read_as_a_multiset_of_powers():
    """同じパワーの体が 2 つ在っても 2 つと数える（集合に潰さない）。"""
    got = K.field_powers(_tok(powers=(5000, 5000, 3000)))
    assert sorted(p for p, _b in got) == [3000, 5000, 5000]
    assert len(got) == 3


def test_the_power_is_rounded_to_the_grid():
    """`1e4` を掛けて戻した量は 10 の単位に丸める（`measurement.md` 罠 14-5）。

    f16 の丸めで 2000 が 2000.0002 になると、**帯が 1 段ずれる**。
    """
    tok = _tok(powers=(4000,))
    tok[2, T.S_POWER] = np.float32(0.4).astype(np.float32) + np.float32(1e-7)
    assert K.field_powers(tok)[0][0] == 4000.0


def test_a_slot_that_moves_is_not_counted_as_lost():
    """**枠が入れ替わっても消えたことにしない**——多重集合で比べる理由そのもの。"""
    before = [(5000.0, False), (3000.0, False)]
    after = [(3000.0, False), (5000.0, False)]     # 枠だけ入れ替わった
    assert K.lost(before, after) == []


def test_only_the_missing_copies_are_counted():
    before = [(5000.0, False), (5000.0, False), (3000.0, False)]
    after = [(5000.0, False)]
    got = K.lost(before, after)
    assert sorted(p for p, _b in got) == [3000.0, 5000.0]     # 2 体だけ消えた


def test_a_new_body_does_not_hide_a_loss():
    """後から出た体が、消えた体を埋めてしまわない（**パワーが違えば別物**）。"""
    before = [(6000.0, False)]
    after = [(2000.0, False), (4000.0, False)]
    assert K.lost(before, after) == [(6000.0, False)]


def test_a_blocker_and_a_vanilla_of_the_same_power_are_different_bodies():
    """ブロッカーかどうかも鍵に入る——`ν` ではブロック項と二重に効きうるので分けて出す。"""
    assert K.lost([(5000.0, True)], [(5000.0, False)]) == [(5000.0, True)]


def test_the_power_bands_are_the_cost_curve_grid():
    assert K.band_of(0) == "p0" and K.band_of(1999) == "p0"
    assert K.band_of(2000) == "p2" and K.band_of(5999) == "p4"
    assert K.band_of(9000) == "p8" and K.band_of(99000) == "p8"


def test_monotone_down_is_what_the_verdict_reads():
    """**事前登録した読み方**——帯で単調に下がれば `ko_p` をパワーの関数にする根拠。"""
    assert K.monotone_down([0.5, 0.4, 0.3])
    assert K.monotone_down([0.4, 0.4, 0.3])        # 等号は許す
    assert not K.monotone_down([0.3, 0.5, 0.2])
    assert not K.monotone_down([0.3])


def _mk(pairs, blocker=False):
    return [{"seed": 1 + i % 5, "turn": 3, "power": p, "blocker": blocker, "gone": g}
            for i, (p, g) in enumerate(pairs)]


def test_a_flat_result_says_the_constant_is_fine():
    """**平らなら定数 0.289 のままでよい**＝身代わり項を単独で入れてよいことになる。

    これは「対が要らなくなる」＝話が進む向きの結論なので、**判定を甘くしない**。
    """
    out = K.summarise(_mk(((1000, 1), (1000, 0), (3000, 1), (3000, 0),
                           (5000, 1), (5000, 0), (7000, 1), (7000, 0))), reps=0)
    assert out["verdict"] == "flat_constant_is_fine"
    assert out["spread"] == pytest.approx(0.0)
    assert out["ko_p_current"] == T.KO_P


def test_the_spread_is_max_minus_min_not_the_two_end_bands():
    """**端点差は「動いていない」の証拠にならない**（2026-09-14 に踏んだ）。

    実測は**山形**（4000〜6000 が峰）で、両端だけ見ると 0.043＝「平ら」と出るのに、
    **峰と谷の差は 0.13 で CI が重ならない**。見る量を間違えると判定が反転する。
    """
    assert K.spread_of([0.24, 0.28, 0.33, 0.24, 0.20]) == pytest.approx(0.13)
    assert K.spread_of([0.30, 0.30]) == pytest.approx(0.0)
    assert K.spread_of([0.30]) is None


def test_a_hump_is_reported_as_non_monotone_not_as_power_dependent():
    """**事前登録した 3 分岐**——単調に下がる／平ら／単調でない。

    山形を `power_dependent` と呼ぶと「高パワーほど死ににくい」という**誤った形**を
    式に入れることになる。**単調でないことを判定が名指しする。**
    """
    hump = _mk([(1000, 0)] * 8 + [(1000, 1)] * 2        # p0: 0.20
               + [(5000, 0)] * 5 + [(5000, 1)] * 5      # p4: 0.50
               + [(9000, 0)] * 9 + [(9000, 1)] * 1)     # p8: 0.10
    out = K.summarise(hump, reps=0)
    assert not out["monotone_down"]
    assert out["verdict"] == "non_monotone_something_else_binds"


def test_a_monotone_fall_is_what_licenses_a_power_dependent_ko():
    down = _mk([(1000, 1)] * 5 + [(1000, 0)] * 5        # p0: 0.50
               + [(5000, 1)] * 3 + [(5000, 0)] * 7      # p4: 0.30
               + [(9000, 1)] * 1 + [(9000, 0)] * 9)     # p8: 0.10
    out = K.summarise(down, reps=0)
    assert out["monotone_down"] and out["verdict"] == "power_dependent"


def test_the_vanilla_curve_is_reported_separately():
    """**ブロッカーは殴られて死ぬ**ので、パワーの効きと混ざる（実測 0.36 対 0.26）。"""
    recs = _mk(((1000, 0), (5000, 0))) + _mk(((1000, 1), (5000, 1)), blocker=True)
    out = K.summarise(recs, reps=0)
    assert out["by_blocker"]["blocker"]["gone_p"] == pytest.approx(1.0)
    assert out["by_blocker"]["vanilla"]["gone_p"] == pytest.approx(0.0)
    assert all(v["gone_p"] == pytest.approx(0.0)
               for v in out["by_power_vanilla"].values() if v)
