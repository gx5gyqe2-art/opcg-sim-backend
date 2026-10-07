"""`pre_settle_asymmetry.py`（T146b/c・T149a）の**符号つきの較正のずれ**の算術（行の重み・局の重み）。
段 7（2026-10-07）: 表の形・CLI・デッキの揮発性の試験は消し、ずれの定義の検算だけを残した。
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

import pre_settle_asymmetry as PA  # noqa: E402


def test_gap_of_is_the_row_weighted_signed_mean_of_the_bin_gaps():
    score = {"calibration": [{"n": 3, "gap": 0.2}, {"n": 1, "gap": -0.4}]}
    assert PA.gap_of(score) == pytest.approx((3 * 0.2 + 1 * -0.4) / 4)
    assert PA.gap_of(None) is None
    assert PA.gap_of({"n": 5}) is None            # calibration が無い（母数不足）


def test_gap_of_is_signed_not_absolute():
    """**非対称を読む本器の核心**——`gap_of` は符号つき（劣勢側の負の gap と優勢側の正の gap が
    打ち消し合わずに区別できる）。絶対値を取ってしまうと本 T の非対称という結論自体が書けなくなる。"""
    over = {"calibration": [{"n": 10, "gap": 0.3}]}
    under = {"calibration": [{"n": 10, "gap": -0.1}]}
    assert PA.gap_of(over) > 0 and PA.gap_of(under) < 0


def _rowg(p, z, seed, stage="early"):
    return {"p": p, "z": z, "d": 0.0, "won": bool(z), "j_me": 0, "stage": stage, "seed": seed}


def test_mean_gap_matches_gap_of_up_to_the_bin_table_s_rounding():
    """**T149a の恒等式**: `mean_gap`（`mean(p)-mean(z)`）は `gap_of`（分位で束ねた n 重み平均）と
    一致する——分位で切っても符号つきの差の合計は保たれる（`calib_bins` が 4 桁に丸めるので
    完全なビット一致ではないが、丸め誤差 1e-3 の範囲で一致する）。"""
    import random
    random.seed(0)
    rows = [_rowg(random.random(), random.randint(0, 1), s) for s in range(30)]
    assert PA.mean_gap(rows) == pytest.approx(np.mean([r["p"] for r in rows]) - np.mean([r["z"] for r in rows]))
    sc = PA.score_group(rows)
    if sc and "calibration" in sc:
        assert PA.mean_gap(rows) == pytest.approx(PA.gap_of(sc), abs=1e-3)


def test_per_game_gap_averages_within_each_seed_first():
    rows = [_rowg(0.9, 1, seed=1), _rowg(0.9, 1, seed=1), _rowg(0.1, 0, seed=2)]
    pg = PA.per_game_gap(rows)
    assert pg == {1: pytest.approx(-0.1), 2: pytest.approx(0.1)}


def test_game_weighted_gap_gives_each_game_one_vote():
    """**核心**: 局 1（行 2 本）と局 2（行 1 本）は行単位では 2:1 だが、局単位では 1:1。"""
    rows = [_rowg(0.9, 1, seed=1)] * 2 + [_rowg(0.1, 0, seed=2)] * 1
    assert PA.mean_gap(rows) != pytest.approx(PA.game_weighted_gap(rows))
    assert PA.game_weighted_gap(rows) == pytest.approx((-0.1 + 0.1) / 2)
