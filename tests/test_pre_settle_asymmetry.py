"""`pre_settle_asymmetry.py`（T146b／T146c・決着前の較正の非対称——優勢側だけの過大評価）の算術を固める。

押さえるのは 4 つ:

1. **`stage_of`**: `j_me`（経過自席ターン）を序盤／中盤／終盤に割る境目（`J_EARLY_MAX`／`J_LATE_MIN`）。
2. **`favorite_split`**: `p > 0.5`（優勢側）と `p < 0.5`（劣勢側）に分ける。`p == 0.5` はどちらにも入れない。
3. **`gap_of`**: 較正表（`win_calib.score` の `calibration`）から行数重みの符号つき平均差を出す。
4. **`stage_split`**: 優勢／劣勢のそれぞれを 3 段に割る。

**基盤健全性ではない**（優勢／劣勢・段の取り違えは非対称の所在を誤って読ませる）。必須側。
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bootstrap  # noqa: F401,E402

_SCRIPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import pre_settle_asymmetry as PA  # noqa: E402


def test_stage_of_splits_on_the_declared_boundaries():
    assert PA.stage_of(0) == "early" and PA.stage_of(PA.J_EARLY_MAX) == "early"
    assert PA.stage_of(PA.J_EARLY_MAX + 1) == "mid" and PA.stage_of(PA.J_LATE_MIN) == "mid"
    assert PA.stage_of(PA.J_LATE_MIN + 1) == "late" and PA.stage_of(99) == "late"


def _row(p, z, j_me=0):
    return {"p": p, "z": z, "d": 0.0, "won": bool(z), "j_me": j_me, "stage": PA.stage_of(j_me)}


def test_favorite_split_uses_p_and_excludes_the_exact_midpoint():
    rows = [_row(0.9, 1), _row(0.6, 0), _row(0.5, 1), _row(0.2, 0), _row(0.1, 1)]
    out = PA.favorite_split(rows)
    assert out["favorite"]["n"] == 2 and out["underdog"]["n"] == 2   # p=0.5 の 1 行はどちらにも入らない


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


def test_stage_split_partitions_favorite_and_underdog_by_stage_without_overlap():
    rows = [_row(0.9, 1, j_me=0), _row(0.8, 1, j_me=2), _row(0.7, 0, j_me=5),
            _row(0.2, 0, j_me=0), _row(0.1, 1, j_me=4)]
    out = PA.stage_split(rows)
    total = sum(out[fav][st]["n"] for fav in ("favorite", "underdog") for st in PA.STAGES)
    assert total == len(rows)                     # 取りこぼし・重複が無い
    assert out["favorite"]["early"]["n"] == 1 and out["favorite"]["mid"]["n"] == 1
    assert out["favorite"]["late"]["n"] == 1
    assert out["underdog"]["early"]["n"] == 1 and out["underdog"]["late"]["n"] == 1


def test_score_group_falls_back_below_the_row_floor():
    assert PA.score_group([_row(0.6, 1)] * 5)["n"] == 5     # n<10 は較正表を出さない（母数不足）
    only_wins = [_row(0.9, 1)] * 20
    sc = PA.score_group(only_wins)
    assert sc["n"] == 20 and "calibration" not in sc         # 片方の結果しか無い行では対数損失を出さない


def test_collect_forces_pre_settle_on_and_restores_the_previous_mode(monkeypatch):
    """**T146b/c**: `collect` は必ず決着前だけを読み（`crossing_bridge.PRE_SETTLE_MODE` を一時的に
    `on` にする）、呼び出し後は元の値に戻す（他の器の既定を壊さない）。"""
    PA.CB.set_pre_settle_mode("off")
    seen = {}

    def fake_collect(dirs, limit_games, theta, mu, mode):
        seen["pre_settle_mode"] = PA.CB.PRE_SETTLE_MODE
        return [], [], {"games": 0}, None, None

    monkeypatch.setattr(PA.CB, "collect", fake_collect)
    monkeypatch.setattr(PA.CB, "sigma_rel_for", lambda dirs: 0.2)
    out = PA.collect(["x"])
    assert seen["pre_settle_mode"] == "on"
    assert PA.CB.PRE_SETTLE_MODE == "off"          # 呼び出し前の値に戻っている
    assert out["n"] == 0 and out["games"] == 0
