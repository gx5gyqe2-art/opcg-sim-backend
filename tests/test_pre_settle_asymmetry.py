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

import numpy as np
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


# ---- T149a: 局を単位にした頑健性 ---------------------------------------------------------------

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


def test_bootstrap_stage_diff_needs_at_least_five_games():
    rows = [_rowg(0.9, 1, seed=s, stage="late") for s in range(3)]
    out = PA.bootstrap_stage_diff(rows)
    assert out["n_games"] == 3 and out["diff"] is None and out["ci95"] is None


def test_bootstrap_stage_diff_detects_a_real_late_minus_early_gap():
    """**T149a**: late の行だけ大きく過大評価する局を作り、`diff` が正で 0 を跨がないことを確かめる
    （合成データでの検算・実データの当否とは別）。"""
    rows = []
    for s in range(20):
        rows.append(_rowg(0.55, 0.5, seed=s, stage="early"))    # 序盤はほぼ較正が合う
        rows.append(_rowg(0.9, 0.5, seed=s, stage="late"))      # 終盤は大きく過大評価
    out = PA.bootstrap_stage_diff(rows, n_boot=500, seed=1)
    assert out["n_games"] == 20
    assert out["diff"] == pytest.approx(0.35, abs=1e-9)
    assert out["excludes_zero"] is True


def test_bootstrap_stage_diff_is_deterministic_given_the_same_seed():
    rows = [_rowg(0.5 + 0.01 * i, i % 2, seed=i, stage=("late" if i % 2 else "early")) for i in range(20)]
    a = PA.bootstrap_stage_diff(rows, n_boot=200, seed=7)
    b = PA.bootstrap_stage_diff(rows, n_boot=200, seed=7)
    assert a["ci95"] == b["ci95"]


def test_robustness_check_splits_favorite_and_underdog_and_reports_game_counts():
    rows = ([_rowg(0.9, 1, seed=s, stage="early") for s in range(6)]
           + [_rowg(0.1, 0, seed=s + 100, stage="late") for s in range(6)])
    out = PA.robustness_check(rows)
    assert set(out) == {"favorite", "underdog"}
    assert out["favorite"]["n_games"] == 6 and out["underdog"]["n_games"] == 6
    assert out["favorite"]["row_weighted_gap"] == pytest.approx(-0.1)


# ---- T149b: 経過 × 残り時間の 2 次元表 ----------------------------------------------------------

def _rows_p(pz_stage_s):
    """`[(p, z, stage, s), ...]` から行を作る（`seed` は連番で局を分ける）。"""
    return [dict(_rowg(p, z, seed=i, stage=st), s=s) for i, (p, z, st, s) in enumerate(pz_stage_s)]


def test_s_axis_table_reports_n_and_cells_below_the_row_floor():
    rows = _rows_p([(0.9, 1, "early", float(i)) for i in range(10)])
    out = PA.s_axis_table(rows, n_q=4)
    assert out["early"]["n"] == 10 and out["early"]["quantiles"] == []   # 10 < 4*5=20 で分位を出さない
    assert out["mid"]["n"] == 0 and out["late"]["n"] == 0


def test_s_axis_table_splits_into_quantiles_that_cover_every_row():
    rows = _rows_p([(0.9, 1, "early", float(i)) for i in range(40)])
    out = PA.s_axis_table(rows, n_q=4)
    cells = out["early"]["quantiles"]
    assert len(cells) == 4
    assert sum(c["n"] for c in cells) == 40                # 取りこぼしなし
    assert [c["q"] for c in cells] == [1, 2, 3, 4]


def test_s_axis_table_detects_a_within_stage_trend_when_present():
    """**T149b の核心**: `s` が大きいほど過大評価が増える合成データを作り、`gap` がその分位で
    単調に増えることを確かめる（実データの当否とは別の器の検算）。"""
    rows = []
    for i in range(40):
        s = float(i)
        p = 0.5 + 0.01 * s          # s が大きいほど予測が高く（過大評価が増える）
        rows.append(dict(_rowg(p, 0.5, seed=i, stage="early"), s=s))
    out = PA.s_axis_table(rows, n_q=4)
    gaps = [c["gap"] for c in out["early"]["quantiles"]]
    assert gaps == sorted(gaps) and gaps[0] < gaps[-1]


def test_s_axis_table_is_flat_when_only_stage_drives_the_gap():
    """**T149b の対照**: `s` を段の中でランダムに散らし、過大評価は段だけで決まる合成データを作ると、
    段の中の分位を追っても `gap` はほぼ平ら（経過そのものが主因という読みの検算）。"""
    import random
    random.seed(3)
    rows = []
    for i in range(40):
        s = random.random() * 10           # s は段の中でランダム（gap と無関係）
        rows.append(dict(_rowg(0.8, 0.5, seed=i, stage="late"), s=s))   # gap は段だけで固定
    out = PA.s_axis_table(rows, n_q=4)
    gaps = [c["gap"] for c in out["late"]["quantiles"]]
    assert max(gaps) - min(gaps) < 0.05      # ほぼ平ら（固定した gap=0.3 の周りに収まる）


# ---- T149c: 生存の選択（序盤から続く優勢 vs 終盤で新しく優勢）--------------------------------------

def _rowgw(p, z, seed, who, stage="early"):
    return {"p": p, "z": z, "d": 0.0, "won": bool(z), "j_me": 0, "stage": stage, "seed": seed, "who": who}


def test_early_leader_of_picks_the_first_early_row_per_game_and_skips_ties():
    rows = [_rowgw(0.9, 1, seed=1, who=0, stage="early"),
            _rowgw(0.2, 1, seed=1, who=0, stage="mid"),      # 同じ局・後の行は無視（最初だけ使う）
            _rowgw(0.5, 1, seed=2, who=0, stage="early"),    # ちょうど 0.5 は判定不能
            _rowgw(0.3, 0, seed=3, who=1, stage="mid")]       # 序盤の行が無い局
    out = PA.early_leader_of(rows)
    assert out == {1: 0}


def test_early_leader_of_flips_the_seat_when_this_row_s_own_seat_is_the_underdog():
    """`p<0.5` の行は自分の席が劣勢＝優勢はもう片方の席（`1 - who`）。"""
    rows = [_rowgw(0.2, 0, seed=5, who=0, stage="early")]
    assert PA.early_leader_of(rows) == {5: 1}


def test_survivorship_split_separates_persistent_from_flipped():
    rows = [
        _rowgw(0.9, 1, seed=1, who=0, stage="early"),        # 局 1: 序盤は席 0 が優勢
        _rowgw(0.8, 1, seed=1, who=0, stage="late"),         # 終盤も席 0 が優勢＝persistent
        _rowgw(0.2, 0, seed=2, who=0, stage="early"),        # 局 2: 序盤は席 1 が優勢（席 0 視点で劣勢）
        _rowgw(0.7, 0, seed=2, who=0, stage="late"),         # 終盤は席 0 が優勢＝flipped
        _rowgw(0.6, 1, seed=3, who=0, stage="late"),         # 局 3: 序盤の行が無い＝no_early_data
    ]
    out = PA.survivorship_split(rows)
    assert out["persistent"]["n"] == 1 and out["flipped"]["n"] == 1
    assert out["no_early_data"]["n"] == 1
    assert out["persistent"]["gap"] == pytest.approx(0.8 - 1)
    assert out["flipped"]["gap"] == pytest.approx(0.7 - 0)


def test_survivorship_split_counts_games_not_just_rows():
    rows = [_rowgw(0.9, 1, seed=1, who=0, stage="early"),
            _rowgw(0.8, 1, seed=1, who=0, stage="late"),
            _rowgw(0.7, 1, seed=1, who=0, stage="late")]     # 同じ局から late の行が 2 本
    out = PA.survivorship_split(rows)
    assert out["persistent"]["n"] == 2 and out["persistent"]["n_games"] == 1


# ---- T149f: 優勢の継続ターン数（連続変数・persistent/flipped の 2 値化の置き換え）------------------

def _rowgwj(p, z, seed, who, stage, j_me):
    return {"p": p, "z": z, "d": 0.0, "won": bool(z), "j_me": j_me, "stage": stage, "seed": seed, "who": who}


def test_leader_streak_of_counts_consecutive_favorable_turns():
    rows = [_rowgwj(0.9, 1, seed=1, who=0, stage="early", j_me=0),
            _rowgwj(0.9, 1, seed=1, who=0, stage="early", j_me=1),
            _rowgwj(0.4, 1, seed=1, who=0, stage="mid", j_me=2),      # 優勢が切れる（p<=0.5）
            _rowgwj(0.9, 1, seed=1, who=0, stage="late", j_me=3)]
    out = PA.leader_streak_of(rows)
    assert out[(1, 0, 0)] == 1
    assert out[(1, 0, 1)] == 2
    assert out[(1, 0, 2)] == 0
    assert out[(1, 0, 3)] == 1


def test_leader_streak_of_is_independent_per_seed_and_who():
    rows = [_rowgwj(0.9, 1, seed=1, who=0, stage="early", j_me=0),
            _rowgwj(0.9, 1, seed=1, who=1, stage="early", j_me=0),
            _rowgwj(0.9, 1, seed=2, who=0, stage="early", j_me=0)]
    out = PA.leader_streak_of(rows)
    assert out[(1, 0, 0)] == 1 and out[(1, 1, 0)] == 1 and out[(2, 0, 0)] == 1


def test_streak_axis_table_reports_n_and_no_quantiles_below_the_row_floor():
    rows = [_rowgwj(0.9, 1, seed=i, who=0, stage="late", j_me=0) for i in range(10)]
    out = PA.streak_axis_table(rows, n_q=4)
    assert out["n"] == 10 and out["quantiles"] == []       # 10 < 4*5=20 で分位を出さない


def test_streak_axis_table_splits_into_quantiles_that_cover_every_late_favorite_row():
    rows = []
    for i in range(40):
        k = i % 10 + 1                                     # 継続ターン数 1..10 を繰り返す
        for j in range(k - 1):
            rows.append(_rowgwj(0.9, 1, seed=i, who=0, stage="early", j_me=j))
        rows.append(_rowgwj(0.9, 1, seed=i, who=0, stage="late", j_me=k - 1))
    out = PA.streak_axis_table(rows, n_q=4)
    cells = out["quantiles"]
    assert len(cells) == 4
    assert sum(c["n"] for c in cells) == 40                 # 取りこぼしなし
    assert [c["q"] for c in cells] == [1, 2, 3, 4]


def test_streak_axis_table_detects_a_trend_when_gap_depends_on_streak():
    """**T149f の核心**: 継続ターン数が長いほど過大評価が増える合成データを作り、`gap` がその分位で
    単調に増えることを確かめる（実データの当否とは別の器の検算・`test_s_axis_table_detects_...` と同じ形）。"""
    rows = []
    for i in range(40):
        k = i % 10 + 1
        for j in range(k - 1):
            rows.append(_rowgwj(0.9, 1, seed=i, who=0, stage="early", j_me=j))
        p = 0.5 + 0.03 * k                                  # 継続が長いほど過大評価が増える
        rows.append(_rowgwj(p, 0.0, seed=i, who=0, stage="late", j_me=k - 1))
    out = PA.streak_axis_table(rows, n_q=4)
    gaps = [c["gap"] for c in out["quantiles"]]
    assert gaps == sorted(gaps) and gaps[0] < gaps[-1]
