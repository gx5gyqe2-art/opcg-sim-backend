"""生成の後半の低確率の温度（§7.73 WP `gen-explore`・`record_gen --temp-late`／`--explore-z`）の契約。

基盤健全性（`cpu_infra`）: ゲームプレイの正しさではなく**生成器の内部機構**を見る。

守る性質:
  1. 既定（`temp_late` を渡さない）と `temp_late=0` は**1 ビットも同じ**（乱数を引かない）——
     dump の全列と局メタが一致し、列 `explore` は積まれない。
  2. `temp_late=1` なら後半（`temp_turns` より後）のメイン窓でも訪問分布から引く＝
     `explore=2`（最善でない手）が立ち、窓・コミットの行には立たない。
  3. `--explore-z root`（(ii)）: 最善でない手を打った行**まで**の価値の教師を、その手番の根の評価に替える
     （席が違えば符号を反転・最後の最善でない手より後は勝敗のまま）。元の勝敗は `z_game` に残る。
"""
import numpy as np
import pytest

import conftest  # noqa: F401
import _bootstrap  # noqa: F401

from opcg_sim.loop import record_gen as G

pytestmark = pytest.mark.cpu_infra

SEED = 930028
SIMS = 16


def _play(seed, explore=None, temp_turns=4):
    G._init_worker(SIMS, None, 0.25, temp_turns, "synth_roles", True, 0.0, 0.0, None, 0.0,
                   None, None, explore)
    return G.play_one(seed)


def test_zero_temp_late_is_bit_identical_to_the_default():
    ref = _play(SEED)
    got = _play(SEED, {"z": "game", "temp_late": 0.0})
    assert ref is not None and got is not None
    got.pop("explore")                                  # 0 でも渡せば印の列は積む（中身は序盤だけ）
    got["_meta"].pop("explore")
    assert set(got) == set(ref)
    assert "explore" not in ref and "z_game" not in ref, "既定は列を足さない"
    for key in sorted(set(got) - {"_meta"}):
        a, b = got[key], ref[key]
        assert a.dtype == b.dtype, key
        assert np.array_equal(a, b), f"temp_late=0 で列 {key} が変わっている"
    assert got["_meta"]["winner"] == ref["_meta"]["winner"]


def test_full_temp_late_marks_late_non_best_moves_on_main_rows_only():
    r = _play(SEED, {"z": "game", "temp_late": 1.0})
    assert r is not None
    ex = r["explore"]
    assert ex.dtype == np.int8 and len(ex) == len(r["z"])
    assert (ex == 2).sum() > 0, "後半でも最善でない手が出る"
    assert (r["kind"][ex > 0] == 0).all(), "印はメイン窓の行だけ"
    assert (r["turn"][ex == 2] > 4).all() and (r["turn"][ex == 1] <= 4).all()
    assert "z_game" not in r, "game の扱いでは教師を替えない"


def test_root_teacher_is_written_and_keeps_the_game_result():
    r = _play(SEED, {"z": "root", "temp_late": 1.0})
    assert r is not None
    assert np.array_equal(np.abs(r["z_game"]), np.ones(len(r["z"]))), "z_game は勝敗のまま"
    want = G.explore_z(r["z_game"], r["who"], r["step"], r["pol_v0"].astype(np.float32), r["explore"])
    assert np.array_equal(r["z"], want)


def test_explore_z_relabels_rows_up_to_each_late_non_best_move():
    z = np.array([1, -1, 1, -1, 1, -1], np.float32)       # p1 が勝った局・席は交互
    who = np.array([0, 1, 0, 1, 0, 1], np.int8)
    step = np.array([0, 1, 2, 3, 4, 5])
    v0 = np.array([0.0, 0.0, 0.5, 0.0, -0.25, 0.0], np.float32)
    explore = np.array([0, 1, 2, 0, 2, 0], np.int8)        # 1（序盤）は替えない
    got = G.explore_z(z, who, step, v0, explore)
    # 行 0〜2 は行 2（p1・0.5）の根の評価・行 3〜4 は行 4（p1・-0.25）・行 5 は勝敗のまま
    assert got.tolist() == [0.5, -0.5, 0.5, 0.25, -0.25, -1.0]
    # 並びに依らない
    perm = np.array([5, 3, 1, 0, 2, 4])
    got2 = G.explore_z(z[perm], who[perm], step[perm], v0[perm], explore[perm])
    assert np.array_equal(got2, got[perm])
    # 後半の印が無ければ何も替えない
    assert np.array_equal(G.explore_z(z, who, step, v0, np.zeros(6, np.int8)), z)
