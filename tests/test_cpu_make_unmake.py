"""② make/unmake を実探索へ統合した際の**等価性**ゲート（docs/SPEC.md §2.5.2）。

make/unmake は探索の内部最適化（per-node clone を「適用→採点→巻き戻し」へ置換）であって、
**選ぶ手・評価値は clone 方式と完全同一**でなければならない。本テストは:

  1. `decide` の選択手が `_USE_MAKE_UNMAKE` の True/False で一致する（多 seed・難易度 normal/hard）。
  2. `_scored_search` の深掘りスコア（手→値）が両方式で一致する。
  3. `decide` が入力 manager を一切変更しない（make/unmake の巻き戻しが完全＝探索後に盤面が無傷）。

実行: OPCG_LOG_SILENT=1 python -m pytest tests/test_cpu_make_unmake.py -q -s -p no:cacheprovider
"""
import copy
import random

import conftest  # noqa: F401
import pytest

from opcg_sim.src.core import cpu_ai, journal
import cpu_arena
import test_cpu_puzzles as P


@pytest.fixture(scope="module")
def db():
    return cpu_arena._load_db()


def _states(db, n=6):
    out = []
    for s in range(n):
        gm = P._new_gm(db, seed=s)
        if P._fast_forward_to_p1_main(gm):
            out.append(gm)
    return out


def _decide_sig(gm, difficulty):
    mv = cpu_ai.decide(gm, gm.p1, difficulty, random.Random(0))
    return cpu_ai._move_sig(mv) if mv is not None else None


@pytest.mark.parametrize("difficulty", ["normal", "hard"])
def test_make_unmake_picks_same_move(db, difficulty):
    """make/unmake ON/OFF で decide の選択手が完全一致する（内部最適化＝方策不変）。"""
    states = _states(db)
    assert states, "p1 メイン局面が作れなかった"
    orig = cpu_ai._USE_MAKE_UNMAKE
    try:
        diffs = []
        for i, gm in enumerate(states):
            cpu_ai._USE_MAKE_UNMAKE = False
            base = _decide_sig(copy.deepcopy(gm), difficulty)
            cpu_ai._USE_MAKE_UNMAKE = True
            mu = _decide_sig(copy.deepcopy(gm), difficulty)
            if base != mu:
                diffs.append((i, base, mu))
        assert not diffs, f"選択手が不一致: {diffs}"
    finally:
        cpu_ai._USE_MAKE_UNMAKE = orig


def test_make_unmake_same_deep_scores(db):
    """`_scored_search` の深掘りスコア（手sig→値）が両方式で完全一致する。"""
    states = _states(db, n=4)
    orig = cpu_ai._USE_MAKE_UNMAKE
    try:
        for gm in states:
            moves = gm.get_legal_actions(gm.p1)
            if len(moves) <= 1:
                continue
            cpu_ai._USE_MAKE_UNMAKE = False
            g1 = copy.deepcopy(gm)
            base = cpu_ai._scored_search(g1, "p1", g1.get_legal_actions(g1.p1),
                                         see_opp_hand=True, opp_public_only=False)
            cpu_ai._USE_MAKE_UNMAKE = True
            g2 = copy.deepcopy(gm)
            mu = cpu_ai._scored_search(g2, "p1", g2.get_legal_actions(g2.p1),
                                       see_opp_hand=True, opp_public_only=False)
            b = {cpu_ai._move_sig(m): round(v, 6) for v, m in base}
            u = {cpu_ai._move_sig(m): round(v, 6) for v, m in mu}
            assert b == u, f"深掘りスコア不一致:\n base={b}\n mu  ={u}"
    finally:
        cpu_ai._USE_MAKE_UNMAKE = orig


def test_decide_leaves_manager_unchanged(db):
    """make/unmake 統合後も decide は入力 manager を変更しない（巻き戻しが完全＝探索後に盤面無傷）。"""
    states = _states(db, n=5)
    orig = cpu_ai._USE_MAKE_UNMAKE
    try:
        cpu_ai._USE_MAKE_UNMAKE = True
        for gm in states:
            before = copy.deepcopy(gm)
            cpu_ai.decide(gm, gm.p1, "hard", random.Random(0))
            diff = journal.deep_diff(before, gm)
            assert diff is None, f"decide が manager を変更した: {diff}"
    finally:
        cpu_ai._USE_MAKE_UNMAKE = orig
