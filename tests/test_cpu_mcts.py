"""MCTS（Phase 1 MVP・docs/SPEC.md §2.5.7）の健全性スモーク。

本 MVP は現行 α-β `hard` を温存した**独立経路**（`cpu_mcts.decide_mcts`）であり、本番 `decide` は
変更しない。よって品質ゲートはまず「壊さない・合法手を返す・入力盤面を破壊しない」を固定する。
**強さ（対 hard Elo）は自己対戦で別途計測**（MVP 時点では hard 未満＝改善ロードマップは SPEC §2.5.7）。

実行: OPCG_LOG_SILENT=1 python -m pytest tests/test_cpu_mcts.py -q -s -p no:cacheprovider
"""
import copy
import random

import conftest  # noqa: F401
import pytest

from opcg_sim.src.core import cpu_mcts, cpu_ai, journal
import cpu_arena
import test_cpu_puzzles as P


@pytest.fixture(scope="module")
def db():
    return cpu_arena._load_db()


def _states(db, n=4):
    out = []
    for s in range(n):
        gm = P._new_gm(db, seed=s)
        if P._fast_forward_to_p1_main(gm):
            out.append(gm)
    return out


def test_mcts_returns_legal_move(db):
    """decide_mcts は常に**現局面の合法手**を返す（不正手を打たない）。"""
    states = _states(db)
    assert states
    for gm in states:
        legal = gm.get_legal_actions(gm.p1)
        legal_sigs = {cpu_ai._move_sig(m) for m in legal}
        mv = cpu_mcts.decide_mcts(gm, gm.p1, "hard", random.Random(0), iterations=60)
        assert mv is not None
        # 選択分岐（RESOLVE_SELECTION）も合法経路なので、合法手 sig かルート候補に含まれること。
        node_sigs = {cpu_ai._move_sig(m) for m in cpu_mcts._node_moves(gm, "p1")}
        assert cpu_ai._move_sig(mv) in (legal_sigs | node_sigs)


def test_mcts_leaves_manager_unchanged(db):
    """decide_mcts は入力 manager を変更しない（1 反復ごとに clone＝実盤面を触らない）。"""
    states = _states(db)
    for gm in states:
        before = copy.deepcopy(gm)
        cpu_mcts.decide_mcts(gm, gm.p1, "hard", random.Random(0), iterations=60)
        diff = journal.deep_diff(before, gm)
        assert diff is None, f"decide_mcts が manager を変更した: {diff}"


def test_mcts_deterministic_with_seeded_rng(db):
    """同一 seed・同一反復数なら同じ手を返す（決定論＝再現性）。"""
    gm = _states(db, n=1)[0]
    a = cpu_mcts.decide_mcts(copy.deepcopy(gm), gm.p1, "hard", random.Random(7), iterations=80)
    b = cpu_mcts.decide_mcts(copy.deepcopy(gm), gm.p1, "hard", random.Random(7), iterations=80)
    assert cpu_ai._move_sig(a) == cpu_ai._move_sig(b)


def test_mcts_single_move_shortcut(db):
    """合法手が 1 つなら探索せず即返す。"""
    gm = _states(db, n=1)[0]
    only = gm.get_legal_actions(gm.p1)[:1]
    mv = cpu_mcts.decide_mcts(gm, gm.p1, "hard", random.Random(0), iterations=80, moves=only)
    assert cpu_ai._move_sig(mv) == cpu_ai._move_sig(only[0])
