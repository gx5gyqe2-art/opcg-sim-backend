"""MCTS（ターン粒度マクロアクション木・docs/SPEC.md §2.5.7）の健全性スモーク。

MCTS は現行 α-β `hard` を温存した**独立経路**（`cpu_mcts.decide_mcts_macro` / `mcts_plan_turn`）であり、
本番 `decide` は変更しない。よって品質ゲートはまず「壊さない・合法手を返す・入力盤面を破壊しない・決定論」を
固定する。**強さ（対 hard Elo）は自己対戦で別途計測**（経緯と現状は SPEC §2.5.7）。

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


# --- マクロアクション（ターン粒度）MCTS の健全性 ----------------------------------

def test_macro_plan_turn_legal_and_unchanged(db):
    """mcts_plan_turn は合法な手列を返し（先頭手は現局面で合法）、入力 manager を変更しない。"""
    states = _states(db)
    for gm in states:
        legal = {cpu_ai._move_sig(m) for m in gm.get_legal_actions(gm.p1)}
        before = copy.deepcopy(gm)
        plan = cpu_mcts.mcts_plan_turn(gm, gm.p1, "hard", random.Random(0), iterations=60, horizon=2)
        assert journal.deep_diff(before, gm) is None, "mcts_plan_turn が manager を変更した"
        if plan:
            assert cpu_ai._move_sig(plan[0]) in legal


def test_macro_decide_legal_and_replay(db):
    """decide_mcts_macro は合法手を返し、計画を queue にキャッシュして逐次 replay する。"""
    gm = _states(db, n=1)[0]
    legal = {cpu_ai._move_sig(m) for m in gm.get_legal_actions(gm.p1)}
    cache = {}
    mv = cpu_mcts.decide_mcts_macro(gm, gm.p1, "hard", random.Random(0),
                                    cache=cache, iterations=60, horizon=2)
    assert mv is not None and cpu_ai._move_sig(mv) in legal
    assert "queue" in cache  # 残り計画手をキャッシュ（replay 用）


def test_macro_deterministic_with_seeded_rng(db):
    """同一 seed・同一反復数ならマクロ計画も同じ手列を返す（再現性）。"""
    gm = _states(db, n=1)[0]
    a = cpu_mcts.mcts_plan_turn(copy.deepcopy(gm), gm.p1, "hard", random.Random(3), iterations=80, horizon=2)
    b = cpu_mcts.mcts_plan_turn(copy.deepcopy(gm), gm.p1, "hard", random.Random(3), iterations=80, horizon=2)
    assert [cpu_ai._move_sig(m) for m in a] == [cpu_ai._move_sig(m) for m in b]


# --- Phase 2: 決定化（公平モード） ------------------------------------------------

def test_determinize_preserves_self_and_counts(db):
    """_determinize_opponent: 自分の手札は不変・相手の手札枚数は保存・入力 manager は不変。"""
    gm = _states(db, n=1)[0]
    me_hand = [cpu_ai._move_sig({"action_type": "H", "payload": {"uuid": c.uuid}}) for c in gm.p1.hand]
    opp_n = len(gm.p2.hand)
    before = copy.deepcopy(gm)
    det = cpu_mcts._determinize_opponent(gm, "p1", random.Random(0))
    assert journal.deep_diff(before, gm) is None, "_determinize_opponent が入力 manager を変更した"
    det_me = [cpu_ai._move_sig({"action_type": "H", "payload": {"uuid": c.uuid}}) for c in det.p1.hand]
    assert det_me == me_hand, "自分の手札が変わった（公平モードでも自分は不変であるべき）"
    assert len(det.p2.hand) == opp_n, "相手の手札枚数が変わった"


def test_macro_fair_mode_plan_legal(db):
    """公平モード（MCTS_DETERMINIZE=True）でも返すターンプランは**実ゲームで合法**（自分の手は実物）。"""
    gm = _states(db, n=1)[0]
    legal = {cpu_ai._move_sig(m) for m in gm.get_legal_actions(gm.p1)}
    orig = cpu_mcts.MCTS_DETERMINIZE
    try:
        cpu_mcts.MCTS_DETERMINIZE = True
        before = copy.deepcopy(gm)
        plan = cpu_mcts.mcts_plan_turn(gm, gm.p1, "hard", random.Random(0), iterations=60, horizon=2)
        assert journal.deep_diff(before, gm) is None, "公平モードで入力 manager を変更した"
        if plan:
            assert cpu_ai._move_sig(plan[0]) in legal
    finally:
        cpu_mcts.MCTS_DETERMINIZE = orig


def test_macro_multiworld_plan_legal(db):
    """複数世界アンサンブル（公平モード・worlds>1）でも返すプランは実ゲームで合法・manager 不変。"""
    gm = _states(db, n=1)[0]
    legal = {cpu_ai._move_sig(m) for m in gm.get_legal_actions(gm.p1)}
    orig = cpu_mcts.MCTS_DETERMINIZE
    try:
        cpu_mcts.MCTS_DETERMINIZE = True
        before = copy.deepcopy(gm)
        plan = cpu_mcts.mcts_plan_turn(gm, gm.p1, "hard", random.Random(0),
                                       iterations=90, horizon=2, worlds=3)
        assert journal.deep_diff(before, gm) is None
        if plan:
            assert cpu_ai._move_sig(plan[0]) in legal
    finally:
        cpu_mcts.MCTS_DETERMINIZE = orig
