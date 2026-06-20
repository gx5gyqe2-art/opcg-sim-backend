"""④ PV/killer 着手順序再利用の**等価性＋効果**ゲート（docs/SPEC.md §2.5.3）。

PV 順序付けは探索の内部最適化（ビーム選別**後**の子集合の中で killer 手を先頭へ寄せて α-β カットを
早める）であって、**集合は変えず順序のみ**＝予算が拘束しないフル探索では **選ぶ手・深掘りスコアは
`_USE_PV_ORDER` の True/False で完全一致**でなければならない（α-β のカットは値を変えない）。本テストは:

  1. `decide` の選択手が PV ON/OFF で一致する（多 seed・難易度 normal/hard）。
  2. `_scored_search` の深掘りスコア（手→値）が両方式で一致する（既定予算＋増量予算の両方）。
  3. 効果方向の保証＝PV ON は探索ノード数が OFF 以下（決して悪化しない）かつ合計で削減がある。
  4. killer 表ユーティリティ（`_record_killer`／`_pv_reorder`）の単体不変条件（集合不変・順序のみ）。

実行: OPCG_LOG_SILENT=1 python -m pytest tests/test_cpu_pv_order.py -q -s -p no:cacheprovider
"""
import copy
import random

import conftest  # noqa: F401
import pytest

from opcg_sim.src.core import cpu_ai
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


def _deep_scores(gm, boost=None):
    g = copy.deepcopy(gm)
    old = cpu_ai.HARD_PER_MOVE_BUDGET
    if boost is not None:
        cpu_ai.HARD_PER_MOVE_BUDGET = boost
    try:
        r = cpu_ai._scored_search(g, "p1", g.get_legal_actions(g.p1),
                                  see_opp_hand=True, opp_public_only=False)
    finally:
        cpu_ai.HARD_PER_MOVE_BUDGET = old
    return {cpu_ai._move_sig(m): round(v, 6) for v, m in r}


def _decide_sig(gm, difficulty):
    mv = cpu_ai.decide(gm, gm.p1, difficulty, random.Random(0))
    return cpu_ai._move_sig(mv) if mv is not None else None


@pytest.mark.parametrize("difficulty", ["normal", "hard"])
def test_pv_order_picks_same_move(db, difficulty):
    """PV ON/OFF で decide の選択手が完全一致する（内部最適化＝方策不変）。"""
    states = _states(db)
    assert states, "p1 メイン局面が作れなかった"
    orig = cpu_ai._USE_PV_ORDER
    try:
        diffs = []
        for i, gm in enumerate(states):
            cpu_ai._USE_PV_ORDER = False
            base = _decide_sig(copy.deepcopy(gm), difficulty)
            cpu_ai._USE_PV_ORDER = True
            pv = _decide_sig(copy.deepcopy(gm), difficulty)
            if base != pv:
                diffs.append((i, base, pv))
        assert not diffs, f"選択手が不一致: {diffs}"
    finally:
        cpu_ai._USE_PV_ORDER = orig


@pytest.mark.parametrize("boost", [None, 2000])
def test_pv_order_same_deep_scores(db, boost):
    """`_scored_search` の深掘りスコア（手sig→値）が PV ON/OFF で完全一致する（既定＋増量予算）。"""
    states = _states(db, n=6)
    orig = cpu_ai._USE_PV_ORDER
    try:
        for i, gm in enumerate(states):
            if len(gm.get_legal_actions(gm.p1)) <= 1:
                continue
            cpu_ai._USE_PV_ORDER = False
            random.seed(0)
            off = _deep_scores(gm, boost)
            cpu_ai._USE_PV_ORDER = True
            random.seed(0)
            on = _deep_scores(gm, boost)
            assert off == on, f"深掘りスコア不一致 (state {i}, boost={boost}):\n off={off}\n on ={on}"
    finally:
        cpu_ai._USE_PV_ORDER = orig


def test_pv_order_reduces_or_equals_nodes(db):
    """効果方向の保証: PV ON の探索ノード数は OFF **以下**（決して悪化しない）かつ合計で削減がある。

    増量予算（カットのみがノード削減要因＝settle が混ざらない領域）で `_search` 呼び出し回数を計数。
    """
    states = _states(db, n=6)
    boost = 2000
    orig = cpu_ai._USE_PV_ORDER

    def _count(gm):
        g = copy.deepcopy(gm)
        old = cpu_ai.HARD_PER_MOVE_BUDGET
        cpu_ai.HARD_PER_MOVE_BUDGET = boost
        real = cpu_ai._search
        cnt = [0]

        def wrap(*a, **k):
            cnt[0] += 1
            return real(*a, **k)
        cpu_ai._search = wrap
        try:
            cpu_ai._scored_search(g, "p1", g.get_legal_actions(g.p1),
                                  see_opp_hand=True, opp_public_only=False)
        finally:
            cpu_ai._search = real
            cpu_ai.HARD_PER_MOVE_BUDGET = old
        return cnt[0]

    try:
        tot_off = tot_on = 0
        for gm in states:
            if len(gm.get_legal_actions(gm.p1)) <= 1:
                continue
            cpu_ai._USE_PV_ORDER = False
            random.seed(0)
            n_off = _count(gm)
            cpu_ai._USE_PV_ORDER = True
            random.seed(0)
            n_on = _count(gm)
            assert n_on <= n_off, f"PV ON がノード増加（悪化）: off={n_off} on={n_on}"
            tot_off += n_off
            tot_on += n_on
        assert tot_on < tot_off, f"PV でノード削減が無い: off={tot_off} on={tot_on}"
    finally:
        cpu_ai._USE_PV_ORDER = orig


def test_record_killer_mru_and_cap():
    """`_record_killer`: 最近使用が先頭・重複は前出し・上限 `_KILLER_SLOTS`・None は no-op。"""
    killers = {}
    a = {"action_type": "A", "payload": {"uuid": "x"}}
    b = {"action_type": "B", "payload": {"uuid": "y"}}
    c = {"action_type": "C", "payload": {"uuid": "z"}}
    cpu_ai._record_killer(killers, 3, a)
    cpu_ai._record_killer(killers, 3, b)
    assert killers[3][0] == cpu_ai._move_sig(b)  # 最近使用が先頭
    cpu_ai._record_killer(killers, 3, a)         # 既出は前出し
    assert killers[3][0] == cpu_ai._move_sig(a)
    cpu_ai._record_killer(killers, 3, c)         # 上限 2 を超えたら最古を捨てる
    assert len(killers[3]) == cpu_ai._KILLER_SLOTS
    cpu_ai._record_killer(None, 0, a)            # None は no-op（例外を出さない）


def test_pv_reorder_preserves_set():
    """`_pv_reorder`: 子の**集合（要素）は不変**＝順序のみ変化。killer は先頭・非killer は元順を保持。"""
    children = [(1.0, {"action_type": "A", "payload": {"uuid": "a"}}),
                (0.9, {"action_type": "B", "payload": {"uuid": "b"}}),
                (0.8, {"action_type": "C", "payload": {"uuid": "c"}})]
    kill = [cpu_ai._move_sig(children[2][1])]  # C を killer に
    out = cpu_ai._pv_reorder(children, kill)
    assert [cpu_ai._move_sig(m) for _v, m in out][0] == kill[0]      # killer が先頭
    assert sorted(id(m) for _v, m in out) == sorted(id(m) for _v, m in children)  # 集合不変
    # 残り（A,B）は元の順序を保持
    rest = [cpu_ai._move_sig(m) for _v, m in out[1:]]
    assert rest == [cpu_ai._move_sig(children[0][1]), cpu_ai._move_sig(children[1][1])]
    # killer 無し / 空 killer は原型そのまま
    assert cpu_ai._pv_reorder(children, ()) is children
    none_match = cpu_ai._pv_reorder(children, [("Z", "zz", ())])
    assert none_match is children
