"""OPCGGame アダプタの高速単体検証（CI内）。重い対局単調性は test_gate_b（slow）。"""
import numpy as np

import conftest  # noqa: F401
from opcg_game import OPCGGame
from az_mcts_tree import TreeMCTS
from cpu_selfplay import _load_db


def test_determinize_pimc():
    """相手の伏せ手札を再サンプル＝枚数保存・中身変化・自分の手札不変（チート除去）。"""
    g = OPCGGame()
    db = _load_db()
    m = g.new_game(db, 1)
    me = g.current_player(m)
    opp = "p2" if me == "p1" else "p1"
    opp_hand = lambda mm: [c.master.card_id for c in (mm.p2 if opp == "p2" else mm.p1).hand]
    me_hand = lambda mm: [c.master.card_id for c in (mm.p1 if me == "p1" else mm.p2).hand]
    before_opp, before_me = opp_hand(m), me_hand(m)
    d = g.determinize(m, me, np.random.default_rng(0))
    assert len(opp_hand(d)) == len(before_opp), "相手手札枚数が保存されない"
    assert opp_hand(d) != before_opp, "相手手札の中身が変わっていない（再サンプル未実施）"
    assert me_hand(d) == before_me, "自分の手札が変わった（チート/破壊）"


def test_value_in_range_and_terminal():
    g = OPCGGame()
    db = _load_db()
    m = g.new_game(db, 2)
    name = g.current_player(m)
    v = g.value(m, name)
    assert -1.0 <= v <= 1.0 and abs(v) < 1.0, f"非終局 value が範囲外/飽和: {v}"


def test_mcts_returns_legal_move():
    """低 sims でも合法な move を1つ返す（アダプタ駆動が壊れていない）。"""
    g = OPCGGame()
    db = _load_db()
    m = g.new_game(db, 3)
    name = g.current_player(m)
    legal = g.legal_actions(m)
    mcts = TreeMCTS(g, value_fn=g.value, n_sims=12,
                    determinize_fn=lambda s, r: g.determinize(s, name, r),
                    rng=np.random.default_rng(0))
    move, N, _ = mcts.run(m)
    assert move in legal, "返り値が合法手でない"
    assert N is not None and int(np.sum(N)) > 0, "訪問が記録されていない"
