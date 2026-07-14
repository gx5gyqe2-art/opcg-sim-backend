"""TreeMCTS 終局値の深さ減衰（±max(TERM_FLOOR, 1 − TERM_DECAY·depth)）の単体検証。

減衰が無いと敗勢の探索は全候補 q=−1 に飽和し「カウンターで粘る」と「素通しで即負け」が
無差別になる（docs/reports/cpu_learned_mark_review_20260711.md §F2・マーク@63）。
決定的な有向グラフゲーム（汎用 make/unmake IF・OPCG 非依存）で
「勝ちは速い方・負けは遅い方」を選ぶことを確認する。
"""
import numpy as np

import conftest  # noqa: F401
from az_mcts_tree import TreeMCTS

pytestmark = __import__("pytest").mark.cpu_infra   # 基盤健全性（探索内部機構の性質テスト）


class _GraphState:
    __slots__ = ("s",)

    def __init__(self, s):
        self.s = s

    def clone(self):
        return _GraphState(self.s)


class _GraphGame:
    """nodes[s] = (to_move("0"/"1") or None, winner or None, {move: next_s})。

    汎用 make/unmake IF（apply_inplace/unmake）で TreeMCTS に適合（test_az_mcts_tree と同型）。
    非終局の葉価値は 0（評価器に依らず終局距離だけで差が付く構成）。
    """

    def __init__(self, nodes):
        self.nodes = nodes

    def current_player(self, st):
        return self.nodes[st.s][0]

    def is_terminal(self, st):
        return self.nodes[st.s][0] is None

    def winner(self, st):
        return self.nodes[st.s][1]

    def legal_actions(self, st):
        return sorted(self.nodes[st.s][2])

    def value(self, st, to_move):
        w = self.winner(st)
        if w is None:
            return 0.0
        return 1.0 if w == to_move else -1.0

    def apply_inplace(self, st, to_move, move):
        token = st.s
        st.s = self.nodes[st.s][2][move]
        return token

    def unmake(self, st, token):
        st.s = token


def _chain(nodes, name, start_player, length, winner):
    """name0(P start) → name1(P 反転) → … → 終局(winner)。forced 一本道を張る。"""
    p = start_player
    for i in range(length):
        nxt = f"{name}{i + 1}" if i + 1 < length else f"{name}_end"
        nodes[f"{name}{i}"] = (p, None, {"go": nxt})
        p = "1" if p == "0" else "0"
    nodes[f"{name}_end"] = (None, winner, {})


def _run(nodes, root, decay=0.05, sims=300):
    game = _GraphGame(nodes)
    mcts = TreeMCTS(game, value_fn=game.value, n_sims=sims,
                    rng=np.random.default_rng(0), term_decay=decay, term_floor=0.5)
    move, N, legal = mcts.run(_GraphState(root))
    return move, dict(zip(legal, mcts.last_stats["Q"]))


def test_prefers_slower_loss():
    """両候補とも負け確定なら、終局が遠い方（=粘る手・カウンター相当）を選ぶ。"""
    nodes = {"root": ("0", None, {"fast": "F_end", "slow": "S0"})}
    nodes["F_end"] = (None, "1", {})           # 即負け（depth1）
    _chain(nodes, "S", "1", 4, "1")            # 4手先で負け（depth5）
    move, q = _run(nodes, "root")
    assert move == "slow", f"負け確定で粘る方を選ばない: {move} Q={q}"
    assert q["slow"] > q["fast"], "遅い負けの Q が速い負けを上回らない（飽和が解消されていない）"


def test_prefers_faster_win():
    """両候補とも勝ち確定なら、終局が近い方（最短リーサル）を選ぶ。"""
    nodes = {"root": ("0", None, {"now": "W_end", "later": "L0"})}
    nodes["W_end"] = (None, "0", {})           # 即勝ち（depth1）
    _chain(nodes, "L", "1", 4, "0")            # 4手先で勝ち（depth5）
    move, q = _run(nodes, "root")
    assert move == "now", f"最短の勝ちを選ばない: {move} Q={q}"


def test_terminal_backup_is_scaled_exactly():
    """終局へ直行する edge の backup は毎訪問 ±scale(depth)＝Q が正確に一致する。

    減衰 0 は従来どおり −1（飽和）・減衰 0.05 は depth1 で −0.95。"""
    nodes = {"root": ("0", None, {"fast": "F_end", "slow": "S0"})}
    nodes["F_end"] = (None, "1", {})
    _chain(nodes, "S", "1", 4, "1")
    _, q0 = _run(nodes, "root", decay=0.0)
    assert abs(q0["fast"] - (-1.0)) < 1e-9
    _, q5 = _run(nodes, "root", decay=0.05)
    assert abs(q5["fast"] - (-0.95)) < 1e-9


def test_floor_bounds_decay():
    """減衰は TERM_FLOOR で下げ止まる（深い終局でも符号情報が非終局評価に埋もれない）。"""
    game = _GraphGame({"root": ("0", None, {})})
    mcts = TreeMCTS(game, value_fn=game.value, term_decay=0.05, term_floor=0.5)
    assert abs(mcts._term_scale(0) - 1.0) < 1e-9
    assert abs(mcts._term_scale(1) - 0.95) < 1e-9
    assert abs(mcts._term_scale(40) - 0.5) < 1e-9    # 1 − 0.05·40 = −1 → floor
    assert abs(mcts._term_scale(1000) - 0.5) < 1e-9
