"""ノード型 TreeMCTS の単体検証＝OPCG用MCTSの backup符号則/選択が正しい。

三目並べを TreeMCTS の IF に薄く適合（apply に actor_name 引数・value 提供）して検証＝
高速・決定的に「即勝ち手を選ぶ/相手の勝ちをブロック」を確認（OPCGの重い対局に依らず符号則を担保）。
"""
import numpy as np

import conftest  # noqa: F401
import tictactoe as T
from az_mcts_tree import TreeMCTS


class _TTTTree:
    """TicTacToe を TreeMCTS の Game IF に適合（apply に actor_name・value を追加）。"""
    def __init__(self):
        self.g = T.TicTacToe()

    def current_player(self, s):
        return None if self.g.is_terminal(s) else str(self.g.current_player(s))

    def is_terminal(self, s):
        return self.g.is_terminal(s)

    def winner(self, s):
        w = self.g.winner(s)
        return None if w is None else str(w)

    def legal_actions(self, s):
        return self.g.legal_actions(s)

    def apply(self, s, move, actor_name):
        return self.g.apply(s, move)

    def value(self, s, to_move):
        w = self.g.winner(s)
        if w is None:
            return 0.0
        return 1.0 if str(w) == to_move else -1.0


def test_tree_mcts_takes_winning_move():
    game = _TTTTree()
    state = ((1, 1, 0, 2, 2, 0, 0, 0, 0), 0)   # player0 が cell2 で勝ち
    mcts = TreeMCTS(game, value_fn=game.value, n_sims=300, rng=np.random.default_rng(0))
    move, N = mcts.run(state)
    assert move == 2, f"即勝ち手(2)を選べない: move={move} N={N}"


def test_tree_mcts_blocks_opponent_win():
    game = _TTTTree()
    state = ((1, 1, 0, 2, 0, 0, 0, 0, 0), 1)   # player1 は cell2 でブロック
    mcts = TreeMCTS(game, value_fn=game.value, n_sims=400, rng=np.random.default_rng(0))
    move, _ = mcts.run(state)
    assert move == 2, f"相手の即勝ちをブロックできない: {move}"
