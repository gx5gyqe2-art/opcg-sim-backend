"""ノード型 TreeMCTS（make/unmake版・唯一の実装）の単体検証＝PUCT backup符号則/選択が正しい。

三目並べを TreeMCTS の**汎用 make/unmake IF**（`apply_inplace`/`unmake` を提供）に薄く適合して検証＝
高速・決定的に「即勝ち手を選ぶ/相手の勝ちをブロック」を確認（OPCGの重い対局・journal機構に依らず符号則を担保）。
TreeMCTS は game が `apply_inplace`/`unmake` を持てば OPCG journal 経路ではなくこの汎用経路を使う（mcts.py 参照）。
"""
import numpy as np

import conftest  # noqa: F401
from az_mcts_tree import TreeMCTS

# 勝ちライン（盤面 index 0..8）。
_LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8),
          (0, 3, 6), (1, 4, 7), (2, 5, 8),
          (0, 4, 8), (2, 4, 6)]


def _winner_board(board):
    for a, b, c in _LINES:
        if board[a] != 0 and board[a] == board[b] == board[c]:
            return board[a] - 1   # 0|1
    return None


class _TTTState:
    """可変盤面（make/unmake 用）。board=[0空/1=p0/2=p1]×9・player=0|1。"""
    __slots__ = ("board", "player")

    def __init__(self, board, player):
        self.board = list(board)
        self.player = player

    def clone(self):
        return _TTTState(self.board, self.player)


class _TTTMakeUnmake:
    """TicTacToe を TreeMCTS の汎用 make/unmake IF に適合。

    `apply_inplace(state, to_move, move)->token` と `unmake(state, token)` を提供する＝
    TreeMCTS が OPCG journal ではなくこの経路で make/unmake する（OPCG非依存で符号則を検証）。
    手番は文字列（"0"/"1"・終局は None）＝TreeMCTS が backup 符号則に使う。
    """
    def current_player(self, s):
        return None if self.is_terminal(s) else str(s.player)

    def is_terminal(self, s):
        return _winner_board(s.board) is not None or all(c != 0 for c in s.board)

    def winner(self, s):
        w = _winner_board(s.board)
        return None if w is None else str(w)

    def legal_actions(self, s):
        if _winner_board(s.board) is not None:
            return []
        return [i for i in range(9) if s.board[i] == 0]

    def value(self, s, to_move):
        w = self.winner(s)
        if w is None:
            return 0.0
        return 1.0 if w == to_move else -1.0

    def apply_inplace(self, s, to_move, move):
        """その場で石を置き手番を進める。undo_token = 埋めたマス index。"""
        assert s.board[move] == 0, "不正手（埋まったマス）"
        s.board[move] = s.player + 1
        s.player = 1 - s.player
        return move

    def unmake(self, s, token):
        """apply_inplace を巻き戻す（マスを空に・手番を戻す）。"""
        s.board[token] = 0
        s.player = 1 - s.player


def test_tree_mcts_takes_winning_move():
    game = _TTTMakeUnmake()
    state = _TTTState((1, 1, 0, 2, 2, 0, 0, 0, 0), 0)   # player0 が cell2 で勝ち
    mcts = TreeMCTS(game, value_fn=game.value, n_sims=300, rng=np.random.default_rng(0))
    move, N, _ = mcts.run(state)
    assert move == 2, f"即勝ち手(2)を選べない: move={move} N={N}"


def test_tree_mcts_blocks_opponent_win():
    game = _TTTMakeUnmake()
    state = _TTTState((1, 1, 0, 2, 0, 0, 0, 0, 0), 1)   # player1 は cell2 でブロック
    mcts = TreeMCTS(game, value_fn=game.value, n_sims=400, rng=np.random.default_rng(0))
    move, _, _ = mcts.run(state)
    assert move == 2, f"相手の即勝ちをブロックできない: {move}"
