"""三目並べ（GATE A 用の自明に既知最適なゲーム）。

AZループ機械（自己対戦→学習→重み更新＋NN誘導MCTS）の**実装正しさ**を検証するための
最小ゲーム。完全プレイ＝必ず引き分け、というオラクルが既知なので、学習したエージェントが
最適へ収束するか＝ループが破綻していないかを判定できる（docs/.../cpu_rl_pilot_plan_20260629.md GATE A）。

Game プロトコル（az_mcts/az_loop が duck-typing で呼ぶ・OPCG アダプタも同IF を実装する）:
  n_actions: int                      固定アクション空間サイズ
  feat_dim: int                       encode の次元
  initial_state() -> state            開始局面（不変・hashable）
  current_player(state) -> 0|1        手番
  legal_actions(state) -> list[int]   合法アクション index
  apply(state, action) -> state       遷移（新 state を返す・元を破壊しない）
  is_terminal(state) -> bool
  winner(state) -> 0|1|None           勝者（引き分け/未終了は None）
  encode(state) -> np.ndarray[feat_dim]  手番視点の特徴
  determinize(state, rng) -> state    不完全情報の決定化（完全情報ゲームは恒等）
"""
import numpy as np

# 勝ちライン（盤面 index 0..8）。
_LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8),
          (0, 3, 6), (1, 4, 7), (2, 5, 8),
          (0, 4, 8), (2, 4, 6)]


class TicTacToe:
    n_actions = 9
    feat_dim = 18   # 手番視点: 自分の石9 + 相手の石9

    def initial_state(self):
        # state = (board(9-tuple: 0空/1=p0/2=p1), player(0|1))
        return ((0,) * 9, 0)

    def current_player(self, state):
        return state[1]

    def legal_actions(self, state):
        board, _ = state
        if self._winner_board(board) is not None:
            return []
        return [i for i in range(9) if board[i] == 0]

    def apply(self, state, action):
        board, player = state
        assert board[action] == 0, "不正手（埋まったマス）"
        nb = list(board)
        nb[action] = player + 1
        return (tuple(nb), 1 - player)

    def is_terminal(self, state):
        board, _ = state
        return self._winner_board(board) is not None or all(c != 0 for c in board)

    def winner(self, state):
        return self._winner_board(state[0])

    def encode(self, state):
        board, player = state
        me = player + 1            # player0 -> 1, player1 -> 2
        opp = 2 if me == 1 else 1
        v = np.zeros(18, dtype=np.float32)
        for i, c in enumerate(board):
            if c == me:
                v[i] = 1.0
            elif c == opp:
                v[9 + i] = 1.0
        return v

    def determinize(self, state, rng):
        return state   # 完全情報＝恒等

    @staticmethod
    def _winner_board(board):
        for a, b, c in _LINES:
            if board[a] != 0 and board[a] == board[b] == board[c]:
                return board[a] - 1   # 0|1
        return None


def perfect_action(game, state, rng, memo=None):
    """ミニマックス完全プレイ（オラクル相手）。最適手が複数なら rng で1つ選ぶ。"""
    if memo is None:
        memo = {}

    def mm(s):
        if game.is_terminal(s):
            w = game.winner(s)
            if w is None:
                return 0
            return 1 if w == game.current_player(s) else -1  # 手番から見た終局値
        if s in memo:
            return memo[s]
        best = -2
        for a in game.legal_actions(s):
            val = -mm(game.apply(s, a))   # 相手番の値を反転
            if val > best:
                best = val
        memo[s] = best
        return best

    best, best_acts = -2, []
    for a in game.legal_actions(state):
        val = -mm(game.apply(state, a))
        if val > best:
            best, best_acts = val, [a]
        elif val == best:
            best_acts.append(a)
    return best_acts[rng.integers(len(best_acts))]
