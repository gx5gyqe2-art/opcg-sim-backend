"""NN誘導 PUCT MCTS（GATE A〜パイロット共通部品・docs/.../cpu_rl_pilot_plan_20260629.md §3.1）。

2人零和・交互手番。各葉で net.evaluate(encode(state), legal) → (事前確率, value)。
選択は PUCT: argmax_a Q(a) + c_puct·P(a)·√(ΣN)/(1+N(a))。backup は手番交代で符号反転。

**PIMC×MCTS の water-oil 回避（レビュー確定事項）**: 不完全情報は「1回の探索中は決定化した
1つの世界線を固定」する（簡略 ISMCTS）。完全情報ゲームは determinize が恒等＝無影響。
ルートに Dirichlet ノイズ（自己対戦の探索多様化）。返り値＝訪問回数分布（policy 教師）。
"""
import math

import numpy as np


class MCTS:
    def __init__(self, game, net, c_puct=1.5, n_sims=50,
                 dirichlet_alpha=0.3, dirichlet_eps=0.25, rng=None):
        self.game = game
        self.net = net
        self.c_puct = c_puct
        self.n_sims = n_sims
        self.da = dirichlet_alpha
        self.de = dirichlet_eps
        self.rng = rng or np.random.default_rng(0)
        self._reset()

    def _reset(self):
        self.P = {}   # state -> {a: prior}
        self.N = {}   # state -> {a: visit}
        self.W = {}   # state -> {a: value-sum}
        self.legal = {}

    def run(self, root_state, add_noise=True):
        """root から n_sims 回探索し、合法手上の訪問分布（np.ndarray[n_actions]）を返す。"""
        self._reset()
        # 不完全情報: この探索の間だけ世界線を固定（完全情報は恒等）。
        root = self.game.determinize(root_state, self.rng)
        self._expand(root)
        if add_noise and self.legal[root]:
            self._add_root_noise(root)
        for _ in range(self.n_sims):
            self._simulate(root)
        counts = np.zeros(self.game.n_actions)
        for a, n in self.N[root].items():
            counts[a] = n
        s = counts.sum()
        if s > 0:
            counts = counts / s
        return counts

    def _expand(self, state):
        legal = self.game.legal_actions(state)
        self.legal[state] = legal
        if not legal:
            self.P[state] = {}; self.N[state] = {}; self.W[state] = {}
            return 0.0
        priors, v = self.net.evaluate(self.game.encode(state), legal)
        self.P[state] = priors
        self.N[state] = {a: 0 for a in legal}
        self.W[state] = {a: 0.0 for a in legal}
        return v

    def _add_root_noise(self, root):
        legal = self.legal[root]
        noise = self.rng.dirichlet([self.da] * len(legal))
        for a, nz in zip(legal, noise):
            self.P[root][a] = (1 - self.de) * self.P[root][a] + self.de * nz

    def _simulate(self, state):
        """state（手番視点）の value を返す。再帰下降＋backup。"""
        if self.game.is_terminal(state):
            return self._terminal_value(state)
        if state not in self.P:
            return self._expand(state)
        # PUCT 選択
        legal = self.legal[state]
        Ns = sum(self.N[state].values())
        sqrtN = math.sqrt(Ns) if Ns > 0 else 1.0
        best_a, best_u = None, -1e18
        for a in legal:
            n = self.N[state][a]
            q = (self.W[state][a] / n) if n > 0 else 0.0
            u = q + self.c_puct * self.P[state][a] * sqrtN / (1 + n)
            if u > best_u:
                best_u, best_a = u, a
        nxt = self.game.apply(state, best_a)
        v_child = self._simulate(nxt)       # 子（相手手番）視点の value
        v = -v_child                        # 現手番視点へ反転
        self.N[state][best_a] += 1
        self.W[state][best_a] += v
        return v

    def _terminal_value(self, state):
        w = self.game.winner(state)
        if w is None:
            return 0.0
        return 1.0 if w == self.game.current_player(state) else -1.0
