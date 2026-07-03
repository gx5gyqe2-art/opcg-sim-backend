"""ノード型 NN/評価器誘導 PUCT MCTS（非hashable・可変状態・非交互手番ゲーム用＝OPCG）。

docs/.../cpu_rl_pilot_plan_20260629.md GATE B。az_mcts（状態キー辞書・三目並べ用）と違い、
状態を**ノードに保持**し edge ごとに1回だけ apply＝OPCG の clone コストを最小化（visit毎に再cloneしない）。

一般化した backup 符号則（OPCG は同一プレイヤーが連続するため必須）:
  _simulate は「そのノードの手番視点」の value を返す。子の値を親へ畳む時、
  **手番が変わらなければ同符号・変われば反転**（2人零和）。三目並べ（常に交互）は常に反転＝az_mcts と一致。

PIMC: 探索開始時に determinize_fn で世界線を1つ固定（簡略 ISMCTS＝water-oil 回避）。
value_fn(state, to_move)->[-1,1] が固定評価器（GATE B は L1）。priors は既定一様（policy head 後付け可）。
"""
import math

import numpy as np


class _Node:
    __slots__ = ("state", "to_move", "legal", "P", "N", "W",
                 "children", "expanded", "terminal", "term_val")

    def __init__(self, state, to_move, terminal, term_val):
        self.state = state
        self.to_move = to_move
        self.terminal = terminal
        self.term_val = term_val
        self.legal = None
        self.P = None
        self.N = None
        self.W = None
        self.children = None
        self.expanded = False


class TreeMCTS:
    def __init__(self, game, value_fn, priors_fn=None, c_puct=1.5, n_sims=100,
                 determinize_fn=None, rng=None, dirichlet_alpha=0.3, dirichlet_eps=0.0):
        self.game = game
        self.value_fn = value_fn
        self.priors_fn = priors_fn         # (state, legal)->np.array|None（None=一様）
        self.c_puct = c_puct
        self.n_sims = n_sims
        self.determinize_fn = determinize_fn
        self.rng = rng or np.random.default_rng(0)
        # ルートDirichletノイズ＝自己対戦の探索多様化（eps>0で有効）。eval は eps=0 で決定的。
        # 小シャードのオンライン化で経験バッファを失う設計の「忘却防止の生命線」（レビュー指摘）。
        self.da = dirichlet_alpha
        self.de = dirichlet_eps

    def run(self, real_state):
        """探索し (最良move, 訪問数N[K], ルート合法手list) を返す。N と legal は同順。"""
        state = self.determinize_fn(real_state, self.rng) if self.determinize_fn else real_state
        to_move = self.game.current_player(state)
        root = _Node(state, to_move, self.game.is_terminal(state), 0.0)
        self._expand(root)
        if not root.legal:
            return None, None, []
        if self.de > 0.0 and len(root.legal) > 1:   # ルートに探索ノイズを混ぜる（自己対戦時）
            noise = self.rng.dirichlet([self.da] * len(root.legal))
            root.P = (1 - self.de) * root.P + self.de * noise
        for _ in range(self.n_sims):
            self._simulate(root)
        best = int(np.argmax(root.N))
        return root.legal[best], root.N, root.legal

    def _expand(self, node):
        if node.terminal:
            node.expanded = True
            return self.value_fn(node.state, node.to_move) if node.to_move else 0.0
        legal = self.game.legal_actions(node.state)
        node.legal = legal
        n = len(legal)
        node.N = np.zeros(n)
        node.W = np.zeros(n)
        node.children = [None] * n
        if self.priors_fn is not None:
            p = self.priors_fn(node.state, legal)
            node.P = p if p is not None else np.full(n, 1.0 / max(n, 1))
        else:
            node.P = np.full(n, 1.0 / max(n, 1))
        node.expanded = True
        return self.value_fn(node.state, node.to_move)

    def _simulate(self, node):
        """node 手番視点の value を返す。"""
        if node.terminal:
            return node.term_val
        if not node.expanded:
            return self._expand(node)
        if not node.legal:   # 合法手なし＝終局扱い
            return self.value_fn(node.state, node.to_move)
        # PUCT 選択
        Ns = node.N.sum()
        sqrtN = math.sqrt(Ns) if Ns > 0 else 1.0
        Q = np.where(node.N > 0, node.W / np.maximum(node.N, 1), 0.0)
        U = Q + self.c_puct * node.P * sqrtN / (1.0 + node.N)
        a = int(np.argmax(U))
        child = node.children[a]
        if child is None:
            child = self._make_child(node, a)
            node.children[a] = child
        v_child = self._simulate(child)
        # backup 符号則: 手番が変われば反転。
        v = v_child if child.to_move == node.to_move else -v_child
        node.N[a] += 1
        node.W[a] += v
        return v

    def _make_child(self, node, a):
        move = node.legal[a]
        nxt = self.game.apply(node.state, move, node.to_move)
        if nxt is None:
            # 例外手＝この手を実質禁止（極大に選ばれ続けないよう終局・自分視点最悪値の葉に）。
            dead = _Node(node.state, node.to_move, True, -1.0)
            dead.expanded = True
            return dead
        term = self.game.is_terminal(nxt)
        to_move = self.game.current_player(nxt)
        term_val = 0.0
        if term:
            # 終局ノードの value は「そのノードの手番視点」。手番不在なら勝者から決める。
            w = self.game.winner(nxt)
            ref = to_move or node.to_move
            term_val = 0.0 if w is None else (1.0 if w == ref else -1.0)
            to_move = to_move or ref
        return _Node(nxt, to_move, term, term_val)
