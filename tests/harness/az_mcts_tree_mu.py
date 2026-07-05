"""make/unmake 版 ノード型 PUCT MCTS（clone廃止・cProfile実測でcloneが自己対戦の79%＝これを消す）。

`az_mcts_tree.TreeMCTS` はエッジごとに `GameManager.clone()`（deepcopy）で子状態を作り保持する。
本実装は **状態をノードに持たせず**、1シミュレーションをルートの作業状態から降下しながら
`_apply_move_inplace` で**その場適用**し、per-ply の `journal.transaction()` 退出で**自動巻き戻し**する
（製品α-βの `cpu_ai._recurse_child` と同一パターン＝`test_cpu_make_unmake.py` で clone 同値を実証済み）。
クローンは determinize の 1手1回だけ（探索中は0）。訪問数 N/W はゲーム状態外の numpy 配列なので巻き戻し不変。

RNG 一貫性（重要）: 確率効果（デッキ再シャッフル等）は探索中の apply でグローバル `random` を消費するが
journal は RNG を巻き戻さない。素の再適用だと「訪問ごとに引き直し」でノード統計が崩れる。そこで run() が
**各シミュレーション冒頭でグローバル `random` を基準状態へ戻す**ため、木では経路が一意→ノードのエッジ apply は
毎回同一 RNG から始まり**エッジごと固定（coherent）**＝clone版（子状態キャッシュ）と同一意味論になる。
検証: 確率を消費しない局面では clone版と **bit 一致**（`mu_mcts_probe.py` パリティ max|ΔN|=0）、確率局面でも
**エッジ単位の apply 後盤面が探索内で一貫**（不一致 0）。確率遷移の乱数実現は clone版と独立（同一サンプリング
方式・別 seed 相当）＝分布は同一で bit 一致ではない。

API は `TreeMCTS` と同じ（drop-in 差し替え可）: run(real_state) -> (best_move, N[K], legal[K])。
"""
import math
import random

import numpy as np

from opcg_sim.src.core import cpu_ai, journal
from opcg_sim.src.core.journal import JournaledList


class _Node:
    __slots__ = ("to_move", "legal", "P", "N", "W", "children", "expanded", "terminal", "term_val")

    def __init__(self):
        self.to_move = None
        self.legal = None
        self.P = None
        self.N = None
        self.W = None
        self.children = None
        self.expanded = False
        self.terminal = False
        self.term_val = 0.0


class TreeMCTSMakeUnmake:
    def __init__(self, game, value_fn, priors_fn=None, c_puct=1.5, n_sims=100,
                 determinize_fn=None, rng=None, dirichlet_alpha=0.3, dirichlet_eps=0.0):
        self.game = game
        self.value_fn = value_fn
        self.priors_fn = priors_fn
        self.c_puct = c_puct
        self.n_sims = n_sims
        self.determinize_fn = determinize_fn
        self.rng = rng or np.random.default_rng(0)
        self.da = dirichlet_alpha
        self.de = dirichlet_eps

    def run(self, real_state):
        # 作業状態＝determinize のクローン（無ければ 1回だけ clone して呼び出し側を汚さない）。
        mgr = self.determinize_fn(real_state, self.rng) if self.determinize_fn else real_state.clone()
        root = _Node()
        self._expand(root, mgr)
        if not root.legal:
            return None, None, []
        if self.de > 0.0 and len(root.legal) > 1:
            noise = self.rng.dirichlet([self.da] * len(root.legal))
            root.P = (1 - self.de) * root.P + self.de * noise
        # 確率効果（デッキ再シャッフル等）は探索中の apply でグローバル `random` を消費するが、
        # journal はゲーム状態しか巻き戻さない（RNG は戻らない）。素の再適用だと「訪問ごとに引き直し」
        # になりノード統計の意味が崩れる（clone版は子状態をキャッシュ＝エッジごと固定）。そこで
        # **各シミュレーションの冒頭でグローバル `random` を基準状態へ戻す**と、あるノードに至る経路は
        # 木では一意なので、そのノードから出るエッジの apply は毎回同一 RNG から始まる＝**エッジごと固定**
        # （coherent・clone版と同一意味論）。確率を消費しない局面ではリセットは no-op＝clone と bit 一致。
        base_rng_state = random.getstate()
        for _ in range(self.n_sims):
            random.setstate(base_rng_state)
            self._simulate(root, mgr)
        random.setstate(base_rng_state)   # 探索の RNG 消費を実ゲームへ漏らさない（決定論・再現性）
        best = int(np.argmax(root.N))
        return root.legal[best], root.N, root.legal

    def _expand(self, node, mgr):
        """mgr は node の状態にある。葉価値（node.to_move 視点）を返す。"""
        g = self.game
        if g.is_terminal(mgr):
            tm = g.current_player(mgr)
            w = g.winner(mgr)
            node.to_move = tm
            node.terminal = True
            node.expanded = True
            ref = tm if tm is not None else node.to_move
            if w is None or ref is None:
                node.term_val = 0.0
            else:
                node.term_val = 1.0 if w == ref else -1.0
            # clone版 _expand のルート終局と同じ規約（実プレイのルートは非終局＝実害なし）。
            return self.value_fn(mgr, tm) if tm else 0.0
        node.to_move = g.current_player(mgr)
        legal = g.legal_actions(mgr)
        node.legal = legal
        n = len(legal)
        node.N = np.zeros(n)
        node.W = np.zeros(n)
        node.children = [None] * n
        if self.priors_fn is not None:
            p = self.priors_fn(mgr, legal)
            node.P = p if p is not None else np.full(n, 1.0 / max(n, 1))
        else:
            node.P = np.full(n, 1.0 / max(n, 1))
        node.expanded = True
        return self.value_fn(mgr, node.to_move)

    def _new_child_after_apply(self, node, mgr):
        """apply 直後の mgr（子状態）から子ノードの終局情報を確定（clone版 _make_child と同規約）。"""
        child = _Node()
        g = self.game
        term = g.is_terminal(mgr)
        tm = g.current_player(mgr)
        if term:
            w = g.winner(mgr)
            ref = tm if tm is not None else node.to_move
            child.terminal = True
            child.expanded = True
            child.term_val = 0.0 if w is None else (1.0 if w == ref else -1.0)
            child.to_move = tm if tm is not None else ref
        else:
            child.to_move = tm
        return child

    def _simulate(self, node, mgr):
        """node 手番視点の value を返す。mgr は node の状態にある（呼び出し前提）。"""
        if node.terminal:
            return node.term_val
        if not node.expanded:
            return self._expand(node, mgr)
        if not node.legal:
            return self.value_fn(mgr, node.to_move)
        # PUCT 選択（clone版と同一式）
        Ns = node.N.sum()
        sqrtN = math.sqrt(Ns) if Ns > 0 else 1.0
        Q = np.where(node.N > 0, node.W / np.maximum(node.N, 1), 0.0)
        U = Q + self.c_puct * node.P * sqrtN / (1.0 + node.N)
        a = int(np.argmax(U))
        move = node.legal[a]
        child = node.children[a]

        vbox = [0.0]
        saved_events = mgr.action_events
        with journal.transaction():                 # ← unmake（退出時に降下分を巻き戻す）
            mgr.action_events = JournaledList()
            dead = False
            try:
                cpu_ai._apply_move_inplace(mgr, node.to_move, move)
            except Exception:
                dead = True
            if dead:
                if child is None:
                    child = _Node()
                    child.expanded = True
                    child.terminal = True
                    child.term_val = -1.0            # 例外手＝自分視点最悪（clone版と同規約）
                    child.to_move = node.to_move
                    node.children[a] = child
                vbox[0] = child.term_val
            else:
                if child is None:
                    child = self._new_child_after_apply(node, mgr)
                    node.children[a] = child
                vbox[0] = self._simulate(child, mgr)
        mgr.action_events = saved_events            # transient（値に無関係・念のため復元）

        v = vbox[0] if child.to_move == node.to_move else -vbox[0]
        node.N[a] += 1
        node.W[a] += v
        return v
