"""ノード型 NN/評価器誘導 PUCT MCTS（make/unmake・clone廃止・自己対戦/本番の単一の正）。

**旧clone版からの移行（2026-07）**: 以前はエッジごとに `GameManager.clone()`（deepcopy）で子状態を
作り保持していた（cProfile 実測で自己対戦の 79% が clone）。本実装は **状態をノードに持たせず**、
1シミュレーションをルートの作業状態から降下しながら**その場適用**し、per-ply の巻き戻しで
**自動 unmake** する（製品α-βの `cpu_ai._recurse_child` と同一パターン＝`test_cpu_make_unmake.py` で
clone 同値を実証済み）。クローンは determinize の 1手1回だけ（探索中は0）。訪問数 N/W はゲーム状態外の
numpy 配列なので巻き戻し不変。

**apply/unmake の2経路（`__init__` で1回判定）**:
- **OPCG（既定）**: `journal.transaction()` 退出で自動巻き戻し＋`cpu_ai._apply_move_inplace` でその場適用。
  OPCGGame（`apply_inplace`/`unmake` を持たない）は必ずこちら＝旧mu版とバイト不変（本番挙動不変）。
- **汎用**: ゲームが `apply_inplace(state, to_move, move)->undo_token` と `unmake(state, token)` を提供する
  場合はそれで make/unmake する。OPCG journal 機構に依存しない任意ゲーム（三目並べ等）を回せる＝
  旧clone版が持っていた「汎用参照」性を引き継ぐ（`test_az_mcts_tree.py` が backup符号則を汎用検証）。

一般化した backup 符号則（OPCG は同一プレイヤーが連続するため必須）:
  _simulate は「そのノードの手番視点」の value を返す。子の値を親へ畳む時、
  **手番が変わらなければ同符号・変われば反転**（2人零和）。三目並べ（常に交互）は常に反転。

RNG 一貫性（重要）: 確率効果（デッキ再シャッフル等）は探索中の apply でグローバル `random` を消費するが
journal は RNG を巻き戻さない。素の再適用だと「訪問ごとに引き直し」でノード統計が崩れる。そこで run() が
**各シミュレーション冒頭でグローバル `random` を基準状態へ戻す**ため、木では経路が一意→ノードのエッジ apply は
毎回同一 RNG から始まり**エッジごと固定（coherent）**＝旧clone版（子状態キャッシュ）と同一意味論になる。
確率を消費しない局面（汎用ゲーム含む）ではリセットは no-op。

PIMC: 探索開始時に determinize_fn で世界線を1つ固定（簡略 ISMCTS＝water-oil 回避）。
value_fn(state, to_move)->[-1,1] が固定評価器（learned は value net）。priors は既定一様（policy head 後付け可）。

API: run(real_state) -> (best_move, N[K], legal[K])。
"""
import math
import random

import numpy as np

from opcg_sim.src.core import cpu_ai, journal
from opcg_sim.src.core.journal import JournaledList
from .config import C_PUCT, DIRICHLET_ALPHA


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


class TreeMCTS:
    def __init__(self, game, value_fn, priors_fn=None, c_puct=C_PUCT, n_sims=100,
                 determinize_fn=None, rng=None, dirichlet_alpha=DIRICHLET_ALPHA, dirichlet_eps=0.0):
        self.game = game
        self.value_fn = value_fn
        self.priors_fn = priors_fn
        self.c_puct = c_puct
        self.n_sims = n_sims
        self.determinize_fn = determinize_fn
        self.rng = rng or np.random.default_rng(0)
        self.da = dirichlet_alpha
        self.de = dirichlet_eps
        # apply/unmake 経路を1回だけ判定（ホットループで分岐しない）。ゲームが make/unmake IF を
        # 提供する＝汎用経路（三目並べ等・OPCG journal に非依存）。OPCGGame は持たない＝journal経路。
        self._generic = hasattr(game, "apply_inplace") and hasattr(game, "unmake")

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
        # 各シミュレーション冒頭でグローバル `random` を基準へ戻す（確率効果のエッジ固定＝coherence。
        # 確率非消費局面では no-op）。詳細はモジュール docstring「RNG 一貫性」。
        base_rng_state = random.getstate()
        for _ in range(self.n_sims):
            random.setstate(base_rng_state)
            self._simulate(root, mgr)
        random.setstate(base_rng_state)   # 探索の RNG 消費を実ゲームへ漏らさない（決定論・再現性）
        best = int(np.argmax(root.N))
        # トレース用の root 統計（訪問数・行動価値 Q=W/N）を残す（`cpu_learned.decide` が等価手マージと
        # トレース候補一覧に読む）。無いと等価手マージが効かず trace["candidates"] も欠落する。
        self.last_stats = {"legal": root.legal, "N": root.N.copy(),
                           "Q": root.W / np.maximum(root.N, 1.0)}
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
            # 旧clone版 _expand のルート終局と同じ規約（実プレイのルートは非終局＝実害なし）。
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
        """apply 直後の mgr（子状態）から子ノードの終局情報を確定（旧clone版 _make_child と同規約）。"""
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

    def _dead_child(self, node, a):
        """例外手＝この手を実質禁止（自分視点最悪値の終局葉に固定）。旧clone版 _make_child と同規約。"""
        child = node.children[a]
        if child is None:
            child = _Node()
            child.expanded = True
            child.terminal = True
            child.term_val = -1.0
            child.to_move = node.to_move
            node.children[a] = child
        return child.term_val

    def _descend_journal(self, node, a, move, mgr):
        """OPCG: journal.transaction() 退出で自動巻き戻し。子の value（子手番視点）を返す。"""
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
                vbox[0] = self._dead_child(node, a)
            else:
                child = node.children[a]
                if child is None:
                    child = self._new_child_after_apply(node, mgr)
                    node.children[a] = child
                vbox[0] = self._simulate(child, mgr)
        mgr.action_events = saved_events            # transient（値に無関係・念のため復元）
        return vbox[0]

    def _descend_generic(self, node, a, move, mgr):
        """汎用: ゲーム提供の apply_inplace/unmake で make/unmake。子の value（子手番視点）を返す。"""
        try:
            token = self.game.apply_inplace(mgr, node.to_move, move)
        except Exception:
            return self._dead_child(node, a)
        try:
            child = node.children[a]
            if child is None:
                child = self._new_child_after_apply(node, mgr)
                node.children[a] = child
            return self._simulate(child, mgr)
        finally:
            self.game.unmake(mgr, token)

    def _simulate(self, node, mgr):
        """node 手番視点の value を返す。mgr は node の状態にある（呼び出し前提）。"""
        if node.terminal:
            return node.term_val
        if not node.expanded:
            return self._expand(node, mgr)
        if not node.legal:
            return self.value_fn(mgr, node.to_move)
        # PUCT 選択（旧clone版と同一式）
        Ns = node.N.sum()
        sqrtN = math.sqrt(Ns) if Ns > 0 else 1.0
        Q = np.where(node.N > 0, node.W / np.maximum(node.N, 1), 0.0)
        U = Q + self.c_puct * node.P * sqrtN / (1.0 + node.N)
        a = int(np.argmax(U))
        move = node.legal[a]

        if self._generic:
            v_child = self._descend_generic(node, a, move, mgr)
        else:
            v_child = self._descend_journal(node, a, move, mgr)

        child = node.children[a]
        v = v_child if child.to_move == node.to_move else -v_child
        node.N[a] += 1
        node.W[a] += v
        return v
