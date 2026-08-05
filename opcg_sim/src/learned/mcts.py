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

終局値の深さ減衰（2026-07-11・マークレビュー F2）: terminal は ±max(TERM_FLOOR, 1 − TERM_DECAY·depth) で
backup する（L1 の ±(W_WIN − ply) と同原理）。減衰が無いと敗勢の探索が全候補 q=−1 に飽和し、
「カウンターで1手粘る」と「素通しで即負け」が無差別になる（防御崩壊）。非終局の葉価値は素通し。

API: run(real_state) -> (best_move, N[K], legal[K])。
"""
import math
import random

import numpy as np

from opcg_sim.src.core import cpu_ai, journal
from opcg_sim.src.core.journal import JournaledList
from .config import (C_PUCT, DIRICHLET_ALPHA, TERM_DECAY, TERM_FLOOR,
                     SERVE_QUIESCE, QUIESCE_MAX_PLIES, TREE_BOX_BATTLE)


def in_battle(mgr):
    """戦闘が未解決か（＝葉評価してはいけない「騒がしい」局面か）。

    カウンター/ブロッカー選択の最中は、どの札を切ったかで手札が1枚減る点は同じでも
    **命が助かるかどうかは解決後にしか盤面へ現れない**。OPCG 以外のゲーム（active_battle を
    持たない）では常に False＝静止探索は no-op。"""
    try:
        return bool(getattr(mgr, "active_battle", None))
    except Exception:
        return False


def quiesce_choice(mgr, legal, priors_fn=None):
    """静止探索の延長で採る手の index（**policy 最良手 → PASS → 先頭手**・pure）。

    葉の意味論は「**解決した時点の手札と盤面**」（ユーザ指摘 2026-08-04）。打ち切る位置を
    人為的に決める（例: 常に PASS）と、実際には選ばれない継続を評価してしまう。

    priors 不在時に **先頭手を機械的に採ってはいけない**（2026-08-04 実測の落とし穴）:
    m1@15 では先頭が「もう1枚カウンターを足す」で、**どちらの枝も残りを足してしまい
    両方助かって判別不能**になった。判別が要る唯一の場所で盲目になるため PASS へ落とす。"""
    if priors_fn is not None and len(legal) > 1:
        p = priors_fn(mgr, legal)
        if p is not None:
            return int(np.argmax(p))
    for i, mv in enumerate(legal):
        if (mv or {}).get("action_type") == "PASS":
            return i
    return 0


def resolve_battle_inplace(game, mgr, priors_fn=None, max_plies=QUIESCE_MAX_PLIES):
    """戦闘が解決するまで mgr をその場で進める（**巻き戻さない**・適用手数を返す）。

    **探索（葉評価）と教師（コーパス符号化）が同一の解決規約を使うための単一の正**。
    別々に実装すると必ずずれ、教師が「ネットが実際に見ることのない盤面」を教えることになる
    （2026-08-04 の train/serve skew 対策）。巻き戻し・乱数復元は呼び出し側の責務
    （探索は transaction 内で使い、教師はクローン上で使うため要件が異なる）。"""
    n = 0
    for _ in range(max_plies):
        if game.is_terminal(mgr) or not in_battle(mgr):
            break
        name = game.current_player(mgr)
        legal = game.legal_actions(mgr) if name else None
        if not legal:
            break
        try:
            cpu_ai._apply_move_inplace(mgr, name, legal[quiesce_choice(mgr, legal, priors_fn)])
        except Exception:
            break
        n += 1
    return n


def resolved_branch_values(game, mgr, name, legal, value_fn, priors_fn=None,
                           max_plies=QUIESCE_MAX_PLIES):
    """各合法手を「戦闘を解決した出口盤面」まで進めて評価した value 列（`mgr` は不変）。

    **戦闘を1つの箱として畳む**（ユーザ整理 2026-08-05）: 箱の中の手順そのものは評価対象に
    せず、箱の**出口**（解決後の盤面・手札・ライフ）だけを value で比べる。どの出口になるかは
    ネットの予測ではなく**エンジンの実計算**（7000 攻撃に 2000 を足せば 8000 で凌ぐ、は算術的に
    確定する）。判断しているのは葉評価と同じ value ネット自身で、別系統の防御ロジックではない。

    枝の残り手は `resolve_battle_inplace`（policy 最良手→PASS→先頭手）で進める＝探索の静止探索・
    教師コーパスと**同一の解決規約**（train/serve skew 防止の単一の正）。

    値は `name` 視点。適用に失敗した枝は None（呼び出し側で除外）。全枝を**同一の乱数列**から
    評価し（CRN）、抜けるときに乱数状態を元へ戻す＝実ゲームへ探索の消費を漏らさない。"""
    base_rng_state = random.getstate()
    vals = []
    for mv in legal:
        random.setstate(base_rng_state)      # CRN: 枝間の差だけを見る（確率効果を共通化）
        saved_events = mgr.action_events
        v = None
        try:
            with journal.transaction():      # 退出で盤面を巻き戻す（呼び出し側の mgr は不変）
                mgr.action_events = JournaledList()
                cpu_ai._apply_move_inplace(mgr, name, mv)
                resolve_battle_inplace(game, mgr, priors_fn, max_plies)
                v = value_fn(mgr, name)
        except Exception:
            v = None
        finally:
            mgr.action_events = saved_events
        vals.append(v)
    random.setstate(base_rng_state)
    return vals


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
                 determinize_fn=None, rng=None, dirichlet_alpha=DIRICHLET_ALPHA, dirichlet_eps=0.0,
                 term_decay=TERM_DECAY, term_floor=TERM_FLOOR,
                 quiesce=None, quiesce_max_plies=QUIESCE_MAX_PLIES, box_battle=None):
        self.game = game
        self.value_fn = value_fn
        self.priors_fn = priors_fn
        self.c_puct = c_puct
        self.n_sims = n_sims
        self.determinize_fn = determinize_fn
        self.rng = rng or np.random.default_rng(0)
        self.da = dirichlet_alpha
        self.de = dirichlet_eps
        # 終局値の深さ減衰（L1 の ±(W_WIN − ply) と同原理）: 速い勝ちを優先し、敗勢では
        # 抵抗して長い方の負けを選ぶ（全候補 −1 飽和の無差別を解消・config.TERM_DECAY 参照）。
        self.term_decay = term_decay
        self.term_floor = term_floor
        # 静止探索（config.SERVE_QUIESCE）: 戦闘中の葉は解決まで進めてから評価する。
        self.quiesce = SERVE_QUIESCE if quiesce is None else quiesce
        self.quiesce_max_plies = quiesce_max_plies
        # 木の中の箱化（config.TREE_BOX_BATTLE）: 戦闘窓ノードを出口 value 最良の1手へ畳む。
        self.box_battle = TREE_BOX_BATTLE if box_battle is None else box_battle
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

    _in_battle = staticmethod(in_battle)   # 後方互換の別名（定義はモジュール関数が正）

    def _quiesce_choice(self, mgr, legal):
        """後方互換の薄いラッパ（定義はモジュール関数 `quiesce_choice` が正）。"""
        return quiesce_choice(mgr, legal, self.priors_fn)

    def _leaf_value(self, mgr, to_move):
        """葉の評価。戦闘中なら**解決するまで進めてから**評価する（静止探索・v35）。

        **葉の意味論＝解決した時点の手札と盤面**（ユーザ指摘 2026-08-04）。延長は policy 最良手で
        進める（`_quiesce_choice`）。打ち切る位置を人為的に決めると実際には選ばれない継続を
        評価してしまうため、解決まで進めてその盤面を見る。

        延長は `journal.transaction()` で巻き戻す。延長中の apply は確率効果で global random を
        消費しうるため**乱数状態も復元**する（消費したままだとエッジ固定＝CRN 一貫性が壊れ、
        ノード統計が訪問ごとにブレる）。
        """
        if not self.quiesce or self._generic or not in_battle(mgr):
            return self.value_fn(mgr, to_move)
        rng_state = random.getstate()
        saved_events = mgr.action_events
        v = None
        try:
            with journal.transaction():
                mgr.action_events = JournaledList()
                resolve_battle_inplace(self.game, mgr, self.priors_fn, self.quiesce_max_plies)
                v = self.value_fn(mgr, to_move)
        finally:
            mgr.action_events = saved_events
            random.setstate(rng_state)   # 延長の乱数消費を漏らさない（CRN 一貫性）
        return v if v is not None else self.value_fn(mgr, to_move)

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
        leaf_v = None
        if self.box_battle and not self._generic and len(legal) > 1 and in_battle(mgr):
            # **木の中の箱化**（v35・2026-08-05）: 戦闘窓は「どの出口（解決後の盤面・手札・
            # ライフ）になるか」で決まる局所判断なので、子を並べて訪問を配らず**出口 value 最良の
            # 1手へ畳む**＝戦闘全体が1本のマクロ手になる。二人零和では相手は最善応手を返すのが
            # 正しく、PUCT の訪問混合は収束前の副産物（保険ではない）＝畳む方がミニマックスに近い。
            # 幅は失われない（木には別の攻撃順・別盤面の戦闘が無数にあり、各々が別の箱を持つ）。
            # 副次効果として、カウンターの組合せに費やしていた訪問がメイン判断へ回る。
            vals = resolved_branch_values(g, mgr, node.to_move, legal, self.value_fn,
                                          self.priors_fn, self.quiesce_max_plies)
            ok = [i for i, v in enumerate(vals) if v is not None]
            if ok:
                b = max(ok, key=lambda i: vals[i])
                legal = [legal[b]]
                leaf_v = vals[b]       # 葉見積もりも同じ出口＝木と読み出しで規約が一致する
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
        return leaf_v if leaf_v is not None else self._leaf_value(mgr, node.to_move)

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

    def _term_scale(self, depth):
        """終局値の深さ減衰係数 ∈ [term_floor, 1]。depth=root からの手数（terminal ノードの深さ）。"""
        return max(self.term_floor, 1.0 - self.term_decay * depth)

    def _descend_journal(self, node, a, move, mgr, depth):
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
                vbox[0] = self._simulate(child, mgr, depth)
        mgr.action_events = saved_events            # transient（値に無関係・念のため復元）
        return vbox[0]

    def _descend_generic(self, node, a, move, mgr, depth):
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
            return self._simulate(child, mgr, depth)
        finally:
            self.game.unmake(mgr, token)

    def _simulate(self, node, mgr, depth=0):
        """node 手番視点の value を返す。mgr は node の状態にある（呼び出し前提）。

        depth＝root からの手数。終局値のみ `_term_scale(depth)` で減衰する（非終局の
        葉価値は素通し＝ネット/評価器の見積もりに深さバイアスを足さない）。
        """
        if node.terminal:
            return node.term_val * self._term_scale(depth)
        if not node.expanded:
            return self._expand(node, mgr)
        if not node.legal:
            return self._leaf_value(mgr, node.to_move)
        # PUCT 選択（旧clone版と同一式）
        Ns = node.N.sum()
        sqrtN = math.sqrt(Ns) if Ns > 0 else 1.0
        Q = np.where(node.N > 0, node.W / np.maximum(node.N, 1), 0.0)
        U = Q + self.c_puct * node.P * sqrtN / (1.0 + node.N)
        a = int(np.argmax(U))
        move = node.legal[a]

        if self._generic:
            v_child = self._descend_generic(node, a, move, mgr, depth + 1)
        else:
            v_child = self._descend_journal(node, a, move, mgr, depth + 1)

        child = node.children[a]
        v = v_child if child.to_move == node.to_move else -v_child
        node.N[a] += 1
        node.W[a] += v
        return v
