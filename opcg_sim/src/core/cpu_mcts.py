"""ルールモード CPU の MCTS（モンテカルロ木探索）エンジン — Phase 1（docs/SPEC.md §2.5.7）。

狙い: 単ターン α-β＋ビーム（`cpu_ai._search`）の「人手調整 eval × 浅い水平線」を超えて、
**有望な手順を選択的に深く伸ばす**ことで人間並みの強さを目指す。現行 α-β `hard` は温存（A/B 基準＋
フォールバック）し、MCTS は独立経路（`decide_mcts`）として段階導入する。

実装（Phase 1）:
  - **完全情報 PUCT**（AlphaZero 風）: 相手手札も読む（`see_opp_hand=True`＝hard と同じ情報方針）。
    隠れ情報の決定化（ISMCTS・Phase 2）はまだ行わない。
  - **方策ネットの代わりに 1-ply 評価のソフトマックスを prior**に使う＝少反復でも有望手へ集中（低反復で
    弱い純 UCT の弱点を直撃）。**価値ネットの代わりに静的 `evaluate` を葉の価値**に使う（ランダム
    プレイアウトは遅く分散が大きいため）。両者とも既存の `cpu_ai` 資産を流用＝学習不要。
  - 価値は**ルート手番側の視点に固定**した [0,1] スカラ（勝=1/負=0／盤面は tanh 圧縮）。選択は 2 人零和
    PUCT（acting==root は Q／他は 1-Q ＋ `MCTS_CPUCT·prior·√N_parent/(1+N_child)`）。
  - prior の 1-ply 評価は make/unmake（`_score_move_1ply`）で安く算出。木の降下は 1 反復ごとにルートを
    clone してパス再生（undo 不要＝正しさ優先。反復速度の最適化は Phase 1.5）。
  - ルートの最終手は**最多訪問数の子**（robust child・標準）。
"""
import math
import random
from typing import Any, Dict, List, Optional

from .cpu_ai import (evaluate, _player_by_name, _selection_moves, _prune_don_moves,
                     _prune_futile_attacks, _apply_move_inplace, _score_move_1ply, _move_sig,
                     _settle_eval)

# PUCT 探索定数。値は [0,1] 正規化なので AlphaZero の ~1.0-2.5 近傍から。自己対戦でチューニング予定。
MCTS_CPUCT = 1.8
# 盤面評価値（数千スケール）を [0,1] へ圧縮する tanh のスケール（葉の価値用）。
MCTS_VALUE_SCALE = 8000.0
# prior のソフトマックス温度（eval 単位）。小さいほど最善手へ尖る／大きいほど均す。
MCTS_PRIOR_TEMP = 2500.0
# 1 反復のシミュレーションで適用する手の上限（暴走防止＝木の最大深さの安全網）。
MCTS_MAX_PLY = 60
# 既定の反復回数（レイテンシ非依存の検証用。本番はデッドライン制御＝Phase 1.5）。
MCTS_DEFAULT_ITERS = 400
# 葉をターン境界へ整流してから評価するか（`_settle_eval`）。**原理的には正しい**（α-β と評価点を揃え
# 「ターン途中の甘い局面」バイアスを除去）が、settle は葉ごとに最大 _SETTLE_LIMIT 手の本物の効果解決を
# 走らせるため**MCTS の反復数×手数で計算が爆発し、実用反復数では計測不能なほど遅い**（実測：8局50反復が
# 10 分でも完走せず）。よって既定 OFF＝高速な mid-turn eval。turn-boundary 整流の効率化（make/unmake 降下／
# ターン粒度マクロアクション木）は Phase 1.5 の課題＝それが入るまでは OFF。
MCTS_SETTLE_LEAVES = False


def _leaf_value(state, root_name: str, see_opp_hand: bool, profile, plan, depth: int) -> float:
    """葉の価値を `root_name` 視点の [0,1] へ。勝=1/負=0／盤面は tanh 圧縮で中庸 0.5 付近。

    **ターン境界へ整流してから評価**（`cpu_ai._settle_eval`）＝α-β と同じ静止点で採点する。これをしないと
    「自分のターン途中／戦闘途中の甘い局面」（行動したがまだ反撃されていない）を過大評価し、その偏りが
    backup で木全体へ伝播する（§2.5.2 の horizon／手番パリティバイアスと同根）。`state` はクローンなので
    `_settle_eval` が破壊的に整流してよい。win は ply（=depth）割引で最短の止めを優先する。
    """
    if state.winner is not None:
        return 1.0 if state.winner == root_name else 0.0
    if MCTS_SETTLE_LEAVES:
        raw = _settle_eval(state, root_name, see_opp_hand, profile, plan, ply=depth)
    else:
        raw = evaluate(state, root_name, see_opp_hand=see_opp_hand, profile=profile, plan=plan)
    return 0.5 * (1.0 + math.tanh(raw / MCTS_VALUE_SCALE))


def _node_moves(manager, actor_name: str) -> List[Dict[str, Any]]:
    """ノードの合法手を `cpu_ai._search` と同じ方針で生成（選択分岐＋無意味手の枝刈り）。"""
    sel = _selection_moves(manager, actor_name)
    if sel:
        return sel
    actor = _player_by_name(manager, actor_name)
    moves = manager.get_legal_actions(actor)
    moves = _prune_don_moves(manager, actor_name, moves)
    moves = _prune_futile_attacks(manager, actor_name, moves)
    return moves


class _Node:
    """MCTS 木のノード（状態は保持せず、ルートからの手列を再生して再構成する）。

    `children is None` = 未展開（まだ子 stub を作っていない）。stub は (move, prior) のみ持ち、
    初訪問で `_expand` され actor/children/terminal が確定する。
    """
    __slots__ = ("move", "prior", "actor", "children", "N", "W", "terminal")

    def __init__(self, move, prior):
        self.move = move                 # parent からこのノードへ至る手（ルートは None）
        self.prior = prior               # parent の方策がこの手に与えた事前確率
        self.actor: Optional[str] = None # このノードで行動する手番側（展開時に確定）
        self.children: Optional[List["_Node"]] = None  # 子 stub 群（未展開は None）
        self.N = 0                       # 訪問回数
        self.W = 0.0                     # ルート視点の価値の総和（Q=W/N）
        self.terminal = False


def _expand(node: _Node, manager, root_name: str, see_opp_hand: bool, profile, plan) -> None:
    """`node`（= `manager` の状態）を展開: 手番側・終局を確定し、子 stub を 1-ply prior 付きで作る。"""
    if manager.winner is not None:
        node.terminal = True
        node.children = []
        return
    pa = manager.pending_actor_action()
    if not pa:
        node.terminal = True
        node.children = []
        return
    actor = pa[0]
    node.actor = actor
    moves = _node_moves(manager, actor)
    if not moves:
        node.terminal = True
        node.children = []
        return
    # 各手の 1-ply 評価（root 視点・make/unmake で安く）→ acting 視点のソフトマックスで prior に。
    sign = 1.0 if actor == root_name else -1.0
    scored = []
    for m in moves:
        v = _score_move_1ply(manager, actor, m, root_name,
                             see_opp_hand=see_opp_hand, profile=profile, plan=plan)
        if v is None:
            continue
        scored.append((m, v))
    if not scored:
        node.terminal = True
        node.children = []
        return
    mx = max(sign * v for _m, v in scored)
    exps = [math.exp((sign * v - mx) / MCTS_PRIOR_TEMP) for _m, v in scored]
    z = sum(exps) or 1.0
    node.children = [_Node(m, e / z) for (m, _v), e in zip(scored, exps)]


def _puct_select(node: _Node, root_name: str, rng) -> _Node:
    """展開済みノードの子を PUCT で選ぶ。手番側が自分に有利な方（acting==root は Q／他は 1-Q）を最大化。

    未訪問子の Q は親推定値（FPU・root 視点）で埋める＝prior 項が探索順を主導する。
    """
    is_root_turn = (node.actor == root_name)
    parent_q = node.W / node.N if node.N > 0 else 0.5
    sqrtN = math.sqrt(node.N + 1.0)
    best, best_score = None, -1e18
    for ch in node.children:
        q = (ch.W / ch.N) if ch.N > 0 else parent_q   # FPU = 親推定
        exploit = q if is_root_turn else (1.0 - q)
        u = MCTS_CPUCT * ch.prior * sqrtN / (1.0 + ch.N)
        score = exploit + u
        if score > best_score or (score == best_score and rng.random() < 0.5):
            best, best_score = ch, score
    return best


def _simulate(root: _Node, root_state, root_name: str,
              see_opp_hand: bool, profile, plan, rng) -> None:
    """1 反復: ルートを clone → PUCT で降下し未展開の葉を展開・評価 → 経路へ backup。"""
    state = root_state.clone()
    state.action_events = []
    node = root
    path = [node]
    depth = 0

    while True:
        if node.terminal or depth >= MCTS_MAX_PLY:
            break
        if node.children is None:               # 未展開の葉 → ここで展開して評価へ
            _expand(node, state, root_name, see_opp_hand, profile, plan)
            break
        child = _puct_select(node, root_name, rng)
        if child is None:
            break
        try:
            _apply_move_inplace(state, node.actor, child.move, stop_at_select=True)
        except Exception:
            # 適用失敗手は以後選ばれないよう prior を 0 に落として打ち切る（稀）。
            child.prior = 0.0
            break
        node = child
        path.append(node)
        depth += 1

    v = _leaf_value(state, root_name, see_opp_hand, profile, plan, depth)
    for nd in path:
        nd.N += 1
        nd.W += v


def decide_mcts(manager, player, difficulty: str = "hard", rng: Optional[random.Random] = None,
                iterations: Optional[int] = None, profile=None, plan=None,
                see_opp_hand: Optional[bool] = None,
                moves: Optional[List[Dict[str, Any]]] = None) -> Optional[Dict[str, Any]]:
    """MCTS（完全情報 PUCT）でルート局面の最善手を返す（合法手が無ければ None）。

    `cpu_ai.decide` と同じ move dict を返す。`see_opp_hand` 既定は hard と同じ True。
    `iterations` は反復回数（既定 `MCTS_DEFAULT_ITERS`）。`moves` を渡すとルート候補をそれに限定。
    """
    rng = rng or random
    name = player.name
    if see_opp_hand is None:
        see_opp_hand = (difficulty == "hard")
    iters = iterations if iterations is not None else MCTS_DEFAULT_ITERS

    root_moves = moves if moves is not None else _node_moves(manager, name)
    if not root_moves:
        return None
    if len(root_moves) == 1:
        return root_moves[0]

    # ルートを展開（手番側＝name・root_moves に prior を付与）。
    root = _Node(None, 1.0)
    root.actor = name
    sign = 1.0
    scored = []
    for m in root_moves:
        v = _score_move_1ply(manager, name, m, name, see_opp_hand=see_opp_hand, profile=profile, plan=plan)
        scored.append((m, v if v is not None else float("-inf")))
    finite = [v for _m, v in scored if v != float("-inf")]
    mx = max(finite) if finite else 0.0
    exps = [math.exp((v - mx) / MCTS_PRIOR_TEMP) if v != float("-inf") else 0.0 for _m, v in scored]
    z = sum(exps) or 1.0
    root.children = [_Node(m, e / z) for (m, _v), e in zip(scored, exps)]

    for _ in range(iters):
        _simulate(root, manager, name, see_opp_hand, profile, plan, rng)

    if not root.children:
        return root_moves[0]
    # robust child = 最多訪問。同数は Q（root 視点）で割り、さらに同点は乱択。
    best = max(root.children, key=lambda c: (c.N, c.W / c.N if c.N else 0.0, rng.random()))
    return best.move
