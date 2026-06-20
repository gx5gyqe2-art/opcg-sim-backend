"""ルールモード CPU の MCTS（モンテカルロ木探索）エンジン — Phase 1 MVP（docs/SPEC.md §2.5.7）。

狙い: 単ターン α-β＋ビーム（`cpu_ai._search`）の「人手調整 eval × 浅い水平線」を超えて、
**有望な手順を選択的に深く伸ばす**ことで人間並みの強さを目指す土台を作る。本 MVP は:

  - **完全情報 UCT**（Phase 1）: 相手手札も読む（hard と同じ情報方針＝`see_opp_hand=True`）。
    隠れ情報の決定化（ISMCTS・Phase 2）はまだ行わない（公平化・頑健化は次段）。
  - **rollout の代わりに静的 `evaluate` を葉の価値**に使う（複雑なカードゲームのランダム
    プレイアウトは遅く・分散が大きいため、既に質の高い評価関数を価値関数として流用する＝
    AlphaZero から方策/価値ネットを抜いた素朴版に相当）。
  - 状態遷移は `cpu_ai` の共通適用パス（`_apply_move_inplace`）＋ `GameManager.clone()` を再利用。
    1 反復ごとにルートを clone してパスを再生する（undo 不要＝実装単純・正しさ優先）。

設計の要:
  - 木のノード = 「ある手番側が行動する決定点」。手番側 = `pending_actor_action()[0]`。
  - 価値は**ルート手番側の視点に固定**した [0,1] のスカラ（勝=1/負=0／盤面は tanh で圧縮）。
    選択は手番側が自分に有利な方を採る＝acting==root は Q を、acting≠root は 1-Q を最大化（2人零和 UCT）。
  - ルートの最終手は**最多訪問数の子**（robust child・UCT 標準）。

現行 `cpu_ai.decide`（α-β）は**温存**＝MCTS はフラグ/難易度で切り替える独立経路。自己対戦 Elo
（`tests/cpu_arena.py`／`elo_settle_ab.py`）で対 hard の強さを計測してから本番採用可否を判断する。
"""
import math
import random
from typing import Any, Dict, List, Optional, Tuple

from . import cpu_ai
from .cpu_ai import (evaluate, _player_by_name, _selection_moves, _prune_don_moves,
                     _prune_futile_attacks, _apply_move_inplace, _move_sig, W_WIN)

# UCT 探索定数。値は [0,1] に正規化するので標準的な ~sqrt(2) 近傍から。自己対戦でチューニング予定。
MCTS_C = 1.2
# 盤面評価値（数千スケール）を [0,1] へ圧縮する tanh のスケール。W_LIFE=6000 等に対し、数千の優劣差が
# 0.5±0.3 程度に乗るよう設定（勝敗は別途 1.0/0.0）。自己対戦でチューニング予定。
MCTS_VALUE_SCALE = 8000.0
# 1 反復のシミュレーション中に適用する手の上限（暴走防止＝木の最大深さの安全網）。
MCTS_MAX_PLY = 60
# 既定の反復回数（レイテンシ非依存の検証用。本番はデッドラインで制御予定＝Phase 1.5）。
MCTS_DEFAULT_ITERS = 400


def _value01(manager, root_name: str, see_opp_hand: bool, profile, plan) -> float:
    """`root_name` 視点の盤面価値を [0,1] へ。勝=1/負=0／盤面は tanh 圧縮で中庸 0.5 付近。"""
    if manager.winner is not None:
        return 1.0 if manager.winner == root_name else 0.0
    ev = evaluate(manager, root_name, see_opp_hand=see_opp_hand, profile=profile, plan=plan)
    return 0.5 * (1.0 + math.tanh(ev / MCTS_VALUE_SCALE))


def _node_moves(manager, actor_name: str) -> List[Dict[str, Any]]:
    """ノードの合法手を `cpu_ai._search` と同じ方針で生成する。

    単一/多対象選択は `_selection_moves` で分岐し、それ以外は `get_legal_actions` ＋ 無意味手の枝刈り
    （`_prune_don_moves`／`_prune_futile_attacks`）＝α-β と同じ候補集合で揃える。
    """
    sel = _selection_moves(manager, actor_name)
    if sel:
        return sel
    actor = _player_by_name(manager, actor_name)
    moves = manager.get_legal_actions(actor)
    moves = _prune_don_moves(manager, actor_name, moves)
    moves = _prune_futile_attacks(manager, actor_name, moves)
    return moves


class _Node:
    """MCTS 木のノード（状態は保持せず、ルートからの手列を再生して再構成する）。"""
    __slots__ = ("parent", "move", "actor", "children", "untried", "N", "W", "terminal")

    def __init__(self, parent, move, actor, untried, terminal):
        self.parent = parent
        self.move = move                 # parent からこのノードへ至る手（ルートは None）
        self.actor = actor               # このノードで行動する手番側（terminal なら None）
        self.children: List["_Node"] = []
        self.untried: List[Dict[str, Any]] = untried   # 未展開の手
        self.N = 0                       # 訪問回数
        self.W = 0.0                     # ルート視点の価値の総和（Q=W/N）
        self.terminal = terminal


def _ucb_select(node: _Node, root_name: str, rng) -> _Node:
    """完全展開ノードの子を UCT で選ぶ。手番側が自分に有利な方（acting==root は Q／他は 1-Q）を最大化。"""
    is_root_turn = (node.actor == root_name)
    logN = math.log(node.N + 1.0)
    best, best_score = None, -1.0
    for ch in node.children:
        q = ch.W / ch.N if ch.N > 0 else 0.5
        exploit = q if is_root_turn else (1.0 - q)
        explore = MCTS_C * math.sqrt(logN / (ch.N + 1e-9))
        score = exploit + explore
        if score > best_score or (score == best_score and rng.random() < 0.5):
            best, best_score = ch, score
    return best


def _make_node(parent, move, manager, root_name: str) -> _Node:
    """`manager`（move 適用後の状態）に対応するノードを生成（手番側・未展開手・終局を判定）。"""
    if manager.winner is not None:
        return _Node(parent, move, None, [], terminal=True)
    pa = manager.pending_actor_action()
    if not pa:
        return _Node(parent, move, None, [], terminal=True)
    actor = pa[0]
    untried = list(_node_moves(manager, actor))
    return _Node(parent, move, actor, untried, terminal=(not untried))


def _simulate(root_node: _Node, root_state, root_name: str,
              see_opp_hand: bool, profile, plan, rng) -> None:
    """1 反復: ルートを clone → 選択/展開でパスを下る → 葉を evaluate → 経路へ backup。"""
    state = root_state.clone()
    state.action_events = []
    node = root_node
    path = [node]
    plies = 0

    # 1) 選択: 完全展開かつ非終局のノードを UCT で下る（手を state に適用しながら）。
    while (not node.terminal) and (not node.untried) and node.children and plies < MCTS_MAX_PLY:
        node = _ucb_select(node, root_name, rng)
        try:
            _apply_move_inplace(state, node.parent.actor, node.move, stop_at_select=True)
        except Exception:
            break
        path.append(node)
        plies += 1

    # 2) 展開: 未展開の手が残っていれば 1 つ展開して子を追加。
    if (not node.terminal) and node.untried and plies < MCTS_MAX_PLY:
        mv = node.untried.pop(rng.randrange(len(node.untried)))
        try:
            _apply_move_inplace(state, node.actor, mv, stop_at_select=True)
        except Exception:
            mv = None
        if mv is not None:
            child = _make_node(node, mv, state, root_name)
            node.children.append(child)
            node = child
            path.append(node)

    # 3) 価値: 葉（展開先 or 終局 or 深さ上限）を root 視点で評価。
    v = _value01(state, root_name, see_opp_hand, profile, plan)

    # 4) backup: 経路の全ノードへ訪問数と価値（root 視点）を加算。
    for nd in path:
        nd.N += 1
        nd.W += v


def decide_mcts(manager, player, difficulty: str = "hard", rng: Optional[random.Random] = None,
                iterations: Optional[int] = None, profile=None, plan=None,
                see_opp_hand: Optional[bool] = None,
                moves: Optional[List[Dict[str, Any]]] = None) -> Optional[Dict[str, Any]]:
    """MCTS でルート局面の最善手を返す（合法手が無ければ None）。`cpu_ai.decide` と同じ move dict を返す。

    Phase 1 MVP: 完全情報 UCT（`see_opp_hand` 既定は hard と同じ True）。`iterations` は反復回数
    （既定 `MCTS_DEFAULT_ITERS`）。`moves` を渡すとルート候補をそれに限定する（guard driver 用）。
    """
    rng = rng or random
    name = player.name
    if see_opp_hand is None:
        see_opp_hand = (difficulty == "hard")
    iters = iterations if iterations is not None else MCTS_DEFAULT_ITERS

    # ルート候補: 明示指定が無ければノードと同じ生成（選択分岐＋枝刈り）。
    if moves is None:
        moves = _node_moves(manager, name)
    if not moves:
        return None
    if len(moves) == 1:
        return moves[0]

    root = _Node(None, None, name, list(moves), terminal=False)
    for _ in range(iters):
        if not root.untried and not root.children:
            break
        _simulate(root, manager, name, see_opp_hand, profile, plan, rng)

    if not root.children:
        return moves[0]
    # robust child = 最多訪問。同数は Q（root 視点）で割り、さらに同点は乱択。
    best = max(root.children, key=lambda c: (c.N, c.W / c.N if c.N else 0.0, rng.random()))
    return best.move
