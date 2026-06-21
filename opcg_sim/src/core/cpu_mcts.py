"""ルールモード CPU の MCTS（モンテカルロ木探索）エンジン＝**ターン粒度マクロアクション木**（docs/SPEC.md §2.5.7）。

狙い: 単ターン α-β＋ビーム（`cpu_ai._search`）の「浅い水平線」を超えて、**有望なターン計画を選択的に深く
読む**ことで強さを目指す。現行 α-β `hard` は温存し、MCTS は独立経路（`decide_mcts_macro`）として導入する。

設計（マクロアクション木）:
  - **ノード=ターン境界の状態／エッジ=1 ターン丸ごとのプレイ**。これで (a) **葉が常にターン境界**＝静的
    `evaluate` が偏りなく・整流不要で安い、(b) **木の深さ=ターン数**＝激減し少 clone で複数ターン先を読める。
    （各ノードを 1 手にする素朴な MCTS は「ターン途中の甘い局面」を過大評価し弱かった＝本設計がその是正。）
  - ターンは全列挙不能なので**確率的サンプリング**（1-ply 評価のソフトマックス方策 `_sample_turn_plan`）で
    候補を生成し、**progressive widening** で木を広げる。選択は 2 人零和 UCB、葉は境界 `evaluate`（[0,1] 圧縮）。
  - ルートの最善子=最多訪問のターンプラン＝**その手列を丸ごと返す**（`mcts_plan_turn`）。逐次 1 手 IF は
    `decide_mcts_macro`（`cpu_ai.decide_cached` と同型＝queue に計画を持ち replay）。
  - **Phase 2 公平化（決定化・ISMCTS-lite）**: `MCTS_DETERMINIZE` で相手の伏せ手札を覗かず再サンプリング。
    `MCTS_WORLDS`>1 で複数世界アンサンブル（真の ISMCTS 近似）。
  - `plan`/`profile`（§2.5.5/§2.5.4）は任意入力。葉評価に勝ち筋項（逆算リーサル・telegraph 致死等）を効かせる。
"""
import math
import random
import time
from typing import Any, Dict, List, Optional

from .cpu_ai import (evaluate, _player_by_name, _selection_moves, _prune_don_moves,
                     _prune_futile_attacks, _apply_move_inplace, _score_move_1ply, _move_sig,
                     TURN_ACTION_CAP)
from . import action_api
from . import cpu_features
from . import cpu_value_model

# 自分のターン中に割り込む相手の戦闘応答（ブロック/カウンター）を既定 PASS で畳んで自分のターンを
# 最後まで続けるか。OFF だと攻撃宣言で相手応答待ち（SELECT_COUNTER/BLOCKER）の**戦闘途中の局面**が葉に
# なり、評価が ±~1245(生eval) ブレる（実測・葉の ~20%）＝探索を増やすほど誤収束（optimizer's curse）。
# ON で葉が常にきれいなターン境界（相手 MAIN）になり評価が健全化する。相手の防御は「相手の手番（次の
# マクロノード）」で読まれるので、ここで PASS 既定にしても二重には無視しない（α-β の settle と同方針）。
MCTS_RESOLVE_COMBAT = True
_ACT_PASS = action_api.CONST.get('c_to_s_interface', {}).get('BATTLE_ACTIONS', {}).get('TYPES', {}).get('PASS', 'PASS')

# 盤面評価値（数千スケール）を [0,1] へ圧縮する tanh のスケール（葉の価値用）。
MCTS_VALUE_SCALE = 8000.0
MCTS_MACRO_HORIZON = 3       # 何ターン先まで読むか（自分→相手→自分…の境界数）。
MCTS_MACRO_TEMP = 1800.0     # ターンサンプリングのソフトマックス温度（eval 単位・小さいほど最善寄り）。
MCTS_MACRO_C = 1.4           # UCB 探索定数（[0,1] 正規化）。
MCTS_PW_C = 2.0              # progressive widening 係数（子数 ≈ 1 + PW_C·N^PW_ALPHA）。
MCTS_PW_ALPHA = 0.5
MCTS_MACRO_ITERS = 160       # 既定反復回数（ターンあたり 1 回の探索＝手列全体を一括計画）。
MCTS_MIN_ITERS = 1           # 壁時計デッドライン使用時でも最低限回す反復数（最悪 overshoot を抑えるため小さく）。
# Phase 2 公平化: 相手の伏せ手札を再サンプリングして探索（チート除去）。既定 OFF＝完全情報。
MCTS_DETERMINIZE = False
# 複数世界アンサンブル（真の ISMCTS 近似）の世界数。既定 1＝単一世界。
MCTS_WORLDS = 1


def _value_boundary(manager, root_name: str, see_opp_hand: bool, profile=None, plan=None) -> float:
    """ターン境界状態の価値を `root_name` 視点 [0,1] へ（整流不要＝境界は元々静止点）。勝1/負0。

    `plan`/`profile` を供給すると `evaluate` の勝ち筋項（逆算リーサル・telegraph 致死・脅威キーワード・
    engine_aware のリーダー有効パワー等）が作動する（macro は葉が常にターン境界＝telegraph 致死が効く）。
    未供給は素の J 値評価（従来同値）。
    """
    if manager.winner is not None:
        return 1.0 if manager.winner == root_name else 0.0
    ev = evaluate(manager, root_name, see_opp_hand=see_opp_hand, profile=profile, plan=plan)
    base = 0.5 * (1.0 + math.tanh(ev / MCTS_VALUE_SCALE))
    # 学習価値葉のブレンド（§2.5.7 残5・既定OFF）。`OPCG_VALUE_BLEND`>0 かつモデル同梱時のみ作動＝
    # option value（手札の答え在庫×相手脅威）を学習価値で補う。α=0 では推論を一切走らせない＝現状同値。
    a = cpu_value_model.blend_alpha()
    if a > 0.0 and cpu_value_model.is_available():
        try:
            p = cpu_value_model.predict_winprob(
                cpu_features.extract_features(manager, root_name, see_opp_hand=see_opp_hand))
            if p is not None:
                return (1.0 - a) * base + a * p
        except Exception:
            pass
    return base


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


def _weighted_choice(items, weights, rng):
    z = sum(weights)
    if z <= 0:
        return rng.choice(items)
    r = rng.random() * z
    acc = 0.0
    for it, w in zip(items, weights):
        acc += w
        if r <= acc:
            return it
    return items[-1]


def _auto_resolve_opp(state, me: str) -> None:
    """自分（`me`）のターン中に割り込む**相手の戦闘応答（ブロック/カウンター）＋それに連なる選択**を既定で
    畳み、自分の手番に戻すか相手の本来のターン（相手 MAIN）/終局まで進める。

    これで自分のターンプランが「攻撃→相手応答待ち」で止まらず最後まで流れ、葉が常にきれいなターン境界
    （相手 MAIN）になる＝戦闘途中の楽観/ノイズ評価を排除する。`MCTS_RESOLVE_COMBAT=False` で no-op（旧挙動）。
    """
    if not MCTS_RESOLVE_COMBAT:
        return
    for _ in range(16):
        if state.winner is not None:
            return
        pa = state.pending_actor_action()
        if not pa:
            return
        pid, act = pa
        if pid == me or act == "MAIN_ACTION":
            return  # 自分の手番に戻った／相手の本来のターン＝境界
        actor = _player_by_name(state, pid)
        state.action_events = []
        try:
            if act in ("SELECT_BLOCKER", "SELECT_COUNTER"):
                action_api.apply_battle_action(state, actor, _ACT_PASS, None)
            else:  # 戦闘に連なる相手側の選択（トリガー等）は既定解決
                pend = state.get_pending_request()
                payload = state.default_interaction_payload(pend)
                action_api.apply_game_action(state, actor, action_api.ACT_RESOLVE_SELECTION, payload)
        except Exception:
            return


def _settle_combat(state, me: str) -> None:
    """進行中の戦闘を既定解決で安定境界まで畳む（`_auto_resolve_opp` の一般化＝**自分の応答も**畳む）。

    `_auto_resolve_opp` は相手の戦闘応答だけを既定 PASS で畳むが（自分のターン中に攻撃宣言で止まらない
    ため）、本関数は **`me` 自身の戦闘応答（カウンター/ブロック）も既定 PASS で畳む**。用途は防御手の採点：
    カウンター宣言直後の「戦闘途中・解決前」の盤面（リーダーが一時的に強い・楽観）で評価せず、**戦闘を最後
    まで解決した実結果**（ライフ増減・カード消費・トリガー）で採点するため（中盤戦闘の楽観排除＝Fix A の
    防御側版）。MAIN_ACTION（ターン境界）に達したら停止。`MCTS_RESOLVE_COMBAT=False` で no-op（旧挙動）。
    """
    if not MCTS_RESOLVE_COMBAT:
        return
    for _ in range(24):
        if state.winner is not None:
            return
        pa = state.pending_actor_action()
        if not pa:
            return
        pid, act = pa
        if act == "MAIN_ACTION":
            return  # ターン境界＝安定静止点
        actor = _player_by_name(state, pid)
        state.action_events = []
        try:
            if act in ("SELECT_BLOCKER", "SELECT_COUNTER"):
                action_api.apply_battle_action(state, actor, _ACT_PASS, None)
            else:  # 戦闘に連なる選択（トリガー等・どちら側でも）は既定解決
                pend = state.get_pending_request()
                payload = state.default_interaction_payload(pend)
                action_api.apply_game_action(state, actor, action_api.ACT_RESOLVE_SELECTION, payload)
        except Exception:
            return


def _score_defense_move(state, player_name: str, move: Dict[str, Any], see_opp_hand: bool) -> Optional[float]:
    """防御の戦闘応答手を、**適用後に戦闘を解決してから**評価した 1-ply 値（生 eval スケール）。

    `_score_move_1ply` は手適用直後（戦闘解決前）の盤面を評価するため、届かない部分カウンター（例: +1000 で
    必要 +3001 に届かない）でも「リーダーが一時的に +1000」で得に見え、サンプラがそれを生成してしまう。本関数は
    クローンへ適用後 `_settle_combat` で戦闘を畳み、**実結果（ライフは結局減る・カードだけ損）**で採点する＝
    無駄カウンターを正しく低評価し、サンプリングから締め出す。返り値は `_score_move_1ply` と同じ生 eval
    スケール（ソフトマックス温度 `MCTS_MACRO_TEMP` と整合）。失敗（例外）は None。
    """
    clone = state.clone()
    clone.action_events = []
    try:
        _apply_move_inplace(clone, player_name, move, stop_at_select=True)
    except Exception:
        return None
    _settle_combat(clone, player_name)
    return evaluate(clone, player_name, see_opp_hand=see_opp_hand, profile=None, plan=None)


def _sample_turn_plan(state, player_name: str, rng, see_opp_hand: bool,
                      deadline: Optional[float] = None) -> List[Dict[str, Any]]:
    """`state`（`player_name` の手番境界）から**1 ターンを確率的にプレイ**し、適用した手列を返す。

    各手番の決定は 1-ply 評価のソフトマックス（温度 `MCTS_MACRO_TEMP`）でサンプリング＝多様な候補ターンを
    生む（最善付近に集中しつつ揺らす）。`state` は破壊的に進む（TURN_END／相手介入／キャップで停止）。
    （prior 採点は plan/profile を渡さない＝候補手×決定×反復で 1-ply 評価が走るため、plan 付き eval だと
    レイテンシが破綻する。plan の価値は葉評価 `_value_boundary` 側だけに効かせる。）
    `deadline`（壁時計）超過時は途中で打ち切る＝**1 反復が巨大盤面で数十秒に膨らむのを内側から防ぐ**
    （上限保証の要。打ち切られた部分プランも合法な手列なので replay 可）。
    """
    plan: List[Dict[str, Any]] = []
    for _ in range(TURN_ACTION_CAP + 4):
        if deadline is not None and time.time() >= deadline:
            break  # 締切超過＝この手番サンプルを早期打ち切り（部分プランを返す）
        _auto_resolve_opp(state, player_name)   # 相手の戦闘応答を畳んで自分の手番へ戻す
        pa = state.pending_actor_action()
        if not pa or pa[0] != player_name:
            break  # 相手の本来のターン/終局＝ターン境界
        moves = _node_moves(state, player_name)
        if not moves:
            break
        # 相手のターン中・自分が防御側で戦闘中の判断は**戦闘を解決してから**採点する＝届かない部分カウンター・
        # 届かないリーダー能力（例: 【相手のアタック時】捨てて+2000）等の「戦闘途中の楽観」を排除する
        # （Fix A の防御側版）。カウンター/ブロックだけでなく、カウンター段階の前に解決される**誘発能力の
        # interaction やその対象選択も含めて全部**対象にするため、pending の種別ではなく「active_battle あり
        # ＆相手の手番」で判定する（リーダー能力経由の無駄防御も締め出す＝凡ミスの別入口を塞ぐ）。
        ab = getattr(state, "active_battle", None)
        combat_resp = (ab is not None and state.turn_player is not None
                       and state.turn_player.name != player_name) \
                      or pa[1] in ("SELECT_COUNTER", "SELECT_BLOCKER")
        if len(moves) == 1:
            mv = moves[0]
        else:
            scored = []
            for m in moves:
                if combat_resp:
                    v = _score_defense_move(state, player_name, m, see_opp_hand)
                else:
                    v = _score_move_1ply(state, player_name, m, player_name,
                                         see_opp_hand=see_opp_hand, profile=None, plan=None)
                if v is not None:
                    scored.append((m, v))
            if not scored:
                break
            mx = max(v for _m, v in scored)
            ws = [math.exp((v - mx) / MCTS_MACRO_TEMP) for _m, v in scored]
            mv = _weighted_choice([m for m, _v in scored], ws, rng)
        try:
            _apply_move_inplace(state, player_name, mv, stop_at_select=True)
        except Exception:
            break
        plan.append(mv)
        if mv.get("action_type") == "TURN_END":
            break
    return plan


def _apply_turn_plan(state, player_name: str, plan: List[Dict[str, Any]]) -> bool:
    """記録済みのターンプランを `state` に**決定論的に再生**する（選択降下の再構成用）。

    完全情報・同一開始状態なら手は合法に再現される。万一不一致（効果内 RNG 等）なら適用を止めて False。
    記録は自分の手のみ＝相手の戦闘応答は `_auto_resolve_opp` で各手の前後に既定再現する（サンプル時と一致）。
    """
    for mv in plan:
        _auto_resolve_opp(state, player_name)
        pa = state.pending_actor_action()
        if not pa or pa[0] != player_name:
            return False
        legal = {_move_sig(m) for m in _node_moves(state, player_name)}
        if _move_sig(mv) not in legal:
            return False
        try:
            _apply_move_inplace(state, player_name, mv, stop_at_select=True)
        except Exception:
            return False
    _auto_resolve_opp(state, player_name)   # 末尾の戦闘も畳んで境界へ
    return True


class _MacroNode:
    """マクロ木のノード（=ターン境界状態）。`plan`=ここへ至るターンプラン（ルートは None）。"""
    __slots__ = ("plan", "actor", "children", "N", "W", "terminal")

    def __init__(self, plan, actor, terminal):
        self.plan = plan
        self.actor: Optional[str] = actor   # この境界で手番を持つ側（terminal は None）
        self.children: List["_MacroNode"] = []
        self.N = 0
        self.W = 0.0
        self.terminal = terminal


def _macro_node_from_state(plan, manager) -> _MacroNode:
    if manager.winner is not None:
        return _MacroNode(plan, None, terminal=True)
    pa = manager.pending_actor_action()
    if not pa:
        return _MacroNode(plan, None, terminal=True)
    return _MacroNode(plan, pa[0], terminal=False)


def _ucb_select_macro(node: _MacroNode, root_name: str, rng) -> _MacroNode:
    is_root_turn = (node.actor == root_name)
    logN = math.log(node.N + 1.0)
    best, best_score = None, -1e18
    for ch in node.children:
        q = ch.W / ch.N if ch.N > 0 else 0.5
        exploit = q if is_root_turn else (1.0 - q)
        explore = MCTS_MACRO_C * math.sqrt(logN / (ch.N + 1e-9))
        s = exploit + explore
        if s > best_score or (s == best_score and rng.random() < 0.5):
            best, best_score = ch, s
    return best


def _macro_simulate(root: _MacroNode, root_state, root_name: str,
                    horizon: int, see_opp_hand: bool, rng, profile=None, plan=None,
                    deadline: Optional[float] = None) -> None:
    """1 反復: ルートを clone → progressive widening でターンプランを展開/選択しながら境界を下り、
    horizon ターン先（or 終局）の境界を `evaluate` → 経路へ backup。`deadline` はターンサンプリングへ伝播。"""
    state = root_state.clone()
    state.action_events = []
    node = root
    path = [node]
    turns = 0

    while (not node.terminal) and turns < horizon:
        max_children = 1 + int(MCTS_PW_C * (node.N ** MCTS_PW_ALPHA))
        if len(node.children) < max_children:
            # 展開: 新しいターンプランをサンプリング（state は次境界まで進む）。
            tp = _sample_turn_plan(state, node.actor, rng, see_opp_hand, deadline)
            if not tp:
                break
            child = _macro_node_from_state(tp, state)
            node.children.append(child)
            node = child
            path.append(node)
            turns += 1
            break  # 展開した子＝葉として評価
        # 選択: 既存子を UCB で選び、そのターンプランを再生。
        child = _ucb_select_macro(node, root_name, rng)
        if child is None or not _apply_turn_plan(state, node.actor, child.plan):
            break
        node = child
        path.append(node)
        turns += 1

    v = _value_boundary(state, root_name, see_opp_hand, profile, plan)
    for nd in path:
        nd.N += 1
        nd.W += v


def _determinize_opponent(manager, me_name: str, rng):
    """`manager` のクローンを返し、**相手の伏せ手札を相手の山札＋手札プールから再サンプリング**する（公平化）。

    自分（`me_name`）の手札・場・山札順は不変（自分の手は実物＝返すプランが実ゲームで合法）。相手の手札枚数は
    保存し、中身だけ「相手のライブラリ（山札＋現手札）からランダムに同数」へ差し替える＝公開情報と整合する
    “ありえる手”＝チート除去。journal 非作動の top-level で呼ぶ前提（plain mutation）。
    """
    clone = manager.clone()
    opp = clone.p2 if clone.p1.name == me_name else clone.p1
    pool = list(opp.hand) + list(opp.deck)
    if not pool:
        return clone
    rng.shuffle(pool)
    n_hand = len(opp.hand)
    new_hand, new_deck = pool[:n_hand], pool[n_hand:]
    opp.hand[:] = new_hand   # JournaledList のスライス代入（top-level＝journal 非作動）
    opp.deck[:] = new_deck
    return clone


def _macro_search_root(root_state, name: str, iters: int, H: int,
                       see_opp_hand: bool, rng, profile=None, plan=None,
                       deadline: Optional[float] = None) -> Optional["_MacroNode"]:
    """`root_state`（`name` の手番境界・決定化済みでもよい）で 1 世界ぶん探索し、ルートノードを返す。

    `deadline`（time.time() の絶対時刻・本番のみ）を渡すと、最低 `MCTS_MIN_ITERS` を確保した上で時刻超過で
    反復を打ち切る＝レイテンシ上限を保証する（決定性は壊れるので tests/自己対戦は None）。
    """
    pa = root_state.pending_actor_action()
    if not pa or pa[0] != name:
        return None
    root = _MacroNode(None, name, terminal=False)
    for i in range(iters):
        if deadline is not None and i >= MCTS_MIN_ITERS and time.time() >= deadline:
            break
        _macro_simulate(root, root_state, name, H, see_opp_hand, rng, profile, plan, deadline)
    return root if root.children else None


def mcts_plan_turn(manager, player, difficulty: str = "hard", rng: Optional[random.Random] = None,
                   iterations: Optional[int] = None, horizon: Optional[int] = None,
                   see_opp_hand: Optional[bool] = None, worlds: Optional[int] = None,
                   profile=None, plan=None, determinize: Optional[bool] = None,
                   deadline_ms: Optional[int] = None) -> List[Dict[str, Any]]:
    """マクロ MCTS で `player` の**このターンの手列（ターンプラン）**を返す（計画キャッシュ的に逐次 replay 可）。

    ルートは `player` の手番境界。子=候補ターンプラン。最善子（最多訪問）の手列を返す。合法手が無ければ空。
    `worlds`>1（公平モード時）は K 通りの決定化世界で探索し、先頭手で集約した最頻・最多訪問プランを返す
    （単一世界の仮定ブレを平均で打ち消す＝真の ISMCTS 近似）。`plan`/`profile` は任意（葉評価の勝ち筋項）。
    `determinize`（指定時はモジュール既定 `MCTS_DETERMINIZE` を上書き）＝公平モード（相手手札を覗かない）。
    `deadline_ms`（本番のみ）を渡すと壁時計で反復を打ち切りレイテンシ上限を保証する（決定性は壊れるので
    tests/自己対戦は None＝固定反復で再現可能）。
    """
    rng = rng or random
    name = player.name
    if see_opp_hand is None:
        see_opp_hand = (difficulty == "hard")
    iters = iterations if iterations is not None else MCTS_MACRO_ITERS
    H = horizon if horizon is not None else MCTS_MACRO_HORIZON
    W = worlds if worlds is not None else MCTS_WORLDS
    det_on = MCTS_DETERMINIZE if determinize is None else determinize
    deadline = (time.time() + deadline_ms / 1000.0) if deadline_ms else None

    pa = manager.pending_actor_action()
    if not pa or pa[0] != name:
        return []

    # 単一世界（既定 or 完全情報）: 1 世界探索して最多訪問プラン。
    if W <= 1 or not det_on:
        root_state = _determinize_opponent(manager, name, rng) if det_on else manager
        root = _macro_search_root(root_state, name, iters, H, see_opp_hand, rng, profile, plan, deadline)
        if root is None:
            return []
        best = max(root.children, key=lambda c: (c.N, c.W / c.N if c.N else 0.0, rng.random()))
        return best.plan

    # 複数世界アンサンブル: 各世界で探索し、ルート子を先頭手 sig で集約（訪問数を合算）。
    per_world = max(1, iters // W)
    tally: Dict[tuple, List[Any]] = {}        # first_sig -> [総訪問, 最良訪問, 最良プラン]
    for wi in range(W):
        if deadline is not None and wi > 0 and time.time() >= deadline:
            break
        det = _determinize_opponent(manager, name, rng)
        root = _macro_search_root(det, name, per_world, H, see_opp_hand, rng, profile, plan, deadline)
        if root is None:
            continue
        for ch in root.children:
            if not ch.plan:
                continue
            fsig = _move_sig(ch.plan[0])
            ent = tally.get(fsig)
            if ent is None:
                tally[fsig] = [ch.N, ch.N, ch.plan]
            else:
                ent[0] += ch.N
                if ch.N > ent[1]:
                    ent[1], ent[2] = ch.N, ch.plan
    if not tally:
        return []
    # 先頭手の総訪問が最大のものを採用＝世界をまたいで最も支持されたターン入り。プランはその先頭手の
    # 最良単一世界プラン（一貫した手列）を返す。
    best_first = max(tally.items(), key=lambda kv: (kv[1][0], kv[1][1], rng.random()))
    return best_first[1][2]


def decide_mcts_macro(manager, player, difficulty: str = "hard", rng: Optional[random.Random] = None,
                      cache: Optional[Dict[str, Any]] = None, iterations: Optional[int] = None,
                      horizon: Optional[int] = None, worlds: Optional[int] = None,
                      profile=None, plan=None, deadline_ms: Optional[int] = None,
                      moves: Optional[List[Dict[str, Any]]] = None) -> Optional[Dict[str, Any]]:
    """マクロ MCTS の**逐次 1 手**インターフェース（`cpu_ai.decide_cached` と同型）。

    `cache={"queue":[...]}` を対局ごとに保持する。queue の次手が現局面で合法なら即返す（replay）。
    空/不正なら `mcts_plan_turn` でこのターンを一括計画して queue 化し先頭を返す。`moves` 指定時は
    候補をそれに限定（guard driver 用・単一手はそのまま）。
    """
    rng = rng or random
    if cache is None:
        cache = {}
    legal = moves if moves is not None else manager.get_legal_actions(player)
    if not legal:
        return None
    if len(legal) == 1:
        cache["queue"] = []
        return legal[0]
    legal_by_sig = {_move_sig(m): m for m in legal}

    q = cache.get("queue")
    if q:
        nxt = q[0]
        if _move_sig(nxt) in legal_by_sig:
            cache["queue"] = q[1:]
            return legal_by_sig[_move_sig(nxt)]
        cache["queue"] = None  # 前提崩れ＝再計画

    tplan = mcts_plan_turn(manager, player, difficulty, rng, iterations=iterations,
                           horizon=horizon, worlds=worlds, profile=profile, plan=plan,
                           deadline_ms=deadline_ms)
    if tplan and _move_sig(tplan[0]) in legal_by_sig:
        cache["queue"] = tplan[1:]
        return legal_by_sig[_move_sig(tplan[0])]
    # 計画が空/先頭不正＝安全側で 1-ply 最良手にフォールバック（稀）。
    cache["queue"] = None
    best = max(legal, key=lambda m: (_score_move_1ply(manager, player.name, m, player.name,
                                                      see_opp_hand=(difficulty == "hard"),
                                                      profile=profile, plan=plan) or float("-inf")))
    return best
