"""ルールモード CPU（AI）の意思決定エンジン（docs/SPEC.md §2.5.2）。

設計:
  - 合法手は `GameManager.get_legal_actions` を単一の真実源として用いる。
  - 各候補手を `GameManager.clone()` 上で適用し、`evaluate` で盤面優劣を採点して選ぶ。
    クローン上では自分側の効果対話を既定解決でドレインしてから採点する。
  - ステートレス（毎ステップ再計画）。ポーリング駆動でも desync に強い。

評価関数（J値理論ベース・docs/SPEC.md §2.5.2）:
  「J値 = 白の枚数 = デッキ残 + トラッシュ」を下げ、相手の J値を上げるゲーム、という
  Jin 氏「J値理論」に整合する形で盤面を採点する。J値を下げる = 黒（手札・ライフ・場・ステージ）
  にカードが多い状態なので、本評価は黒リソースの重み付き和を主軸に、理論の以下の機微を加える:
    - ライフの非線形価値（薄いほど 1 枚の限界価値が跳ね上がる＝45[J] ラインの危険）。
    - 手札のカウンター値（防御リソース＝相手の +1[J] をいなす力）。
    - 場のアクティブキャラ（＝将来のアタック＝相手の J値を上げる圧力）とブロッカー（最終防御）。
  KO・カウンター誘発・ハンデス等の「相手 +1[J]」は、相手側の枚数・パワーが下がることで自然に
  差分へ反映される（明示の J値項は黒リソースと相補で二重計上になるため置かない）。

難易度:
  easy   : ランダム合法手。
  normal : 貪欲 1-ply（evaluate 最良）。
  hard   : フルクローン多 ply 先読み（α-β ＋ ビーム）。自分の連続手を読み、ターン終了後は
           相手の最善応答（ブロック/カウンター/相手ターン）まで min ノードとして辿る。
           探索木内で `winner` に到達する手順を最高評価とすることでリーサルを認識する。

公平性メモ: hard はユーザ選択により「最強」設定で、先読みの相手応答シミュレーションに
クローン上の相手手札（隠れ情報）も用いる。完全な視点マスクを敷く easy/normal とは別方針
（docs/SPEC.md §2.2/§6 参照）。
"""
import random
from typing import Any, Dict, List, Optional, Tuple

# 評価重み（盤面 1000=パワー1段相当に正規化）
W_LIFE = 6000.0          # ライフ 1 枚の基礎価値（最重要）
W_LIFE_LOW = 4000.0      # 希少域（最初の 2 枚）への上乗せ＝非線形・45[J] ラインの危険
W_HAND = 700.0           # 手札 1 枚の基礎価値
W_COUNTER = 0.6          # 手札のカウンター値 1 点あたり（防御リソース）
W_FIELD_COUNT = 1500.0   # 場のキャラ 1 体の存在価値
W_FIELD_POWER = 1.0      # 場の総パワー
W_DON_ACTIVE = 200.0     # アクティブドン!! 1 枚
W_BLOCKER = 1200.0       # ブロッカー 1 体（最終防御）
W_ATTACKER = 400.0       # アクティブキャラ 1 体＝将来のアタック圧（相手 +1[J] 機会）
W_WIN = 1.0e9            # 勝敗

_EPS = 1.0  # これ未満の改善ならターンを畳む（無限ループ防止＋無意味手の抑制）
_DRAIN_LIMIT = 12        # クローン上で自分側対話を解決する最大回数

# hard 先読みのパラメータ。clone（≈4-5ms）が支配的なので、1 手あたりのレイテンシ（≈1 秒）に
# 収まるよう NODE_BUDGET でクローン総数を厳しく制限する（DEPTH/BEAM は探索の形を決める）。
HARD_DEPTH = 5           # 探索する意思決定ノードの ply 数
HARD_BEAM = 3            # 各ノードで展開する候補手の数（1-ply 評価上位 K）
HARD_NODE_BUDGET = 250   # 1 回の decide で生成するクローン総数の上限（レイテンシ/暴走防止）


def _other(manager, name: str):
    return manager.p2 if manager.p1.name == name else manager.p1


def _player_by_name(manager, name: str):
    return manager.p1 if manager.p1.name == name else manager.p2


def _side_score(p, is_turn: bool) -> float:
    """1 プレイヤー側の素点（J値理論ベース：黒リソースの重み付き和）。"""
    score = 0.0

    # ライフ: 非線形（薄いほど 1 枚の限界価値が高い）。最初の 2 枚に厚く上乗せする。
    life_n = len(p.life)
    score += life_n * W_LIFE
    score += min(life_n, 2) * W_LIFE_LOW

    # 手札: 枚数 ＋ カウンター値（防御に回せる力＝相手の +1[J] を打ち消す資源）。
    score += len(p.hand) * W_HAND
    for c in p.hand:
        try:
            score += (c.current_counter or 0) * W_COUNTER
        except Exception:
            pass

    # ドン!!（アクティブ）。
    score += len(p.don_active) * W_DON_ACTIVE

    # 場のキャラ: 存在価値 ＋ パワー ＋ ブロッカー（最終防御）＋ アクティブ＝将来の攻め圧。
    score += len(p.field) * W_FIELD_COUNT
    for c in p.field:
        try:
            score += c.get_power(is_turn) * W_FIELD_POWER
        except Exception:
            score += (c.master.power or 0) * W_FIELD_POWER
        if not c.is_rest:
            score += W_ATTACKER
            if c.has_keyword("ブロッカー"):
                score += W_BLOCKER
    return score


def evaluate(manager, me_name: str) -> float:
    """`me_name` 視点の盤面優劣スコア（高いほど自分有利）。"""
    if manager.winner == me_name:
        return W_WIN
    if manager.winner is not None:
        return -W_WIN
    me = _player_by_name(manager, me_name)
    opp = _other(manager, me_name)
    is_my_turn = manager.turn_player.name == me_name
    return _side_score(me, is_my_turn) - _side_score(opp, not is_my_turn)


def _pending_keys():
    from . import action_api
    pending_props = action_api.CONST.get('PENDING_REQUEST_PROPERTIES', {})
    return pending_props.get('PLAYER_ID', 'player_id'), pending_props.get('ACTION', 'action')


def _drain_own_interactions(manager, actor_name: str) -> None:
    """クローン上で actor 側の効果対話を既定解決でドレインする（採点を安定させるため）。

    相手の意思決定（ブロック/カウンター等）は解決しない（相手に委ねる）。
    """
    from . import action_api
    KEY_PID, KEY_ACTION = _pending_keys()
    for _ in range(_DRAIN_LIMIT):
        pending = manager.get_pending_request()
        if not pending or pending[KEY_PID] != actor_name:
            return
        action = pending[KEY_ACTION]
        # メイン/マリガン/戦闘は「意思決定」なのでドレインしない（呼び出し側が1手として扱う）。
        if action in ("MAIN_ACTION", "MULLIGAN", "SELECT_BLOCKER", "SELECT_COUNTER"):
            return
        payload = manager.default_interaction_payload(pending)
        actor = _player_by_name(manager, actor_name)
        manager.action_events = []
        try:
            action_api.apply_game_action(manager, actor, action_api.ACT_RESOLVE_SELECTION, payload)
        except Exception:
            return


def _apply_clone(manager, actor_name: str, move: Dict[str, Any]):
    """move を新しいクローンへ適用し、actor 側の対話をドレインしたクローンを返す。

    シミュレーションが例外を出す手は None を返す（呼び出し側で除外する）。
    """
    from . import action_api
    clone = manager.clone()
    actor = _player_by_name(clone, actor_name)
    clone.action_events = []
    try:
        if move["kind"] == "battle":
            action_api.apply_battle_action(clone, actor, move["action_type"], move.get("card_uuid"))
        else:
            action_api.apply_game_action(clone, actor, move["action_type"], move.get("payload", {}))
        _drain_own_interactions(clone, actor_name)
    except Exception:
        return None
    return clone


def _simulate_and_eval(manager, actor_name: str, move: Dict[str, Any]) -> float:
    """move をクローン上で適用し、actor 側の対話をドレインしてから評価する（1-ply）。"""
    clone = _apply_clone(manager, actor_name, move)
    if clone is None:
        return float("-inf")
    return evaluate(clone, actor_name)


def _search(manager, root_name: str, depth: int, alpha: float, beta: float,
            budget: List[int], ply: int = 0) -> float:
    """α-β ＋ ビームのフルクローン先読み。`root_name` 視点の最善到達値を返す。

    手番が root のノードは max（自分の最善手）、相手のノードは min（相手の最善応答）。
    探索木内で `winner` に到達した手順は ±(W_WIN − ply) となり、リーサル認識として機能する
    （ply 割引により「より早く勝つ／より遅く負ける」手順が優先され、最短の止めを選ぶ）。
    相手応答にはクローン上の相手手札（隠れ情報）を用いる＝hard の「最強」方針。
    """
    if manager.winner is not None:
        return (W_WIN - ply) if manager.winner == root_name else -(W_WIN - ply)
    if depth <= 0 or budget[0] <= 0:
        return evaluate(manager, root_name)

    KEY_PID, _ = _pending_keys()
    pending = manager.get_pending_request()
    if not pending:
        return evaluate(manager, root_name)
    actor_name = pending[KEY_PID]
    actor = _player_by_name(manager, actor_name)
    moves = manager.get_legal_actions(actor)
    if not moves:
        return evaluate(manager, root_name)
    is_max = (actor_name == root_name)

    # 子ノードを生成し、1-ply 評価でビーム選別（best-first で α-β の枝刈り効率を上げる）。
    children: List[Tuple[float, Any]] = []
    for m in moves:
        if budget[0] <= 0:
            break
        budget[0] -= 1
        child = _apply_clone(manager, actor_name, m)
        if child is None:
            continue
        children.append((evaluate(child, root_name), child))
    if not children:
        return evaluate(manager, root_name)
    children.sort(key=lambda x: x[0], reverse=is_max)
    children = children[:HARD_BEAM]

    if is_max:
        value = float("-inf")
        for _leaf, child in children:
            value = max(value, _search(child, root_name, depth - 1, alpha, beta, budget, ply + 1))
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value
    else:
        value = float("inf")
        for _leaf, child in children:
            value = min(value, _search(child, root_name, depth - 1, alpha, beta, budget, ply + 1))
            beta = min(beta, value)
            if alpha >= beta:
                break
        return value


def _scored_hard(manager, name: str, moves: List[Dict[str, Any]]) -> List[Tuple[float, Dict[str, Any]]]:
    """hard: 各ルート手を 1 手適用し、子局面から多 ply 先読みした値で採点する。"""
    budget = [HARD_NODE_BUDGET]
    out: List[Tuple[float, Dict[str, Any]]] = []
    for m in moves:
        child = _apply_clone(manager, name, m)
        if child is None:
            out.append((float("-inf"), m))
            continue
        budget[0] -= 1
        # ルート手で 1 手消費しているので ply=1 から探索する（早い勝ちを優先）。
        v = _search(child, name, HARD_DEPTH - 1, float("-inf"), float("inf"), budget, ply=1)
        out.append((v, m))
    return out


# 1 ターン内に CPU が取れる手の総数上限（暴走/無限ループの最終防壁）。
TURN_ACTION_CAP = 60
# 同一の起動効果/ドン付与をこのターン内に繰り返してよい回数の上限。
REPEAT_CAP = 3


def _move_sig(move: Dict[str, Any]) -> tuple:
    payload = move.get("payload") or {}
    return (move.get("action_type"), payload.get("uuid") or move.get("card_uuid"),
            tuple(payload.get("target_ids", []) or []))


def decide(manager, player, difficulty: str = "normal", rng: Optional[random.Random] = None,
           moves: Optional[List[Dict[str, Any]]] = None) -> Optional[Dict[str, Any]]:
    """`player` が取るべき次の 1 手を返す（合法手が無ければ None）。

    `moves` を渡すとその候補集合から選ぶ（ガード driver が絞り込んだ手を渡す用途）。
    """
    rng = rng or random
    if moves is None:
        moves = manager.get_legal_actions(player)
    if not moves:
        return None
    if len(moves) == 1:
        return moves[0]

    if difficulty == "easy":
        return rng.choice(moves)

    name = player.name
    end_move = next((m for m in moves if m.get("action_type") == "TURN_END"), None)

    if difficulty == "hard":
        scored = _scored_hard(manager, name, moves)
    else:
        scored = [(_simulate_and_eval(manager, name, m), m) for m in moves]
    # 同点はランダムタイブレーク（決定論にしたい場合は呼び出し側で seed 済み rng を渡す）。
    rng.shuffle(scored)
    best_score, best_move = max(scored, key=lambda x: x[0])

    # 非ターン終了手が end を有意に上回らなければターンを畳む（進行保証）。
    if end_move is not None and best_move is not end_move:
        end_score = next((s for s, m in scored if m is end_move), None)
        if end_score is not None and best_score <= end_score + _EPS:
            return end_move
    return best_move


def decide_guarded(manager, player, difficulty: str = "normal", rng: Optional[random.Random] = None,
                   mem: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """ターン内メモリ `mem` を用いた暴走防止つきの意思決定。

    `mem` は呼び出し側が対局ごとに保持する dict（ステートレスな /cpu/step でも CPU_GAMES に
    保持して渡す）。同一ターン内で:
      - 取った手の総数が TURN_ACTION_CAP を超えたら強制 TURN_END
      - 同じ起動効果/ドン付与を REPEAT_CAP 回行ったら候補から除外（イガラム等の無限ループ防止）
    これにより「効果に per-turn 制限が無い/付け忘れ」のカードでも CPU ターンが必ず終わる。
    """
    rng = rng or random
    if mem is None:
        mem = {}
    if mem.get("turn") != manager.turn_count:
        mem["turn"] = manager.turn_count
        mem["counts"] = {}
        mem["total"] = 0

    moves = manager.get_legal_actions(player)
    if not moves:
        return None
    end_move = next((m for m in moves if m.get("action_type") == "TURN_END"), None)

    # 総数キャップ: 上限超過ならターンを畳む（畳めない＝対話中等なら通常選択）。
    if end_move is not None and mem.get("total", 0) >= TURN_ACTION_CAP:
        return end_move

    # 繰り返しキャップ: 起動効果/ドン付与の同一手を上限まで使い切ったら除外する。
    counts = mem.get("counts", {})
    repeatable = {"ACTIVATE_MAIN", "ATTACH_DON"}
    filtered = [m for m in moves
                if not (m.get("action_type") in repeatable and counts.get(_move_sig(m), 0) >= REPEAT_CAP)]
    if not filtered:
        filtered = [end_move] if end_move is not None else moves

    move = decide(manager, player, difficulty, rng, moves=filtered)
    if move is not None:
        sig = _move_sig(move)
        counts[sig] = counts.get(sig, 0) + 1
        mem["counts"] = counts
        mem["total"] = mem.get("total", 0) + 1
    return move
