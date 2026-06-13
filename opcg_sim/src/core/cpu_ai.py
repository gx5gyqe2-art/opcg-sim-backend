"""ルールモード CPU（AI）の意思決定エンジン（docs/CPU_BATTLE_PLAN.md §2.3/2.4/2.5）。

設計:
  - 合法手は `GameManager.get_legal_actions` を単一の真実源として用いる。
  - 各候補手を `GameManager.clone()` 上で適用し、`evaluate` で盤面優劣を採点して選ぶ
    （1-ply 先読み）。クローン上では自分側の効果対話を既定解決でドレインしてから採点する。
  - ステートレス（毎ステップ再計画）。ポーリング駆動でも desync に強い。

難易度:
  easy   : ランダム合法手
  normal : 貪欲 1-ply（evaluate 最良）
  hard   : 貪欲 1-ply ＋ 攻撃時の被カウンター/反撃を考慮した評価（より強い重み）

公平性メモ: 現状クローンは相手手札も含むため理論上は隠れ情報を参照しうる。評価関数は
公開情報（盤面・ライフ・手札枚数）主体にとどめ、相手手札の中身は読まない方針。完全な
視点マスクは将来拡張（§2.2/§6）。
"""
import random
from typing import Any, Dict, List, Optional

from ..models.enums import CardType

# 評価重み（盤面 1000=パワー1段相当に正規化）
W_LIFE = 6000.0          # ライフ 1 枚の価値（最重要）
W_HAND = 800.0           # 手札 1 枚
W_FIELD_COUNT = 1500.0   # 場のキャラ 1 体の存在価値
W_FIELD_POWER = 1.0      # 場の総パワー
W_DON_ACTIVE = 200.0     # アクティブドン!! 1 枚
W_BLOCKER = 1200.0       # ブロッカー 1 体
W_WIN = 1.0e9            # 勝敗

_EPS = 1.0  # これ未満の改善ならターンを畳む（無限ループ防止＋無意味手の抑制）
_DRAIN_LIMIT = 12        # クローン上で自分側対話を解決する最大回数


def _other(manager, name: str):
    return manager.p2 if manager.p1.name == name else manager.p1


def _player_by_name(manager, name: str):
    return manager.p1 if manager.p1.name == name else manager.p2


def _side_score(p, is_turn: bool) -> float:
    """1 プレイヤー側の素点。"""
    score = 0.0
    score += len(p.life) * W_LIFE
    score += len(p.hand) * W_HAND
    score += len(p.field) * W_FIELD_COUNT
    score += len(p.don_active) * W_DON_ACTIVE
    for c in p.field:
        try:
            score += c.get_power(is_turn) * W_FIELD_POWER
        except Exception:
            score += (c.master.power or 0) * W_FIELD_POWER
        if c.has_keyword("ブロッカー") and not c.is_rest:
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


def _drain_own_interactions(manager, actor_name: str) -> None:
    """クローン上で actor 側の効果対話を既定解決でドレインする（採点を安定させるため）。

    相手の意思決定（ブロック/カウンター等）は解決しない（相手に委ねる）。
    """
    from . import action_api
    pending_props = action_api.CONST.get('PENDING_REQUEST_PROPERTIES', {})
    KEY_PID = pending_props.get('PLAYER_ID', 'player_id')
    KEY_ACTION = pending_props.get('ACTION', 'action')
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


def _simulate_and_eval(manager, actor_name: str, move: Dict[str, Any]) -> float:
    """move をクローン上で適用し、actor 側の対話をドレインしてから評価する。

    シミュレーションが例外を出す手は選ばない（-inf）。
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
        return float("-inf")
    return evaluate(clone, actor_name)


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

    scored = [(_simulate_and_eval(manager, name, m), m) for m in moves]
    # 同点はランダムタイブレーク（決定論にしたい場合は呼び出し側で seed 済み rng を渡す）。
    rng.shuffle(scored)
    best_score, best_move = max(scored, key=lambda x: x[0])

    if end_move is not None:
        end_score = _simulate_and_eval(manager, name, end_move)
        # 非ターン終了手が end を有意に上回らなければターンを畳む（進行保証）。
        if best_move is end_move or best_score <= end_score + _EPS:
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
