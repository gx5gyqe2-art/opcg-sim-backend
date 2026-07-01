"""OPCG アクションの正準符号化（P3 policy head 用・docs/.../cpu_rl_pilot_plan_20260629.md P3）。

heterogeneous な合法手（攻撃/プレイ/ドン付与/カウンター/効果起動…）を**ポインタ/marginal方式**で扱う:
巨大疎な直積空間を作らず、各合法手を**固定長の特徴ベクトル**へ符号化し、policy は
[状態埋め込み, 各手の特徴] からスコアを出して合法手上で softmax する（可変個の手に自然対応）。

action_features = [action_type one-hot] ++ [関与カードの特徴(rl_encoder._char_feats と同型)] ++ [所有者flag, 対象有flag]。
列挙した action_type（self-play 実測）: MULLIGAN/KEEP_HAND/TURN_END/PASS/PLAY/ATTACK/
ATTACH_DON/SELECT_COUNTER/SELECT_BLOCKER/ACTIVATE_MAIN/RESOLVE_EFFECT_SELECTION。
"""
import numpy as np

from . import encoder as E

ACTION_TYPES = [
    "MULLIGAN", "KEEP_HAND", "TURN_END", "PASS", "PLAY", "ATTACK",
    "ATTACH_DON", "SELECT_COUNTER", "SELECT_BLOCKER", "ACTIVATE_MAIN",
    "RESOLVE_EFFECT_SELECTION",
]
_AT_IDX = {t: i for i, t in enumerate(ACTION_TYPES)}
_CARD_FEAT = E.PER_CHAR        # _char_feats の次元
# [type one-hot] + [card feats] + [owner_mine, has_card, has_target]
ACTION_DIM = len(ACTION_TYPES) + _CARD_FEAT + 3


def action_key(move):
    """探索木のための hashable な手の同一性キー（dict は非hashable）。"""
    at = move.get("action_type")
    payload = move.get("payload") or {}
    uuid = move.get("card_uuid") or payload.get("uuid")
    tgt = payload.get("target_ids")
    tgt = tuple(tgt) if isinstance(tgt, (list, tuple)) else tgt
    sel = payload.get("selected_uuids")
    sel = tuple(sel) if isinstance(sel, (list, tuple)) else sel
    return (move.get("kind"), at, uuid, tgt, sel, payload.get("index"),
            payload.get("position"), payload.get("declared_value"), payload.get("accepted"))


def _find_card(manager, uuid):
    if not uuid:
        return None, None
    for owner, pl in (("me", manager.p1), ("opp", manager.p2)):
        for zone in (pl.field, pl.hand, [pl.leader] if pl.leader else []):
            for c in zone:
                if c is not None and getattr(c, "uuid", None) == uuid:
                    return c, pl
    return None, None


def action_features(manager, move, me_name):
    """手番 me_name 視点で 1 手を ACTION_DIM 次元へ符号化（関与カードは self/opp を区別）。"""
    f = np.zeros(ACTION_DIM, dtype=np.float32)
    at = move.get("action_type")
    if at in _AT_IDX:
        f[_AT_IDX[at]] = 1.0
    payload = move.get("payload") or {}
    uuid = move.get("card_uuid") or payload.get("uuid")
    card, owner_pl = _find_card(manager, uuid)
    base = len(ACTION_TYPES)
    if card is not None:
        try:
            f[base:base + _CARD_FEAT] = E._char_feats(card)
        except Exception:
            pass
        f[base + _CARD_FEAT] = 1.0 if (owner_pl is not None and owner_pl.name == me_name) else 0.0
        f[base + _CARD_FEAT + 1] = 1.0   # has_card
    if payload.get("target_ids"):
        f[base + _CARD_FEAT + 2] = 1.0   # has_target
    return f


def legal_action_matrix(manager, moves, me_name):
    """合法手リスト → [K, ACTION_DIM] 行列（policy 入力）。"""
    if not moves:
        return np.zeros((0, ACTION_DIM), dtype=np.float32)
    return np.stack([action_features(manager, mv, me_name) for mv in moves])
