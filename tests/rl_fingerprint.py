"""カード“効果フィンガープリント”エンコーダ（汎化学習の中核・pre-flight プローブ用）。

docs/reports/cpu_rl_generalization_plan_20260701.md ①。狙いは **カードを識別子(card_id)ではなく
“振る舞い”で表す** こと。既存 encoder.py の card_idx（ID埋め込み＝丸暗記）は分布外カードで未訓練
ベクトル＝ゴミ入力になり V が崩れる。パーサが吐く構造化効果（TriggerType/ActionType/TargetQuery）を
固定長の behavior ベクトルへ落とせば、「コスト5以下をKOして1回復する未知のカード」は同じ振る舞いの
既知カードと同じ入力＝分布内になる。

本モジュールは研究用（tests/）。プローブで汎化retention を確認してから本番 encoder へ移植する。
決定的（カード静的属性のみ参照・RNG不使用）。numpy 実装。
"""
import numpy as np

# --- フィンガープリントの語彙（固定・順序が次元を決める）---
COLORS = ["赤", "緑", "青", "紫", "黒", "黄"]
TYPES = ["LEADER", "CHARACTER", "EVENT", "STAGE"]
KEYWORDS = ["ブロッカー", "速攻", "ダブルアタック", "バニッシュ"]

# 意味のあるトリガー種別（存在ビット）。稀種別は UNKNOWN 落ちで無視。
TRIGGERS = ["ON_PLAY", "ON_ATTACK", "ON_KO", "ACTIVATE_MAIN", "TRIGGER",
            "ON_OPP_ATTACK", "OPPONENT_ATTACK", "ON_BLOCK", "TURN_END",
            "PASSIVE", "YOUR_TURN", "OPPONENT_TURN", "ON_LEAVE", "GAME_START"]

# アクション種別を意味バケットへ集約（存在ビット）。列挙の細部より“何をする効果か”を残す。
ACTION_BUCKETS = {
    "removal":   ["KO"],
    "rest_lock": ["REST", "FREEZE", "LOCK", "ATTACK_DISABLE", "PREVENT_REST"],
    "bounce":    ["MOVE_CARD", "DECK_BOTTOM", "DECK_TOP"],
    "draw":      ["DRAW"],
    "discard":   ["DISCARD", "TRASH_FROM_DECK"],
    "search":    ["LOOK", "LOOK_LIFE", "REVEAL"],
    "life":      ["LIFE_RECOVER", "LIFE_MANIPULATE", "FACE_UP_LIFE", "ORDER_LIFE"],
    "power":     ["BP_BUFF", "SET_BASE_POWER", "SWAP_POWER"],
    "cost_mod":  ["COST_BUFF", "COST_CHANGE", "SET_COST"],
    "negate":    ["NEGATE_EFFECT", "DISABLE_ABILITY"],
    "keyword":   ["KEYWORD", "GRANT_KEYWORD", "GRANT_EFFECT"],
    "play_in":   ["PLAY_CARD", "EXECUTE_MAIN_EFFECT"],
    "don":       ["ATTACH_DON", "RAMP_DON", "REST_DON", "RETURN_DON", "FREEZE_DON"],
    "redirect":  ["REDIRECT_ATTACK"],
    "big":       ["VICTORY", "EXTRA_TURN", "DEAL_DAMAGE", "DAMAGE"],
}
ACTION_KEYS = list(ACTION_BUCKETS.keys())

# 静的 numeric: cost, power, counter, life, n_abilities, has_ability_cost, any_up_to
N_STATIC = 7
CARD_DIM = (N_STATIC + len(COLORS) + len(TYPES) + len(KEYWORDS)
            + len(TRIGGERS) + len(ACTION_KEYS))


def _enum_name(x):
    return getattr(x, "name", str(x))


def _iter_actions(node):
    """Sequence/Branch/GameAction を再帰的に辿って GameAction を列挙する。"""
    if node is None:
        return
    acts = getattr(node, "actions", None)          # Sequence
    if acts is not None:
        for a in acts:
            yield from _iter_actions(a)
        return
    # Branch: 条件分岐の各枝
    for attr in ("then_branch", "else_branch", "then", "else_", "branches"):
        b = getattr(node, attr, None)
        if b is not None:
            if isinstance(b, (list, tuple)):
                for x in b:
                    yield from _iter_actions(x)
            else:
                yield from _iter_actions(b)
    if getattr(node, "type", None) is not None:    # GameAction 本体
        yield node
    sub = getattr(node, "sub_effect", None)
    if sub is not None:
        yield from _iter_actions(sub)


def card_fingerprint(master):
    """カード master → 振る舞いベクトル[CARD_DIM]（float32）。"""
    f = np.zeros(CARD_DIM, dtype=np.float32)
    if master is None:
        return f
    i = 0
    # 静的 numeric
    f[0] = float(getattr(master, "cost", 0) or 0) / 10.0
    f[1] = float(getattr(master, "power", 0) or 0) / 10000.0
    f[2] = float(getattr(master, "counter", 0) or 0) / 2000.0
    f[3] = float(getattr(master, "life", 0) or 0) / 5.0
    abilities = list(getattr(master, "abilities", []) or [])
    f[4] = min(len(abilities), 4) / 4.0
    i = N_STATIC
    # 色 multi-hot（colors は Color enum なので .value で比較）
    cols = {getattr(x, "value", x) for x in (getattr(master, "colors", []) or [])}
    for j, c in enumerate(COLORS):
        if c in cols:
            f[i + j] = 1.0
    i += len(COLORS)
    # 種別 one-hot
    tname = _enum_name(getattr(master, "type", None))
    for j, t in enumerate(TYPES):
        if tname == t:
            f[i + j] = 1.0
    i += len(TYPES)
    # キーワード
    for j, kw in enumerate(KEYWORDS):
        try:
            if any(kw in (getattr(k, "value", str(k)) if not isinstance(k, str) else k)
                   for k in (getattr(master, "keywords", []) or [])):
                f[i + j] = 1.0
        except Exception:
            pass
    i += len(KEYWORDS)
    # トリガー種別 & アクションバケット（全 ability を走査）
    trig_hit = set()
    act_names = set()
    has_cost = False
    any_up_to = False
    for ab in abilities:
        trig_hit.add(_enum_name(getattr(ab, "trigger", None)))
        if getattr(ab, "cost", None) is not None:
            has_cost = True
        for act in _iter_actions(getattr(ab, "effect", None)):
            act_names.add(_enum_name(getattr(act, "type", None)))
            tq = getattr(act, "target", None)
            if tq is not None and getattr(tq, "is_up_to", False):
                any_up_to = True
    for j, tr in enumerate(TRIGGERS):
        if tr in trig_hit:
            f[i + j] = 1.0
    i += len(TRIGGERS)
    for j, key in enumerate(ACTION_KEYS):
        if any(a in act_names for a in ACTION_BUCKETS[key]):
            f[i + j] = 1.0
    i += len(ACTION_KEYS)
    f[5] = 1.0 if has_cost else 0.0
    f[6] = 1.0 if any_up_to else 0.0
    return f


def build_fingerprints(db):
    """card_id → fingerprint[CARD_DIM] の辞書（決定的）。"""
    fp = {}
    for cid in db.raw_db.keys():
        m = db.get_card(cid)
        if m is not None:
            fp[cid] = card_fingerprint(m)
    return fp
