"""効果セマンティクス特徴（EffFeat）: カードDBの効果ASTから決定的に計算する固定特徴テーブル。

docs/reports/effect_semantics_v3_plan_20260708.md §1（改訂1）。カード1枚 →
`[能力スロット1 (51) | 能力スロット2 (51) | カード静的 (14)] = 116次元`（float32・0/1）。
学習しない＝新カードはASTから自動でゼロショット特徴化される（埋め込みの共起転移と違い**意味**が転移する）。
決定的（ASTのみ参照・RNG不使用）。PAD行(idx=0)は全ゼロ。vocab と同様 db スナップショットと対で固定し、
ネット側で npz に保存する（DBドリフトによるサイレント破壊を防ぐ）。

設計の一次資料: docs/reports/effect_feature_inventory_20260708.md（全2652枚の実測頻度・
BUFFのstatus×値スケール判別・ATTACH_DON=99の全体センチネル等の解決済み論点を含む）。
"""
import numpy as np

# ---- 能力スロット（51次元）のレイアウト --------------------------------------
TRIGGERS = ["ON_PLAY", "ACTIVATE_MAIN", "TRIGGER", "PASSIVE", "ON_ATTACK", "COUNTER",
            "ON_KO", "YOUR_TURN", "OPPONENT_TURN", "ON_OPP_ATTACK", "TURN_END", "ON_BLOCK"]
# 効果連鎖（effect + sub_effect）に現れる ActionType の主要クラス。BUFF は別ブロック。
ACTIONS = ["KO", "PLAY_CARD", "DRAW", "BOUNCE", "DECK_BOTTOM", "REST", "ACTIVE",
           "RAMP_DON", "ATTACH_DON", "GRANT_KEYWORD", "DISCARD", "TRASH_FROM_DECK",
           "HEAL", "ATTACK_DISABLE", "PREVENT_LEAVE", "NEGATE", "VICTORY"]
_ACTION_ALIAS = {  # enum名 → ACTIONS クラス（それ以外は OTHER）
    "KEYWORD": "GRANT_KEYWORD", "LIFE_RECOVER": "HEAL",
    "NEGATE_EFFECT": "NEGATE", "DISABLE_ABILITY": "NEGATE",
}
N_TRIG = len(TRIGGERS) + 1          # +OTHER = 13
N_ACT = len(ACTIONS) + 1            # +OTHER = 18
# BUFF細分6: [パワー+~1k, +~2k, +3k以上, パワーデバフ, コスト操作, パワー上書き]
# misc2: [DRAW2枚以上, ATTACH_DON全体(=99センチネル)]
# condition6: [HAS_DON, TURN_LIMIT, リソース閾値, ロック, 履歴, その他条件]
# duration1: [持続効果あり]  target2: [相手対象, 自分対象]  cost3: [ドン系, 手札系, 他]
ABILITY_DIM = N_TRIG + N_ACT + 6 + 2 + 6 + 1 + 2 + 3   # = 51
STATIC_DIM = 4 + 2 + 4 + 4                              # 種別4+カウンター2+コスト帯4+印刷KW4 = 14
FEATURE_DIM = 2 * ABILITY_DIM + STATIC_DIM              # = 116

_RESOURCE_CONDS = {"LIFE_COUNT", "HAND_COUNT", "FIELD_COUNT", "TRASH_COUNT", "DECK_COUNT",
                   "DON_COUNT", "LIFE_HAND_SUM", "FIELD_COST_SUM", "RESTED_COUNT",
                   "LIFE_COUNT_COMPARE", "HAND_COUNT_COMPARE", "DON_COUNT_COMPARE",
                   "FIELD_COUNT_COMPARE", "LIFE_COUNT_BOTH"}
_LOCK_CONDS = {"LEADER_TRAIT", "LEADER_NAME", "LEADER_COLOR", "LEADER_ATTRIBUTE",
               "HAS_TRAIT", "HAS_ATTRIBUTE", "HAS_UNIT", "HAS_CHARACTER", "FIELD_ALL_TRAIT"}
_HISTORY_CONDS = {"EVENT_THIS_TURN", "CHAR_KOED_THIS_TURN", "OPPONENT_REMOVAL", "PREV_ACTION"}
_STRUCT_CONDS = {"AND", "OR", "NOT", "NONE"}   # 構造ノード＝特徴にしない（子を辿るだけ）
_DON_COSTS = {"RETURN_DON", "REST_DON"}
_PRINTED_KEYWORDS = ["ブロッカー", "速攻", "ダブルアタック", "バニッシュ"]


def _walk_conditions(cond, out):
    """条件ツリーを再帰し ConditionType 名の集合を out へ集める（構造ノードは辿るだけ）。"""
    if cond is None:
        return
    t = getattr(cond, "type", None)
    if t is not None and t.name not in _STRUCT_CONDS:
        out.add(t.name)
    for a in (getattr(cond, "args", None) or []):
        _walk_conditions(a, out)


def _walk_actions(node, out):
    """効果木の GameAction を out(list) へ集める。

    EffectNode は3種: GameAction（.type・.sub_effect 連鎖）／Sequence（.actions 列）／
    Choice（.options 分岐）。Sequence/Choice は子を辿るだけ（それ自体は特徴にしない）。
    """
    if node is None:
        return
    for child in getattr(node, "actions", None) or []:      # Sequence
        _walk_actions(child, out)
    for child in getattr(node, "options", None) or []:      # Choice
        _walk_actions(child, out)
    if getattr(node, "type", None) is not None:             # GameAction
        out.append(node)
        _walk_actions(getattr(node, "sub_effect", None), out)


def _ability_vec(ability):
    """能力1つ → 51次元（レイアウトは冒頭コメント）。"""
    v = np.zeros(ABILITY_DIM, dtype=np.float32)
    off = 0
    # trigger
    tt = getattr(ability, "trigger", None)
    name = tt.name if tt is not None else None
    v[off + (TRIGGERS.index(name) if name in TRIGGERS else N_TRIG - 1)] = 1.0
    off += N_TRIG

    acts = []
    _walk_actions(getattr(ability, "effect", None), acts)
    persist = False
    opp_target = self_target = False
    for a in acts:
        t = a.type.name
        d = str(getattr(a, "duration", "INSTANT") or "INSTANT")
        if d != "INSTANT":
            persist = True
        tq = getattr(a, "target", None)
        pl = getattr(getattr(tq, "player", None), "name", None) if tq is not None else None
        if pl in ("OPPONENT", "ALL"):
            opp_target = True
        if pl in ("SELF", "OWNER", "ALL"):
            self_target = True
        base = getattr(getattr(a, "value", None), "base", 0) or 0
        if t == "BUFF":
            # 判別= status×値スケール（inventory §5-1）: |値|<100 はコスト操作系。
            status = str(getattr(a, "status", None) or "")
            if "POWER_OVERRIDE" in status:
                v[off + N_ACT + 5] = 1.0
            elif "BLOCKER_DISABLE" in status:
                v[off + ACTIONS.index("ATTACK_DISABLE")] = 1.0
            elif "COST" in status or abs(base) < 100:
                v[off + N_ACT + 4] = 1.0
            elif base < 0:
                v[off + N_ACT + 3] = 1.0
            else:
                bucket = 0 if base <= 1000 else (1 if base <= 2000 else 2)
                v[off + N_ACT + bucket] = 1.0
            continue
        cls = _ACTION_ALIAS.get(t, t)
        v[off + (ACTIONS.index(cls) if cls in ACTIONS else N_ACT - 1)] = 1.0
        if t == "DRAW" and base >= 2:
            v[off + N_ACT + 6] = 1.0
        if t == "ATTACH_DON" and base >= 99:
            v[off + N_ACT + 7] = 1.0
    off += N_ACT + 6 + 2

    conds = set()
    _walk_conditions(getattr(ability, "condition", None), conds)
    v[off + 0] = 1.0 if "HAS_DON" in conds else 0.0
    v[off + 1] = 1.0 if "TURN_LIMIT" in conds else 0.0
    v[off + 2] = 1.0 if conds & _RESOURCE_CONDS else 0.0
    v[off + 3] = 1.0 if conds & _LOCK_CONDS else 0.0
    v[off + 4] = 1.0 if conds & _HISTORY_CONDS else 0.0
    known = {"HAS_DON", "TURN_LIMIT"} | _RESOURCE_CONDS | _LOCK_CONDS | _HISTORY_CONDS
    v[off + 5] = 1.0 if (conds - known) else 0.0
    off += 6

    v[off] = 1.0 if persist else 0.0
    off += 1
    v[off + 0] = 1.0 if opp_target else 0.0
    v[off + 1] = 1.0 if self_target else 0.0
    off += 2

    costs = []
    _walk_actions(getattr(ability, "cost", None), costs)
    for a in costs:
        t = a.type.name
        zone = getattr(getattr(getattr(a, "target", None), "zone", None), "name", None)
        if t in _DON_COSTS:
            v[off + 0] = 1.0
        elif t == "DISCARD" or zone == "HAND":
            v[off + 1] = 1.0
        else:
            v[off + 2] = 1.0
    return v


def _static_vec(card):
    """カード静的14次元: 種別4 + カウンター{1000,2000}2 + コスト帯{0-2,3-4,5-6,7+}4 + 印刷KW4。"""
    v = np.zeros(STATIC_DIM, dtype=np.float32)
    tname = card.type.name
    for i, t in enumerate(("LEADER", "CHARACTER", "EVENT", "STAGE")):
        if tname == t:
            v[i] = 1.0
    counter = getattr(card, "counter", 0) or 0
    if counter >= 2000:
        v[5] = 1.0
    elif counter >= 1000:
        v[4] = 1.0
    cost = getattr(card, "cost", 0) or 0
    v[6 + min(3, (0 if cost <= 2 else 1 if cost <= 4 else 2 if cost <= 6 else 3))] = 1.0
    kws = getattr(card, "keywords", None) or set()
    for i, k in enumerate(_PRINTED_KEYWORDS):
        if k in kws:
            v[10 + i] = 1.0
    return v


def card_features(card):
    """カード1枚 → FEATURE_DIM(116) ベクトル。能力はパース順で slot1/slot2（3つ目は slot2 へOR併合）。"""
    v = np.zeros(FEATURE_DIM, dtype=np.float32)
    abilities = list(getattr(card, "abilities", None) or [])
    if abilities:
        v[:ABILITY_DIM] = _ability_vec(abilities[0])
    for ab in abilities[1:]:
        v[ABILITY_DIM:2 * ABILITY_DIM] = np.maximum(v[ABILITY_DIM:2 * ABILITY_DIM], _ability_vec(ab))
    v[2 * ABILITY_DIM:] = _static_vec(card)
    return v


def build_efffeat(db, vocab):
    """vocab（card_id→idx・0=PAD）に整合した EffFeat テーブル [vocab+1, FEATURE_DIM] を返す。"""
    table = np.zeros((len(vocab) + 1, FEATURE_DIM), dtype=np.float32)
    for cid, idx in vocab.items():
        c = db.get_card(cid)
        if c is not None:
            table[idx] = card_features(c)
    return table
