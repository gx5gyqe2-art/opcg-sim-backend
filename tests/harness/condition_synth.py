"""H-5: 条件/コスト充足盤面の合成ハーネス。

effect_coverage / quality_map は汎用盤面で能力を発動する。そのため
**ゲート条件が成立しない**（COND_FALSE）・**任意コストを払えない**（COST_UNMET）
能力は発動自体が起きず、「テキスト通り動くか」を一切検証できない盲点だった
（合計 ~490 能力）。

本ツールは各能力の `ability.condition` と `ability.cost` を読み、それらを満たす
ように汎用盤面を変形してから再実行する。変形後も盤面が一切動かない能力は
`SATISFIED_NO_CHANGE` として真のバグ候補に挙げる。条件/コストの形が未対応で
盤面を作れなかった能力は `UNHANDLED`（バグではなく合成の限界）に分類する。

実行:
    OPCG_LOG_SILENT=1 python tests/condition_synth.py
    OPCG_LOG_SILENT=1 python tests/condition_synth.py --show SATISFIED_NO_CHANGE
    OPCG_LOG_SILENT=1 python tests/condition_synth.py --card OP01-001
"""
import argparse
import os
import sys
from collections import Counter, defaultdict
from typing import List, Optional

import os as _os, sys as _sys  # noqa: E402  test bootstrap (sys.path + google stub)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap  # noqa: E402,F401

import effect_coverage as cov
from engine_helpers import make_master
from opcg_sim.src.models.models import CardInstance, DonInstance
from opcg_sim.src.models.enums import (
    ActionType, CardType, CompareOperator, ConditionType, Player, Zone,
)
from opcg_sim.src.models.effect_types import Branch, Choice, GameAction, Sequence
from opcg_sim.src.utils.loader import CardLoader

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    "opcg_sim", "data")


# ---------------------------------------------------------------------------
# 盤面ビルダー（条件/コストを満たすカードを注入する）
# ---------------------------------------------------------------------------

def _mk(traits=None, cost=1, power=2000, ctype=CardType.CHARACTER, name="合成", attr=None,
        owner="P1") -> CardInstance:
    kw = dict(card_id="SYN", name=name, cost=cost, power=power, type=ctype,
              traits=traits or [])
    if attr is not None:
        from opcg_sim.src.models.enums import Attribute
        for a in Attribute:
            if a.value == attr:
                kw["attribute"] = a
                break
    return CardInstance(make_master(**kw), owner)


def _set_zone_count(player, zone_attr: str, target: int, owner: str, protect=None):
    """指定ゾーンの枚数を target に合わせる（不足は合成カードで補い、過剰は削る）。

    protect（ソースカード等）は削らない。ON_PLAY ではソースが手札に居るため、
    保護しないと登場対象を消してしまい play_card_action が空振りする。
    """
    zone = getattr(player, zone_attr)
    while len(zone) < target:
        if zone_attr == "don_active":
            zone.append(DonInstance(owner_id=owner))
        else:
            zone.append(_mk(owner=owner))
    i = len(zone) - 1
    while len(zone) > target and i >= 0:
        if protect is not None and zone[i] is protect:
            i -= 1
            continue
        zone.pop(i)
        i -= 1


def _target_int(cond) -> int:
    import re
    if isinstance(cond.value, int):
        return cond.value
    nums = re.findall(r'\d+', cond.raw_text or "")
    return int(nums[0]) if nums else 1


def _satisfy_value(op: CompareOperator, threshold: int) -> int:
    """演算子 op と閾値 threshold を満たす具体値を返す。"""
    if op in (CompareOperator.GE, CompareOperator.EQ, CompareOperator.HAS):
        return max(threshold, 0)
    if op == CompareOperator.GT:
        return threshold + 1
    if op == CompareOperator.LE:
        return threshold
    if op == CompareOperator.LT:
        return max(threshold - 1, 0)
    if op == CompareOperator.NEQ:
        return threshold + 1
    return max(threshold, 0)


# 注入ゾーンを持つ単純カウント条件 → (player_attr, zone_attr)
_COUNT_ZONE = {
    ConditionType.LIFE_COUNT: "life",
    ConditionType.HAND_COUNT: "hand",
    ConditionType.TRASH_COUNT: "trash",
    ConditionType.DECK_COUNT: "deck",
    ConditionType.HAS_DON: "don_active",
}


def _satisfy_condition(gm, controller, cond, source) -> Optional[bool]:
    """cond を満たすよう盤面を変形する。

    返り値: True=満たした / False=この OR 枝は無理 / None=未対応（合成不可）。
    """
    if cond is None or cond.type in (ConditionType.NONE, ConditionType.TURN_LIMIT,
                                     ConditionType.GENERIC):
        return True
    if cond.type == ConditionType.AND:
        results = [_satisfy_condition(gm, controller, s, source) for s in cond.args]
        if any(r is None for r in results):
            return None
        return all(results)
    if cond.type == ConditionType.OR:
        # いずれか1枝を満たせれば良い
        handled_any = False
        for s in cond.args:
            r = _satisfy_condition(gm, controller, s, source)
            if r is True:
                return True
            if r is not None:
                handled_any = True
        return False if handled_any else None

    opp = gm.p2 if controller is gm.p1 else gm.p1
    tp = opp if cond.player == Player.OPPONENT else controller
    op = cond.operator
    thr = _target_int(cond)

    if cond.type in _COUNT_ZONE:
        _set_zone_count(tp, _COUNT_ZONE[cond.type], _satisfy_value(op, thr), tp.name,
                        protect=source)
        return True

    if cond.type == ConditionType.DON_COUNT:
        # active を基準に総ドン数を合わせる
        _set_zone_count(tp, "don_active", _satisfy_value(op, thr), tp.name, protect=source)
        return True

    if cond.type == ConditionType.FIELD_COUNT:
        q = cond.target
        n = _satisfy_value(op, thr if thr else 1)
        traits = list(getattr(q, "traits", []) or []) if q else []
        cmax = getattr(q, "cost_max", None) if q else None
        cmin = getattr(q, "cost_min", None) if q else None
        cost = cmin if cmin else (cmax if cmax else 5)
        is_rest = getattr(q, "is_rest", None) if q else None
        fp = opp if (q and q.player == Player.OPPONENT) else tp
        for _ in range(max(n - len(fp.field), 0)):
            c = _mk(traits=traits, cost=cost, owner=fp.name)
            if is_rest is not None:
                c.is_rest = is_rest
            fp.field.append(c)
        return True

    if cond.type in (ConditionType.HAS_TRAIT, ConditionType.HAS_ATTRIBUTE,
                     ConditionType.HAS_UNIT, ConditionType.HAS_CHARACTER):
        # 不在条件（「…がいない場合」= EQ 0 / 単純文字列の EQ）は注入すると壊れる。
        # 汎用盤面は既に該当カードを持たないため、何もしないで満たす。
        if cond.type == ConditionType.HAS_CHARACTER:
            cv = cond.value
            absent = (op == CompareOperator.EQ and (
                isinstance(cv, str) or (isinstance(cv, tuple) and not isinstance(cv[1], str))))
            if absent:
                return True
        q = cond.target
        traits = list(getattr(q, "traits", []) or []) if q else []
        attrs = list(getattr(q, "attributes", []) or []) if q else []
        is_rest = getattr(q, "is_rest", None) if q else None
        if not traits and cond.type == ConditionType.HAS_TRAIT and isinstance(cond.value, str):
            traits = [cond.value]
        attr = attrs[0] if attrs else None
        name = cond.value if (cond.type == ConditionType.HAS_CHARACTER
                              and isinstance(cond.value, str)) else "合成"
        if isinstance(cond.value, tuple):
            name = cond.value[0] if isinstance(cond.value[0], str) else "合成"
        n = thr if thr else 1
        for _ in range(n):
            c = _mk(traits=traits, attr=attr, name=name, owner=tp.name)
            if is_rest is not None:
                c.is_rest = is_rest
            tp.field.append(c)
        return True

    if cond.type in (ConditionType.LEADER_TRAIT, ConditionType.LEADER_NAME,
                     ConditionType.LEADER_COLOR, ConditionType.LEADER_ATTRIBUTE):
        if not tp.leader:
            return None
        m = tp.leader.master
        traits = list(m.traits)
        name = m.name
        attr_val = m.attribute.value
        from opcg_sim.src.models.enums import Color, Attribute
        colors = list(m.colors)
        if cond.type == ConditionType.LEADER_TRAIT and isinstance(cond.value, str):
            traits = traits + [cond.value]
        elif cond.type == ConditionType.LEADER_NAME and isinstance(cond.value, str):
            name = cond.value
        elif cond.type == ConditionType.LEADER_COLOR:
            for c in Color:
                if c.value == cond.value:
                    colors = [c]
        elif cond.type == ConditionType.LEADER_ATTRIBUTE and isinstance(cond.value, str):
            for a in Attribute:
                if a.value == cond.value:
                    attr_val = a.value
        nm = make_master(card_id=m.card_id, name=name, type=CardType.LEADER,
                         traits=traits, life=5)
        # colors / attribute を差し替え（make_master は単色赤・斬がデフォルト）
        object.__setattr__(nm, "colors", colors)
        for a in Attribute:
            if a.value == attr_val:
                object.__setattr__(nm, "attribute", a)
        tp.leader.master = nm
        return True

    if cond.type == ConditionType.CONTEXT:
        cv = cond.value
        if cv in ("MY_TURN", "SELF_TURN"):
            gm.turn_player = controller
        elif cv == "OPPONENT_TURN":
            gm.turn_player = opp
        return True

    if cond.type == ConditionType.SOURCE_STATE:
        sv = cond.value
        if sv == "IS_RESTED":
            source.is_rest = True
        elif sv == "IS_ACTIVE":
            source.is_rest = False
        elif sv == "ENTERED_THIS_TURN":
            source.is_newly_played = True
        else:
            return None
        return True

    if cond.type == ConditionType.LEADER_STATE:
        if not tp.leader:
            return None
        if cond.value == "IS_ACTIVE":
            tp.leader.is_rest = False
        elif cond.value == "IS_RESTED":
            tp.leader.is_rest = True
        else:
            return None
        return True

    # 未対応の条件型（DON_COUNT_COMPARE / PREV_ACTION / REVEALED_* 等）は合成不可
    return None


def _satisfy_cost(gm, controller, node, source) -> Optional[bool]:
    """コストノードの対象を満たすカードを注入する。"""
    if node is None:
        return True
    if isinstance(node, GameAction):
        if node.type == ActionType.REST_DON:
            need = node.value.base if node.value else 1
            _set_zone_count(controller, "don_active",
                            max(need, len(controller.don_active)), controller.name)
            return True
        q = node.target
        if q is None:
            return True
        zone = q.zone if not isinstance(q.zone, list) else (q.zone[0] if q.zone else Zone.FIELD)
        traits = list(getattr(q, "traits", []) or [])
        names = list(getattr(q, "names", []) or [])
        cmax = getattr(q, "cost_max", None)
        cmin = getattr(q, "cost_min", None)
        cost = cmin if cmin else (cmax if cmax else 2)
        name = names[0] if names else "合成"
        fp = gm.p2 if (q.player == Player.OPPONENT and controller is gm.p1) else controller
        zone_attr = {Zone.HAND: "hand", Zone.TRASH: "trash", Zone.FIELD: "field",
                     Zone.LIFE: "life", Zone.DECK: "deck"}.get(zone)
        if zone_attr is None:
            return None
        need = getattr(q, "count", 1) or 1
        if need < 0:
            need = 1
        for _ in range(need):
            getattr(fp, zone_attr).append(
                _mk(traits=traits, cost=cost, name=name, owner=fp.name))
        return True
    if isinstance(node, Sequence):
        results = [_satisfy_cost(gm, controller, a, source) for a in node.actions]
        if any(r is None for r in results):
            return None
        return all(results)
    if isinstance(node, Choice):
        for opt in node.options:
            if _satisfy_cost(gm, controller, opt, source) is True:
                return True
        return None
    return True


# ---------------------------------------------------------------------------
# 分類
# ---------------------------------------------------------------------------

def _ability_has_gate(ab) -> bool:
    c = getattr(ab, "condition", None)
    has_cond = c is not None and getattr(c, "type", None) not in (None, ConditionType.NONE)
    return has_cond or getattr(ab, "cost", None) is not None


def classify_satisfied(master, ability, trig: str) -> str:
    """1能力を、条件/コストを満たす盤面で発動して分類する。"""
    try:
        gm, p1, p2, source = cov._build_test_state(
            master, source_in_hand=(trig == "ON_PLAY"))
    except Exception as e:
        return f"SETUP_ERROR:{e}"

    cond_ok = _satisfy_condition(gm, p1, getattr(ability, "condition", None), source)
    cost_ok = _satisfy_cost(gm, p1, getattr(ability, "cost", None), source)
    if cond_ok is None or cost_ok is None:
        return "UNHANDLED"
    if cond_ok is False:
        return "UNHANDLED"  # OR 枝を満たせなかった

    # 合成後に実際の評価器で条件・コスト実行可能性を再確認する。満たせていなければ
    # 合成漏れ（測定限界）であり、真のバグ候補ではないため UNHANDLED とする。
    from opcg_sim.src.core.effects.resolver import EffectResolver
    verifier = EffectResolver(gm)
    cond = getattr(ability, "condition", None)
    if cond is not None and getattr(cond, "type", None) not in (None, ConditionType.NONE):
        if not verifier._check_condition(p1, cond, source):
            return "UNHANDLED"
    cost = getattr(ability, "cost", None)
    if cost is not None and not verifier._can_satisfy_node(p1, cost, source):
        return "UNHANDLED"

    sb = cov._stat_snap(p1, p2)
    fb = cov._zone_fingerprint(p1, p2)
    try:
        if trig == "ON_PLAY":
            gm.play_card_action(p1, source)
        else:
            gm.resolve_ability(p1, ability, source)
    except Exception:
        return "ERROR"

    record: dict = {}
    cov._smart_drain(gm, record=record)
    sa = cov._stat_snap(p1, p2)
    fa = cov._zone_fingerprint(p1, p2)
    # ON_PLAY はソース自身の手札→場移動を「実行」とみなさない（プレイ行為そのもの）
    ignore = frozenset({source.uuid}) if trig == "ON_PLAY" else frozenset()

    if gm.active_interaction:
        return "INTERACTIVE"
    # ソースの移動だけを除いたグロス移動、またはステータス変化があれば実行された
    fb2 = {k: v for k, v in fb.items() if not k.endswith(source.uuid)}
    fa2 = {k: v for k, v in fa.items() if not k.endswith(source.uuid)}
    if cov._moved(fb2, fa2) or cov._stat_changed(sb, sa, ignore):
        return "SATISFIED_EXECUTED"
    if bool(getattr(gm, "action_events", [])):
        return "SATISFIED_EXECUTED"
    return "SATISFIED_NO_CHANGE"


def collect(card_filter: Optional[str] = None):
    db = CardLoader(os.path.join(DATA, "opcg_cards.json"))
    db.load()
    card_ids = sorted(db.raw_db.keys())
    if card_filter:
        card_ids = [c for c in card_ids if c == card_filter]

    buckets = defaultdict(list)
    total = len(card_ids)
    for i, cid in enumerate(card_ids, 1):
        if i % 200 == 0:
            sys.stderr.write(f"\r進行中: {i}/{total}...")
            sys.stderr.flush()
        master = db.get_card(cid)
        if master is None or not master.abilities:
            continue
        for ab in master.abilities:
            if not _ability_has_gate(ab):
                continue  # ゲートなし能力は effect_coverage が既にカバー
            trig = ab.trigger.name if hasattr(ab.trigger, "name") else str(ab.trigger)
            if trig in ("PASSIVE", "YOUR_TURN", "OPPONENT_TURN"):
                continue  # 静的/継続系は別ハーネス
            status = classify_satisfied(master, ab, trig)
            buckets[status].append((cid, master.name, trig,
                                    (getattr(ab.effect, "raw_text", "") or "")[:48]))
    sys.stderr.write(f"\r完了: {total} カード処理済み\n")
    return buckets


def run(show: Optional[str] = None, card_filter: Optional[str] = None):
    buckets = collect(card_filter)
    print("=== 条件/コスト充足盤面での実行分類（H-5）===")
    order = ["SATISFIED_EXECUTED", "SATISFIED_NO_CHANGE", "INTERACTIVE",
             "UNHANDLED", "ERROR"]
    for k in order:
        mark = "  ★真のバグ候補" if k == "SATISFIED_NO_CHANGE" else (
            "  (合成不可=測定限界)" if k == "UNHANDLED" else "")
        print(f"  {k:<20}: {len(buckets.get(k, [])):4d}{mark}")
    for k in list(buckets):
        if k.startswith("SETUP_ERROR") or k == "ERROR":
            for cid, name, trig, raw in buckets[k][:5]:
                print(f"    {k}: {cid} {trig} {name}")
    print()

    targets = [show] if show else ["SATISFIED_NO_CHANGE"]
    for t in targets:
        items = buckets.get(t, [])
        print(f"--- {t} ({len(items)} 件) ---")
        for cid, name, trig, raw in items[:80]:
            print(f"  {cid:<12} {trig:<14} {name}  | {raw}")
        if len(items) > 80:
            print(f"  ... 他 {len(items) - 80} 件")
        print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", default=None)
    ap.add_argument("--card", default=None)
    args = ap.parse_args()
    run(show=args.show, card_filter=args.card)
