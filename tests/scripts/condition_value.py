"""**能力の条件を判定する**（P2 の続き・`game_theory.md` §14.1.3）。

ユーザ確認 2026-09-14「**効果は条件＋コスト＋アクションの要素で考えるという認識で良い？**」
——**その通りだが、条件は足す項ではなく掛かる側**である:

```
能力の価値 = P(条件が成り立つ) × ( Σ実行内容 − Σコスト )
```

## 要点 1——**条件は確率ではなく述語**（判る場ではそのまま判定する）

初版の計画は「条件の成立率を棋譜から測って割り引く」だったが、**それは筋が悪い**。
**記録は判断点の状態を持っている**（ライフ・手札・場・ドン・トラッシュ・デッキ・ターン・
手番・両リーダー）ので、**確率を推定せずにその場で真偽を決められる**。
**推定を入れるのは、状態から決められない条件だけ**。

## 要点 2——**起動メインの候補では `P = 1`**（割り引いてはいけない）

エンジンは `rules/legal.rs::has_activatable_main` で**条件・ターン回数・コスト充足・
効果が空振りでないこと**の 4 つを確かめてから `ACTIVATE_MAIN` を候補に出す。
**＝候補に出ている時点で条件は成立している。** 一律に割り引くと**そこだけ過小**になる。

| 値の使いどころ | 扱い |
|---|---|
| **起動メインの候補** | **1.0**（エンジンが検査済み） |
| **登場・イベントの `PLAY` 候補** | **状態から判定**（`PLAY` の合法性はコストだけ・`ON_PLAY` の条件は解決時に判定される） |
| 常在（`PASSIVE`） | 状態から判定 |
| **カード単体**（手札の中・選択の利得の `W`） | 状態が無い＝**判定しない**（1.0・**上限として読む**） |

## 要点 3——**判定はエンジンの実装を写す**（規則を発明しない）

正本は `rust/opcg_engine/src/effects/cond.rs`。写したもの:

- `compare`（`EQ/NEQ/GT/LT/GE/LE`・`HAS` は常に偽）
- `offset_threshold`（「相手より N 枚以上 少ない/多い」＝ `LE/LT` なら `opp − N`・他は `opp + N`）
- `TURN_COUNT` は**手番プレイヤー自身の第 N ターン**（`(turn_count + 1) // 2`）
- `FIELD_COUNT` は**ステージも数える**（`field.len() + stage.is_some()`）。
  **`target` に絞り込みが付いていれば枚数が判らない**ので判定しない
- `DON_COUNT` は `active + rested + attached`。**記録は付与ドンを別に持つ**ので
  **区間で判定する**（下限と上限で答えが変わるなら判定しない）
- `CONTEXT` は `MY_TURN/SELF_TURN` と `OPPONENT_TURN` だけ見て**他は真**
- `TURN_LIMIT` は**ここでは常に真**（回数は解決側が守る）

## 母数はエンジン（`ConditionType` の 42 メンバ）

**出現から表を作らない**（`measurement.md` の罠 25）。テストが**全メンバがちょうど 1 つの
類に入ること**をラチェットする。

実行例:
  OPCG_LOG_SILENT=1 python tests/scripts/condition_value.py --in ~/w41
"""
import argparse
import json
import os
import sys
import time
from collections import Counter

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

#: 状態から厳密に判定できる条件（記録の列に写るもの）
STATE_KINDS = ("LIFE_COUNT", "HAND_COUNT", "FIELD_COUNT", "DON_COUNT", "TRASH_COUNT",
               "DECK_COUNT", "TURN_COUNT", "LIFE_COUNT_COMPARE", "HAND_COUNT_COMPARE",
               "FIELD_COUNT_COMPARE", "DON_COUNT_COMPARE", "LIFE_COUNT_BOTH",
               "LIFE_HAND_SUM", "CONTEXT")
#: デッキ（リーダー）から決まる条件
DECK_KINDS = ("LEADER_NAME", "LEADER_TRAIT", "LEADER_COLOR", "LEADER_ATTRIBUTE")
#: **確率ではない**条件——回数制限・空・総称。`cond.rs` と同じく**常に真**として通す。
META_KINDS = ("TURN_LIMIT", "NONE", "GENERIC", "OTHER")
#: 合成
COMPOSE_KINDS = ("AND", "OR", "NOT")
#: **盤面の札 id・レスト・ドンの内訳から決まる条件**（T72・2026-09-17・ユーザ決定「読める条件を増やしましょうか」）。
#: 状態に `my_field_ids`／`opp_field_ids`（場のキャラの札 id・`card_idx` の枠 2〜6／7〜11）・`*_field_rest`・`*_don_total`／`*_don_active`・
#: `source_rested`・`cards` が在れば判定する（無ければ従来どおり `None`＝1.0）。
BOARD_KINDS = ("HAS_DON", "HAS_CHARACTER", "SOURCE_STATE", "FIELD_ALL_TRAIT", "RESTED_COUNT")
#: **状態から決められない**条件（カード単位の細部・履歴）。判定しない（1.0 に倒す）。
OPAQUE_KINDS = ("HAS_TRAIT", "HAS_UNIT", "HAS_ATTRIBUTE",
                "IS_RESTED", "LEADER_STATE",
                "FIELD_COST_SUM", "EVENT_THIS_TURN",
                "CHAR_KOED_THIS_TURN", "PREV_ACTION", "REVEALED_CARD_TRAIT",
                "OPPONENT_REMOVAL", "DECLARED_COST_MATCH")

_CLASSES = (("state", STATE_KINDS), ("deck", DECK_KINDS), ("meta", META_KINDS),
            ("compose", COMPOSE_KINDS), ("board", BOARD_KINDS), ("opaque", OPAQUE_KINDS))
#: ドンの上限（規則）——区間判定の上側に使う
DON_MAX = 10


def engine_conditions():
    """**エンジンの条件の全集合**（`ConditionType` の 42 メンバ）。"""
    from opcg_sim.src.models.enums import ConditionType
    return tuple(sorted({m.name for m in ConditionType}))


def family_of(kind):
    """その条件がどの類か。**エンジンに無い名前は `absent`**（架空の名前を止める）。"""
    for fam, kinds in _CLASSES:
        if kind in kinds:
            return fam
    return "absent"


def compare(current, op, target):
    """`cond.rs::compare` を写す（`HAS` は常に偽）。"""
    op = str(op or "").upper()
    if current is None or target is None:
        return None
    if op == "EQ":
        return current == target
    if op in ("NEQ", "NE"):
        return current != target
    if op == "GT":
        return current > target
    if op == "LT":
        return current < target
    if op == "GE":
        return current >= target
    if op == "LE":
        return current <= target
    return False                      # HAS ほか


def offset_threshold(opp_count, cond):
    """`cond.rs::offset_threshold` を写す（`LE/LT` なら引く・他は足す）。"""
    v = cond.get("value")
    off = int(v) if isinstance(v, bool) or isinstance(v, int) else 0
    op = str(cond.get("operator") or "").upper()
    return opp_count - off if op in ("LE", "LT") else opp_count + off


def _int_value(cond):
    """しきい値（int でなければ `None`＝判定しない）。

    エンジンは `threshold_or_raw_text` で本文から数を拾う場合があるが、
    **ここでは拾わない**（拾い方を誤ると静かに間違うので、判定しないほうを選ぶ）。
    """
    v = cond.get("value")
    if isinstance(v, bool):
        return int(v)
    return int(v) if isinstance(v, int) else None


def _mine(cond):
    """その条件は誰の資源を見るか（`SELF` なら自分）。"""
    return str(cond.get("player") or "SELF").upper() != "OPPONENT"


def _decide_interval(lo, hi, op, target):
    """**区間で判定する**——下限と上限で答えが変わるなら `None`（判定しない）。

    付与ドンが記録から読めないときの `DON_COUNT` に使う。
    """
    a = compare(lo, op, target)
    b = compare(hi, op, target)
    return a if a == b else None


def holds(cond, st):
    """**その条件は成り立つか**（`True`／`False`／判らなければ `None`）。

    `st` は判断点の状態（`state_from_scalars` が作る）。`None` を渡せば
    **状態が無い**扱いで、状態に依る条件は全部 `None` になる。
    """
    if not isinstance(cond, dict):
        return None
    kind = str(cond.get("type") or "")
    fam = family_of(kind)
    if fam == "compose":
        subs = [holds(c, st) for c in (cond.get("args") or [])]
        if kind == "AND":
            if any(s is False for s in subs):
                return False
            return None if any(s is None for s in subs) else True
        if kind == "OR":
            if any(s is True for s in subs):
                return True
            return None if any(s is None for s in subs) else False
        s = subs[0] if subs else None
        return None if s is None else (not s)
    if fam == "meta":
        return True                       # 回数制限は解決側が守る（`cond.rs` と同じ）
    if fam == "opaque" or fam == "absent":
        return None
    if not st:
        return None
    if fam == "deck":
        return _holds_deck(kind, cond, st)
    if fam == "board":
        return _holds_board(kind, cond, st)
    return _holds_state(kind, cond, st)


def _field_ids(st, mine, cond_target=None):
    """その席の場のキャラの札 id（`is_rest` の絞り込みが在ればレストの列で絞る）。無ければ `None`。"""
    p = "my_" if mine else "opp_"
    ids = st.get(p + "field_ids")
    if ids is None:
        return None
    ids = list(ids)
    want_rest = (cond_target or {}).get("is_rest")
    if want_rest is not None:
        rests = st.get(p + "field_rest")
        if rests is None or len(rests) != len(ids):
            return None
        ids = [c for c, r in zip(ids, rests) if bool(r) == bool(want_rest)]
    return ids


def _matching(target, ids, st):
    """絞り込みに合う札 id（`search_price.eligible_deck_cards`・`cards` は状態から）。読めなければ `None`。"""
    cards = st.get("cards")
    if cards is None:
        return None
    try:
        import search_price as SP
    except Exception:
        return None
    t = {k: v for k, v in (target or {}).items() if k != "is_rest"}
    return SP.eligible_deck_cards(t, list(ids), cards)


def _holds_board(kind, cond, st):
    """盤面の札 id・レスト・ドンの内訳から決まる条件（T72）。"""
    mine = _mine(cond)
    op = cond.get("operator")
    p = "my_" if mine else "opp_"
    if kind == "HAS_DON":
        # 【ドン!!×N】＝この札に N 枚付いていれば。**付けられるか**（アクティブなドン ≥ N）で読む＝上限（付ける費用は数えない）
        n = _int_value(cond)
        a = st.get(p + "don_active")
        return None if (n is None or a is None) else int(a) >= int(n)
    if kind == "HAS_CHARACTER":
        want = str(cond.get("value") or "")
        ids = _field_ids(st, mine)
        if not want or ids is None:
            return None
        try:
            from theory_order import card_identity
        except Exception:
            return None
        names = []
        for c in ids:
            names.extend([str(x) for x in ((card_identity(c) or {}).get("names") or [])])
        info = st.get("my_leader" if mine else "opp_leader") or {}
        names.extend([str(x) for x in (info.get("names") or [])])
        return any(want == n or want in n for n in names)
    if kind == "SOURCE_STATE":
        r = st.get("source_rested")
        if r is None:
            return None
        v = str(cond.get("value") or "").upper()
        if v in ("IS_RESTED", "RESTED", "REST"):
            return bool(r)
        if v in ("IS_ACTIVE", "ACTIVE"):
            return not bool(r)
        return None
    if kind == "FIELD_ALL_TRAIT":
        v = cond.get("value")
        trait = str(v[0] if isinstance(v, (list, tuple)) and v else v or "")
        ids = _field_ids(st, mine)
        if not trait or ids is None:
            return None
        try:
            from theory_order import card_identity
        except Exception:
            return None
        return all(trait in [str(t) for t in ((card_identity(c) or {}).get("traits") or [])] for c in ids)
    if kind == "RESTED_COUNT":
        rests = st.get(p + "field_rest")
        val = _int_value(cond)
        if rests is None or val is None:
            return None
        return compare(int(sum(1 for r in rests if r)), op, val)   # 場のキャラのレスト数（リーダー・ドンは数えない＝下限）
    return None


def _holds_deck(kind, cond, st):
    """リーダーから決まる条件。**その席のリーダーの素性**を見る。"""
    who = "my_leader" if _mine(cond) else "opp_leader"
    info = st.get(who) or None
    if not info:
        return None
    want = cond.get("value")
    if kind == "LEADER_NAME":
        names = [str(n) for n in (info.get("names") or [])]
        return any(str(want) == n or str(want) in n for n in names)
    if kind == "LEADER_TRAIT":
        return str(want) in [str(t) for t in (info.get("traits") or [])]
    if kind == "LEADER_COLOR":
        return str(want) in [str(c) for c in (info.get("colors") or [])]
    if kind == "LEADER_ATTRIBUTE":
        return str(want) == str(info.get("attribute") or "")
    return None


def _holds_state(kind, cond, st):
    """状態から決まる条件（`cond.rs` の対応する枝を写す）。"""
    mine = _mine(cond)
    op = cond.get("operator")
    val = _int_value(cond)
    p = "my_" if mine else "opp_"
    q = "opp_" if mine else "my_"
    if kind == "CONTEXT":
        v = str(cond.get("value") or "")
        if v in ("MY_TURN", "SELF_TURN"):
            return bool(st.get("is_my_turn"))
        if v == "OPPONENT_TURN":
            return not st.get("is_my_turn")
        return True                       # 他の値は `cond.rs` も真で通す
    if kind == "LIFE_COUNT":
        return compare(st.get(p + "life"), op, val)
    if kind == "HAND_COUNT":
        return compare(st.get(p + "hand"), op, val)
    if kind == "TRASH_COUNT":
        return compare(st.get(p + "trash"), op, val)
    if kind == "DECK_COUNT":
        return compare(st.get(p + "deck"), op, val)
    if kind == "LIFE_COUNT_BOTH":
        a, b = st.get("my_life"), st.get("opp_life")
        return compare(None if a is None or b is None else a + b, op, val)
    if kind == "LIFE_HAND_SUM":
        a, b = st.get(p + "life"), st.get(p + "hand")
        return compare(None if a is None or b is None else a + b, op, val)
    if kind == "TURN_COUNT":
        t = st.get("turn")
        # **手番プレイヤー自身の第 N ターン**（先攻 1,3,5 → 1,2,3）
        return compare(None if t is None else (int(t) + 1) // 2, op, val)
    if kind == "FIELD_COUNT":
        if cond.get("target"):
            # **T72**: 絞り込み付きは場の札 id で数える（無ければ従来どおり判定しない）
            ids = _field_ids(st, mine, cond.get("target"))
            if ids is None:
                return None
            got = _matching(cond.get("target"), ids, st)
            if got is None:
                return None
            return compare(len(got), op, val if val is not None else 1)
        n = st.get(p + "field")
        if n is None:
            return None
        return compare(int(n) + int(bool(st.get(p + "stage"))), op, val)
    if kind == "DON_COUNT":
        tot = st.get(p + "don_total")
        if tot is not None:
            return compare(int(round(float(tot))), op, val)      # T72: 総在庫（付与込み）が読めれば区間ではなく値で
        lo = st.get(p + "don")
        if lo is None:
            return None
        hi = st.get(p + "don_max", DON_MAX)
        return _decide_interval(int(lo), int(hi), op, val)
    if kind in ("LIFE_COUNT_COMPARE", "HAND_COUNT_COMPARE", "FIELD_COUNT_COMPARE",
                "DON_COUNT_COMPARE"):
        key = {"LIFE_COUNT_COMPARE": "life", "HAND_COUNT_COMPARE": "hand",
               "FIELD_COUNT_COMPARE": "field", "DON_COUNT_COMPARE": "don"}[kind]
        a, b = st.get(p + key), st.get(q + key)
        if a is None or b is None:
            return None
        return compare(int(a), op, offset_threshold(int(b), cond))
    return None


#: **判らない条件をどう扱うか**（§0.4 の感度の切替）。既定 `1.0`＝割り引かない＝**上限**。
#: `0.0` にすると「判らないものは全部成り立たない」＝**下限**になる。
#: **結果を良くするために動かさない**（幅として読むためだけ）。
UNKNOWN_FACTOR = 1.0


def set_unknown_factor(v):
    """感度の切替（`theory_bridge --cond-unknown` が呼ぶ）。"""
    global UNKNOWN_FACTOR
    UNKNOWN_FACTOR = float(v)
    return UNKNOWN_FACTOR


def factor(ab, st, offered=False, unknown=None):
    """**能力に掛かる係数**（`1.0` か `0.0`）。

    `offered=True`（エンジンが候補に出した起動メイン）は**検査済みなので 1.0**。
    **判らない条件は既定で割り引かない**（`UNKNOWN_FACTOR = 1.0`）——推定値を入れずに
    上限として読む。下限が要るときは `unknown=0.0`（または `set_unknown_factor(0.0)`）。
    """
    if offered:
        return 1.0
    got = holds((ab or {}).get("condition"), st)
    if got is False:
        return 0.0
    if got is None and (ab or {}).get("condition"):
        return float(UNKNOWN_FACTOR if unknown is None else unknown)
    return 1.0


def state_from_scalars(sc, my_leader=None, opp_leader=None, my_stage=False,
                      opp_stage=False, cards=None):
    """判断点の状態を `scalars` から作る（列は `encode/scalars.rs` の正本どおり）。

    | 列 | 中身 |
    |---|---|
    | 0,1 | ライフ（自/相） |
    | 2,3 / 4,5 | ドン active/rested（自/相） |
    | 6,7 | 手札 | 8,9 | 場のキャラ数 | 10 | ターン | 11 | 手番 |
    | 16,17 | デッキ ÷50（自/相） | 18,19 | トラッシュ ÷20（自/相） |

    **ドンは `active + rested` しか無い**（`don_total` は付与も足す）ので
    **上限を `DON_MAX` に置いて区間で判定する**。
    """
    def f(i):
        return float(sc[i])
    st = {"my_life": int(round(f(0))), "opp_life": int(round(f(1))),
          "my_don": int(round(f(2) + f(3))), "opp_don": int(round(f(4) + f(5))),
          "my_hand": int(round(f(6))), "opp_hand": int(round(f(7))),
          "my_field": int(round(f(8))), "opp_field": int(round(f(9))),
          "turn": int(round(f(10))), "is_my_turn": bool(round(f(11))),
          "my_deck": int(round(f(16) * 50)), "opp_deck": int(round(f(17) * 50)),
          "my_trash": int(round(f(18) * 20)), "opp_trash": int(round(f(19) * 20)),
          "my_stage": bool(my_stage), "opp_stage": bool(opp_stage)}
    st["my_don_max"] = DON_MAX
    st["opp_don_max"] = DON_MAX
    st["my_don_active"] = int(round(f(2))); st["opp_don_active"] = int(round(f(4)))   # T72: 【ドン!!×N】の判定
    # **ドンの内訳**（T41・2026-09-15）——レスト（列 3/5）と**ドンデッキ残**（列 66/67 ÷10・v9）。
    # 効果の値付けが「N 枚まで」を実際に動かせる枚数で打ち切るのに使う。列が無い古い記録は省く
    st["my_don_rested"] = int(round(f(3))); st["opp_don_rested"] = int(round(f(5)))
    if len(sc) > 67:
        st["my_don_deck"] = int(round(f(66) * 10)); st["opp_don_deck"] = int(round(f(67) * 10))
    st["my_leader"] = leader_info(my_leader, cards)
    st["opp_leader"] = leader_info(opp_leader, cards)
    return st


_LEADERS = {}


def leader_info(cid, cards=None):
    """リーダーの素性（名前・特徴・色・属性）。判らなければ `None`。"""
    if not cid:
        return None
    if cid in _LEADERS:
        return _LEADERS[cid]
    try:
        if cards is None:
            from opcg_sim.loop import decks as D
            cards = D.load_db()
        m = cards.get_card(cid)
    except Exception:
        m = None
    if m is None:
        _LEADERS[cid] = None
        return None
    def _v(x):
        return getattr(x, "value", x)
    _LEADERS[cid] = {
        "names": list(getattr(m, "all_names", None) or [getattr(m, "name", "")]),
        "traits": [str(_v(t)) for t in (getattr(m, "traits", None) or [])],
        "colors": [str(_v(c)) for c in (getattr(m, "colors", None) or [])],
        "attribute": str(_v(getattr(m, "attribute", "")) or "")}
    return _LEADERS[cid]


def walk_conditions(cond):
    """条件木を平らに取り出す（`args` を辿る）。"""
    out = []
    if not isinstance(cond, dict):
        return out
    if cond.get("type"):
        out.append(cond)
    for c in (cond.get("args") or []):
        out.extend(walk_conditions(c))
    return out


def census(cards):
    """**エンジンの全条件**について、類と同梱での出現数・契機ごとの内訳を出す。"""
    seen = Counter()
    by_trigger = {}
    for cid, c in cards.items():
        for ab in (c.get("abilities") or []):
            cd = ab.get("condition")
            if not cd:
                continue
            trg = str(ab.get("trigger") or ab.get("timing"))
            for x in walk_conditions(cd):
                k = str(x.get("type"))
                seen[k] += 1
                by_trigger.setdefault(trg, Counter())[k] += 1
    rows = [{"condition": k, "family": family_of(k), "occurrences": seen[k]}
            for k in engine_conditions()]
    return {"rows": rows, "in_engine": len(rows),
            "unclassified": [r["condition"] for r in rows if r["family"] == "absent"],
            "outside_engine": sorted(set(seen) - set(engine_conditions())),
            "by_trigger": {t: dict(c.most_common(8)) for t, c in by_trigger.items()}}


def decidable(cards, st, triggers=("ON_PLAY",)):
    """**その状態で、条件付きの能力の何割を判定できたか**（`None` が残る割合）。

    **`0` と `None` の区別と同じ趣旨**——「成り立たない」と「判らない」を混ぜない。
    """
    n = fal = tru = unk = 0
    kinds = Counter()
    for cid, c in cards.items():
        for ab in (c.get("abilities") or []):
            if (ab.get("trigger") or ab.get("timing")) not in triggers:
                continue
            cd = ab.get("condition")
            if not cd:
                continue
            n += 1
            got = holds(cd, st)
            if got is None:
                unk += 1
                for x in walk_conditions(cd):
                    if holds(x, st) is None:
                        kinds[str(x.get("type"))] += 1
            elif got:
                tru += 1
            else:
                fal += 1
    return {"conditional": n, "true": tru, "false": fal, "undecided": unk,
            "decided_share": round((tru + fal) / max(1, n), 4),
            "false_share": round(fal / max(1, n), 4),
            "undecided_kinds": dict(kinds.most_common(10))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="src", default="", help="n_records のディレクトリ（任意）")
    ap.add_argument("--effects", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)

    t0 = time.time()
    import effect_value as EV
    cards = EV.load_cards(a.effects or None)
    out = {"census": census(cards)}
    if a.src:
        import numpy as np
        rows = []
        meta = json.load(open(os.path.join(os.path.expanduser(a.src), "meta_games.json"),
                              encoding="utf-8"))
        leaders = {int(g["seed"]): g["leaders"] for g in meta["games"]}
        for fn in sorted(os.listdir(os.path.expanduser(a.src))):
            if not fn.endswith(".npz"):
                continue
            z = np.load(os.path.join(os.path.expanduser(a.src), fn), allow_pickle=True)
            if "scalars" not in z.files:
                continue
            sc, who, seed = z["scalars"], z["who"], z["seed"]
            for i in range(len(sc)):
                pair = leaders.get(int(seed[i]))
                if not pair:
                    continue
                me = 0 if str(who[i]) in ("p1", "P1", "0") else 1
                rows.append(state_from_scalars(sc[i], pair[me], pair[1 - me]))
        out["rows"] = len(rows)
        # **状態ごとに判定できた割合**を、いくつかの代表的な行で出す
        samples = rows[:: max(1, len(rows) // 200)][:200]
        agg = Counter()
        share = []
        for st in samples:
            d = decidable(cards, st)
            share.append(d["decided_share"])
            agg.update(d["undecided_kinds"])
        out["decidable"] = {
            "samples": len(samples),
            "decided_share_mean": round(sum(share) / max(1, len(share)), 4),
            "undecided_kinds": dict(agg.most_common(10))}
        out["no_state"] = decidable(cards, None)
    out["seconds"] = round(time.time() - t0, 1)
    txt = json.dumps(out, ensure_ascii=False, indent=2)
    print(txt)
    if a.out:
        with open(os.path.expanduser(a.out), "w", encoding="utf-8") as fh:
            fh.write(txt + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
