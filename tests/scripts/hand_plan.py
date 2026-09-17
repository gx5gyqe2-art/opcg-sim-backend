"""**手札の価値＝計画の価値 `H`、札 1 枚の価値＝計画の増分 `ΔH`**（T66・読み取り専用・2026-09-16）。

ユーザ提案 2026-09-16「サーチの目的はゲームを通じて必要な札を集めるため」「出せる札が手札にあるかどうかも関わる」。
札の価値を静的な `v`（T64）ではなく、**残りターンのドンで何を出せるかの最良の計画**の中で測る:

```
H(手札, 状態) = max_計画 Σ_t s^t · Σ_{t に出す札} v(札)     制約: t に出す札のコスト合計 ≤ cap_t・各札は 1 回
cap_0 = 今アクティブなドン（このターンの残り）   cap_t = min(10, 総在庫 + t)（t = 1, 2）   cap_3 = 10（それより後の枠）   s = 1 − ko_p
ΔH(札) = H(手札 ∪ {札}) − H(手札)                                  （必要な札ほど大きい・出せない札は 0）
```

`v` は `hand_spend.use_value`（既にコスト·δ を引いてある＝余ったドンの付与の価値 δ×余りは計画に依らない定数なので落とす）。
先の盤面は今の盤面で近似（`v` は相手リーダーのパワーと R だけに依る）。**回帰しない・当てはめない**（新定数ゼロ）。

測るもの: 探して手に入れた札の `ΔH`（探した行の次の判断点の手札で）対 引いた札の `ΔH`（自席ターン開始の手札で）。
静的 `v` の差（T65: ≈0）と並べ、**サーチが計画の穴を埋めているか**を読む。副産物: 次のターンに出せる札が在るか（`playable_next`）。

使い方: `python tests/scripts/hand_plan.py --in <n_records ディレクトリ>... [--out x.json]`
"""
import argparse
import collections
import functools
import json
import os
import sys
import time

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from opcg_sim.learned.train import plan_labels as PL  # noqa: E402
import effect_value as EV  # noqa: E402
import guard_afford as GA  # noqa: E402
import theory_order as TO  # noqa: E402
from theory_order import KO_P, MU, POL_COLS, SC_MY_DON, SC_MY_LIFE, SC_OPP_LEADER_POWER, SC_OPP_LIFE  # noqa: E402
from theory_bridge import ROW_COLS, _extra, move_family  # noqa: E402
from hand_spend import hand_ids, spent_cards, use_value  # noqa: E402
import hand_guard as HG  # noqa: E402
from onplay_parts import look_k  # noqa: E402
from price_realised import don_stock, primary_action  # noqa: E402

#: 計画の長さ＝今・次・その次の 3 ターン＋「それより後」の 1 枠（上限 10・割引 s^3）。ユーザ提案「ゲームを通じて必要な札」
#: ＝3 ターン先より後に出す札（フィニッシャー）も計画に入れるため。厳密 DP は 11^4 状態。
PLAN_TURNS = 4
#: ドンの上限（規則の 10 枚）
DON_CAP = 10


def caps_of(don_active, don_total, turns=PLAN_TURNS, r_turns=None):
    """t = 0 は今アクティブなドン・t ≥ 1 は総在庫 + t（上限 10）。

    **「それより後」の枠の容量は残りターン数で決まる**（T68・2026-09-17）: `r_turns`（式が置く残りターン `R`・1〜5）を
    渡せば `10 × max(1, round(R) − (turns − 1))`＝3 ターン先より後に残るターンの数だけ 10 ドンのターンがある。
    渡さなければ従来どおり 1 ターンぶん（10）。最終盤（R ≤ 3）は 1 ターンぶんのまま＝「最終盤まで使う札が揃っていれば
    探す価値が下がる」がそのまま出る。"""
    out = [max(0, int(round(don_active)))]
    for t in range(1, turns - 1):
        out.append(int(min(DON_CAP, max(0, int(round(don_total)) + t))))
    if turns >= 2:
        later = 1 if r_turns is None else max(1, int(round(float(r_turns))) - (turns - 1))
        out.append(DON_CAP * later)                        # 「それより後」の枠＝上限まで出せる（割引 s^(turns−1)）
    return out


def search_context(sc, tok_row, ci_row, idx2cid, cards, deck):
    """**探す能力の価格に要る状態**（T68）＝今の手札（`hand_items`）・ドンの枠（`caps`・`R` 依存の後ろ枠）・来る攻撃・
    受ける損・自分のデッキ（`search_price.deck_of` で seed から復元した並び・`None` なら価格は従来の `sel(k)` に落ちる）。"""
    olp = float(sc[SC_OPP_LEADER_POWER]) * 1e4 or 5000.0
    r = max(1.0, min(5.0, float(sc[SC_OPP_LIFE])))
    xs = HG.incoming(tok_row); take = HG.take_cost_of(float(sc[SC_MY_LIFE]))
    field = own_field_ids(ci_row, idx2cid)
    items = hand_items(tok_row, ci_row, idx2cid, cards, olp, r)
    items = apply_inflow(items, deck, xs, take, cards, olp, r, field=field)        # T70
    return {"hand_items": items,
            "caps": caps_of(float(sc[SC_MY_DON]), don_stock(sc, tok_row, "me"), r_turns=r),
            "xs": xs, "take": take,
            "deck": (None if deck is None else list(deck)), "olp": olp, "r": r, "field": field}


def own_field_ids(ci_row, idx2cid):
    """自分の場のキャラの札 id（枠 2〜6・空は落とす）。"""
    return [c for c in (idx2cid.get(int(x)) for x in np.asarray(ci_row)[GA.SLOT_OWN_FIELD]) if c]


def plan_value(items, caps, s=1.0 - KO_P):
    """**厳密 DP**: `items` = [(コスト, v), …]（v は使ったときの価値・None は 0）・`caps` = ターンごとのドン。
    状態＝各ターンの使ったドン。札ごとに「出さない／t に出す」を選ぶ。"""
    caps = [int(c) for c in caps]
    T = len(caps)
    disc = [s ** t for t in range(T)]
    best = {tuple([0] * T): 0.0}
    for cost, v in items:
        vs = [max(0.0, v_at(v, t)) for t in range(T)]     # T70: v はターンごとの並びでもよい（相方待ちの札）
        c = int(round(cost))
        nxt = dict(best)
        if max(vs) <= 0.0:
            continue                                   # 価値 0 の札は出しても計画は増えない
        for used, val in best.items():
            for t in range(T):
                if vs[t] > 0.0 and used[t] + c <= caps[t]:
                    u2 = list(used); u2[t] += c; u2 = tuple(u2)
                    cand = val + disc[t] * vs[t]
                    if cand > nxt.get(u2, -1.0):
                        nxt[u2] = cand
        best = nxt
    return max(best.values()) if best else 0.0


def v_at(v, t):
    """札の価値の t ターン目の値（数なら同じ・並びなら t 番目・並びより先は最後の値・None は 0）。"""
    if v is None:
        return 0.0
    if isinstance(v, (list, tuple)):
        if not v:
            return 0.0
        return float(v[min(int(t), len(v) - 1)])
    return float(v)


def v_scalar(v, s=1.0 - KO_P):
    """札の価値を 1 つの数に（守る費用＝切ったときの機会費用に使う）＝`max_t s^t v_t`。"""
    if not isinstance(v, (list, tuple)):
        return 0.0 if v is None else float(v)
    return max([0.0] + [(s ** t) * float(x) for t, x in enumerate(v)])


def delta_h(items, extra, caps, s=1.0 - KO_P):
    """`ΔH(札)` = 札を足した計画 − 元の計画（≥ 0）。"""
    return plan_value(items + [extra], caps, s) - plan_value(items, caps, s)


def playable_next(items, caps):
    """次のターン（t=1）に出せる札が在るか（コスト ≤ cap_1 かつ v > 0）。"""
    if len(caps) < 2:
        return None
    return any(v_at(v, 1) > 0.0 and int(round(c)) <= caps[1] for c, v in items)


def _items(cids, cards, olp, r):
    out = []
    for cid in cids:
        info = cards.info(cid) or {}
        out.append((float(info.get("cost") or 0.0), use_value(cid, info, olp, r)))
    return out


def hand_items(tok_row, ci_row, idx2cid, cards, olp, r):
    """手札の枠ごとに `{cid, cost, v, counter}`（空の枠は落とす・カウンター値は枠のトークンから）。"""
    out = []
    ci = np.asarray(ci_row)
    for j, slot in enumerate(range(GA.SLOT_HAND.start, GA.SLOT_HAND.stop)):
        cid = idx2cid.get(int(ci[slot]))
        if not cid:
            continue
        info = cards.info(cid) or {}
        out.append({"cid": str(cid), "cost": float(info.get("cost") or 0.0), "v": use_value(cid, info, olp, r),
                    "counter": HG.counter_of(tok_row, slot), "event": bool(info.get("event"))})
    return out


#: **相方が後で来る期待**（T70・2026-09-17・ユーザ提案「ドロー＋サーチ＋相手の攻撃によるライフで出せる札が確保できる期待値も加味」）:
#: `on`（既定）＝「手札から出す」効果を持つ札（相方待ちの札）の `v` を **ターンごとの並び** `v_t = base + P(t までに相方が来る) × E[相方の値]` に
#: する（`P = 1 − (1 − p)^N(t)`・`p` = 残りの山の合う札の割合・`N(t)` = t ターンで手札に入る枚数＝ドロー 1 ＋ 受けるライフ ＋ サーチの当たり）。
#: 相方が今の手札に在れば `P = 1`。`off`＝旧（静的 `v`・効果は満額）。**新定数ゼロ**（率は全部デッキと来る攻撃から出る）。
INFLOW_MODES = ("off", "on")
INFLOW_MODE = "on"
#: 自分のターンに引く枚数（規則）
DRAWS_PER_TURN = 1.0


def set_inflow_mode(mode):
    global INFLOW_MODE
    if mode not in INFLOW_MODES:
        raise ValueError("inflow mode は %s のどれか" % (INFLOW_MODES,))
    INFLOW_MODE = mode
    return INFLOW_MODE


def add_inflow_arg(ap):
    ap.add_argument("--inflow", default=None, choices=INFLOW_MODES,
                    help="**T70** 相方待ちの札の v をターンごと（相方が来る確率つき）にする（`on`・既定）か旧の静的 v（`off`）か")


def apply_inflow_mode(a):
    if getattr(a, "inflow", None) is not None:
        set_inflow_mode(a.inflow)
    return INFLOW_MODE


def arrival_prob(p, n):
    """`n` 枚入るうちに合う札（割合 `p`）が 1 枚以上来る確率 `1 − (1 − p)^n`。"""
    p = min(1.0, max(0.0, float(p))); n = max(0.0, float(n))
    if p >= 1.0:
        return 1.0 if n > 0 else 0.0
    return 1.0 - (1.0 - p) ** n


def expected_taken(items, xs, take_cost):
    """相手のターン 1 回に受ける攻撃の本数（守る規則で止めない攻撃＝止められない・受ける方が安い）＝ライフの札が手札に入る枚数。"""
    items = [(it["counter"], v_scalar(it["v"])) for it in items]
    n = 0.0
    for x in sorted(xs, reverse=True):
        if x < -TO.PWR_EPS:
            continue
        cost, idx = HG.guard_cost_min_v(items, x)
        if cost is None or float(take_cost) - float(cost) <= 0.0:
            n += 1.0
            continue
        items = [it for i, it in enumerate(items) if i not in idx]
    return n


def expected_search_hits(items, deck, cards):
    """手札の探す札ごとの「合う札が k 枚の中に在る確率」の和（`search_price.search_actions`・当たれば 1 枚入る）。"""
    if not deck:
        return 0.0
    import search_price as SP
    tot = 0.0
    for it in items:
        c = EV._all_cards().get(it["cid"])
        for ab in (c or {}).get("abilities") or []:
            if (ab.get("trigger") or ab.get("timing")) not in EV.CHAR_ON_PLAY_TRIGGERS:
                continue
            found = SP.search_actions(EV.walk_actions(ab.get("effect")))
            if found is None:
                continue
            k, target, _act = found
            f = len(SP.eligible_deck_cards(target, deck, cards)) / float(len(deck))
            tot += arrival_prob(f, k)
            break
    return tot


def inflow_per_turn(items, xs, take_cost, deck, cards):
    """1 ターン（自分のターン＋相手のターン）に手札へ入る枚数の期待値＝ドロー ＋ 受けるライフ ＋ サーチの当たり。"""
    return DRAWS_PER_TURN + expected_taken(items, xs, take_cost) + expected_search_hits(items, deck, cards)


def _ctx_with_hand(hand_cids, cards, olp, r, field=()):
    """`use_value` に渡す状態（手札の札 id だけ・`v` は要らない）。"""
    items = [{"cid": str(c), "cost": float((cards.info(c) or {}).get("cost") or 0.0), "v": None, "counter": 0.0,
              "event": bool((cards.info(c) or {}).get("event"))} for c in hand_cids]
    return {"search_ctx": {"hand_items": items, "cards": cards, "deck": [], "olp": olp, "r": r, "caps": [], "xs": [], "take": 0.0,
                           "field": list(field)}}


def _base_value(cid, info, cards, olp, r, field=()):
    """相方待ちの札の「手札から出す」効果を 0 にした値（`use_value` に空の手札の状態を渡す＝コスト付きなら払わない＝0）。"""
    return use_value(cid, info, olp, r, st=_ctx_with_hand([], cards, olp, r, field))


def _value_with_partner(cid, info, cards, olp, r, partner, field=()):
    """相方 1 枚が手札に在るときの札の値（効果のコスト・条件・「払わない自由」込み＝`ability_value` をそのまま通す）。"""
    return use_value(cid, info, olp, r, st=_ctx_with_hand([partner], cards, olp, r, field))


def inflow_item(item, others, deck, xs, take_cost, cards, olp, r, turns=PLAN_TURNS, field=()):
    """1 枚の `v` を相方待ちの並びにする（相方待ちの札でなければそのまま）。`others` は同じ手札の残り。
    相方が来たときの取り分は `use_value(相方が手札に在る状態) − base`＝効果のコスト・条件・払わない自由を通した値。"""
    import search_price as SP
    target = SP.enabler_target(item["cid"])
    if target is None:
        return item
    info = cards.info(item["cid"]) or {}
    base = _base_value(item["cid"], info, cards, olp, r, field)
    if base is None:
        return item
    out = dict(item)
    out["v_static"] = item["v"]
    in_hand = SP.eligible_hand_cards(target, others, cards)
    if in_hand:                                                  # 相方が今在る＝P = 1（どの t でも・一番良い相方）
        best = max(((_value_with_partner(item["cid"], info, cards, olp, r, c, field) or 0.0) - base) for c in in_hand)
        out["v"] = [max(0.0, base + max(0.0, best))] * turns
        out["p_partner"] = 1.0
        return out
    if not deck:
        out["v"] = [base] * turns
        out["p_partner"] = 0.0
        return out
    pool = SP.eligible_deck_cards(target, deck, cards)
    if not pool:
        out["v"] = [base] * turns
        out["p_partner"] = 0.0
        return out
    p = len(pool) / float(len(deck))
    gains = {}
    for c in pool:
        if c not in gains:
            gains[c] = max(0.0, (_value_with_partner(item["cid"], info, cards, olp, r, c, field) or 0.0) - base)
    gain = float(np.mean([gains[c] for c in pool]))
    n_per = inflow_per_turn(others, xs, take_cost, deck, cards)
    out["v"] = [max(0.0, base + arrival_prob(p, n_per * t) * gain) for t in range(turns)]
    out["p_partner"] = p
    out["inflow_per_turn"] = n_per
    return out


def apply_inflow(items, deck, xs, take_cost, cards, olp, r, turns=PLAN_TURNS, field=()):
    """手札の全部の札に `inflow_item` を当てる（`off` ならそのまま）。`field` は自分の場の札 id（コストを払えるかの判定）。"""
    if INFLOW_MODE != "on":
        return list(items)
    items = list(items)
    return [inflow_item(it, items[:k] + items[k + 1:], deck, xs, take_cost, cards, olp, r, turns, field) for k, it in enumerate(items)]


def card_deltas(rest, card, caps, xs, take_cost):
    """1 枚の `ΔH_play`・`ΔG_guard`・`ΔH = max` と、その札がカウンター札か（2000 以上か【カウンター】イベント）。"""
    plan_items = [(it["cost"], it["v"]) for it in rest]
    guard_items = [(it["counter"], v_scalar(it["v"])) for it in rest]
    dh = delta_h(plan_items, (card["cost"], card["v"]), caps)
    dg = HG.delta_g(guard_items, (card["counter"], v_scalar(card["v"])), xs, take_cost)
    return {"dh": dh, "dg": dg, "dtotal": max(dh, dg), "counter": card["counter"],
            "counter_card": bool(card["counter"] >= 2000.0 - TO.PWR_EPS or (card["event"] and card["counter"] > 0.0))}


def added_card_gains(sc_after, tok_after, ci_before, ci_after, idx2cid, cards, deck=None):
    """**窓の中で手札に入った札の `max(ΔH_play, ΔG_guard)`**（T69・物差しに手札の質を入れる）。
    入った先の手札（`ci_after`・他の入った札も含む）で読む。同じ札が 2 枚入れば別の枠を当てる。戻り値は `[(cid, gain), …]`。"""
    before = hand_ids(ci_before, idx2cid)
    after = hand_ids(ci_after, idx2cid)
    added = spent_cards(after, before)                      # after − before（多重集合）
    if not added:
        return []
    olp = float(sc_after[SC_OPP_LEADER_POWER]) * 1e4 or 5000.0
    r = max(1.0, min(5.0, float(sc_after[SC_OPP_LIFE])))
    caps = caps_of(float(sc_after[SC_MY_DON]), don_stock(sc_after, tok_after, "me"), r_turns=r)
    xs = HG.incoming(tok_after)
    take = HG.take_cost_of(float(sc_after[SC_MY_LIFE]))
    items = apply_inflow(hand_items(tok_after, ci_after, idx2cid, cards, olp, r), deck, xs, take, cards, olp, r,
                         field=own_field_ids(ci_after, idx2cid))   # T70
    used = set()
    out = []
    for cid in added:
        k_ = next((q for q, it in enumerate(items) if it["cid"] == cid and q not in used), None)
        if k_ is None:
            continue
        used.add(k_)
        card = items[k_]
        rest = items[:k_] + items[k_ + 1:]
        out.append((cid, float(card_deltas(rest, card, caps, xs, take)["dtotal"])))
    return out


def collect(dirs, limit_games=0):
    cards = PL.Cards()
    idx2cid = {i: c for c, i in GA._vocab().items()}
    from theory_bridge import _seat_decks                    # 遅延（橋は本器を import しない）
    import search_price as SP
    rec_decks = SP.record_decks(dirs) if INFLOW_MODE == "on" else {}
    searches, draws = [], []
    stats = {"games": 0, "search_rows": 0, "draw_rows": 0, "no_next": 0, "inflow": INFLOW_MODE, "search_deck_ok": 0, "search_deck_bad": 0}
    games = 0
    for rows, pol, ex, L, ptr, idx in PL.iter_games(dirs, row_cols=ROW_COLS, pol_cols=POL_COLS, extra_fn=_extra):
        games += 1
        if limit_games and games > limit_games:
            break
        stats["games"] += 1
        order = list(idx)
        decks = _seat_decks(rec_decks, int(rows["seed"][idx[0]]), rows, ex, idx, idx2cid, stats)   # T70
        by_main, by_all = {}, {}
        for n, i in enumerate(order):
            w = int(rows["who"][i])
            by_all.setdefault(w, []).append(n)
            if int(rows["kind"][i]) == 0:
                by_main.setdefault(w, []).append(n)
        nxt = {a: b for ns in by_main.values() for a, b in zip(ns, ns[1:])}
        prev_any = {b: a for ns in by_all.values() for a, b in zip(ns, ns[1:])}
        seen_first = set()
        for n, i in enumerate(order):
            w, t = int(rows["who"][i]), int(rows["turn"][i])
            if t < 1 or not PL.is_own_turn(w, t) or int(rows["kind"][i]) != 0:
                continue
            sc, tok = ex["sc"][i], ex["tok"][i]
            olp = float(sc[SC_OPP_LEADER_POWER]) * 1e4 or 5000.0
            r = max(1.0, min(5.0, float(sc[SC_OPP_LIFE])))
            caps = caps_of(float(sc[SC_MY_DON]), don_stock(sc, tok, "me"), r_turns=r)   # T68: 後ろ枠は R 依存
            hand = hand_ids(ex["ci"][i], idx2cid)
            # --- 引いた札（ターン開始・直前の行からライフが減っていない） ---
            if (w, t) not in seen_first:
                seen_first.add((w, t))
                p = prev_any.get(n)
                if p is not None:
                    ip = order[p]
                    if int(round(float(ex["sc"][ip][SC_MY_LIFE]))) == int(round(float(sc[SC_MY_LIFE]))):
                        added = spent_cards(hand, hand_ids(ex["ci"][ip], idx2cid))
                        xs = HG.incoming(tok); take = HG.take_cost_of(float(sc[SC_MY_LIFE]))
                        hitems = apply_inflow(hand_items(tok, ex["ci"][i], idx2cid, cards, olp, r), decks.get(w), xs, take, cards, olp, r,
                                              field=own_field_ids(ex["ci"][i], idx2cid))
                        for cid in added:
                            k_ = next((q for q, it in enumerate(hitems) if it["cid"] == cid), None)
                            if k_ is None:
                                continue
                            card = hitems[k_]; rest = hitems[:k_] + hitems[k_ + 1:]
                            items = [(it["cost"], it["v"]) for it in rest]
                            d = card_deltas(rest, card, caps, xs, take)
                            draws.append({"cid": cid, "v": v_scalar(card["v"]), "dh": d["dh"], "dg": d["dg"], "dtotal": d["dtotal"],
                                          "counter_card": d["counter_card"], "turn": t, "hand_n": len(hand),
                                          "playable_next_before": playable_next(items, caps)})
                        stats["draw_rows"] += 1
            # --- 探した行 ---
            k = int(L[i]); ch = int(rows["pol_chosen"][i])
            if k < 1 or ch < 0 or ch >= k:
                continue
            b = int(ptr[i]) + ch
            if move_family(json.loads(pol["pol_sig"][b])) != "play":
                continue
            cid = str(pol["pol_cid"][b]) or None
            if not cid or primary_action(cid, EV.CHAR_ON_PLAY_TRIGGERS) != "LOOK":
                continue
            j = nxt.get(n)
            if j is None or int(rows["turn"][order[j]]) != t:
                stats["no_next"] += 1
                continue
            stats["search_rows"] += 1
            i2 = order[j]
            sc2, tok2 = ex["sc"][i2], ex["tok"][i2]
            caps2 = caps_of(float(sc2[SC_MY_DON]), don_stock(sc2, tok2, "me"), r_turns=r)
            after = hand_ids(ex["ci"][i2], idx2cid)
            added = spent_cards(after, hand)
            xs2 = HG.incoming(tok2); take2 = HG.take_cost_of(float(sc2[SC_MY_LIFE]))
            hitems2 = apply_inflow(hand_items(tok2, ex["ci"][i2], idx2cid, cards, olp, r), decks.get(w), xs2, take2, cards, olp, r,
                                   field=own_field_ids(ex["ci"][i2], idx2cid))
            got = []
            for c2 in added:
                k_ = next((q for q, it in enumerate(hitems2) if it["cid"] == c2), None)
                if k_ is None:
                    continue
                card = hitems2[k_]; rest = hitems2[:k_] + hitems2[k_ + 1:]
                items = [(it["cost"], it["v"]) for it in rest]
                d = card_deltas(rest, card, caps2, xs2, take2)
                got.append({"cid": c2, "v": v_scalar(card["v"]), "dh": d["dh"], "dg": d["dg"], "dtotal": d["dtotal"],
                            "counter_card": d["counter_card"], "playable_next_before": playable_next(items, caps2)})
            searches.append({"cid": cid, "k": look_k(cid), "turn": t, "found": len(added), "got": got})
    return searches, draws, stats


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def block(ss, draw_dh, draw_v, draw_dg=None, draw_dt=None):
    n = len(ss)
    if n == 0:
        return {"n": 0}
    got = [g for s in ss for g in s["got"]]
    dh = [g["dh"] for g in got]
    dg = [g["dg"] for g in got]
    dt = [g["dtotal"] for g in got]
    vs = [g["v"] for g in got if g["v"] is not None]
    return {"n": n, "found": float(np.mean([s["found"] >= 1 for s in ss])), "k_mean": _mean([s["k"] for s in ss]),
            "dh_searched": _mean(dh), "dh_draw": draw_dh,
            # **守る備え**（T67）: 探した札の ΔG・引いた札の ΔG・max(ΔH, ΔG)
            "dg_searched": _mean(dg), "dg_draw": draw_dg, "premium_dg": (None if not dg or draw_dg is None else float(np.mean(dg) - draw_dg)),
            "dtotal_searched": _mean(dt), "dtotal_draw": draw_dt,
            "premium_total": (None if not dt or draw_dt is None else float(np.mean(dt) - draw_dt)),
            "counter_card_share": (float(np.mean([g["counter_card"] for g in got])) if got else None),
            "guard_motivated_share": (float(np.mean([g["dg"] > g["dh"] + 1e-12 for g in got])) if got else None),
            # **計画の増分で見た選択の利得**（探した札 − 引いた札）対 静的 v の差
            "premium_dh": (None if not dh or draw_dh is None else float(np.mean(dh) - draw_dh)),
            "v_searched": _mean(vs), "v_draw": draw_v,
            "premium_v": (None if not vs or draw_v is None else float(np.mean(vs) - draw_v)),
            "sel_k": _mean([EV._sel_premium(int(s["k"])) for s in ss]),
            # 探した札が「次のターンに出せる札が無かった手札」に来た割合
            "hole_before": _mean([None if g["playable_next_before"] is None else float(not g["playable_next_before"]) for g in got]),
            "dh_zero_share": (float(np.mean([d <= 1e-9 for d in dh])) if dh else None)}


def summarise(searches, draws, min_card=8):
    draw_dh = _mean([d["dh"] for d in draws]); draw_v = _mean([d["v"] for d in draws])
    draw_dg = _mean([d["dg"] for d in draws]); draw_dt = _mean([d["dtotal"] for d in draws])
    out = {"draws": {"n": len(draws), "dh_mean": draw_dh, "dg_mean": draw_dg, "dtotal_mean": draw_dt, "v_mean": draw_v, "mu": MU,
                     "dh_zero_share": (float(np.mean([d["dh"] <= 1e-9 for d in draws])) if draws else None),
                     "counter_card_share": (float(np.mean([d["counter_card"] for d in draws])) if draws else None),
                     "hole_before": _mean([None if d["playable_next_before"] is None else float(not d["playable_next_before"]) for d in draws])},
           "all": block(searches, draw_dh, draw_v, draw_dg, draw_dt)}
    for k in (3, 4, 5):
        ss = [s for s in searches if int(s["k"]) == k]
        if ss:
            out["k=%d" % k] = block(ss, draw_dh, draw_v, draw_dg, draw_dt)
    by = collections.defaultdict(list)
    for s in searches:
        by[s["cid"]].append(s)
    out["by_card"] = {c: block(ss, draw_dh, draw_v, draw_dg, draw_dt) for c, ss in sorted(by.items(), key=lambda kv: -len(kv[1])) if len(ss) >= min_card}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="src", nargs="+", required=True, help="n_records のディレクトリ")
    ap.add_argument("--limit-games", type=int, default=0)
    ap.add_argument("--min-card", type=int, default=8)
    TO.add_nu_mode_arg(ap)
    TO.add_surv_mode_arg(ap)
    TO.add_cbar_mode_arg(ap)
    add_inflow_arg(ap)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    TO.apply_nu_mode(a)
    TO.apply_surv_mode(a)
    TO.apply_cbar_mode(a)
    apply_inflow_mode(a)
    t0 = time.time()
    searches, draws, stats = collect(a.src, a.limit_games)
    res = {"nu_mode": a.nu_mode, "surv_mode": a.surv_mode, "cbar_mode": a.cbar_mode, "inflow": INFLOW_MODE, "plan_turns": PLAN_TURNS,
           "stats": stats, "summary": summarise(searches, draws, a.min_card), "seconds": round(time.time() - t0, 1)}
    txt = json.dumps(res, ensure_ascii=False, indent=2)
    print(txt)
    if a.out:
        with open(os.path.expanduser(a.out), "w", encoding="utf-8") as fh:
            fh.write(txt + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
