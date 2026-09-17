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
    return {"hand_items": hand_items(tok_row, ci_row, idx2cid, cards, olp, r),
            "caps": caps_of(float(sc[SC_MY_DON]), don_stock(sc, tok_row, "me"), r_turns=r),
            "xs": HG.incoming(tok_row), "take": HG.take_cost_of(float(sc[SC_MY_LIFE])),
            "deck": (None if deck is None else list(deck)), "olp": olp, "r": r,
            "field": [c for c in (idx2cid.get(int(x)) for x in np.asarray(ci_row)[GA.SLOT_OWN_FIELD]) if c]}


def plan_value(items, caps, s=1.0 - KO_P):
    """**厳密 DP**: `items` = [(コスト, v), …]（v は使ったときの価値・None は 0）・`caps` = ターンごとのドン。
    状態＝各ターンの使ったドン。札ごとに「出さない／t に出す」を選ぶ。"""
    caps = [int(c) for c in caps]
    T = len(caps)
    disc = [s ** t for t in range(T)]
    best = {tuple([0] * T): 0.0}
    for cost, v in items:
        v = 0.0 if v is None else float(v)
        c = int(round(cost))
        nxt = dict(best)
        if v <= 0.0:
            continue                                   # 価値 0 の札は出しても計画は増えない
        for used, val in best.items():
            for t in range(T):
                if used[t] + c <= caps[t]:
                    u2 = list(used); u2[t] += c; u2 = tuple(u2)
                    cand = val + disc[t] * v
                    if cand > nxt.get(u2, -1.0):
                        nxt[u2] = cand
        best = nxt
    return max(best.values()) if best else 0.0


def delta_h(items, extra, caps, s=1.0 - KO_P):
    """`ΔH(札)` = 札を足した計画 − 元の計画（≥ 0）。"""
    return plan_value(items + [extra], caps, s) - plan_value(items, caps, s)


def playable_next(items, caps):
    """次のターン（t=1）に出せる札が在るか（コスト ≤ cap_1 かつ v > 0）。"""
    if len(caps) < 2:
        return None
    return any((v or 0.0) > 0.0 and int(round(c)) <= caps[1] for c, v in items)


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


def card_deltas(rest, card, caps, xs, take_cost):
    """1 枚の `ΔH_play`・`ΔG_guard`・`ΔH = max` と、その札がカウンター札か（2000 以上か【カウンター】イベント）。"""
    plan_items = [(it["cost"], it["v"]) for it in rest]
    guard_items = [(it["counter"], it["v"]) for it in rest]
    dh = delta_h(plan_items, (card["cost"], card["v"]), caps)
    dg = HG.delta_g(guard_items, (card["counter"], card["v"]), xs, take_cost)
    return {"dh": dh, "dg": dg, "dtotal": max(dh, dg), "counter": card["counter"],
            "counter_card": bool(card["counter"] >= 2000.0 - TO.PWR_EPS or (card["event"] and card["counter"] > 0.0))}


def collect(dirs, limit_games=0):
    cards = PL.Cards()
    idx2cid = {i: c for c, i in GA._vocab().items()}
    searches, draws = [], []
    stats = {"games": 0, "search_rows": 0, "draw_rows": 0, "no_next": 0}
    games = 0
    for rows, pol, ex, L, ptr, idx in PL.iter_games(dirs, row_cols=ROW_COLS, pol_cols=POL_COLS, extra_fn=_extra):
        games += 1
        if limit_games and games > limit_games:
            break
        stats["games"] += 1
        order = list(idx)
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
                        hitems = hand_items(tok, ex["ci"][i], idx2cid, cards, olp, r)
                        xs = HG.incoming(tok); take = HG.take_cost_of(float(sc[SC_MY_LIFE]))
                        for cid in added:
                            k_ = next((q for q, it in enumerate(hitems) if it["cid"] == cid), None)
                            if k_ is None:
                                continue
                            card = hitems[k_]; rest = hitems[:k_] + hitems[k_ + 1:]
                            items = [(it["cost"], it["v"]) for it in rest]
                            d = card_deltas(rest, card, caps, xs, take)
                            draws.append({"cid": cid, "v": card["v"], "dh": d["dh"], "dg": d["dg"], "dtotal": d["dtotal"],
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
            hitems2 = hand_items(tok2, ex["ci"][i2], idx2cid, cards, olp, r)
            xs2 = HG.incoming(tok2); take2 = HG.take_cost_of(float(sc2[SC_MY_LIFE]))
            got = []
            for c2 in added:
                k_ = next((q for q, it in enumerate(hitems2) if it["cid"] == c2), None)
                if k_ is None:
                    continue
                card = hitems2[k_]; rest = hitems2[:k_] + hitems2[k_ + 1:]
                items = [(it["cost"], it["v"]) for it in rest]
                d = card_deltas(rest, card, caps2, xs2, take2)
                got.append({"cid": c2, "v": card["v"], "dh": d["dh"], "dg": d["dg"], "dtotal": d["dtotal"],
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
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    TO.apply_nu_mode(a)
    TO.apply_surv_mode(a)
    TO.apply_cbar_mode(a)
    t0 = time.time()
    searches, draws, stats = collect(a.src, a.limit_games)
    res = {"nu_mode": a.nu_mode, "surv_mode": a.surv_mode, "cbar_mode": a.cbar_mode, "plan_turns": PLAN_TURNS,
           "stats": stats, "summary": summarise(searches, draws, a.min_card), "seconds": round(time.time() - t0, 1)}
    txt = json.dumps(res, ensure_ascii=False, indent=2)
    print(txt)
    if a.out:
        with open(os.path.expanduser(a.out), "w", encoding="utf-8") as fh:
            fh.write(txt + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
