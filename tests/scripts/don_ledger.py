#!/usr/bin/env python3
"""**ドンの帳尻**（T109・ユーザ指示 2026-09-19「使用できるドンと使ったドンの整合が取れるように」）。

**4 つのゾーン**（`journal.rs::DonZone`）＝**アクティブ・レスト・付与・ドンデッキ**。ドンは
**この 4 つの間を動くだけで、増えも減りもしない**——だから**席ごとの合計は不変量**であり、
その合計は**リーダーのルール効果で決まる定数**（既定 10 枚・OP15-058 エネルだけ 6 枚）。

## 規則としての「使う」（エンジンの全経路を数え上げた・2026-09-19）

| 使い道 | ゾーンの動き | 場所 |
|---|---|---|
| **追加**（ドン!!フェイズ） | デッキ → アクティブ（1 ターン目 1 枚・以降 2 枚） | `rules/turn.rs::don_phase` |
| **追加**（効果） | デッキ → アクティブ／レスト | `effects/actions/don.rs` |
| **支払い**（登場・イベント・起動） | アクティブ → レスト（足りなければ付与 → レスト） | `ops.rs::pay_cost` |
| **付与** | アクティブ → 付与 | `rules/actions.rs`・`ops.rs::attach_don` |
| **マイナス**（戻す） | レスト／アクティブ／付与 → デッキ | `ops.rs::return_don` |
| **付与ドンの移動** | 付与 → レスト（持ち主が場を離れる）／付与 → アクティブ（リフレッシュ） | `ops.rs`・`turn.rs` |
| **リフレッシュ** | レスト → アクティブ（自分のターン開始） | `rules/turn.rs` |

**「使えるドン」＝自分のターン開始時のアクティブ**（リフレッシュ＋追加の後）。**支払いも付与も
ここから出る**＝**財布は 1 つ**。理論の側がこれを守っているかを測るのが本器の目的。

## 記録から読める 4 ゾーン（`encode/scalars.rs`）

`sc[2]`／`sc[3]` 自アクティブ／自レスト・`sc[4]`／`sc[5]` 相手・`sc[14]`／`sc[15]` リーダーの付与（×5）・
`sc[66]`／`sc[67]` ドンデッキ（×10）・**枠ごとの付与**はトークンの列 2（×5）。

使い方:

    python tests/scripts/don_ledger.py --in <records_dir> [--games N] [--json out.json]
"""

import argparse
import json
import os
import re
import sys

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from opcg_sim.learned.train import plan_labels as PL  # noqa: E402
import guard_afford as GA  # noqa: E402
import theory_order as TO  # noqa: E402
from theory_bridge import POL_COLS, ROW_COLS, _extra  # noqa: E402

#: **既定のドンデッキ枚数**（規則）。`py_game.rs::leader_don_deck_size`・`gamestate._apply_leader_don_deck_rule`。
DON_BUDGET_DEFAULT = 10
#: リーダーの「ルール上、自分のドン!!デッキはN枚になる」（`!!` と `‼` の両方・エンジンと同じ）。
_DON_DECK_RE = re.compile(r"ドン(?:!!|‼)デッキは(\d+)枚")

#: `scalars.rs` の列（4 ゾーン ×2 席）
SC_MY_ACTIVE, SC_MY_RESTED, SC_OPP_ACTIVE, SC_OPP_RESTED = 2, 3, 4, 5
SC_MY_LEADER_DON, SC_OPP_LEADER_DON = 14, 15
SC_MY_DON_DECK, SC_OPP_DON_DECK = 66, 67
#: 付与の目盛り（`attached_don / 5`）とドンデッキの目盛り（`len / 10`）
ATT_SCALE, DECK_SCALE = 5.0, 10.0
#: トークンの「付与ドン」列（`encode/tokens.rs`・枠ごと）
TOK_ATTACHED = 2
#: **枚数の許容**——記録は float32 で `/5`・`/10` された値なので、掛け戻すと 1 枚あたり 1e-7 ずれる
#: （6 項足すと 1e-6 に届く）。**数えているのは整数の枚数**なので 0.01 で切る。
DON_EPS = 0.01

_BUDGET = {}


def budget_of(cid, db=None):
    """**その席のドンの総数**＝リーダーのルール効果（既定 10・OP15-058 は 6）。**規則であって打ち筋ではない**。"""
    key = str(cid or "")
    if key in _BUDGET:
        return _BUDGET[key]
    out = DON_BUDGET_DEFAULT
    if key:
        if db is None:
            from opcg_sim.loop import decks as D
            db = D.load_db()
        m = db.get_card(key)
        if m is not None:
            hit = _DON_DECK_RE.search(str(getattr(m, "effect_text", "") or ""))
            if hit:
                out = int(hit.group(1))
    _BUDGET[key] = out
    return out


def zones_of(sc, tok, side="me"):
    """**1 行から読む 4 ゾーン**（`side`＝`me`／`opp`）。返すのは `{active, rested, attached, deck, total}`。

    **付与はリーダーの列＋その席の枠**（`tok` の列 2）。相手の枠は 7〜11・自分の枠は 2〜6。"""
    sc = np.asarray(sc)
    tok = np.asarray(tok)
    if side == "me":
        act, rest = float(sc[SC_MY_ACTIVE]), float(sc[SC_MY_RESTED])
        lead = float(sc[SC_MY_LEADER_DON]) * ATT_SCALE
        slots = range(TO.SLOT_OWN_FIELD.start, TO.SLOT_OWN_FIELD.stop)
        deck = float(sc[SC_MY_DON_DECK]) * DECK_SCALE
    else:
        act, rest = float(sc[SC_OPP_ACTIVE]), float(sc[SC_OPP_RESTED])
        lead = float(sc[SC_OPP_LEADER_DON]) * ATT_SCALE
        slots = range(TO.SLOT_OPP_FIELD.start, TO.SLOT_OPP_FIELD.stop)
        deck = float(sc[SC_OPP_DON_DECK]) * DECK_SCALE
    att = lead + sum(float(tok[s][TOK_ATTACHED]) * ATT_SCALE for s in slots)
    return {"active": act, "rested": rest, "attached": att, "deck": deck,
            "total": act + rest + att + deck}


def flows_between(a, b):
    """**2 つの行のあいだにドンが動いたぶん**を使い道の型に割り振る（`a` → `b`）。

    **1 つの差分に複数の使い道が混ざりうる**ので、**符号から決まるぶんだけ**を名前で数える
    （どれが先かは記録に無い＝**推測しない**）:

    * `added`＝デッキが減ったぶん（**追加**・ドン!!フェイズか効果）
    * `minus`＝デッキが増えたぶん（**マイナス**＝戻した）
    * `attached`＝付与が増えたぶん（**付与**）
    * `unattached`＝付与が減ったぶん（**付与ドンの移動**＝持ち主が場を離れた／リフレッシュ）
    * `rested`＝レストが増えたぶん（**支払い**の跡）
    * `refreshed`＝レストが減ったぶん（**リフレッシュ**）
    """
    d = {k: b[k] - a[k] for k in ("active", "rested", "attached", "deck")}
    q = {k: (0.0 if abs(v) < DON_EPS else v) for k, v in d.items()}
    return {"added": max(0.0, -q["deck"]), "minus": max(0.0, q["deck"]),
            "attached": max(0.0, q["attached"]), "unattached": max(0.0, -q["attached"]),
            "rested": max(0.0, q["rested"]), "refreshed": max(0.0, -q["rested"]),
            "d_active": d["active"], "conserved": abs(sum(d.values())) < DON_EPS}


def theory_spend(sc, tok, ci_row, idx2cid, cards, olp, theta=None, mu=None):
    """**理論が「使う」ことにしているドン**を 3 つに分けて数える（T109 の問い）。

    * `stock`＝`playable_attack_price` が詰めた札のコストの和（T77・体を出す）
    * `eff`＝`hand_effect_harm` が選んだ札のコスト（T108・効果を撃つ）
    * `attach`＝`attack_value_don` が場の攻め手に**暗黙に付けている**ドン（T45）

    **3 つは同じアクティブから出る**のが規則なので、**合計が `active` を超えていれば二重使用**。
    """
    import crossing_bridge as CB
    import deck_refill as DR
    import hand_plan as HP
    sc_a = np.asarray(sc)
    theta = TO.THETA if theta is None else theta
    mu = TO.MU if mu is None else mu
    avail = float(sc_a[SC_MY_ACTIVE])
    r = max(1.0, min(5.0, float(sc_a[TO.SC_OPP_LIFE])))
    mlp = float(sc_a[TO.SC_MY_LEADER_POWER]) * 1e4 or 5000.0
    items = HP.hand_items(tok, ci_row, idx2cid, cards, float(olp), r) or []
    if CB.DON_PURSE_MODE == "all":
        # **財布 1 つ・付与も同じ財布**（T109）＝ナップサックが払った額そのもの
        g = CB.hand_groups(items, cards, float(olp), theta, mu, mlp, r, with_don=False)
        g = g + CB.attach_groups(tok, float(olp), theta, mu)
        paid = float(CB.purse_plan(g, avail)["paid"])
        return {"available": avail, "stock": paid, "eff": 0.0, "attach": 0.0, "asked": paid}
    if CB.DON_PURSE_MODE == "one":
        paid = float(CB.hand_purse(items, cards, avail, float(olp), theta, mu, mlp, r)[3])
        attach = sum(float(don_for_attacker(float(p), float(olp), theta, mu))
                     for p in TO.own_attackers_of(tok, float(olp)))
        return {"available": avail, "stock": paid, "eff": 0.0, "attach": float(attach),
                "asked": paid + float(attach)}
    _val, _rush, cost_stock = CB.playable_attack_price(items, cards, avail, float(olp), theta, mu,
                                                       want_rush=True, want_cost=True)
    # 効果の側（T108）が使うコスト＝選ばれた 1 枚のコスト
    cost_eff, best = 0.0, 0.0
    cap = int(round(avail))
    for it in items:
        c = int(round(float(it.get("cost") or 0.0)))
        if c > cap:
            continue
        h = DR.card_effect_harm(it["cid"], mlp, r)
        if h > best:
            best, cost_eff = h, float(c)
    # 場の攻め手に暗黙に付いているドン
    attach = 0.0
    for p in TO.own_attackers_of(tok, float(olp)):
        attach += float(don_for_attacker(float(p), float(olp), theta, mu))
    return {"available": avail, "stock": float(cost_stock), "eff": float(cost_eff),
            "attach": float(attach), "asked": float(cost_stock) + float(cost_eff) + float(attach)}


def don_for_attacker(power, target_power, theta=None, mu=None, max_don=None, delta=None):
    """`attack_value_don` が**その攻め手に何枚付けているか**（`argmax_k` の `k`）。器の外に出す理由は
    **理論が暗黙に使っているドンを数える**ため（`attack_value_don` は値だけ返して枚数を捨てる）。"""
    theta = TO.THETA if theta is None else theta
    mu = TO.MU if mu is None else mu
    delta = TO.DELTA if delta is None else delta
    max_don = TO.ATTACK_DON_MAX if max_don is None else max_don
    if TO.ATTACK_DON_MODE != "don":
        return 0
    best, arg = TO.attack_value(float(power), float(target_power), True, theta, mu), 0
    for k in range(1, int(max_don) + 1):
        v = TO.attack_value(float(power) + 1000.0 * k, float(target_power), True, theta, mu) - k * float(delta)
        if v > best:
            best, arg = v, k
    return arg


def collect(dirs, limit_games=0):
    """**記録を 1 行ずつ舐めて（1）不変量（2）使い道の内訳（3）理論の使いすぎ**を測る。"""
    cards = PL.Cards()
    idx2cid = {i: c for c, i in GA._vocab().items()}
    from opcg_sim.loop import decks as D
    db = D.load_db()
    rows_n = bad_n = 0
    by_budget = {}
    kinds = {k: 0.0 for k in ("added", "minus", "attached", "unattached", "rested", "refreshed")}
    turn_avail, turn_used, turn_idle, turn_n = 0.0, 0.0, 0.0, 0
    ask = {"available": 0.0, "stock": 0.0, "eff": 0.0, "attach": 0.0, "asked": 0.0, "n": 0, "over": 0}
    games = 0
    for rows, pol, ex, L, ptr, idx in PL.iter_games(dirs, row_cols=ROW_COLS, pol_cols=POL_COLS,
                                                    extra_fn=_extra):
        games += 1
        if limit_games and games > limit_games:
            break
        first_of = {}          # (w, t) -> 4 ゾーン（そのターン最初の行）
        first_i = {}           # (w, t) -> その行の index
        last_of = {}
        order = [i for i in idx if int(rows["kind"][i]) == 0]
        for i in order:
            w, t = int(rows["who"][i]), int(rows["turn"][i])
            sc, tok, ci = ex["sc"][i], ex["tok"][i], ex["ci"][i]
            for side, seat in (("me", w), ("opp", 1 - w)):
                z = zones_of(sc, tok, side)
                cid = np.asarray(ci)[0 if side == "me" else 1]
                b = budget_of(idx2cid.get(int(cid)), db)
                rows_n += 1
                by_budget[b] = by_budget.get(b, 0) + 1
                if abs(z["total"] - b) > DON_EPS:
                    bad_n += 1
            z = zones_of(sc, tok, "me")
            if (w, t) not in first_of:
                first_of[(w, t)], first_i[(w, t)] = z, i
            last_of[(w, t)] = z
        for key, z0 in first_of.items():
            w, t = key
            if t < 1 or not PL.is_own_turn(w, t):
                continue
            z1 = last_of[key]
            fl = flows_between(z0, z1)
            for k in kinds:
                kinds[k] += fl[k]
            turn_avail += z0["active"]
            turn_idle += z1["active"]
            turn_used += max(0.0, z0["active"] - z1["active"])
            turn_n += 1
        # 理論の使いすぎ（**自席ターン開始の行だけ**＝リフレッシュと追加が済んだ後の財布）
        for key, i in first_i.items():
            w, t = key
            if t < 1 or not PL.is_own_turn(w, t):
                continue
            sc = np.asarray(ex["sc"][i])
            olp = float(sc[TO.SC_OPP_LEADER_POWER]) * 1e4 or 5000.0
            sp = theory_spend(sc, ex["tok"][i], ex["ci"][i], idx2cid, cards, olp)
            for k in ("available", "stock", "eff", "attach", "asked"):
                ask[k] += sp[k]
            ask["n"] += 1
            if sp["asked"] > sp["available"] + 1e-9:
                ask["over"] += 1
    out = {"games": games, "rows": rows_n, "conservation_violations": bad_n,
           "rows_by_budget": {str(k): v for k, v in sorted(by_budget.items())},
           "flow_per_turn": {k: (v / turn_n if turn_n else 0.0) for k, v in kinds.items()},
           "turns": turn_n,
           "available_per_turn": turn_avail / turn_n if turn_n else 0.0,
           "used_per_turn": turn_used / turn_n if turn_n else 0.0,
           "idle_per_turn": turn_idle / turn_n if turn_n else 0.0}
    n = max(1, ask["n"])
    out["theory"] = {"n": ask["n"], "over_rows": ask["over"],
                     "over_share": ask["over"] / n,
                     "available": ask["available"] / n, "stock": ask["stock"] / n,
                     "eff": ask["eff"] / n, "attach": ask["attach"] / n,
                     "asked": ask["asked"] / n,
                     "asked_over_available": ask["asked"] / max(1e-9, ask["available"])}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="ドンの帳尻（4 ゾーンの不変量・使い道・理論の使いすぎ）")
    ap.add_argument("--in", dest="src", nargs="+", required=True)
    ap.add_argument("--games", type=int, default=0)
    ap.add_argument("--don-purse", default="", help="理論側の財布の形（`off`／`one`／`all`）を指定して測る")
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)
    if a.don_purse:
        import crossing_bridge as CB
        CB.set_don_purse_mode(a.don_purse)
    out = collect(a.src, a.games)
    out["don_purse"] = a.don_purse or "off"
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
