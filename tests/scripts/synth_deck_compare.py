"""合成デッキと実デッキの分布の比較表（WP `synth-decks-v2`・`docs/search_efficiency_plan.md` §7.18）。

合成（`--decks synth_roles` など・random リーダー・seed 0..N/2-1 の両席＝N 本）と実デッキ
（`--decks user` の 4 本・ホールドアウト）で、1 デッキあたりの量を並べ、差の大きい順に表にする。

差の物差し（`z`）: |合成の平均 − 実の平均| ÷ max(合成の標準偏差, 実の標準偏差, 0.5)。
実は 4 本しか無いので、**実の範囲 [最小, 最大] に入る合成の割合**も併記する（どちらも記述用）。

`--leader-cond` を付けると、実デッキ 4 リーダーそれぞれで合成を `--per-leader` 本組み、
同じリーダーの実デッキと突き合わせる（リーダーを揃えた比較＝色の差を除いた差）。

    python tests/scripts/synth_deck_compare.py --decks synth_roles --n 2000 --out /tmp/cmp.json
"""
import argparse
import collections
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
os.environ.setdefault("OPCG_LOG_SILENT", "1")

from opcg_sim.loop import decks as D  # noqa: E402
from opcg_sim.loop import deck_roles as R  # noqa: E402
from opcg_sim.loop import deck_synth as DS  # noqa: E402

_DRAW = re.compile(r"カード\d枚(まで)?を引")
_SEARCH = re.compile(r"デッキの上から\d枚")


def is_blocker(m):
    return "ブロッカー" in (getattr(m, "keywords", ()) or ())


def is_trigger(m):
    return bool((getattr(m, "trigger_text", "") or "").strip())


def is_search(m):
    t = getattr(m, "effect_text", "") or ""
    return bool(_SEARCH.search(t)) and "手札に加え" in t


def is_draw(m):
    return bool(_DRAW.search(getattr(m, "effect_text", "") or ""))


def removal_forms(m):
    return {k.split(":")[0] for k in R.classify(m)}


def deck_metrics(db, leader_id, card_ids):
    """1 デッキ → 量の dict（pure）。"""
    lm = db.get_card(leader_id)
    ms = [db.get_card(c) for c in card_ids]
    ltr = DS.traits_of(lm)
    out = {"n_colors": len(DS.colors_of(lm))}
    for k in range(11):
        out[f"cost_{k}"] = sum(1 for m in ms if min(getattr(m, "cost", 0) or 0, 10) == k)
    out["cost_avg"] = sum((m.cost or 0) for m in ms) / len(ms)
    for v in (0, 1000, 2000):
        out[f"ctr_{v}"] = sum(1 for m in ms if (m.counter or 0) == v)
    out["blocker"] = sum(1 for m in ms if is_blocker(m))
    out["trigger"] = sum(1 for m in ms if is_trigger(m))
    out["event"] = sum(1 for m in ms if m.type.name == "EVENT")
    out["stage"] = sum(1 for m in ms if m.type.name == "STAGE")
    fs = [removal_forms(m) for m in ms]
    out["removal"] = sum(1 for f in fs if f)
    for f in R.FORMS:
        out[f"rm_{f}"] = sum(1 for x in fs if f in x)
    out["search"] = sum(1 for m in ms if is_search(m))
    out["draw"] = sum(1 for m in ms if is_draw(m))
    cnt = collections.Counter(card_ids)
    out["distinct"] = len(cnt)
    out["n_4copies"] = sum(1 for v in cnt.values() if v >= 4)
    out["n_1copy"] = sum(1 for v in cnt.values() if v == 1)
    out["leader_trait_share"] = sum(1 for m in ms if DS.traits_of(m) & ltr) / len(ms)
    out["name_ref_leader"] = sum(1 for m in ms if lm.name in (m.effect_text or ""))
    out["vanilla"] = sum(1 for m in ms if not DS.has_effect(m))
    out["dead_rate"] = DS.audit_deck(lm, ms)["dead_rate"]
    return out


def _stat(xs):
    n = len(xs)
    mu = sum(xs) / n
    sd = math.sqrt(sum((x - mu) ** 2 for x in xs) / max(n - 1, 1)) if n > 1 else 0.0
    return mu, sd


def compare(synth_rows, real_rows):
    """量ごとの比較行（差の大きい順）。"""
    out = []
    for k in real_rows[0]:
        s = [r[k] for r in synth_rows]
        r = [x[k] for x in real_rows]
        ms, ss = _stat(s)
        mr, sr = _stat(r)
        z = abs(ms - mr) / max(ss, sr, 0.5)
        lo, hi = min(r), max(r)
        inr = sum(1 for x in s if lo <= x <= hi) / len(s)
        out.append({"metric": k, "synth_mean": round(ms, 3), "synth_sd": round(ss, 3),
                    "real_mean": round(mr, 3), "real_min": lo, "real_max": hi,
                    "z": round(z, 3), "in_real_range": round(inr, 3)})
    out.sort(key=lambda x: -x["z"])
    return out


def synth_rows(db, decks, n):
    rows = []
    for seed in range((n + 1) // 2):
        la, lb = D.leader_pair(db, seed, "random")
        (l1, d1), (l2, d2) = D.build_pair(db, la, lb, seed, decks)
        rows.append(deck_metrics(db, l1, d1))
        rows.append(deck_metrics(db, l2, d2))
    return rows[:n]


def legality(db, decks, n):
    """`deck_synth.check_legal` で n 本を検査し、違反を返す。"""
    bad = []
    for seed in range((n + 1) // 2):
        la, lb = D.leader_pair(db, seed, "random")
        for l, d in D.build_pair(db, la, lb, seed, decks):
            why = DS.check_legal(db.get_card(l), [db.get_card(c) for c in d])
            if why:
                bad.append({"seed": seed, "leader": l, "why": why})
    return bad


def leader_cond(db, decks, per_leader):
    out = {}
    for lid, cards in D.user_decks():
        rows, seen = [], set()
        for seed in range(per_leader):
            (l1, d1), _ = D.build_pair(db, lid, lid, seed, decks)
            rows.append(deck_metrics(db, l1, d1))
            seen.add(tuple(sorted(d1)))
        out[lid] = {"distinct_decks": len(seen), "table": compare(rows, [deck_metrics(db, lid, cards)])}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decks", default="synth_roles")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--leader-cond", action="store_true")
    ap.add_argument("--per-leader", type=int, default=50)
    ap.add_argument("--legal", action="store_true", help="合法性の検査もする")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    db = D.load_db()
    real = [deck_metrics(db, l, c) for l, c in D.user_decks()]
    res = {"decks": a.decks, "n": a.n, "table": compare(synth_rows(db, a.decks, a.n), real),
           "real_per_deck": real}
    if a.leader_cond:
        res["leader_cond"] = leader_cond(db, a.decks, a.per_leader)
    if a.legal:
        bad = legality(db, a.decks, a.n)
        res["illegal"] = {"n": len(bad), "examples": bad[:5]}
    print(f"{'metric':22s} {'synth':>8s} {'sd':>6s} {'real':>8s} {'[min,max]':>12s} {'z':>6s} {'inR':>5s}")
    for r in res["table"]:
        print(f"{r['metric']:22s} {r['synth_mean']:8.2f} {r['synth_sd']:6.2f} {r['real_mean']:8.2f} "
              f"{str([r['real_min'], r['real_max']]):>12s} {r['z']:6.2f} {r['in_real_range']:5.2f}")
    if a.legal:
        print("illegal:", res["illegal"]["n"])
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
