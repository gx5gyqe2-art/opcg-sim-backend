"""**探す効果の選択の利得を、札の価値 `v` で実測する**（T65・読み取り専用・2026-09-16）。

T59 で「探す効果の価格 `μ + sel(k)` のうち `μ` は実現と一致し、選択の利得 `sel(k)` は物差しに見えない」と出た（罠 32）。
T64 の `v(札)`（使ったときの価値・`hand_spend.use_value`）で手札の中身が読めるので、**探して手に入れた札の `v`** と
**引いた札の `v`**（基準）を比べる。ユーザ指示 2026-09-16「探すのに使ったコストは考慮してね」＝探し札のドンのコスト・
能力の中の費用（ドン −1・手札を捨てる）を引いた**正味**で読む。

```
探した行:  自席の main 行で LOOK の登場時能力を持つ札を出した行（`primary_action == "LOOK"`）
前／後:    その行の手札 ／ 同じターンの次の判断点（kind 0）の手札
加わった:  後 − 前（探して手に入れた札）        消えた: 前 − 後 から探し札 1 枚を除いたもの（能力で捨てた札）
基準:      自席ターン開始（最初の main 行）の手札 − その席の直前の行の手札（ライフが減っていない行だけ）＝引いた札
実現:      Σ v(加わった) − Σ v(消えた) − コスト·δ（探し札のドン）             価格: `card_value(探し札, ON_PLAY)`（μ + sel(k) − 能力の費用）
利得:      v(加わった) − E[v(引いた)]                                          対 sel(k)（`effect_value._sel_premium`）
```

**回帰しない・当てはめない**（既存の価格だけ・新定数ゼロ）。

使い方: `python tests/scripts/search_gain.py --in <n_records ディレクトリ>... [--out x.json]`
"""
import argparse
import collections
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
from theory_order import DELTA, MU, POL_COLS, SC_MY_LIFE, SC_OPP_LEADER_POWER, SC_OPP_LIFE  # noqa: E402
from theory_bridge import ROW_COLS, _extra, move_family  # noqa: E402
from theory_bridge import is_decision_row as TB_is_decision_row  # noqa: E402  (D-5)
from theory_bridge import add_decision_row_arg, apply_decision_row  # noqa: E402  (D-5)
from hand_spend import hand_ids, spent_cards, use_value  # noqa: E402
from onplay_parts import look_k  # noqa: E402
from price_realised import primary_action  # noqa: E402


def _v(cid, cards, olp, r):
    return use_value(cid, cards.info(cid), olp, r)


def collect(dirs, limit_games=0):
    cards = PL.Cards()
    idx2cid = {i: c for c, i in GA._vocab().items()}
    searches, draws = [], []
    stats = {"games": 0, "search_rows": 0, "no_next": 0, "draw_rows": 0}
    games = 0
    for rows, pol, ex, L, ptr, idx in PL.iter_games(dirs, row_cols=ROW_COLS, pol_cols=POL_COLS, extra_fn=_extra):
        games += 1
        if limit_games and games > limit_games:
            break
        stats["games"] += 1
        order = list(idx)
        # 席ごとの main 行の並び（次の判断点）と、席ごとの全行の並び（直前の行）
        by_seat_main, by_seat_all = {}, {}
        for n, i in enumerate(order):
            w = int(rows["who"][i])
            by_seat_all.setdefault(w, []).append(n)
            if TB_is_decision_row(rows, pol, L, ptr, i):
                by_seat_main.setdefault(w, []).append(n)
        nxt = {}
        for w, ns in by_seat_main.items():
            for a, b in zip(ns, ns[1:]):
                nxt[a] = b
        prev_any = {}
        for w, ns in by_seat_all.items():
            for a, b in zip(ns, ns[1:]):
                prev_any[b] = a
        first_main_seen = set()
        for n, i in enumerate(order):
            w, t = int(rows["who"][i]), int(rows["turn"][i])
            if t < 1 or not PL.is_own_turn(w, t) or not TB_is_decision_row(rows, pol, L, ptr, i):
                continue
            sc = ex["sc"][i]
            olp = float(sc[SC_OPP_LEADER_POWER]) * 1e4 or 5000.0
            r = max(1.0, min(5.0, float(sc[SC_OPP_LIFE])))
            # --- 基準: ターン開始の引き（直前の行からライフが減っていない） ---
            if (w, t) not in first_main_seen:
                first_main_seen.add((w, t))
                p = prev_any.get(n)
                if p is not None:
                    ip = order[p]
                    if int(round(float(ex["sc"][ip][SC_MY_LIFE]))) == int(round(float(sc[SC_MY_LIFE]))):
                        added = spent_cards(hand_ids(ex["ci"][i], idx2cid), hand_ids(ex["ci"][ip], idx2cid))
                        for cid in added:
                            v = _v(cid, cards, olp, r)
                            if v is not None:
                                draws.append({"cid": cid, "v": float(v), "turn": t})
                        stats["draw_rows"] += 1
            # --- 探した行 ---
            k = int(L[i]); ch = int(rows["pol_chosen"][i])
            if k < 1 or ch < 0 or ch >= k:
                continue
            b = int(ptr[i]) + ch
            sig = json.loads(pol["pol_sig"][b])
            if move_family(sig) != "play":
                continue
            cid = str(pol["pol_cid"][b]) or None
            if not cid or primary_action(cid, EV.CHAR_ON_PLAY_TRIGGERS) != "LOOK":
                continue
            j = nxt.get(n)
            if j is None or int(rows["turn"][order[j]]) != t:
                stats["no_next"] += 1
                continue
            stats["search_rows"] += 1
            before = hand_ids(ex["ci"][i], idx2cid)
            after = hand_ids(ex["ci"][order[j]], idx2cid)
            added = spent_cards(after, before)
            removed = spent_cards(before, after)
            if cid in removed:
                removed.remove(cid)                       # 探し札そのもの（出した札）は除く
            info = cards.info(cid) or {}
            price_eff, _u = EV.card_value(cid, EV.CHAR_ON_PLAY_TRIGGERS, no_ability=0.0)
            va = [(_v(c, cards, olp, r)) for c in added]
            vr = [(_v(c, cards, olp, r)) for c in removed]
            # 探し札そのものの体の価値（ν・登場時効果を除く）＝札全体の実現 対 価格 を出すため
            body = TO.nu_of(float(info.get("power") or 0.0), olp, r, is_blocker=bool(info.get("blocker"))) if not info.get("event") else 0.0
            searches.append({"cid": cid, "k": look_k(cid), "cost": float(info.get("cost") or 0.0), "body": float(body),
                             "found": len(added), "added": added, "removed": removed,
                             "v_added": [x for x in va if x is not None], "v_removed": [x for x in vr if x is not None],
                             "unknown_added": sum(1 for x in va if x is None),
                             "price_effect": (None if price_eff is None else float(price_eff)), "turn": t})
    return searches, draws, stats


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def block(ss, draw_mean):
    n = len(ss)
    if n == 0:
        return {"n": 0}
    found = [s["found"] >= 1 for s in ss]
    va = [x for s in ss for x in s["v_added"]]
    vr_sum = [sum(s["v_removed"]) for s in ss]
    gross = [sum(s["v_added"]) for s in ss]                                  # 手に入れた札の v の和（見つからなければ 0）
    eff = [sum(s["v_added"]) - sum(s["v_removed"]) for s in ss]              # 能力の実現（手に入れた − 捨てた）
    # **札全体**: 体 + 能力 − 探し札のドン（ユーザ指示「探すのに使ったコストは考慮」）。価格側も同じ形（体 + 能力の価格 − ドン）
    net = [s["body"] + e - s["cost"] * DELTA for s, e in zip(ss, eff)]
    price_play = [None if s["price_effect"] is None else s["body"] + s["price_effect"] - s["cost"] * DELTA for s in ss]
    prem = [x - draw_mean for x in va] if draw_mean is not None else []
    k_mean = _mean([s["k"] for s in ss])
    sel = _mean([EV._sel_premium(int(s["k"])) for s in ss])
    return {"n": n, "k_mean": k_mean, "found": float(np.mean(found)),
            "v_added_mean": _mean(va), "v_draw_mean": draw_mean,
            # **選択の利得の実現**＝探した札の v − 引いた札の v（1 枚あたり）
            "premium_real": _mean(prem), "sel_k": sel,
            "premium_over_sel": (None if not prem or not sel else float(np.mean(prem) / sel)),
            # 能力の費用（捨てた札の v）と探し札のドン
            "v_removed_mean": _mean(vr_sum), "don_cost_mean": _mean([s["cost"] * DELTA for s in ss]),
            # 能力だけ: 実現（手に入れた − 捨てた）対 価格（μ + sel − 能力の費用）
            "gross_mean": _mean(gross), "effect_real_mean": _mean(eff), "price_effect_mean": _mean([s["price_effect"] for s in ss]),
            "effect_over_price": (lambda p: None if not p else float(np.mean(eff) / p))(_mean([s["price_effect"] for s in ss])),
            # 札全体: 体 + 能力 − ドン（実現）対 体 + 能力の価格 − ドン
            "body_mean": _mean([s["body"] for s in ss]), "net_mean": _mean(net), "price_play_mean": _mean(price_play),
            "net_over_price_play": (lambda p: None if not p else float(np.mean(net) / p))(_mean(price_play)),
            "unknown_added": int(sum(s["unknown_added"] for s in ss))}


def summarise(searches, draws, min_card=8):
    draw_mean = _mean([d["v"] for d in draws])
    out = {"draws": {"n": len(draws), "v_mean": draw_mean, "mu": MU,
                     "v_hist": None},
           "all": block(searches, draw_mean)}
    if draws:
        vs = np.array([d["v"] for d in draws])
        out["draws"]["v_hist"] = {"0": float(np.mean(vs <= 1e-9)), "0-mu": float(np.mean((vs > 1e-9) & (vs <= MU))),
                                  "mu-2mu": float(np.mean((vs > MU) & (vs <= 2 * MU))), ">2mu": float(np.mean(vs > 2 * MU))}
    for k in (3, 4, 5):
        ss = [s for s in searches if int(s["k"]) == k]
        if ss:
            out["k=%d" % k] = block(ss, draw_mean)
    by = collections.defaultdict(list)
    for s in searches:
        by[s["cid"]].append(s)
    out["by_card"] = {c: block(ss, draw_mean) for c, ss in sorted(by.items(), key=lambda kv: -len(kv[1])) if len(ss) >= min_card}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_decision_row_arg(ap)
    ap.add_argument("--in", dest="src", nargs="+", required=True, help="n_records のディレクトリ")
    ap.add_argument("--limit-games", type=int, default=0)
    ap.add_argument("--min-card", type=int, default=8)
    TO.add_nu_mode_arg(ap)
    TO.add_surv_mode_arg(ap)
    TO.add_cbar_mode_arg(ap)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    apply_decision_row(a)
    TO.apply_nu_mode(a)
    TO.apply_surv_mode(a)
    TO.apply_cbar_mode(a)
    t0 = time.time()
    searches, draws, stats = collect(a.src, a.limit_games)
    res = {"nu_mode": a.nu_mode, "surv_mode": a.surv_mode, "cbar_mode": a.cbar_mode, "stats": stats,
           "summary": summarise(searches, draws, a.min_card), "seconds": round(time.time() - t0, 1)}
    txt = json.dumps(res, ensure_ascii=False, indent=2)
    print(txt)
    if a.out:
        with open(os.path.expanduser(a.out), "w", encoding="utf-8") as fh:
            fh.write(txt + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
