#!/usr/bin/env python3
"""**リーダー別の較正と役割ごとの外れ**（2026-10-12・司令塔の依頼・合成 w39 の追加集計・1 回きり・テストなし）。

`theory_map.py` の行（`build_rows`）をそのまま使う。
1. リーダー別（色別も）: ターン先頭の行（決着の宣言の前・時計つき）で、席（局 × 席）ごとに理論の勝率 W の平均と実勝敗を出し、
   リーダーごとに単位の平均 W・実勝率・差（実 − W）。**局数 10 以上**のリーダーだけ並べる（差の大きい順＝負に大きい順）。
2. 役割ごとの外れ（実勝敗 − 行の W・単位ごとに平均してから単位の平均・`theory_map` の問い 5 と同じ式）を、
   「差が −0.08 を下回るリーダー（1 の表に載ったもの）の行」と「それ以外」の 2 群で。閾値 −0.08 は数を見る前に固定（司令塔）。
   行は問い 5 と同じ「理論は小さな損（逸脱 < 手札 1 枚）」の行と、参考に全行。

    OPCG_LOG_SILENT=1 python tests/scripts/theory_map_deckcal.py --in <記録> --work <theory_map の置き場> --out <json>
"""
import argparse
import collections
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import theory_map as TM  # noqa: E402

THRESH = -0.08
MIN_LEADER = 10


def unit_cal(rows):
    """rows → 単位（seed, w）ごとの (平均 W, z)。"""
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["seed"], r["w"])].append(r)
    return {k: (float(np.mean([r["W"] for r in v])), v[0]["z"]) for k, v in by.items()}


def cal_block(units, reps):
    if not units:
        return {"units": 0}
    ws = [w for w, _ in units.values()]
    zs = [z for _, z in units.values()]
    by_seed = collections.defaultdict(list)
    for (s, _w), v in units.items():
        by_seed[s].append(v)
    ci = TM.boot_games(by_seed, lambda us: float(np.mean([z - w for w, z in us])), reps)
    return {"units": len(units), "W": round(float(np.mean(ws)), 4), "win": round(float(np.mean(zs)), 4),
            "diff": round(float(np.mean(zs) - np.mean(ws)), 4), "diff_ci": ci}


def resid_groups(rows, reps):
    groups = collections.defaultdict(list)
    for r in rows:
        groups["ALL"].append(r)
        groups["type:" + r["ptype"]].append(r)
        for ro in r["role"]:
            groups["role:%s:%s" % (r["ptype"], ro)].append(r)
        if r["target"]:
            groups["attack:" + r["target"]].append(r)

    def res(rs):
        by = collections.defaultdict(list)
        for r in rs:
            by[(r["seed"], r["w"])].append(r["z"] - r["W"])
        return float(np.mean([np.mean(v) for v in by.values()])) if by else None

    out = {}
    for nm, rs in sorted(groups.items()):
        units = {(r["seed"], r["w"]) for r in rs}
        e = {"rows": len(rs), "units": len(units)}
        if len(units) < TM.MIN_GAMES:
            e["verdict"] = "局数不足"
        else:
            bys = collections.defaultdict(list)
            for r in rs:
                bys[r["seed"]].append(r)
            e["resid"] = round(res(rs), 4)
            e["resid_ci"] = TM.boot_games(bys, res, reps)
        out[nm] = e
    return out


def split(rows, bad, reps):
    small = [r for r in rows if (r["bv"] - r["pv"]) / MU < 1.0]
    return {
        "bad_leaders": sorted(bad),
        "small_loss": {"bad": resid_groups([r for r in small if r["leader"] in bad], reps),
                       "rest": resid_groups([r for r in small if r["leader"] not in bad], reps)},
        "all_rows": {"bad": resid_groups([r for r in rows if r["leader"] in bad], reps),
                     "rest": resid_groups([r for r in rows if r["leader"] not in bad], reps)},
    }


def main():
    global MU
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--boot", type=int, default=TM.BOOT)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from theory_rs import MU as _MU
    MU = _MU
    dump = TM.load_dump(os.path.join(a.work, "dump.jsonl"))
    clocks = json.load(open(os.path.join(a.work, "clocks.json"), encoding="utf-8"))
    rows_all, align, *_ = TM.build_rows(a.src, dump, clocks)
    rows = [r for r in rows_all if r["pv"] is not None and r["nsc"] >= 2 and r["ptype"] in TM.TYPES
            and r["btype"] in TM.TYPES and r.get("W") is not None]
    first = [r for r in rows if r["first_in_turn"] and not r["settled_after"]]
    reps = a.boot

    # 1. リーダー別・色別の較正
    by_lead = collections.defaultdict(list)
    for r in first:
        by_lead[r["leader"]].append(r)
    leaders = {ld: cal_block(unit_cal(rs), reps) for ld, rs in by_lead.items()}
    games_of = {ld: len({r["seed"] for r in rs}) for ld, rs in by_lead.items()}
    for ld in leaders:
        leaders[ld]["games"] = games_of[ld]
    listed = sorted([ld for ld in leaders if games_of[ld] >= MIN_LEADER], key=lambda d: leaders[d]["diff"])
    by_col = collections.defaultdict(list)
    for r in first:
        for c in r["colors"]:
            by_col[c].append(r)
    colors = {c: cal_block(unit_cal(rs), reps) for c, rs in by_col.items()}
    ref5 = sorted([ld for ld in leaders if games_of[ld] >= 5], key=lambda d: leaders[d]["diff"])

    bad = {ld for ld in listed if leaders[ld]["diff"] < THRESH}
    bad_ref5 = {ld for ld in ref5 if leaders[ld]["diff"] < THRESH}
    bad_col = {c for c, e in colors.items() if e.get("diff", 0) < THRESH}

    out = {
        "record": a.src, "align": dict(align), "rows": len(rows), "first_rows_pre_settle": len(first),
        "mu": MU, "thresh": THRESH, "min_leader_games": MIN_LEADER,
        "note": ("合成 w39 は 1 局ごとにデッキを無作為に作る（同じリーダーでも中身は別）。リーダーは 139 種・1 種あたり最大 9 局で、"
                 "局数 10 以上のリーダーは 0。固定した手順（1 の表に載ったリーダーで分ける）は bad 群が空になる。"
                 "参考として (a) 局数 5 以上のリーダーで同じ閾値、(b) 色で同じ閾値の分け方も出す（いずれも分け方を同じ行の差で選ぶので循環がある）。"),
        "q1_leaders_listed": [dict(leader=ld, **leaders[ld]) for ld in listed],
        "q1_all_leaders": len(leaders), "q1_max_games": max(games_of.values()),
        "q1_ref_leaders_ge5": [dict(leader=ld, **leaders[ld]) for ld in ref5],
        "q1_colors": dict(sorted(colors.items(), key=lambda kv: kv[1].get("diff", 0))),
        "q1_overall": cal_block(unit_cal(first), reps),
        "q2_fixed": split(rows, bad, reps),
        "q2_ref_leaders_ge5": split(rows, bad_ref5, reps),
    }
    sc = split([dict(r, leader=("BAD" if set(r["colors"]) & bad_col else "REST")) for r in rows], {"BAD"}, reps)
    sc["bad_colors"] = sorted(bad_col)
    out["q2_ref_colors"] = sc
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("deckcal: rows %d first %d leaders %d listed %d bad_ref5 %d bad_col %s" %
          (len(rows), len(first), len(leaders), len(listed), len(bad_ref5), sorted(bad_col)))


if __name__ == "__main__":
    raise SystemExit(main())
