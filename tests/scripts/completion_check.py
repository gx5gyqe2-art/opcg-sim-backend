"""**終わりの検算**（2026-10-09・`docs/reports/2026-10-09_completion_check.md`・読み取り専用・診断だけ）。

勝ちの時計に残る遅れ（`r_int = max(1, ⌈min(τ, 30)⌉) − act`）が、実際の守り手が守る側の計算の最善より余分に受けた損害 `E`
を、交点の付近の 1 段あたりの損害 `s` で割ったターン数 `δ = E / s` で説明できるかを測る（新定数ゼロ・当てはめなし）。

入力は `m2_probe.py --clocks-out` の時計の行と `survival_split.py --out` の時計ごとの測り（どちらも `OPCG_M2_PROBE`）。
「余分に受けた」は 2 通り（報告 §1.5）:

* (i) 窓の内・最初の行から: `hitsW_obs − hitsW_A`・`cutsW_obs − cutsW_A`（最初の行の守る側の計算に窓の内の実際の攻撃を入れた最善との差）。
* (ii) 席ごと・残りの全部のターン: 段 1..act の各々の自席ターンの頭の行で、その行の守る側の計算に今のターンの実際の攻撃を入れた最善
  （`now_act`）と実際との差を足す。ライフ 0 の守り手の行（`survival_split` が測らない）はここで解く（`--in` の記録が要る）。

本線（歩きの時計）は同じ行の `tau_walk` で読む（真ん中の直しは歩きを変えない）。

実行例:
  python tests/scripts/completion_check.py --clocks clocks_real.jsonl --recs recs_real.jsonl --in ~/w41 --out cc_real.json
"""
import argparse
import collections
import json
import math
import os
import sys

os.environ.setdefault("OPCG_M2_PROBE", "1")

import numpy as np  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import survival_split as SS  # noqa: E402

CAP = 30.0
FLOOR = 1e-3            # `outer::SLOPE_FLOOR`
EPS = SS.EPS


def r_int(tau, act):
    return max(1, math.ceil(min(float(tau), CAP) - 0.0)) - int(act)


def step_harm(c, tau, mode):
    """交点の付近の 1 段あたりの損害。`cross`＝`⌈τ⌉` 段目の歩きの段の値（床以下なら次の正の段・無ければ θ/τ）・`mean`＝θ/τ。"""
    th = float(c["theta"])
    t = min(float(tau), CAP)
    mean = th / t if t > 1e-9 else None
    if mode == "mean":
        return mean
    sch = [float(x) for x in (c.get("sched") or [])]
    if not sch:
        return mean
    j = max(1, math.ceil(t - 1e-12))
    for q in range(min(j, len(sch)) - 1, len(sch)):
        if sch[q] > FLOOR:
            return sch[q]
    return mean


def turn_delta(rec):
    """`survival_split` の時計の行 1 本から、その自席ターン（段 1）の (実際の本数, 最善の本数, 実際の切った札, 最善の切った札)。"""
    na = rec.get("now_act")
    st = rec["steps"][0] if rec.get("steps") else None
    if na is None or st is None or st.get("n_atk") is None:
        return None
    if rec["won"] and rec["act"] == 1:
        ho = rec.get("hitsW_obs")
        if ho is None:
            return None
    elif st.get("life_lost") is not None:
        ho = max(0, st["life_lost"])
    else:
        return None
    return float(ho), float(na["hits"]), float(st["counters"]), float(na["cuts"])


def turn_delta_life0(c, G):
    """ライフ 0 の守り手の自席ターン（`survival_split` は倒れる確率を測れず外す）: 地平 1 で解く。
    最善の本数＝止められる本数 < 当たる本数なら 1（とどめ）・切った札＝その解の切った札。"""
    w, t = int(c["who"]), int(c["t"])
    T = G["turns"].get((w, t))
    if T is None:
        return None
    din = c["din"]
    xa = [a["x"] for a in T["atk"] if a["leader"] and a["x"] is not None]
    n = sum(1 for x in xa if x >= -float(din["eps"]))
    if n == 0:
        dp_h, dp_c = 0.0, 0.0
    else:
        r = SS.resolve(din, turns=1, xs_first=xa)
        dp_h = 0.0 if float(r["prevented"]) >= n - 1e-9 else 1.0
        dp_c = float(r["cut"])
    if T.get("killed"):
        ho = 1.0
    elif T["def_life1"] is not None:
        ho = float(max(0, T["def_life0"] - T["def_life1"]))
    else:
        return None
    return ho, dp_h, float(T["counters"]), dp_c


def build(clocks_path, recs_path, src):
    clocks = {}
    for line in open(os.path.expanduser(clocks_path)):
        c = json.loads(line)
        if c.get("side") != "me" or "seed" not in c:
            continue
        clocks[(int(c["seed"]), int(c["who"]), int(c["t"]))] = c
    recs = {}
    for line in open(os.path.expanduser(recs_path)):
        r = json.loads(line)
        recs[(r["seed"], r["w"], r["t0"])] = r
    ext = SS.extract(src) if src else {}
    # 自席ターンごとの差（(ii) の材料）
    per_turn, how = {}, collections.Counter()
    for k, c in clocks.items():
        if k in recs:
            d = turn_delta(recs[k])
            how["rec" if d is not None else "rec_none"] += 1
        elif c.get("plan") and c.get("din") and round(float(c["din"]["life"])) < 1 and k[0] in ext:
            d = turn_delta_life0(c, ext[k[0]])
            how["life0" if d is not None else "life0_none"] += 1
        else:
            d = None
            how["no_din"] += 1
        if d is not None:
            per_turn[k] = d
    out = []
    for k, r in recs.items():
        if not r["won"]:
            continue
        c = clocks[k]
        din = c["din"]
        lam, mu = float(din["lam"]), float(din["mu"])
        act = int(r["act"])
        row = {"key": k, "act": act, "H": r["H"], "S": r["S"], "in_h": act <= r["H"],
               "tau_med": float(c["tau"]), "tau_walk": float(c["tau_walk"]), "lam": lam, "mu": mu, "theta": float(c["theta"])}
        # (i)
        if r.get("hitsW_obs") is not None and r.get("hitsW_A") is not None:
            row["i_dh"] = r["hitsW_obs"] - r["hitsW_A"]
            row["i_dc"] = r["cutsW_obs"] - r["cutsW_A"]
        # (ii)
        dh, dc, miss = 0.0, 0.0, 0
        for s in range(1, act + 1):
            d = per_turn.get((k[0], k[1], k[2] + 2 * (s - 1)))
            if d is None:
                miss += 1
                continue
            dh += d[0] - d[1]
            dc += d[2] - d[3]
        row["ii_dh"], row["ii_dc"], row["ii_miss"] = dh, dc, miss
        for rd, tk in (("med", "tau_med"), ("walk", "tau_walk")):
            row["s_cross_" + rd] = step_harm(c, row[tk], "cross")
            row["s_mean_" + rd] = step_harm(c, row[tk], "mean")
        out.append(row)
    return out, dict(how)


def _band_act(a):
    return "1" if a <= 1 else "2" if a == 2 else "3" if a == 3 else "4-5" if a <= 5 else "6+"


def summarise(rows, src_kind, e_kind, s_kind, reading, need_complete=False):
    """1 つの組（測り方・損害・段の値・読み方）の全体と帯ごと。"""
    tk = "tau_" + reading
    sel = []
    for r in rows:
        dh, dc = r.get(src_kind + "_dh"), r.get(src_kind + "_dc")
        if dh is None:
            continue
        if need_complete and src_kind == "ii" and r["ii_miss"]:
            continue
        s = r["s_%s_%s" % (s_kind, reading)]
        if s is None or s <= 0:
            continue
        E = dh * r["lam"] + (dc * r["mu"] if e_kind == "HC" else 0.0)
        tau = r[tk]
        delta = E / s
        sel.append((r, tau, delta))

    def agg(xs):
        if not xs:
            return None
        ri = np.array([r_int(t, r["act"]) for r, t, _ in xs], float)
        rf = np.array([min(t, CAP) - r["act"] for r, t, _ in xs], float)
        dl = np.array([d for _, _, d in xs], float)
        ri2 = np.array([max(1, math.ceil(min(t - d, CAP))) - r["act"] for r, t, d in xs], float)
        mr = float(ri.mean())
        return {"n": len(xs), "r_int": round(mr, 4), "r_frac": round(float(rf.mean()), 4),
                "delta": round(float(dl.mean()), 4), "delta_med": round(float(np.median(dl)), 4),
                "rho": round(float(dl.mean()) / mr, 3) if abs(mr) > 1e-9 else None,
                "r_int_after": round(float(ri2.mean()), 4), "r_frac_after": round(float((rf - dl).mean()), 4),
                "exact_before": round(float((ri == 0).mean()), 4), "exact_after": round(float((ri2 == 0).mean()), 4)}
    res = {"all": agg(sel), "bands": {}}
    bands = {"S>=1/2": lambda r: r["S"] >= 0.5, "S<1/2": lambda r: r["S"] < 0.5,
             "in_h": lambda r: r["in_h"], "out_h": lambda r: not r["in_h"]}
    for b in ("1", "2", "3", "4-5", "6+"):
        bands["act=" + b] = (lambda bb: (lambda r: _band_act(r["act"]) == bb))(b)
    for name, f in bands.items():
        res["bands"][name] = agg([x for x in sel if f(x[0])])
    return res


def verdict(primary):
    a = primary["all"]
    c1 = a["rho"] is not None and 0.8 <= a["rho"] <= 1.2
    c2 = abs(a["r_int_after"]) <= 0.1
    bad = []
    for name, b in primary["bands"].items():
        if b is None or b["n"] < 30:
            continue
        if abs(b["r_int"]) < 0.1:
            if abs(b["r_int_after"]) > 0.1:
                bad.append(name)
            continue
        if np.sign(b["delta"]) != np.sign(b["r_int"]):
            bad.append(name)
    return {"size_0.8_1.2": c1, "after_within_0.1": c2, "sign_bands_ok": not bad, "sign_bands_bad": bad,
            "pass": bool(c1 and c2 and not bad)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--clocks", required=True)
    ap.add_argument("--recs", required=True)
    ap.add_argument("--in", dest="src", nargs="*", default=[])
    ap.add_argument("--out", default="")
    ap.add_argument("--rows-out", default="")
    a = ap.parse_args(argv)
    rows, how = build(a.clocks, a.recs, a.src)
    res = {"winner_clocks": len(rows), "turn_sources": how,
           "ii_complete": int(sum(1 for r in rows if r["ii_miss"] == 0)), "combos": {}}
    # 平均の差（本数・札）
    for sk in ("i", "ii"):
        dh = [r[sk + "_dh"] for r in rows if r.get(sk + "_dh") is not None]
        dc = [r[sk + "_dc"] for r in rows if r.get(sk + "_dc") is not None]
        res["mean_extra_" + sk] = {"hits": round(float(np.mean(dh)), 4), "cuts": round(float(np.mean(dc)), 4), "n": len(dh)}
    for reading in ("med", "walk"):
        for sk in ("i", "ii"):
            for ek in ("H", "HC"):
                for stk in ("cross", "mean"):
                    res["combos"]["%s/%s/%s/%s" % (reading, sk, ek, stk)] = summarise(rows, sk, ek, stk, reading)
        res["combos"]["%s/ii_complete/H/cross" % reading] = summarise(rows, "ii", "H", "cross", reading, need_complete=True)
    res["verdict_primary"] = verdict(res["combos"]["med/i/H/cross"])
    res["verdict_walk"] = verdict(res["combos"]["walk/i/H/cross"])
    txt = json.dumps(res, ensure_ascii=False, indent=1)
    if a.out:
        with open(os.path.expanduser(a.out), "w") as fh:
            fh.write(txt + "\n")
    if a.rows_out:
        with open(os.path.expanduser(a.rows_out), "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
    print(json.dumps({"verdict_primary": res["verdict_primary"], "primary": res["combos"]["med/i/H/cross"]["all"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
