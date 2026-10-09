"""**勝った席だけを直す天井**（2026-10-09・`docs/reports/2026-10-09_ceiling_redo.md`・診断だけ・後知恵の天井）。

`residual_value.py` の上限は群れの時計を両席とも早めた（負けた席の正しい読みまで壊した）。ここでは群れの**勝った席の時計だけ**を
(a) 群れ以外の勝った席と同じ平均の遅れになるよう `Δ` だけ早める／(b) 実際の残り `act − 1/2` に置き換える、の 2 通りで直し、
交点の橋の的中と決着前の予測力（主＝相対の腕・副＝絶対の腕）を直す前と並べる。群れを予測の側の帯（倒れない確率・守る側の地平の段数）で
割った天井も出す。どちらも後知恵（勝った席を知っている）で、予測の側で作れる式ではない。

実行例（`m2_probe` と同じ温めた計画のディスクを使う）:
  OPCG_LOG_SILENT=1 OPCG_PLAN_STORE=<dir> python tests/scripts/ceiling_redo.py --in ~/w41 --clocks clocks.jsonl --out cr.json
"""
import argparse
import json
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import crossing_bridge as CB  # noqa: E402
import residual_value as RV  # noqa: E402
import theory_rs as TR  # noqa: E402
import win_calib as WC  # noqa: E402
from theory_rs import MU, THETA  # noqa: E402

CAP = CB.RACE_CAP
S_BANDS = ((0.5, 0.7), (0.7, 0.9), (0.9, 1.0001))


def load_clocks(path, out):
    """`residual_value.load_clocks` に守る側の地平の段数 `nh` を足す。"""
    with open(path, encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            pd = c.get("pdeath")
            s = max(0.0, 1.0 - sum(float(x) for x in pd)) if (c.get("plan") and pd is not None) else None
            grp = bool(c.get("plan")) and s is not None and s >= 0.5 and c.get("where") in ("past", "capped")
            out[(c["seed"], c["t"], c["who"], c["side"])] = {"grp": grp, "S": s, "where": c.get("where"),
                                                              "nh": c.get("nh"), "tau_row": c.get("tau_row")}


def attach(fr, rows_out, ck):
    for o, r in zip(fr, rows_out):
        for side in ("me", "opp"):
            m = ck.get((r["seed"], r["t"], r["who"], side)) or {}
            o["S_" + side], o["nh_" + side] = m.get("S"), m.get("nh")


def s_band(s):
    for lo, hi in S_BANDS:
        if s is not None and lo <= s < hi:
            return "[%.1f,%.1f)" % (lo, min(hi, 1.0))
    return None


def fixed(fr, delta, mode, sel=None):
    """群れの**勝った席の時計だけ**を直す（`sel(o, side)` でさらに絞る）。`mode`: none／shift（−Δ）／perfect（act − 1/2）。"""
    out = []
    for o in fr:
        t = {}
        for side in ("me", "opp"):
            tau = o["tau_" + side]
            if mode != "none" and o["grp_" + side] and o["win_" + side] and (sel is None or sel(o, side)):
                tau = (o["act_" + side] - 0.5) if mode == "perfect" else max(0.0, tau - delta)
            t[side] = tau
        out.append((t["me"], t["opp"]))
    return out


def calib(fr, taus, mode, sigma_rel):
    rs = [(to - tm, tm, to, 1.0 if o["won"] else 0.0) for o, (tm, to) in zip(fr, taus)]
    z = [x[3] for x in rs]
    old = TR.CLOCK["W_ERR_MODE"]
    try:
        TR.set_w_err_mode(mode)
        s = WC.score(WC.probs_of(rs, sigma_rel if mode == "rel" else None), z)
    finally:
        TR.set_w_err_mode(old)
    return {k: s[k] for k in ("n", "auc", "brier", "logloss", "ece")}


def bias(fr, taus, sel=lambda o, side: True):
    """勝った席の時計の偏り（整数＝主・小数＝副）。"""
    bi, bf = [], []
    for o, (tm, to) in zip(fr, taus):
        for side, tau in (("me", tm), ("opp", to)):
            if not o["win_" + side] or not sel(o, side):
                continue
            tc = min(tau, CAP)
            bi.append(max(1, math.ceil(tc)) - o["act_" + side])
            bf.append(tc - o["act_" + side])
    return {"n": len(bi), "int": round(float(np.mean(bi)), 4) if bi else None,
            "frac": round(float(np.mean(bf)), 4) if bf else None}


def evaluate(fr, fp, delta, sigma_rel, sel=None):
    out = {}
    for mode in ("none", "shift", "perfect"):
        tb = fixed(fr, delta, mode, sel)
        tp = fixed(fp, delta, mode, sel)
        e = {"acc": RV.accuracy(fr, tb)[0], "pre_rel": calib(fp, tp, "rel", sigma_rel),
             "pre_abs": calib(fp, tp, "abs", sigma_rel)}
        if sel is None:
            g = lambda o, side: o["grp_" + side]  # noqa: E731
            e["bias_all_winners"] = bias(fr, tb)
            e["bias_grp_winners"] = bias(fr, tb, g)
            e["bias_rest_winners"] = bias(fr, tb, lambda o, side: not o["grp_" + side])
        out[mode] = e
    for mode in ("shift", "perfect"):
        out[mode]["gain_acc"] = round(out[mode]["acc"] - out["none"]["acc"], 4)
        out[mode]["gain_auc_rel"] = round(out[mode]["pre_rel"]["auc"] - out["none"]["pre_rel"]["auc"], 4)
        out[mode]["gain_auc_abs"] = round(out[mode]["pre_abs"]["auc"] - out["none"]["pre_abs"]["auc"], 4)
    out["judge"] = {"gain_acc_max": max(out["shift"]["gain_acc"], out["perfect"]["gain_acc"]),
                    "gain_auc_rel_max": max(out["shift"]["gain_auc_rel"], out["perfect"]["gain_auc_rel"])}
    return out


def band_table(fr, fp, delta, sigma_rel, key_of):
    keys = sorted({key_of(o, s) for o in fr for s in ("me", "opp") if o["grp_" + s]} - {None}, key=lambda x: (0, float(x), "") if isinstance(x, (int, float)) else (1, 0.0, str(x)))
    out = {}
    for k in keys:
        sel = lambda o, side, k=k: key_of(o, side) == k  # noqa: E731
        n = w = 0
        for o in fr:
            for side in ("me", "opp"):
                if o["grp_" + side] and sel(o, side):
                    n += 1
                    w += int(o["win_" + side])
        g = lambda o, side, sel=sel: o["grp_" + side] and sel(o, side)  # noqa: E731
        ev = evaluate(fr, fp, delta, sigma_rel, sel)
        out[str(k)] = {"clocks": n, "winner_share": round(w / max(1, n), 4),
                       "winner_lag": bias(fr, fixed(fr, 0, "none"), g),
                       "gain_acc_shift": ev["shift"]["gain_acc"], "gain_acc_perfect": ev["perfect"]["gain_acc"],
                       "gain_auc_rel_shift": ev["shift"]["gain_auc_rel"], "gain_auc_rel_perfect": ev["perfect"]["gain_auc_rel"]}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="勝った席だけを直す天井（後知恵）")
    ap.add_argument("--in", dest="src", nargs="+", required=True)
    ap.add_argument("--clocks", nargs="+", required=True, help="`m2_probe.py --clocks-out` の時計の行（複数可）")
    ap.add_argument("--games", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    ck = {}
    for p in a.clocks:
        load_clocks(p, ck)
    sigma_rel = CB.sigma_rel_for(a.src)
    res = {"src": a.src, "sigma_rel": sigma_rel}
    CB.set_pre_settle_mode("off")
    rows_out = CB.collect(a.src, a.games, THETA, MU, "const")[0]
    fr, miss, tdiff = RV.flat(rows_out, ck)
    attach(fr, rows_out, ck)
    CB.set_pre_settle_mode("on")
    rows_pre = CB.collect(a.src, a.games, THETA, MU, "const")[0]
    fp, miss_p, _ = RV.flat(rows_pre, ck)
    attach(fp, rows_pre, ck)
    CB.set_pre_settle_mode("off")
    res["join"] = {"rows": len(rows_out), "rows_pre": len(rows_pre), "clock_missing": miss + miss_p,
                   "tau_row_max_abs_diff": tdiff}
    res["sizes"] = RV.sizes(fr)
    sh = RV.shift_of(fr)
    res["shift"] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in sh.items()}
    d = sh["delta"]
    res["main"] = evaluate(fr, fp, d, sigma_rel)
    res["bands_S"] = band_table(fr, fp, d, sigma_rel, lambda o, side: s_band(o["S_" + side]))
    res["bands_nh"] = band_table(fr, fp, d, sigma_rel, lambda o, side: o["nh_" + side])
    print(json.dumps(res, ensure_ascii=False, indent=1))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
