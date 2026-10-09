"""**残りの芯の値打ち**（2026-10-09・`docs/reports/2026-10-09_residual_value.md`・診断だけ・後知恵の上限）。

地平を生き延びると言う時計（`S` ≥ 1/2）のうち歩きが守る側の計算の地平の外で交わるもの（群れ）の遅れが直ったと仮定した上限で、
交点の橋の的中（`crossing_bridge` の `sign_accuracy`・`theory`）と決着前の予測力（`win_calib --pre-settle on` の `abs` の AUC）が
どれだけ上がるかを測る。群れの入り方は予測の側の特徴だけ（`m2_probe.py --clocks-out` の時計の行: 計画あり・`S` ≥ 1/2・
`where ∈ {past, capped}`）。

- 主の上限: 群れの時計（両席）を `Δ = 群れの勝った席の平均の遅れ − 群れ以外の勝った席の平均の遅れ` だけ早める（報告 §1.3）。
- 副の上限: 群れの勝った席の時計を `act − 1/2` に置き換える（負けた席は主のまま・報告 §1.4）。

既定の値・表は変えない（`crossing_bridge.collect` を 2 度〔決着前で絞らない／絞る〕回し、時計を差し替えて同じ集計をし直すだけ）。

実行例（`m2_probe` と同じ温めた計画のディスクを使う）:
  OPCG_LOG_SILENT=1 OPCG_PLAN_STORE=<dir> python tests/scripts/residual_value.py --in ~/w41 --clocks clocks.jsonl --out rv.json
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
import theory_rs as TR  # noqa: E402
import win_calib as WC  # noqa: E402
from theory_rs import MU, THETA  # noqa: E402

CAP = CB.RACE_CAP


def load_clocks(path):
    """時計の行 → `(seed, t, who, side)` ごとの `(群れか, S, where)`。"""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            pd = c.get("pdeath")
            s = max(0.0, 1.0 - sum(float(x) for x in pd)) if (c.get("plan") and pd is not None) else None
            grp = bool(c.get("plan")) and s is not None and s >= 0.5 and c.get("where") in ("past", "capped")
            out[(c["seed"], c["t"], c["who"], c["side"])] = {"grp": grp, "S": s, "where": c.get("where"),
                                                              "tau_row": c.get("tau_row")}
    return out


def flat(rows_out, ck):
    """行ごとの 2 本の時計（`τ` は 30 で打ち切る）・群れの旗・実際の残り・勝敗。"""
    res, miss, tau_diff = [], 0, 0.0
    for r in rows_out:
        o = {"won": bool(r["won"]), "key": (r["seed"], r["t"], r["who"])}
        for side in ("me", "opp"):
            tau = min(float(r["tau_%s_theory" % side]), CAP)
            m = ck.get((r["seed"], r["t"], r["who"], side))
            if m is None:
                miss += 1
            elif m["tau_row"] is not None:
                tau_diff = max(tau_diff, abs(min(float(m["tau_row"]), CAP) - tau))
            o["tau_" + side] = tau
            o["grp_" + side] = bool(m and m["grp"])
            o["act_" + side] = int(r["t_%s_act" % side])
            o["win_" + side] = o["won"] if side == "me" else not o["won"]
        res.append(o)
    return res, miss, tau_diff


def winner_lags(fr, integer=False, me_only=False):
    g, ng = [], []
    for o in fr:
        for side in (("me",) if me_only else ("me", "opp")):
            if not o["win_" + side]:
                continue
            tau, act = o["tau_" + side], o["act_" + side]
            lag = (max(1, math.ceil(tau)) - act) if integer else (tau - act)
            (g if o["grp_" + side] else ng).append(lag)
    return g, ng


def shift_of(fr, **kw):
    g, ng = winner_lags(fr, **kw)
    return {"delta": float(np.mean(g) - np.mean(ng)), "grp_n": len(g), "grp_lag": float(np.mean(g)),
            "rest_n": len(ng), "rest_lag": float(np.mean(ng))}


def fixed(fr, delta, mode="main"):
    """直した時計の `(τ_me, τ_opp)` の並び。`mode`: none／main（群れを −Δ）／perfect（群れの勝った席を act − 1/2）。"""
    out = []
    for o in fr:
        t = {}
        for side in ("me", "opp"):
            tau = o["tau_" + side]
            if mode != "none" and o["grp_" + side]:
                if mode == "perfect" and o["win_" + side]:
                    tau = o["act_" + side] - 0.5
                else:
                    tau = max(0.0, tau - delta)
            t[side] = tau
        out.append((t["me"], t["opp"]))
    return out


def accuracy(fr, taus, sel=None):
    hit = [((tm <= to) == o["won"]) for o, (tm, to) in zip(fr, taus) if sel is None or sel(o)]
    return (round(float(np.mean(hit)), 4) if hit else None), len(hit)


def calib(fr, taus):
    rs = [(to - tm, tm, to, 1.0 if o["won"] else 0.0) for o, (tm, to) in zip(fr, taus)]
    z = [x[3] for x in rs]
    old = TR.CLOCK["W_ERR_MODE"]
    try:
        TR.set_w_err_mode("abs")
        s = WC.score(WC.probs_of(rs), z)
    finally:
        TR.set_w_err_mode(old)
    return {k: s[k] for k in ("n", "auc", "brier", "logloss", "ece")}


def sizes(fr):
    n = len(fr)
    cl = {(w, s): [0, 0] for w in (True, False) for s in ("me", "opp")}
    for o in fr:
        for s in ("me", "opp"):
            c = cl[(o["win_" + s], s)]
            c[1] += 1
            c[0] += int(o["grp_" + s])
    rows_any = sum(1 for o in fr if o["grp_me"] or o["grp_opp"])
    win_only = sum(1 for o in fr if (o["grp_me"] and o["win_me"] or o["grp_opp"] and o["win_opp"])
                   and not (o["grp_me"] and not o["win_me"] or o["grp_opp"] and not o["win_opp"]))
    lose_only = sum(1 for o in fr if (o["grp_me"] and not o["win_me"] or o["grp_opp"] and not o["win_opp"])
                    and not (o["grp_me"] and o["win_me"] or o["grp_opp"] and o["win_opp"]))
    tot = sum(v[1] for v in cl.values())
    grp = sum(v[0] for v in cl.values())
    return {"rows": n, "clocks": tot, "grp_clocks": grp, "grp_share": round(grp / max(1, tot), 4),
            "by_seat": {("winner" if w else "loser") + "_" + s: {"grp": v[0], "n": v[1], "share": round(v[0] / max(1, v[1]), 4)}
                        for (w, s), v in cl.items()},
            "rows_with_grp": rows_any, "rows_with_grp_share": round(rows_any / max(1, n), 4),
            "rows_grp_winner_side_only": win_only, "rows_grp_loser_side_only": lose_only,
            "rows_grp_both_sides": rows_any - win_only - lose_only}


def evaluate(fr, deltas, with_calib):
    has = lambda o: o["grp_me"] or o["grp_opp"]  # noqa: E731
    out = {}
    for name, (d, mode) in deltas.items():
        taus = fixed(fr, d, mode)
        e = {"delta": round(d, 4), "mode": mode, "acc": accuracy(fr, taus)[0],
             "acc_rows_with_grp": accuracy(fr, taus, has)[0], "acc_rows_without_grp": accuracy(fr, taus, lambda o: not has(o))[0]}
        if with_calib:
            e["calib"] = calib(fr, taus)
        out[name] = e
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="残りの芯の値打ち（後知恵の上限）")
    ap.add_argument("--in", dest="src", nargs="+", required=True)
    ap.add_argument("--clocks", nargs="+", required=True, help="`m2_probe.py --clocks-out` の時計の行（複数可）")
    ap.add_argument("--games", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    ck = {}
    for p in a.clocks:
        ck.update(load_clocks(p))
    res = {"src": a.src}
    # 1) 決着前で絞らない行（交点の橋の的中）——ずらし量はここの勝った席の時計から測る
    CB.set_pre_settle_mode("off")
    rows_out = CB.collect(a.src, a.games, THETA, MU, "const")[0]
    fr, miss, tdiff = flat(rows_out, ck)
    res["join"] = {"rows": len(rows_out), "clock_missing": miss, "tau_row_max_abs_diff": tdiff}
    res["sizes"] = sizes(fr)
    sh = {"main": shift_of(fr), "int": shift_of(fr, integer=True), "me_only": shift_of(fr, me_only=True)}
    res["shift"] = {k: {kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in s.items()} for k, s in sh.items()}
    d0 = sh["main"]["delta"]
    deltas = {"none": (0.0, "none"), "main": (d0, "main"), "perfect": (d0, "perfect"),
              "sens_int": (sh["int"]["delta"], "main"), "sens_me_only": (sh["me_only"]["delta"], "main")}
    res["bridge"] = evaluate(fr, deltas, with_calib=False)
    # 2) 決着前の行（`win_calib --pre-settle on` と同じ母数）・同じ Δ
    CB.set_pre_settle_mode("on")
    rows_pre = CB.collect(a.src, a.games, THETA, MU, "const")[0]
    fp, miss_p, tdiff_p = flat(rows_pre, ck)
    res["join_pre"] = {"rows": len(rows_pre), "clock_missing": miss_p, "tau_row_max_abs_diff": tdiff_p}
    res["sizes_pre"] = sizes(fp)
    res["pre_settle"] = evaluate(fp, deltas, with_calib=True)
    CB.set_pre_settle_mode("off")
    print(json.dumps(res, ensure_ascii=False, indent=1))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
