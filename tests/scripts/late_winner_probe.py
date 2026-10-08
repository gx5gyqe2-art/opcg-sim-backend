"""**地平の外で交わる勝った席の時計の遅れの原因調査**（2026-10-08・`docs/reports/2026-10-08_late_winner.md`・読み取り専用・診断だけ）。

M-2 の計器（`m2_probe.py --clocks-out`）が書いた時計ごとの行（JSON lines）を読み、集計するだけ。計算（守る側の計算の
解き直し・倒れる段の分布 `pdeath`）は Rust（`core::m2probe::resolve_death`）。既定の出力には触らない。

見るもの:

1. 地平の外で交わる時計の「割合」（交点の段 `J` の中で届いた割合 `frac`）の分布と、地平の終わりで歩きが耐久に足りない量
   （`θ − 地平までの和`）。
2. 倒れる段の分布（同じ計画・同じ地平で守る側の計算を「倒れた段にだけ 1」の値段で解き直したもの）: 地平の内で倒れない確率 `S`、
   倒れる段の真ん中 `M`、最後に倒れる段 `Tmax`。歩きの交点と比べる。
3. 診断の別の時計（式の提案の下調べ・既定は変えない）: 浮動小数の誤差の幅を許した歩き・倒れる時刻の真ん中で読む時計。

実行例:
  python tests/scripts/late_winner_probe.py --clocks real.jsonl --out late_real.json
"""
import argparse
import json
import math

import numpy as np

CAP = 30.0


def walk(theta, sched, j0, tol=0.0):
    """`m2probe::walk` の写し（盾 0・`step` 0）。`tol` は届いたとみなす誤差の幅（相対・0 なら既定と同じ）。"""
    f = 0.0
    for j in range(1, int(CAP) + 1):
        add = 0.0 if (j0 + j - 1 <= 1 or not sched) else sched[min(j, len(sched)) - 1]
        need = theta + 0.0 * j + 0.0
        if f + add >= need - tol * max(1.0, abs(theta)):
            short = max(0.0, need - f)
            frac = short / add if add > 1e-6 else 1.0
            return (j - 1) + min(frac, 1.0), j
        f += add
    return CAP, 0


def death_stats(pd):
    """倒れる段の分布 → `(S, M, Tmax, τ_med)`。段 `i`（0 始まり）で倒れる＝時計 `(i, i+1]`。"""
    tot = float(sum(pd))
    s = max(0.0, 1.0 - tot)
    cum, m, tmed = 0.0, None, None
    tmax = None
    for i, p in enumerate(pd):
        if p > 1e-6:
            tmax = i + 1
        if m is None and cum + p >= 0.5 - 1e-12:
            m = i + 1
            tmed = i + ((0.5 - cum) / p if p > 0 else 1.0)
        cum += p
    return s, m, tmax, tmed


def _stats(res):
    a = np.asarray(res, float)
    if not len(a):
        return None
    ai = None
    return {"n": int(len(a)), "bias": round(float(a.mean()), 3), "sigma": round(float(a.std()), 3),
            "within1": round(float((np.abs(a) <= 1.0).mean()), 4), "mae": round(float(np.abs(a).mean()), 3)} if ai is None else None


def _stats_int(taus, acts):
    r = np.array([max(1, math.ceil(min(t, CAP) - 1e-12)) - a for t, a in zip(taus, acts)], float)
    return {"bias_int": round(float(r.mean()), 3), "exact_int": round(float((r == 0).mean()), 4),
            "late_int": round(float((r > 0).mean()), 4), "early_int": round(float((r < 0).mean()), 4)}


def analyse(path):
    rows = [json.loads(l) for l in open(path)]
    out = {"n_clocks": len(rows)}
    W = [c for c in rows if c["winner"] and c.get("plan")]
    out["n_winner_plan"] = len(W)
    for c in W:
        c["tau_c"] = min(float(c["tau"]), CAP)
        c["res"] = c["tau_c"] - c["act"]
        j0 = int(c["j"]) + 1
        c["tau_re"], c["J_re"] = walk(float(c["theta"]), c["sched"], j0)
        c["tau_tol"], c["J_tol"] = walk(float(c["theta"]), c["sched"], j0, tol=1e-9)
        nh = int(c["nh"])
        c["cum_nh"] = float(sum(c["sched"][:nh])) if j0 > 1 else float(sum(c["sched"][1:nh]))
        c["gap_nh"] = float(c["theta"]) - c["cum_nh"]
        pd = c.get("pdeath")
        if pd is not None:
            c["S"], c["M"], c["Tmax"], c["tmed"] = death_stats(pd)
    out["walk_recompute_mismatch"] = int(sum(abs(c["tau_re"] - c["tau_c"]) > 1e-9 for c in W))
    groups = {"in": [c for c in W if c["where"] == "in"], "past": [c for c in W if c["where"] == "past"],
              "capped": [c for c in W if c["where"] == "capped"]}
    out["groups"] = {k: len(v) for k, v in groups.items()}
    out["resid"] = {k: _stats([c["res"] for c in v]) for k, v in groups.items() if v}
    out["resid_int"] = {k: _stats_int([c["tau_c"] for c in v], [c["act"] for c in v]) for k, v in groups.items() if v}
    P = groups["past"]
    # 1. 割合と地平の終わりの不足
    fr = np.array([c["frac"] for c in P], float)
    rel = np.array([c["gap_nh"] / max(1e-12, abs(float(c["theta"]))) for c in P], float)
    out["past_frac"] = {"le_1e-9": round(float((fr <= 1e-9).mean()), 4), "le_1e-3": round(float((fr <= 1e-3).mean()), 4),
                        "le_0.1": round(float((fr <= 0.1).mean()), 4), "median": round(float(np.median(fr)), 4) if len(fr) else None}
    out["past_gap_rel"] = {"le_1e-9": round(float((rel <= 1e-9).mean()), 4), "le_1e-3": round(float((rel <= 1e-3).mean()), 4),
                           "le_0.05": round(float((rel <= 0.05).mean()), 4), "median": round(float(np.median(rel)), 5) if len(rel) else None}
    out["past_J_minus_nh"] = {str(k): int(sum(1 for c in P if c["J"] - c["nh"] == k)) for k in range(1, 5)}
    out["past_act_minus_nh"] = {str(k): int(sum(1 for c in P if c["act"] - c["nh"] == k)) for k in range(-3, 4)}
    # 3a. 誤差の幅を許した歩き
    for k, v in groups.items():
        if not v:
            continue
        out.setdefault("tol_walk", {})[k] = {
            "moved": int(sum(abs(c["tau_tol"] - c["tau_c"]) > 1e-6 for c in v)),
            "moved_to_in": int(sum(1 for c in v if c["J_tol"] and c["J_tol"] <= c["nh"] and c["J"] > c["nh"])),
            "resid": _stats([min(c["tau_tol"], CAP) - c["act"] for c in v]),
            "resid_int": _stats_int([c["tau_tol"] for c in v], [c["act"] for c in v])}
    # 2. 倒れる段の分布
    for k, v in groups.items():
        vv = [c for c in v if c.get("S") is not None]
        if not vv:
            continue
        S = np.array([c["S"] for c in vv], float)
        d = {"n": len(vv), "n_life0": len(v) - len(vv),
             "S_mean": round(float(S.mean()), 4), "S_le_1e-6": round(float((S <= 1e-6).mean()), 4),
             "S_lt_0.5": round(float((S < 0.5).mean()), 4), "S_ge_0.5": round(float((S >= 0.5).mean()), 4)}
        mm = [c for c in vv if c["M"] is not None]
        if mm:
            d["J_minus_M"] = {str(q): int(sum(1 for c in mm if c["J"] - c["M"] == q)) for q in range(-1, 4)}
            d["J_minus_Tmax"] = {str(q): int(sum(1 for c in mm if c["Tmax"] is not None and c["J"] - c["Tmax"] == q)) for q in range(-2, 3)}
            d["act_minus_M"] = {str(q): int(sum(1 for c in mm if c["act"] - c["M"] == q)) for q in range(-3, 4)}
            d["M_eq_Tmax"] = round(float(np.mean([c["M"] == c["Tmax"] for c in mm])), 4)
            # 倒れる時刻の真ん中で読む時計（地平の内で半分以上倒れる時計だけ置き換え・ほかは既定のまま）
            d["resid_tmed"] = _stats([c["tmed"] - c["act"] for c in mm])
            d["resid_tmed_int"] = _stats_int([c["tmed"] for c in mm], [c["act"] for c in mm])
            d["resid_same_rows"] = _stats([c["res"] for c in mm])
            d["resid_same_rows_int"] = _stats_int([c["tau_c"] for c in mm], [c["act"] for c in mm])
            d["resid_min_walk_tmed"] = _stats([min(c["tmed"], c["tau_c"]) - c["act"] for c in mm])
            # 整数ターンの遅れの分解: ⌈τ⌉ − 実際 = (⌈τ⌉ − M) ＋ (M − 実際)。前者＝歩きが分布のどこを言うか、後者＝守る側の計算の真ん中と実際の差
            ci = [max(1, math.ceil(c["tau_c"] - 1e-12)) for c in mm]
            d["decomp_int"] = {"walk_minus_M": round(float(np.mean([a - c["M"] for a, c in zip(ci, mm)])), 3),
                               "M_minus_act": round(float(np.mean([c["M"] - c["act"] for c in mm])), 3),
                               "walk_minus_Tmax": round(float(np.mean([a - c["Tmax"] for a, c in zip(ci, mm)])), 3),
                               "Tmax_minus_act": round(float(np.mean([c["Tmax"] - c["act"] for c in mm])), 3)}
        # 倒れる段の平均（地平の内で倒れない分 S は既定の歩きの整数の段で倒れるとみなす）
        et = [sum((i + 1) * p for i, p in enumerate(c["pdeath"])) + c["S"] * max(1, math.ceil(c["tau_c"] - 1e-12)) for c in vv]
        d["mean_death"] = {"bias": round(float(np.mean([e - c["act"] for e, c in zip(et, vv)])), 3),
                           "sigma": round(float(np.std([e - c["act"] for e, c in zip(et, vv)])), 3),
                           "default_bias_int_same_rows": round(float(np.mean([max(1, math.ceil(c["tau_c"] - 1e-12)) - c["act"] for c in vv])), 3),
                           "default_sigma_int_same_rows": round(float(np.std([max(1, math.ceil(c["tau_c"] - 1e-12)) - c["act"] for c in vv])), 3)}
        out.setdefault("death", {})[k] = d
    # 全部の勝った席で「真ん中で読む時計」に差し替えたときの要約（半分以上が地平の内で倒れる時計だけ差し替え）
    def alt(c):
        if c.get("tmed") is not None:
            return min(c["tmed"], c["tau_c"])
        return c["tau_c"]
    out["all_winner"] = {"default": _stats([c["res"] for c in W]), "default_int": _stats_int([c["tau_c"] for c in W], [c["act"] for c in W]),
                         "tmed_min": _stats([alt(c) - c["act"] for c in W]),
                         "tmed_min_int": _stats_int([alt(c) for c in W], [c["act"] for c in W]),
                         "tol": _stats([min(c["tau_tol"], CAP) - c["act"] for c in W]),
                         "tol_int": _stats_int([c["tau_tol"] for c in W], [c["act"] for c in W])}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clocks", required=True)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    o = analyse(a.clocks)
    s = json.dumps(o, ensure_ascii=False, indent=1)
    if a.out:
        open(a.out, "w").write(s)
    print(s)


if __name__ == "__main__":
    main()
