"""**M-2 の大きさの計器**（2026-10-08・`docs/reports/2026-10-08_m2_probe.md`・読み取り専用・診断だけ）。

ユーザ決定 2026-10-07「式を変える前に、M-2 の残りの大きさを先に測る」。H-4（`rule_don`）の既定では、守る側の計算（動的計画法）が
地平の内側で「切れる間は切らせ、あとは受けさせる」を厳密に値付けしている。手札の残りを見ない 1 本ずつの値段が残るのは 4 か所:
(1) 地平の先の段（`fb`）・(2) 攻め手が引く札の流入（`a_tab`／`ar_tab`）・(3) 効果（`e_tab`・出した札の `eff`）・
(4) 攻め手か守り手の手札が読めず旧来の速さに落ちる道。この器はそれが交点の橋の時計にどれだけ効いているかを数える。

中身（計算は Rust の `core::m2probe`・`drv_cb::m2_clock`。ここは `OPCG_M2_PROBE=1` を立てて `crossing_bridge.collect` を回し、
Rust が返す診断の行を集計するだけ）:

1. 時計（行ごとに自分と相手の 2 本）が耐久に届く段が地平の内か外か（届かない・落ちる道も別に数える）。
2. 交点（届かなければ 30 ターンの打ち切り）までに積んだ損害の出どころの割合（地平の内側の段の損害・地平の先の `fb`・流入・効果・
   落ちる道）。
3. 打ち切り（30 ターンで届かない時計）の数と、その中で一番大きい出どころ。
4. 地平を縮めた検算: 同じ計画のまま守る側の計算を地平 `h − k`（`k` = 1, 2, 3）で解き直し、尾を地平の外の値付けで歩かせた時刻を
   元の地平 `h` の時刻と比べる（A: 元の段の損害を `h − k` で切る・耐久は元のまま／B: 短い地平で解き直した損害と耐久／
   C: 解き直した損害・耐久は元のまま）。**診断だけ**（採る計画も既定の値も変えない）。
5. 診断の別の歩き（引く札を抜く・効果を抜く・地平の先を「受ける」の値段で粗く値付けする）で的中・偏り・σ_T がどう動くか。
   **式の提案ではない**（新定数ゼロ・既定は変えない）。

既定の器（`crossing_bridge` ほか）の出力は変わらない: 変数が無ければ Rust は診断の欄を作らない。

実行例:
  OPCG_LOG_SILENT=1 OPCG_PLAN_STORE=/tmp/ps python tests/scripts/m2_probe.py --in ~/w41 --out ~/m2_w41.json
"""
import argparse
import copy
import json
import math
import os
import sys
import time

os.environ["OPCG_M2_PROBE"] = "1"      # Rust は最初の呼び出しで 1 度だけ読む（import より前に立てる）

import numpy as np  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import crossing_bridge as CB  # noqa: E402

CAP = CB.RACE_CAP
SRC = ("dp", "fb", "flow", "eff", "fallback")
VARIANTS = ("parts", "noflow", "noeff", "core", "take", "take_core")


def _band(act):
    a = int(act)
    return "1" if a <= 1 else "2" if a == 2 else "3" if a == 3 else "4-5" if a <= 5 else "6+"


def _q(xs):
    if not len(xs):
        return None
    a = np.asarray(xs, float)
    return {"n": int(len(a)), "mean": round(float(a.mean()), 4), "q25": round(float(np.quantile(a, 0.25)), 4),
            "median": round(float(np.median(a)), 4), "q75": round(float(np.quantile(a, 0.75)), 4)}


def clocks(rows_out, m2rows):
    """行ごとの 2 本の時計を平らにする（勝った席の時計には実際の残りの自席ターン `act`）。"""
    out = []
    for r, m in zip(rows_out, m2rows):
        won_me = bool(r["won"])
        act_w = r["t_me_act"] if won_me else r["t_opp_act"]
        for side in ("me", "opp"):
            c = dict(m[side])
            c["side"] = side
            c["winner"] = won_me if side == "me" else not won_me
            c["act"] = (r["t_me_act"] if side == "me" else r["t_opp_act"]) if c["winner"] else None
            c["band"] = _band(act_w)
            c["tau_row"] = r["tau_me_theory" if side == "me" else "tau_opp_theory"]
            # 局と行の鍵・攻め手の残りの自席ターン（負けた席では打ち切りの長さ）——`survival_split.py` が記録と突き合わせる
            c["seed"], c["t"], c["who"] = r["seed"], r["t"], r["who"]
            c["own_left"] = r["t_me_act"] if side == "me" else r["t_opp_act"]
            c["plan"] = c["why"] == "plan"
            if c["plan"]:
                c["where"] = "capped" if c["J"] == 0 else ("in" if c["J"] <= c["nh"] else "past")
            else:
                c["where"] = "fallback"
            acc = c["acc"]
            sh = {}
            for s in SRC:
                sh[s] = (c.get("acc_" + s, 0.0) / acc) if acc > 1e-12 else 0.0
            c["share"] = sh
            c["share_out"] = 1.0 - sh["dp"]          # 守る側の計算の外（地平の先・流入・効果・落ちる道）
            out.append(c)
    return out


def where_table(cs):
    def tab(sel):
        n = len(sel)
        o = {"n": n}
        for k in ("in", "past", "capped", "fallback"):
            m = sum(1 for c in sel if c["where"] == k)
            o[k] = m
            o[k + "_share"] = round(m / n, 4) if n else None
        pl = [c for c in sel if c["plan"]]
        o["past_share_of_plan"] = round(sum(1 for c in pl if c["where"] == "past") / len(pl), 4) if pl else None
        return o
    res = {"all": tab(cs), "by_seat": {}, "by_band": {}}
    for w, name in ((True, "winner"), (False, "loser")):
        res["by_seat"][name] = tab([c for c in cs if c["winner"] == w])
    for b in ("1", "2", "3", "4-5", "6+"):
        res["by_band"][b] = tab([c for c in cs if c["band"] == b])
        res["by_band"][b]["winner"] = tab([c for c in cs if c["band"] == b and c["winner"]])
    hz = [c["nh"] for c in cs if c["plan"]]
    res["horizon"] = _q(hz)
    res["horizon_hist"] = {str(h): int(sum(1 for x in hz if x == h)) for h in sorted(set(hz))}
    res["horizon_cut_by_budget"] = int(sum(1 for c in cs if c["plan"] and c.get("horizon") is not None
                                           and c.get("horizon0") is not None and c["horizon"] < c["horizon0"]))
    res["why"] = {k: int(sum(1 for c in cs if c["why"] == k)) for k in sorted(set(c["why"] for c in cs))}
    return res


def share_table(cs):
    def tab(sel):
        if not sel:
            return {"n": 0}
        o = {"n": len(sel)}
        acc = sum(c["acc"] for c in sel)
        o["pooled"] = {s: round(sum(c.get("acc_" + s, 0.0) for c in sel) / acc, 4) if acc > 0 else None for s in SRC}
        o["per_clock"] = {s: _q([c["share"][s] for c in sel]) for s in SRC}
        so = [c["share_out"] for c in sel]
        o["outside"] = _q(so)
        for t in (0.10, 0.25, 0.50):
            o["outside_gt_%d" % int(t * 100)] = round(float(np.mean([x > t for x in so])), 4)
        return o
    pl = [c for c in cs if c["plan"]]
    res = {"plan_clocks": tab(pl), "all_clocks": tab(cs), "by_seat": {}, "by_where": {}, "by_band": {}}
    for w, name in ((True, "winner"), (False, "loser")):
        res["by_seat"][name] = tab([c for c in pl if c["winner"] == w])
        res["by_seat"][name + "_all"] = tab([c for c in cs if c["winner"] == w])
    for k in ("in", "past", "capped"):
        res["by_where"][k] = tab([c for c in pl if c["where"] == k])
    for b in ("1", "2", "3", "4-5", "6+"):
        res["by_band"][b] = tab([c for c in pl if c["band"] == b and c["winner"]])
    return res


def capped_table(cs):
    cp = [c for c in cs if c["tau"] >= CAP]
    dom = {}
    for c in cp:
        k = max(SRC, key=lambda s: c.get("acc_" + s, 0.0))
        dom[k] = dom.get(k, 0) + 1
    o = {"n": len(cp), "winner": int(sum(1 for c in cp if c["winner"])), "loser": int(sum(1 for c in cp if not c["winner"])),
         "fallback": int(sum(1 for c in cp if not c["plan"])), "dominant": dom}
    if cp:
        acc = sum(c["acc"] for c in cp)
        o["pooled"] = {s: round(sum(c.get("acc_" + s, 0.0) for c in cp) / acc, 4) if acc > 0 else None for s in SRC}
        o["theta_mean"] = round(float(np.mean([c["theta"] for c in cp])), 4)
        o["acc30_mean"] = round(float(np.mean([c["acc"] for c in cp])), 4)
        o["acc30_over_theta"] = round(float(np.mean([c["acc"] / c["theta"] for c in cp if c["theta"] > 0])), 4)
        o["horizon"] = _q([c["nh"] for c in cp if c["plan"]])
        o["alive_mean"] = (round(float(np.mean([c["alive"] for c in cp if c["plan"]])), 4)
                           if any(c["plan"] for c in cp) else None)
    return o


def short_table(cs, h_min=3):
    """地平を縮めた検算（同じ計画・`h ≥ h_min` の時計・守る側の計算の解き直しが元と一致した計画だけ）。"""
    sel = [c for c in cs if c["plan"] and c["nh"] >= h_min and c.get("resolve_same")]
    res = {"h_min": h_min, "n_clocks": len(sel),
           "resolve_mismatch": int(sum(1 for c in cs if c["plan"] and not c.get("resolve_same")))}
    for k in (1, 2, 3):
        rows = []
        for c in sel:
            for s in c.get("short") or ():
                if s["k"] == k:
                    rows.append((c, s))
        if not rows:
            continue
        o = {"n": len(rows)}
        full = np.array([min(c["tau_full_plan"], CAP) for c, _ in rows])
        for v in ("a", "b", "c"):
            t = np.array([min(s["tau_" + v], CAP) for _, s in rows])
            d = t - full
            o[v] = {"mean": round(float(d.mean()), 4), "median": round(float(np.median(d)), 4),
                    "mae": round(float(np.abs(d).mean()), 4), "gt05": round(float((np.abs(d) > 0.5).mean()), 4),
                    "gt1": round(float((np.abs(d) > 1.0).mean()), 4),
                    "ceil_changed": round(float((np.ceil(t - 1e-9) != np.ceil(full - 1e-9)).mean()), 4)}
            # 交点が縮めた地平の外に出た時計だけ（外の値付けが実際に効いた時計）
            past = np.array([full_j > s["h"] for (c, s), full_j in zip(rows, [math.ceil(min(c["tau_full_plan"], CAP) - 1e-9) for c, _ in rows])])
            if past.any():
                o[v]["past_only"] = {"n": int(past.sum()), "mean": round(float(d[past].mean()), 4),
                                     "mae": round(float(np.abs(d[past]).mean()), 4)}
        # 段ごと: 解き直した地平の外になった段 j（h−k < j ≤ h）の `fb` 対 元の守る側の計算の損害
        fb_sum = dp_sum = 0.0
        ratios = []
        for c, s in rows:
            pairs = c.get("fbdp") or []
            j_lo = s["h"]                      # 0 始まりの添字で s["h"] 以上 = 縮めた地平の外
            off = 1 if (c["j"] == 0) else 0       # 1 段目を落とした（今は攻撃しない）並びの補正
            for idx, (dpv, fbv) in enumerate(pairs):
                jj = idx + off                     # 0 始まりの段
                if jj >= j_lo:
                    fb_sum += fbv
                    dp_sum += dpv
                    if dpv > 1e-3:
                        ratios.append(fbv / dpv)
        o["steps_fb_over_dp_pooled"] = round(fb_sum / dp_sum, 4) if dp_sum > 0 else None
        o["steps_fb_over_dp"] = _q(ratios)
        res["k%d" % k] = o
    # 地平の内側の全段（どの時計でも）で fb 対 守る側の計算の損害
    fb_sum = dp_sum = 0.0
    ratios = []
    seen = 0
    for c in cs:
        if not c["plan"]:
            continue
        for dpv, fbv in c.get("fbdp") or []:
            seen += 1
            fb_sum += fbv
            dp_sum += dpv
            if dpv > 1e-3:
                ratios.append(fbv / dpv)
    res["all_inside_steps"] = {"n_steps": seen, "fb_over_dp_pooled": round(fb_sum / dp_sum, 4) if dp_sum > 0 else None,
                               "fb_over_dp": _q(ratios)}
    return res


def _theory_metrics(rows_out, ledger, turn_harm, theta_check):
    s = CB.summarise(rows_out, ledger, turn_harm, theta_check)["by_slope"]["theory"]
    return {k: s.get(k) for k in ("sign_accuracy", "bias", "sigma_T", "within1", "mae", "bias_int", "tau_me_median")}


def variant_table(rows_out, m2rows, ledger, turn_harm, theta_check):
    base = _theory_metrics(rows_out, ledger, turn_harm, theta_check)
    res = {"default": base, "capped_default": int(sum((r["tau_me_theory"] >= CAP) + (r["tau_opp_theory"] >= CAP) for r in rows_out))}
    for v in VARIANTS:
        rs = []
        capped = 0
        for r, m in zip(rows_out, m2rows):
            r2 = dict(r)
            for side, key in (("me", "tau_me_theory"), ("opp", "tau_opp_theory")):
                c = m[side]
                if c["why"] == "plan":
                    r2[key] = c["tau_" + v]
            r2["pred_theory"] = r2["tau_me_theory"] <= r2["tau_opp_theory"]
            capped += (r2["tau_me_theory"] >= CAP) + (r2["tau_opp_theory"] >= CAP)
            rs.append(r2)
        o = _theory_metrics(rs, ledger, turn_harm, theta_check)
        o["capped"] = int(capped)
        o["delta"] = {k: (round(o[k] - base[k], 4) if isinstance(o.get(k), (int, float)) and isinstance(base.get(k), (int, float)) else None)
                      for k in ("sign_accuracy", "bias", "sigma_T", "within1", "mae")}
        res[v] = o
    return res


def resid_table(cs):
    """勝った席の時計の残差（`min(τ, 30) − 実際の残り`）と、守る側の計算の外から来た割合の関係。"""
    w = [c for c in cs if c["winner"]]
    res_all = np.array([min(c["tau"], CAP) - c["act"] for c in w], float)
    so = np.array([c["share_out"] for c in w], float)
    o = {"n": len(w), "bias": round(float(res_all.mean()), 4), "sigma_T": round(float(res_all.std()), 4),
         "corr_resid_share_out": round(float(np.corrcoef(res_all, so)[0, 1]), 4) if len(w) > 2 and so.std() > 0 else None,
         "by_share_out": {}}
    for lo, hi, name in ((-1, 0.10, "<=10%"), (0.10, 0.25, "10-25%"), (0.25, 0.50, "25-50%"), (0.50, 2, ">50%")):
        m = (so > lo) & (so <= hi)
        if m.any():
            o["by_share_out"][name] = {"n": int(m.sum()), "bias": round(float(res_all[m].mean()), 4),
                                       "sigma_T": round(float(res_all[m].std()), 4),
                                       "mae": round(float(np.abs(res_all[m]).mean()), 4)}
    for k in ("in", "past", "capped", "fallback"):
        m = np.array([c["where"] == k for c in w])
        if m.any():
            o["by_where_" + k] = {"n": int(m.sum()), "bias": round(float(res_all[m].mean()), 4),
                                  "sigma_T": round(float(res_all[m].std()), 4),
                                  "sq_share": round(float((res_all[m] ** 2).sum() / max(1e-12, (res_all ** 2).sum())), 4)}
    # 残差の二乗和のうち、外の割合が 25% を超える時計が持つ分
    m = so > 0.25
    o["sq_share_outside_gt25"] = round(float((res_all[m] ** 2).sum() / max(1e-12, (res_all ** 2).sum())), 4)
    return o


def checks(rows_out, m2rows, cs):
    bad_row = sum(1 for r, m in zip(rows_out, m2rows)
                  if r["tau_me_theory"] != m["me"]["tau"] or r["tau_opp_theory"] != m["opp"]["tau"])
    bad_walk = sum(1 for c in cs if c["tau"] != c["tau_walk"])
    pl = [c for c in cs if c["plan"]]
    d_parts = [abs(c["tau_parts"] - c["tau"]) for c in pl]
    d_plan = [abs(min(c["tau_full_plan"], CAP) - min(c["tau"], CAP)) for c in pl]
    acc_bad = sum(1 for c in pl if abs(sum(c.get("acc_" + s, 0.0) for s in SRC) - c["acc"]) > 1e-9 * max(1.0, c["acc"]))
    return {"rows": len(rows_out), "m2_rows": len(m2rows), "row_tau_mismatch": bad_row, "walk_tau_mismatch": bad_walk,
            "parts_tau_maxdiff": round(float(max(d_parts)), 12) if d_parts else None,
            "plan_tau_vs_row_tau_maxdiff": round(float(max(d_plan)), 9) if d_plan else None,
            "plan_tau_vs_row_tau_gt1e6": int(sum(1 for x in d_plan if x > 1e-6)),
            "acc_split_mismatch": acc_bad}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="src", nargs="+", required=True, help="n_records のディレクトリ")
    ap.add_argument("--limit-games", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--clocks-out", default="", help="時計ごとの診断の行（JSON lines）の書き出し先（任意）")
    a = ap.parse_args(argv)
    t0 = time.time()
    rows_out, ledger, stats, turn_harm, theta_check = CB.collect(a.src, a.limit_games)
    m2rows = list(CB.M2_ROWS)
    if len(m2rows) != len(rows_out):
        raise SystemExit("m2 の行が rows_out と揃わない（%d 対 %d）" % (len(m2rows), len(rows_out)))
    cs = clocks(rows_out, m2rows)
    res = {"kind": CB.record_kind(a.src), "src": a.src, "games": stats.get("games"),
           "checks": checks(rows_out, m2rows, cs),
           "where": where_table(cs),
           "share": share_table(cs),
           "capped": capped_table(cs),
           "short": short_table(cs),
           "short_h2": short_table(cs, h_min=2),
           "resid": resid_table(cs),
           "variants": variant_table(rows_out, m2rows, ledger, turn_harm, theta_check),
           "seconds": round(time.time() - t0, 1)}
    txt = json.dumps(res, ensure_ascii=False, indent=2)
    print(txt)
    if a.out:
        with open(os.path.expanduser(a.out), "w", encoding="utf-8") as fh:
            fh.write(txt + "\n")
    if a.clocks_out:
        with open(os.path.expanduser(a.clocks_out), "w", encoding="utf-8") as fh:
            for c in cs:
                c2 = copy.copy(c)
                fh.write(json.dumps(c2, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
