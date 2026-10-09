"""**守り手の先読みの浅さを確かめる**（2026-10-09・`docs/reports/2026-10-09_defender_foresight.md`・読み取り専用・診断だけ）。

理論の守る側の計算（A＝地平の全部を見通す最善）と、**目先だけの最善**（B＝判断のたびに今の段だけを最善に守る）を、
同じ盤面・同じ値段・同じ入力で並べる。先読み `k` の守り手は Rust の写し（`core/foresight.rs`・`m2.resolve` の `fore`）で解く。
既定の出力・表・golden には触らない。

測るもの（報告 §1）:

1. **打ち方（ターンごと）**: 自席ターンの頭の行ごとに、その行の守る側の計算の入力（実際の盤面と守り手の手札）へそのターンの実際の
   攻撃を入れ、A（地平 `H`）と B（地平 1）・`k` = 2, 3（地平 `k`）の段 1 の受けた本数・切った札を、実際（ライフの減り・
   リーダーへの攻撃に切ったカウンター）と比べる。
2. **遅れの説明（時計ごと）**: 同じ計画のまま守る側の計算を先読み `k` の守り手で解き直した時計 `τ^k`（地平の内の段の値と耐久を
   差し替えて同じ歩き）と、窓の内で実際の攻撃を入れてその守り手が受ける本数から、終わりの検算の主の組（`completion_check.py`）と
   同じ物差しで残る遅れ・差し引いた後の遅れを出す。

入力は `m2_probe.py --clocks-out` の時計の行（`OPCG_M2_PROBE`・`din` つき）と記録。

実行例:
  python tests/scripts/defender_foresight.py --in ~/w41 --clocks clocks_real.jsonl --out fs_real.json --rows-out fs_real_rows.jsonl
"""
import argparse
import collections
import json
import math
import multiprocessing as mp
import os
import sys

os.environ.setdefault("OPCG_M2_PROBE", "1")

import numpy as np  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import survival_split as SS  # noqa: E402
import completion_check as CC  # noqa: E402
from opcg_sim.learned.train import plan_labels as PL  # noqa: E402

CAP = 30.0
FLOOR = 1e-3            # `outer::SLOPE_FLOOR`
EPS = SS.EPS
KS = (1, 2, 3)


# ---------------------------------------------------------------------------------------------
# 記録: リーダーへの攻撃に切ったカウンター

def leader_counters(dirs):
    """{(seed, 攻め手 w, t): (リーダーへの攻撃に切ったカウンター, 全部のカウンター)}。`SELECT_COUNTER` の行を、そのターンの直前の
    攻撃（`DON_BOX` で対象あり）の対象で振り分ける（対象が `pol_ti == 1`＝リーダー）。"""
    out = {}
    for rows, pol, _ex, L, ptr, idx in PL.iter_games(dirs, row_cols=SS.ROW_COLS, pol_cols=SS.POL_COLS):
        seed = int(rows["seed"][idx[0]])
        last = {}
        for i in idx:
            w, t, k = int(rows["who"][i]), int(rows["turn"][i]), int(rows["kind"][i])
            if SS._own(w, t):
                if k != 0 or int(L[i]) < 1:
                    continue
                j = int(ptr[i]) + int(rows["pol_chosen"][i])
                sg = json.loads(pol["pol_sig"][j])
                if sg and sg[0] == "DON_BOX" and sg[2]:
                    last[(w, t)] = int(pol["pol_ti"][j]) == 1
                continue
            if k == 0:
                continue
            key = (1 - w, t)
            rs = json.loads(rows["sig"][i]) if rows["sig"][i] else None
            if rs and rs[0] == "SELECT_COUNTER":
                a, b = out.get((seed,) + key, (0, 0))
                out[(seed,) + key] = (a + (1 if last.get(key, False) else 0), b + 1)
    return out


# ---------------------------------------------------------------------------------------------
# 時計: 先読み k の守り手で読み替えた歩き

def walk(theta, sched, j0, step=0.0, cap=CAP, floor=FLOOR):
    """`m2probe::walk` の写し（`drv_cb::m2_clock` の `tau_walk`）。"""
    f = 0.0
    for j in range(1, int(cap) + 1):
        add = 0.0 if (j0 + j - 1 <= 1 or not sched) else sched[min(max(j, 1), len(sched)) - 1]
        need = theta + (step if j >= 2 else 0.0)
        if f + add >= need:
            short = max(0.0, need - f)
            frac = short / add if add > floor else 1.0
            return (j - 1) + frac
        f += add
    return cap


def sched_k(c, hA, hk):
    """地平の内の段の値を `hk` の損害で差し替えた歩きの段の値（報告 §1.4）。`hA` は今の守り手の損害（`base` の地平の内）。
    地平の内の段の数が変わるときは、外に出た段を `fb`（`fbdp`）・内に入った段を `hk` で読む。読めない段があれば `None`。"""
    sch = [float(x) for x in c["sched"]]
    base = [float(x) for x in c["base"]]
    n = min(len(base), len(sch))
    j0 = int(c["j"]) + 1
    nhA, nhk = len(hA), len(hk)
    fb_idx = [i for i in range(min(nhA, n)) if not (i == 0 and j0 <= 1)]
    fb = {i: float(p[1]) for i, p in zip(fb_idx, c.get("fbdp") or [])}
    no_now = bool(c["din"].get("no_now"))
    out = list(sch)
    for i in range(n):
        if no_now and i == 0:
            continue
        if i < nhk:
            nb = float(hk[i])
        elif i < nhA:
            if i not in fb:
                if i == 0 and j0 <= 1:      # 頭の段は歩きが足さない
                    continue
                return None
            nb = fb[i]
        else:
            continue
        out[i] = sch[i] - base[i] + nb
    return out


def step_harm(sch, theta, tau):
    """`completion_check.step_harm` の `cross`（交点の段の値・床以下なら次の正の段・無ければ θ/τ）を与えた段の値で。"""
    t = min(float(tau), CAP)
    mean = theta / t if t > 1e-9 else None
    j = max(1, math.ceil(t - 1e-12))
    for q in range(min(j, len(sch)) - 1, len(sch)):
        if sch[q] > FLOOR:
            return sch[q]
    return mean


# ---------------------------------------------------------------------------------------------
# 1 本の時計（自席ターンの頭の行）の測り

def _hits_cuts_W(din, W, ov):
    """窓の内（段 1..W）の受けた本数（とどめを含む）・切った札。解けなければ `None`。"""
    hp = SS.resolve(din, lam=1.0, lam_net=1.0, mu=0.0, nu=[[m, 0.0] for m, _ in din["nu"]], **ov)
    cp = SS.resolve(din, lam=0.0, lam_net=0.0, mu=1.0, nu=[[m, 0.0] for m, _ in din["nu"]], **ov)
    if hp is None or cp is None:
        return None
    pd = SS.resolve(din, death=True, **ov) if round(float(din["life"])) >= 1 else None
    F = float(sum(pd["harms"][:W])) if pd else 0.0
    return float(sum(hp["harms"][:W])) + F, float(sum(cp["harms"][:W]))


def _turn_now(din, xa, turns=None, fore=None):
    """今の段に実際の攻撃 `xa` を入れた段 1 の (受けた本数〔とどめを含む〕, 切った札)。"""
    ov = {"xs_first": xa}
    if turns is not None:
        ov["turns"] = turns
    if fore is not None:
        ov["fore"] = fore
    nu0 = [[m, 0.0] for m, _ in din["nu"]]
    hp = SS.resolve(din, lam=1.0, lam_net=1.0, mu=0.0, nu=nu0, **ov)
    cp = SS.resolve(din, lam=0.0, lam_net=0.0, mu=1.0, nu=nu0, **ov)
    pd = SS.resolve(din, death=True, **ov)
    if hp is None or cp is None or pd is None:
        return None
    h1 = (hp["harms"][0] if hp["harms"] else 0.0) + (pd["harms"][0] if pd["harms"] else 0.0)
    return float(h1), float(cp["harms"][0] if cp["harms"] else 0.0)


def measure(c, G, lc, ks=KS):
    seed, w, t0 = int(c["seed"]), int(c["who"]), int(c["t"])
    din = c["din"]
    H = int(din["turns"])
    pd0 = [float(x) for x in c["pdeath"]]
    won = bool(c["winner"])
    act = int(c["act"]) if won else None
    r = {"key": [seed, w, t0], "won": won, "act": act, "H": H, "S": max(0.0, 1.0 - sum(pd0)),
         "tau": float(c["tau"]), "tau_walk": float(c["tau_walk"]), "theta": float(c["theta"]),
         "lam": float(din["lam"]), "mu": float(din["mu"]), "where": c.get("where"), "life": float(din["life"])}
    T = G["turns"].get((w, t0))
    # --- 1. 打ち方（このターン）
    if T is not None:
        la = [a for a in T["atk"] if a["leader"] and a["x"] is not None]
        xa = [a["x"] for a in la]
        n_hit = sum(1 for x in xa if x >= -float(din["eps"]))
        if T.get("killed"):
            ho = float(T["def_life0"] + 1)
        elif T["def_life1"] is not None:
            ho = float(max(0, T["def_life0"] - T["def_life1"]))
        else:
            ho = None
        cl, call = lc.get((seed, w, t0), (0, 0))
        tr = {"n_hit": n_hit, "obs_hits": ho, "obs_cuts": float(cl), "obs_cuts_all": float(call), "counters_rec": T["counters"]}
        if n_hit >= 1 and round(float(din["life"])) >= 1 and ho is not None:
            a_ = _turn_now(din, xa)
            tr["A"] = a_
            for k in ks:
                tr["k%d" % k] = a_ if k >= H else _turn_now(din, xa, turns=k)
        r["turn"] = tr
    # --- 2. 時計（先読み k の守り手で読み替え）
    rA = SS.resolve(din)
    hA = list(rA["harms"])
    j0 = int(c["j"]) + 1
    r["tau_check"] = walk(float(c["theta"]), [float(x) for x in c["sched"]], j0)
    clk = {}
    for k in ks:
        rk = rA if k >= H else SS.resolve(din, fore=k)
        if rk is None:
            clk["k%d" % k] = None
            continue
        sk = sched_k(c, hA, list(rk["harms"]))
        if sk is None:
            clk["k%d" % k] = None
            continue
        thk = float(c["theta"]) - float(rA["theta"]) + float(rk["theta"])
        clk["k%d" % k] = {"tau": walk(thk, sk, j0), "theta": thk, "cut": float(rk["cut"]), "alive": float(rk["alive"]),
                          "prevented": float(rk["prevented"]), "nh": len(rk["harms"]), "harm_sum": float(sum(rk["harms"])),
                          "sched": sk}
    r["clk"] = clk
    r["cutA"], r["harm_sumA"], r["nhA"], r["sched"] = float(rA["cut"]), float(sum(hA)), len(hA), [float(x) for x in c["sched"]]
    # --- 窓の内（勝った席だけ・終わりの検算の (i)）
    if won:
        W = min(H, act)
        by_step = {}
        ok = True
        ho, co = 0.0, 0.0
        for s in range(1, W + 1):
            Ts = G["turns"].get((w, t0 + 2 * (s - 1)))
            if Ts is None:
                ok = False
                break
            by_step[s] = [a["x"] for a in Ts["atk"] if a["leader"] and a["x"] is not None]
            co += Ts["counters"]
            if Ts["def_life1"] is not None:
                ho += max(0, Ts["def_life0"] - Ts["def_life1"])
            elif Ts.get("killed"):
                ho += Ts["def_life0"] + 1
            else:
                ok = False
                break
        if ok and by_step:
            ov = dict(xs_first=by_step.get(1, list(din["xs_first"])), seq=SS._seq_with(din, W, by_step, "replace"))
            r["hitsW_obs"], r["cutsW_obs"], r["W"] = ho, co, W
            wA = _hits_cuts_W(din, W, ov)
            r["hitsW_A"], r["cutsW_A"] = wA if wA else (None, None)
            for k in ks:
                wk = wA if k >= H else _hits_cuts_W(din, W, dict(ov, fore=k))
                r["hitsW_k%d" % k], r["cutsW_k%d" % k] = wk if wk else (None, None)
    return r


# ---------------------------------------------------------------------------------------------
# 集計

def _r_int(tau, act):
    return CC.r_int(tau, act)


def clock_rows(recs, k):
    """終わりの検算の主の組（勝った席・(i)・本数・交点の段の値）を守り手 `k`（"A" か整数）で。"""
    out = []
    for r in recs:
        if not r["won"] or r.get("hitsW_obs") is None:
            continue
        if k == "A":
            tau, sch, th, hm = r["tau"], r["sched"], r["theta"], r.get("hitsW_A")
        else:
            ck = r["clk"].get("k%d" % k)
            if ck is None:
                continue
            tau, sch, th, hm = ck["tau"], ck["sched"], ck["theta"], r.get("hitsW_k%d" % k)
        if hm is None:
            continue
        s = step_harm(sch, th, tau) if sch else None
        if s is None or s <= 0:
            continue
        delta = (r["hitsW_obs"] - hm) * r["lam"] / s
        out.append({"key": r["key"], "S": r["S"], "where": r["where"], "act": r["act"], "tau": tau, "delta": delta,
                    "r_int": _r_int(tau, r["act"]), "r_after": max(1, math.ceil(min(tau - delta, CAP))) - r["act"],
                    "dh": r["hitsW_obs"] - hm})
    return out


def _agg(xs):
    if not xs:
        return None
    a = np.array([x["r_int"] for x in xs], float)
    b = np.array([x["r_after"] for x in xs], float)
    d = np.array([x["delta"] for x in xs], float)
    dh = np.array([x["dh"] for x in xs], float)
    return {"n": len(xs), "r_int": round(float(a.mean()), 4), "delta": round(float(d.mean()), 4), "r_after": round(float(b.mean()), 4),
            "dh": round(float(dh.mean()), 4), "exact_before": round(float((a == 0).mean()), 4), "exact_after": round(float((b == 0).mean()), 4)}


BANDS = {"all": lambda x: True, "S>=1/2": lambda x: x["S"] >= 0.5, "S<1/2": lambda x: x["S"] < 0.5,
         "S>=1/2&cross_past": lambda x: x["S"] >= 0.5 and x["where"] in ("past", "capped"),
         "cross_in": lambda x: x["where"] == "in", "cross_past": lambda x: x["where"] in ("past", "capped")}


def summarise_clocks(recs, ks=KS):
    names = ["A"] + list(ks)
    rows = {k: clock_rows(recs, k) for k in names}
    # 同じ時計（全部の守り手で読めたもの）だけで並べる
    common = set(tuple(x["key"]) for x in rows["A"])
    for k in ks:
        common &= set(tuple(x["key"]) for x in rows[k])
    res = {}
    for k in names:
        xs = [x for x in rows[k] if tuple(x["key"]) in common]
        res[str(k)] = {b: _agg([x for x in xs if f(x)]) for b, f in BANDS.items()}
        res[str(k)]["n_all_readable"] = len(rows[k])
    return res


def summarise_turns(recs, ks=KS):
    """打ち方の比べ（報告 §1.6）。帯はその行の時計の `S`。"""
    rows = [r for r in recs if r.get("turn") and r["turn"].get("A") is not None
            and all(r["turn"].get("k%d" % k) is not None for k in ks)]
    names = ["A"] + ["k%d" % k for k in ks]

    def agg(xs):
        if not xs:
            return None
        o = {"n": len(xs)}
        oh = np.array([x["turn"]["obs_hits"] for x in xs], float)
        oc = np.array([x["turn"]["obs_cuts"] for x in xs], float)
        oca = np.array([x["turn"]["obs_cuts_all"] for x in xs], float)
        o["obs_hits"], o["obs_cuts"], o["obs_cuts_all"] = round(float(oh.mean()), 4), round(float(oc.mean()), 4), round(float(oca.mean()), 4)
        for nm in names:
            mh = np.array([x["turn"][nm][0] for x in xs], float)
            mc = np.array([x["turn"][nm][1] for x in xs], float)
            o[nm] = {"hits": round(float(mh.mean()), 4), "cuts": round(float(mc.mean()), 4),
                     "mae_hits": round(float(np.abs(oh - mh).mean()), 4), "mae_cuts": round(float(np.abs(oc - mc).mean()), 4),
                     "mae_cuts_all": round(float(np.abs(oca - mc).mean()), 4),
                     "bias_hits": round(float((oh - mh).mean()), 4), "bias_cuts": round(float((oc - mc).mean()), 4)}
        return o
    bands = {"all": lambda r: True, "S>=1/2": lambda r: r["S"] >= 0.5, "S<1/2": lambda r: r["S"] < 0.5,
             "winner": lambda r: r["won"], "loser": lambda r: not r["won"]}
    res = {b: agg([r for r in rows if f(r)]) for b, f in bands.items()}
    res["life0_turns"] = sum(1 for r in recs if r.get("turn") and r["turn"]["n_hit"] >= 1 and round(r["life"]) < 1)
    return res


def summarise_chain(recs, ks=KS):
    """時計ごと（勝った席の時計の段 1..act の和）: 実際 − 守り手の切った札・受けた本数。"""
    by = {tuple(r["key"]): r for r in recs}
    names = ["A"] + ["k%d" % k for k in ks]
    acc = {b: {nm: [] for nm in names} for b in ("all", "S>=1/2", "S<1/2")}
    for r in recs:
        if not r["won"]:
            continue
        seed, w, t0 = r["key"]
        sums = {nm: [0.0, 0.0] for nm in names}
        ok = True
        for s in range(1, r["act"] + 1):
            q = by.get((seed, w, t0 + 2 * (s - 1)))
            tr = q.get("turn") if q else None
            if tr is None or tr.get("obs_hits") is None:
                ok = False
                break
            if tr["n_hit"] == 0:
                continue
            if tr.get("A") is None or any(tr.get(nm) is None for nm in names):
                ok = False
                break
            for nm in names:
                sums[nm][0] += tr["obs_hits"] - tr[nm][0]
                sums[nm][1] += tr["obs_cuts"] - tr[nm][1]
        if not ok:
            continue
        for b in ("all", "S>=1/2" if r["S"] >= 0.5 else "S<1/2"):
            for nm in names:
                acc[b][nm].append(sums[nm])
    out = {}
    for b, d in acc.items():
        out[b] = {nm: ({"n": len(v), "extra_hits": round(float(np.mean([x[0] for x in v])), 4),
                        "extra_cuts": round(float(np.mean([x[1] for x in v])), 4),
                        "abs_extra_cuts": round(float(np.mean([abs(x[1]) for x in v])), 4)} if v else None) for nm, v in d.items()}
    return out


def verdict(turns, clocks):
    t = turns["all"]
    tb = turns["S>=1/2"]
    c1a = t["k1"]["mae_cuts"] < t["A"]["mae_cuts"]
    c1b = t["k1"]["mae_hits"] <= t["A"]["mae_hits"]
    c1c = tb is not None and tb["k1"]["mae_cuts"] < tb["A"]["mae_cuts"]
    RA, RB = clocks["A"]["S>=1/2"]["r_after"], clocks["1"]["S>=1/2"]["r_after"]
    phi = (RA - RB) / RA if abs(RA) > 1e-12 else None
    c2a = phi is not None and phi >= 0.5
    c2b = abs(clocks["1"]["all"]["r_after"]) <= abs(clocks["A"]["all"]["r_after"]) + 0.1
    return {"1a_cuts": c1a, "1b_hits": c1b, "1c_cuts_S>=1/2": c1c, "pass1": bool(c1a and c1b and c1c),
            "R_A": RA, "R_B": RB, "phi": None if phi is None else round(phi, 4), "2a_phi>=0.5": c2a, "2b_all_not_worse": c2b,
            "pass2": bool(c2a and c2b)}


# ---------------------------------------------------------------------------------------------

_G = {}


def _work(args):
    c, ks = args
    try:
        return measure(c, _G["ext"][int(c["seed"])], _G["lc"], ks)
    except Exception as e:  # noqa: BLE001  1 本が解けなくても全体は止めない（数を書く）
        return {"key": [int(c["seed"]), int(c["who"]), int(c["t"])], "error": repr(e)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="src", nargs="+", required=True)
    ap.add_argument("--clocks", nargs="+", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--rows-out", default="")
    ap.add_argument("--rows-in", nargs="*", default=[], help="測りの行を読み直して集計だけする")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=3)
    a = ap.parse_args(argv)
    if a.rows_in:
        recs = [json.loads(l) for p in a.rows_in for l in open(os.path.expanduser(p))]
    else:
        ext = SS.extract(a.src)
        lc = leader_counters(a.src)
        clocks = [json.loads(l) for p in a.clocks for l in open(os.path.expanduser(p))]
        me = [c for c in clocks if c.get("side") == "me" and c.get("plan") and c.get("din") and c.get("pdeath") is not None
              and "seed" in c and int(c["seed"]) in ext]
        if a.limit:
            me = me[:a.limit]
        _G["ext"], _G["lc"] = ext, lc
        recs = []
        with mp.get_context("fork").Pool(a.jobs) as pool:
            for n, r in enumerate(pool.imap(_work, [(c, KS) for c in me], chunksize=8)):
                recs.append(r)
                if (n + 1) % 500 == 0:
                    print("  %d / %d" % (n + 1, len(me)), file=sys.stderr, flush=True)
        if a.rows_out:
            with open(os.path.expanduser(a.rows_out), "w") as fh:
                for r in recs:
                    fh.write(json.dumps(r) + "\n")
    errs = [r for r in recs if "error" in r]
    recs = [r for r in recs if "error" not in r]
    chk = [abs(r["tau_check"] - r["tau_walk"]) for r in recs]
    res = {"clocks": len(recs), "errors": len(errs), "error_sample": [e["error"] for e in errs[:3]],
           "walk_check_max_abs": max(chk) if chk else None,
           "turns": summarise_turns(recs), "chain": summarise_chain(recs), "clocks_sum": summarise_clocks(recs)}
    res["verdict"] = verdict(res["turns"], res["clocks_sum"])
    txt = json.dumps(res, ensure_ascii=False, indent=1)
    if a.out:
        with open(os.path.expanduser(a.out), "w") as fh:
            fh.write(txt + "\n")
    print(json.dumps({"verdict": res["verdict"], "clocks": res["clocks"], "errors": res["errors"],
                      "walk_check": res["walk_check_max_abs"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
