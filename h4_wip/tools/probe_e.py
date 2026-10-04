"""H-4e probe: per attacker turn-start row, the planned attach count, the actual attached DON, the realised harm
and the bridge's speed. usage: probe_e.py FORM OUT.jsonl -- <crossing_bridge args>
FORM: cuttable_forced / rule / rule_don / rule_don_purse / noplan / spdavg (refs: rule_don endurance)."""
import sys, json, math, os
WT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path[:0] = [WT + "/tests/scripts", WT + "/tests", WT]
import numpy as np
import crossing_bridge as CB
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plan_cache  # noqa: F401  (H-4e: shared plans)
import math
_orig = CB.summarise
def patched(rows_out, ledger, turn_harm=None, theta_check=None):
    out = _orig(rows_out, ledger, turn_harm, theta_check)
    extra = {}
    for sv in ("theory", "curve"):
        if not rows_out or ("tau_me_" + sv) not in rows_out[0]:
            continue
        res, resi = [], []
        for r in rows_out:
            if r["won"]:
                tau, act = min(r["tau_me_" + sv], 30.0), r["t_me_act"]
            else:
                tau, act = min(r["tau_opp_" + sv], 30.0), r["t_opp_act"]
            res.append(tau - act); resi.append(math.ceil(tau - 1e-9) - act)
        res = np.array(res); resi = np.array(resi, float)
        extra[sv] = {"n": len(res), "bias": round(float(res.mean()), 3),
                     "bias_int_ceil": round(float(resi.mean()), 3),
                     "sigma_T_int": round(float(resi.std()), 3),
                     "within1_int": round(float((np.abs(resi) <= 1).mean()), 4),
                     "exact_int": round(float((resi == 0).mean()), 4),
                     "mae_int": round(float(np.abs(resi).mean()), 3),
                     "share_tau_integer": round(float(np.mean([abs(x - round(x)) < 1e-9 for x in res])), 4)}
        byk = {}
        for r in rows_out:
            if r["won"]:
                tau, act = min(r["tau_me_" + sv], 30.0), r["t_me_act"]
            else:
                tau, act = min(r["tau_opp_" + sv], 30.0), r["t_opp_act"]
            k = str(int(act)) if act <= 5 else "6+"
            byk.setdefault(k, []).append((tau - act, math.ceil(tau - 1e-9) - act, bool(r["pred_" + sv]) == bool(r["won"])))
        extra[sv]["by_t_act"] = {k: {"n": len(v), "bias": round(float(np.mean([a for a, _, _ in v])), 3),
                                     "bias_int": round(float(np.mean([b for _, b, _ in v])), 3),
                                     "mae": round(float(np.mean([abs(a) for a, _, _ in v])), 3),
                                     "within1": round(float(np.mean([abs(a) <= 1 for a, _, _ in v])), 4),
                                     "hit": round(float(np.mean([c for _, _, c in v])), 4)}
                                 for k, v in sorted(byk.items(), key=lambda kv: (kv[0] == "6+", kv[0]))}
    out["int_turn"] = extra
    return out
CB.summarise = patched


FORM = sys.argv[1]
outp = sys.argv[2]
args = sys.argv[4:] if sys.argv[3] == "--" else sys.argv[3:]
CB.set_theta_hand_mode("rule_don" if FORM in ("noplan", "spdavg") else FORM)
G = {"n": 0, "cur": None}
LOG = {}
LAST = {"plan": None, "actx": None}
SCHED_ID = {}
CAPPED = set()
_orig_iter = CB.PL.iter_games


def iter_wrap(*a, **k):
    G["n"] = 0
    ACT.clear()
    for item in _orig_iter(*a, **k):
        G["n"] += 1
        G["cur"] = item
        yield item


CB.PL.iter_games = iter_wrap

_orig_pf = CB.rule_don_plan_for


def pf(sc, tok, g_hand, attacker):
    plan = _orig_pf(sc, tok, g_hand, attacker)
    LAST["plan"], LAST["actx"] = plan, attacker
    return plan


def speed_plan(plan):
    if plan is None or FORM not in ("noplan", "spdavg"):
        return plan
    if FORM == "noplan":
        return None
    ap = avg_plan(plan, LAST["actx"])
    ap.pop("harm_steps", None)
    for kk in ("sched", "a_time", "a_turn"):
        ap.pop(kk, None)
    return ap


CB.rule_don_plan_for = pf


def locate(sc):
    rows, pol, ex, L, ptr, idx = G["cur"]
    base = ex["sc"]
    try:
        p = sc.__array_interface__["data"][0]; b0 = base.__array_interface__["data"][0]
    except Exception:
        return None
    off = p - b0
    st = base.strides[0]
    if off < 0 or off % st:
        return None
    i = off // st
    if i >= len(base):
        return None
    return int(i)


def avg_plan(plan, actx):
    if plan is None or actx is None:
        return None
    ks = plan["k"]; price = actx["price"]; att1 = actx["att1"]
    gl = (price[0][ks[0]] + ks[0] * CB.DELTA) if att1 else 0.0
    gc = sum(price[j][ks[j]] + ks[j] * CB.DELTA for j in range(1, len(att1)))
    p = dict(plan, attach_lead=float(gl), attach=float(gc))
    p.pop("attach_steps", None)
    return p


_ot, _os, _otg = CB.seat_slope_terms, CB.seat_slope_sched, CB.tau_grow


def slope_of(tt):
    s_board, s_stock, s_flow, s_lead, s_srush, s_frush, s_eff, s_eff1 = tt
    s_hand = s_stock if CB.SLOPE_HAND_MODE == "stock" else s_flow
    return s_board + s_hand + s_eff + s_eff1


def terms(sc, tok, ci, *a, plan=None, **k):
    out = _ot(sc, tok, ci, *a, plan=speed_plan(plan), **k)
    i = locate(sc)
    if i is None:
        return out
    rows = G["cur"][0]
    w, t = int(rows["who"][i]), int(rows["turn"][i])
    key = (G["n"], w, t)
    if key in LOG:
        return out
    pl = LAST["plan"] if plan is not None else None
    rec = {"g": G["n"], "w": w, "t": t, "A": slope_of(out),
           "ksum": (int(sum(pl["k"])) if pl is not None else None),
           "alive": (float(pl["alive"]) if pl is not None else None),
           "h1": (float(pl["harm_steps"][0]) if pl is not None and pl.get("harm_steps") else None),
           "nsteps": (len(pl["harm_steps"]) if pl is not None and pl.get("harm_steps") is not None else None),
           "pl_theta": (float(pl["theta"]) if pl is not None and "theta" in pl else None)}
    LOG[key] = rec
    return out


def sched(sc, tok, ci, *a, plan=None, **k):
    out = _os(sc, tok, ci, *a, plan=speed_plan(plan), **k)
    i = locate(sc)
    if i is not None:
        rows = G["cur"][0]
        rec = LOG.get((G["n"], int(rows["who"][i]), int(rows["turn"][i])))
        if rec is not None and "s0" not in rec:
            rec["s0"] = float(out[0])                  # the walk's first step (per-step view)
            rec["s1"] = float(out[1]) if len(out) > 1 else None
    return out


CB.seat_slope_terms, CB.seat_slope_sched = terms, sched


# actual attached DON per (g, w, t) from the records + game end
ACT = {}
_orig_collect_summ = CB.summarise


def summ(rows_out, ledger, turn_harm=None, theta_check=None):
    out = _orig_collect_summ(rows_out, ledger, turn_harm, theta_check)
    # j -> t mapping per (g, w)
    byw = {}
    for (g, w, t) in LOG:
        byw.setdefault((g, w), []).append(t)
    jt = {}
    for kk, ts in byw.items():
        for j, t in enumerate(sorted(ts)):
            jt[(kk[0], kk[1], j)] = t
    nmatch = 0
    for r in turn_harm or []:
        t = jt.get((r["g"], r["who"], r["j"]))
        if t is None:
            continue
        rec = LOG.get((r["g"], r["who"], t))
        if rec is None:
            continue
        rec["harm"] = float(r["harm"]); rec["slope_theory"] = float(r["slope_theory"]); rec["t_left"] = r["t_left"]
        rec["harm_e2"] = float(r.get("harm_e2", r["harm"])); rec["slope_time"] = float(r.get("slope_time", r["slope_theory"]))
        rec["won"] = bool(r["won"]); nmatch += 1
    with open(outp, "w") as f:
        for key, rec in LOG.items():
            rec.pop("_sid", None)
            rec.update(ACT.get(key, {}))
            f.write(json.dumps(rec) + "\n")
    print("probe: logged %d matched_harm %d" % (len(LOG), nmatch), file=sys.stderr)
    return out


CB.summarise = summ

# attach actuals: computed lazily per game at the time the next game starts (and at the end)
_prev_iter = CB.PL.iter_games


def iter_wrap2(*a, **k):
    for item in _prev_iter(*a, **k):
        rows, pol, ex, L, ptr, idx = item
        g = G["n"]
        tok = ex["tok"]
        z = {}
        per = {}
        for i in idx:
            w, t = int(rows["who"][i]), int(rows["turn"][i])
            zz = float(rows["z"][i])
            if zz != 0.0:
                z[w] = zz > 0
            if t < 1 or not CB.PL.is_own_turn(w, t):
                continue
            att_l = float(tok[i][0, 2]) * 5.0
            att_c = float(np.sum(tok[i][2:7, 2])) * 5.0
            p = per.setdefault((w, t), {"act_lead": 0.0, "act_chars": 0.0, "act_tot": 0.0})
            p["act_lead"] = max(p["act_lead"], att_l); p["act_chars"] = max(p["act_chars"], att_c)
            p["act_tot"] = max(p["act_tot"], att_l + att_c)
        wn = [w for w, v in z.items() if v]
        last = {}
        for (w, t) in per:
            last[w] = max(t, last.get(w, -1))
        for (w, t), p in per.items():
            p["kill_turn"] = bool(len(wn) == 1 and w == wn[0] and t == last[w])
            ACT[(g, w, t)] = p
        yield item


CB.PL.iter_games = iter_wrap2
raise SystemExit(CB.main(args))
