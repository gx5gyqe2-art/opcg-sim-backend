"""Six-form tables for H-4f: plan-based forms from h4g_measure, default/rule from h4e_measure (identical code path)."""
import json, os
import os
SC = os.environ.get("OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
D_E, D_G = SC + "/h4e_measure", SC + "/h4g_measure"
M = ("cuttable_forced", "rule", "rule_don", "rule_don_purse", "noplan", "spdavg")


def _dir(p):
    return D_G if os.path.exists(os.path.join(D_G, p)) else D_E


def ld(p):
    q = os.path.join(_dir(p), p)
    return json.load(open(q)) if os.path.exists(q) and os.path.getsize(q) else None


def rows(p):
    q = os.path.join(_dir(p), p)
    if not os.path.exists(q):
        return None
    return [json.loads(l) for l in open(q) if l.strip()]


print("== crossing bridge (theory)  [hit bias intbias sigT w1 mae | capped | rate check: turn-reader speed / realised(E2)]")
for s in ("real", "syn"):
    for m in M:
        j = ld("cb_%s_%s.json" % (s, m))
        if not j:
            print(s, m, "missing"); continue
        th = j["summary"]["by_slope"]["theory"]; it = j["summary"]["int_turn"]["theory"]; st = j["stats"]
        rc = j["summary"].get("rate_check", {})
        print("%-4s %-16s %.4f %+.3f %+.3f %.3f %.4f %.3f | %3d | %.3f" % (
            s, m, th["sign_accuracy"], th["bias"], it["bias_int_ceil"], th["sigma_T"], th["within1"], th["mae"],
            st["tau_capped"], rc.get("theory_over_real", float("nan"))))
print("== theta/need by turns left")
for s in ("real", "syn"):
    for m in M:
        j = ld("cb_%s_%s.json" % (s, m))
        if not j:
            continue
        b = j["summary"]["theta_check"]["by_turns_left"]
        print("%-4s %-16s " % (s, m) + " ".join("%s:%.3f" % (k, v["theta_over_need"]) for k, v in b.items()))
print("== pre-settle calibration LL / ECE / AUC")
for s in ("real", "syn"):
    for m in M:
        w = ld("wc_%s_%s.json" % (s, m))
        if not w:
            print(s, m, "missing"); continue
        r = w["rel"]
        print("%-4s %-16s LL %.4f ECE %.4f AUC %.4f" % (s, m, r["logloss"], r["ece"], r["auc"]))
print("== theory_bridge dG / dS ; relative ledger rel_K")
for s in ("real", "syn"):
    for m in M[:4]:
        j = ld("tb_%s_%s.json" % (s, m)); l = ld("rl_%s_%s.json" % (s, m))
        print("%-4s %-16s dG %s dS %s rel_K %s" % (
            s, m, "%.4f" % j["summary"]["dG"]["auc"] if j else "-", "%.4f" % j["summary"]["dS"]["auc"] if j else "-",
            "%.4f" % l["arms"]["rel_K"]["auc"] if l else "-"))
print("== kill-now diagnostic: died turns predicted to survive / survived turns predicted to die")
for s in ("real", "syn"):
    j = ld("diag_%s.json" % s)
    if not j:
        print(s, "missing"); continue
    d, v = j["died"], j["survived"]
    for m in ("rule", "rule_don", "rule_don_purse"):
        print("%-4s %-16s %d/%d (%.3f)  %d/%d (%.3f)" % (s, m, d[m], d["n"], d[m] / max(1, d["n"]),
                                                        v["n"] - v[m], v["n"], (v["n"] - v[m]) / max(1, v["n"])))
print("== speed / realised per reader type (realised = turn harm, kill turn priced at remaining endurance for plan forms)")
print("   columns: turn reader all [non-kill] | matched k>0 ; time reader all | matched k>0")
for s in ("real", "syn"):
    ref = rows("pr_%s_rule_don.jsonl" % s) or []
    plan_k = {(r["g"], r["w"], r["t"]): r.get("ksum") for r in ref}
    for m in M:
        R = rows("pr_%s_%s.jsonl" % (s, m))
        if not R:
            print(s, m, "missing"); continue
        R = [r for r in R if "harm" in r]

        def ratio(T, key):
            h = sum(r.get("harm_e2", r["harm"]) for r in T)
            return sum(r.get(key, r["slope_theory"]) for r in T) / h if h else float("nan")

        def matched(r):
            k = plan_k.get((r["g"], r["w"], r["t"]))
            return k is not None and k > 0 and int(round(r.get("act_tot", 0.0))) == k
        nk = [r for r in R if not r.get("kill_turn")]
        mt = [r for r in R if matched(r)]
        print("%-4s %-16s %.3f [%.3f] | %.3f (n%d) ; %.3f | %.3f" % (
            s, m, ratio(R, "slope_theory"), ratio(nk, "slope_theory"), ratio(mt, "slope_theory"), len(mt),
            ratio(R, "slope_time"), ratio(mt, "slope_time")))
