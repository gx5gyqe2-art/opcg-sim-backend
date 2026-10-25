"""WP audit-verify-opp: 確かめの新旧を並べる（1 回きり）。出力 table.json。"""
import os
import json
import sys
sys.path.insert(0, os.getcwd())
sys.path.insert(0, "tests/scripts")
import move_judge as MJ

D = "n1_results/audit_verify_opp"


def rd(p):
    return [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()]


def bl(rows):
    out = {}
    for r in rows:
        k = f"{r['seed']}:{r['step']}"
        if "error" in r:
            out[k] = {"error": r["error"]}
            continue
        pool = MJ.paired(r["r13d"]["a"] + r["r20"]["a"], r["r13d"]["b"] + r["r20"]["b"])
        out[k] = {"d13": r["r13d"]["diff"], "ci13": r["r13d"]["ci"], "d20": r["r20"]["diff"], "ci20": r["r20"]["ci"],
                  "wa": pool["wa"], "wb": pool["wb"], "pool": pool["diff"], "pool_ci": pool["ci"],
                  "verified": bool(pool["ci"][0] is not None and pool["ci"][0] > 0 and r["r13d"]["diff"] > 0 and r["r20"]["diff"] > 0)}
    return out


T = {}
T["blunder_old_record"] = bl(rd("n1_results/blunder_audit/verify.jsonl"))
T["blunder_mover"] = bl(rd(f"{D}/blunder_verify_mover.jsonl"))
T["blunder_both"] = bl(rd(f"{D}/blunder_verify_both.jsonl"))
T["reaudit_old_record"] = bl(rd("n1_results/reaudit_wr/blunder/verify.jsonl"))
T["reaudit_mover"] = bl(rd(f"{D}/reaudit_blunder_verify_mover.jsonl"))
T["reaudit_both"] = bl(rd(f"{D}/reaudit_blunder_verify_both.jsonl"))


CPU = {f"{d['seed']}:{d['step']}": d["rows"][0]["move"] for d in rd(f"{D}/defense_points6.jsonl") if "game" not in d}


def df(rows):
    out = {}
    for r in rows:
        k = f"{r['seed']}:{r['step']}"
        p = r.get("playout")
        out[k] = {"miss_ko_full": r.get("miss_ko_full"), "miss_otsu_full": r.get("miss_otsu_full"), "error": r.get("error"),
                  "cpu_move": CPU.get(k)}
        if p:
            out[k].update({"safe_move": p["safe_move"], "kept": p["kept"]})
            for m in ("real", "fair"):
                q = p[m]
                out[k][m] = {"w_cpu": q["wa"], "w_safe": q["wb"], "diff": q["diff"], "ci": q["ci"],
                             "cpu_lost_this_turn": q["cpu_lost_this_turn"], "safe_lost_this_turn": q["safe_lost_this_turn"]}
    return out


T["defense_both"] = df(rd(f"{D}/defense_verify_both.jsonl"))
T["defense_mover"] = df(rd(f"{D}/defense_verify_mover.jsonl"))
json.dump(T, open(f"{D}/table.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
for k, v in T.items():
    print("==", k)
    for kk, vv in v.items():
        print(" ", kk, json.dumps(vv, ensure_ascii=False))
