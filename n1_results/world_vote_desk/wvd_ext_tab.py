"""標本外の確かめ（PREREG_ext.md）の集計: ext_scan.jsonl・ext_judge.jsonl → ext_summary.json（読み取りだけ）。"""
import json, math, os
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
D = "n1_results/world_vote_desk"
S = [r for r in map(json.loads, open(f"{D}/ext_scan.jsonl")) if not r.get("skip")]
J = [r for r in map(json.loads, open(f"{D}/ext_judge.jsonl"))]


def ms(x):
    n = len(x)
    if n < 2:
        return {"n": n, "mean": round(sum(x) / n, 4) if n else None, "ci": None}
    m = sum(x) / n; sd = math.sqrt(sum((v - m) ** 2 for v in x) / (n - 1)); h = 1.96 * sd / math.sqrt(n)
    return {"n": n, "mean": round(m, 4), "ci": [round(m - h, 4), round(m + h, 4)]}


tot = len(S); dis = [r for r in S if r["disagree"]]
cpu4 = sum(r["cpu4"] for r in S)
out = {"points": tot, "reproduce": sum(r["now_is_record"] for r in S), "disagree": len(dis), "disagree_rate": round(len(dis) / tot, 4), "forms": {}}
for name, th in (("A_all_disagree", None), ("gap>=0.01", 0.01), ("gap>=0.03", 0.03), ("B_gap>=0.05", 0.05), ("gap>=0.1", 0.1)):
    sel = [r for r in J if th is None or (r["gap4"] or 0) >= th]
    d = [r["diff"] for r in sel if r["diff"] is not None]
    rd = [r for r in dis if th is None or (r["gap4"] or 0) >= th]
    m = ms(d)
    out["forms"][name] = {"reread_points": len(rd), "reread_rate": round(len(rd) / tot, 4), "changed": len(sel), "changed_rate": round(len(sel) / tot, 4),
                          "void_pairs": sum(1 for r in sel if r["diff"] is None), "changed_diff": m,
                          "per_tree_point": None if m["mean"] is None else round(m["mean"] * len(d) / tot, 5),
                          "big_gain(>=0.2)": [(r["seed"], r["step"], r["diff"]) for r in sel if r["diff"] is not None and r["diff"] >= 0.2],
                          "big_loss(<=-0.2)": [(r["seed"], r["step"], r["diff"]) for r in sel if r["diff"] is not None and r["diff"] <= -0.2],
                          "cpu_inc_over_tree_add4": round(sum(r["cpu8"] - r["cpu4"] for r in rd) / cpu4, 4)}
json.dump(out, open(f"{D}/ext_summary.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
