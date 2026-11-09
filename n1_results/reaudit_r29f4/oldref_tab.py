"""損の候補 17 点を、今の審判（r29f4／r28x4）と前回の審判（r13d／r20）で打ち直し・確かめた結果を並べる（読み取りだけ）。
通る＝2 審判まとめた差の区間の下限 > 0 かつ 2 審判とも差 > 0（器の summary と同じ規則）。"""
import json, sys
sys.path.insert(0, "tests/scripts"); sys.path.insert(0, ".")
import move_judge as MJ
D = "n1_results/reaudit_r29f4"
def load(p): return {(x["seed"], x["step"]): x for x in map(json.loads, open(p))}
P = {(p["seed"], p["step"]): p for p in map(json.loads, open(f"{D}/blunder/points.jsonl")) if p.get("cand")}
new_rc, new_vf = load(f"{D}/blunder/recheck.jsonl"), load(f"{D}/blunder/verify.jsonl")
old_rc, old_vf = load(f"{D}/blunder_oldref/recheck.jsonl"), load(f"{D}/blunder_oldref/verify.jsonl")
def ok(x):
    q = MJ.paired(x["r13d"]["a"] + x["r20"]["a"], x["r13d"]["b"] + x["r20"]["b"])
    return (q["ci"][0] > 0 and x["r13d"]["diff"] > 0 and x["r20"]["diff"] > 0), round(q["diff"], 3), [round(c, 3) for c in q["ci"]]
cnt = {"new_rc": 0, "new_vf": 0, "new_both": 0, "old_rc": 0, "old_vf": 0, "old_both": 0}
rows = []
for k in sorted(P):
    a, b, c, d = ok(new_rc[k]), ok(new_vf[k]), ok(old_rc[k]), ok(old_vf[k])
    cnt["new_rc"] += a[0]; cnt["new_vf"] += b[0]; cnt["new_both"] += a[0] and b[0]
    cnt["old_rc"] += c[0]; cnt["old_vf"] += d[0]; cnt["old_both"] += c[0] and d[0]
    rows.append({"pt": f"{k[0]}:{k[1]}", "dtype": P[k]["dtype"], "new_rc": a, "new_vf": b, "old_rc": c, "old_vf": d})
    print(f"{k[0]}:{k[1]:<4} {P[k]['dtype']:<30} new rc {a} vf {b} | old rc {c} vf {d}")
print(cnt)
json.dump({"rows": rows, "counts": cnt}, open(f"{D}/oldref_tab.json", "w"), ensure_ascii=False, indent=1)
