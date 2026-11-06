"""損の候補を 1 行ずつ（判定・打ち直し・確かめ）。読み取りだけ。"""
import json, sys
B = "n1_results/reaudit_r29f4/blunder"
rd = lambda f: [json.loads(l) for l in open(f"{B}/{f}")]
pts = [p for p in rd("points.jsonl") if p.get("cand")]
rc = {(x["seed"], x["step"]): x for x in rd("recheck.jsonl")}
vf = {(x["seed"], x["step"]): x for x in rd("verify.jsonl")}
import importlib.util, os
sys.path.insert(0, "tests/scripts"); sys.path.insert(0, ".")
import move_judge as MJ
for p in sorted(pts, key=lambda x: (x["seed"], x["step"])):
    k = (p["seed"], p["step"]); r = rc.get(k); v = vf.get(k)
    def pool(x):
        if not x or "r13d" not in x: return None
        q = MJ.paired(x["r13d"]["a"] + x["r20"]["a"], x["r13d"]["b"] + x["r20"]["b"]); return (round(q["diff"], 3), [round(c, 3) for c in q["ci"]], x["r13d"]["diff"], x["r20"]["diff"])
    print(p["seed"], p["step"], p["kind"], p["dtype"], "T", p.get("turn"), "life", p.get("my_life"), p.get("opp_life"),
          "d1", p["ref"]["r13d"]["diff"], "d2", (p["ref"].get("r20") or {}).get("diff"),
          "| rc", pool(r), "| vf", pool(v), (v or {}).get("error"))
