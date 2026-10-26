"""§7.49 の集計: 上限ごとの証明の数・本当の配り方（上限 160k）での仕分け・節点・秒、と 1 手の時間。"""
import gzip, json, os, sys
O = os.path.dirname(os.path.abspath(__file__))

def load(p):
    out = {}
    for l in gzip.open(p, "rt"):
        d = json.loads(l)
        if "rs" in d:
            out[(d["seed"], d["seat"], d["turn"], d["step"])] = d["rs"]
    return out

true = load(os.path.join(O, "judge_true_n160k.jsonl.gz")) if os.path.exists(os.path.join(O, "judge_true_n160k.jsonl.gz")) else {}
base = None
rows = {}
for t in ("n20k", "n40k", "n80k", "n160k"):
    J = load(os.path.join(O, f"judge_{t}.jsonl.gz"))
    wins = {k for k, r in J.items() if r["status"] == "win"}
    cls = {}
    for k in wins:
        s = true.get(k, {}).get("status", "missing")
        cls[s] = cls.get(s, 0) + 1
    new = wins - base if base is not None else set()
    ncls = {}
    for k in new:
        s = true.get(k, {}).get("status", "missing")
        ncls[s] = ncls.get(s, 0) + 1
    lat = json.load(open(os.path.join(O, f"lat_{t}.json")))
    rows[t] = {"points": len(J), "win": len(wins), "win_new_vs_n20k": len(new), "new_by_true_deal": ncls,
               "win_by_true_deal": cls,
               "false_proofs": sorted(["%s:%s:%s:%s" % k for k in wins if true.get(k, {}).get("status") == "no"]),
               "nodes": sum(r.get("nodes", 0) for r in J.values()), "sec": round(sum(r.get("sec", 0.0) for r in J.values()), 1),
               "latency": {k: lat[k] for k in ("moves", "mean", "p50", "p90", "p99", "max", "wall_sec")},
               "latency_lethal": {"fired": lat["lethal"]["fired"], "root_tried": lat["lethal"]["root_tried"],
                                  "root_status": lat["lethal"]["root_status"],
                                  "root_nodes_mean": lat["lethal"]["root_nodes"]["mean"],
                                  "root_sec_mean": lat["lethal"]["root_sec"]["mean"], "root_sec_p99": lat["lethal"]["root_sec"]["p99"]}}
    if base is None:
        base = wins
b = rows["n20k"]
for t, r in rows.items():
    r["nodes_ratio"] = round(r["nodes"] / b["nodes"], 3)
    r["mean_ratio"] = round(r["latency"]["mean"] / b["latency"]["mean"], 3)
    r["p99_ratio"] = round(r["latency"]["p99"] / b["latency"]["p99"], 3)
json.dump(rows, open(os.path.join(O, "summary.json"), "w"), ensure_ascii=False, indent=1)
print(json.dumps(rows, ensure_ascii=False, indent=1))
