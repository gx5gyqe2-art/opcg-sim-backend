"""WP weak-points §4: ライフの枠の配り方の形ごとに、今の既定（judge_now）と比べて減った証明を「本当の配り方で勝てない（的）／勝てる（損）」に分ける。"""
import json, os, sys
O = "n1_results/weak_points/lethal"


def load(p):
    d = {}
    import gzip
    for l in (gzip.open(p, "rt") if p.endswith(".gz") else open(p)):
        r = json.loads(l)
        if "rs" in r:
            d[(r["seed"], r["seat"], r["turn"], r["step"])] = r["rs"]
    return d


base, true = load(f"{O}/judge_now.jsonl.gz"), load(f"{O}/judge_now_true.jsonl.gz")
out = {}
for v in ("off", "both", "mode0", "add"):
    p = f"{O}/judge_life_{v}.jsonl.gz"
    lg = f"{O}/judge_life_{v}.log"
    if not os.path.exists(p) or not os.path.exists(lg) or '"judge"' not in open(lg).read():
        continue
    x = load(p)
    bw = {k for k, r in base.items() if r["status"] == "win"}
    xw = {k for k, r in x.items() if r["status"] == "win"}
    lost = sorted(bw - xw)
    new = sorted(xw - bw)
    tgt = [k for k in lost if true.get(k, {}).get("status") == "no"]
    loss = [k for k in lost if true.get(k, {}).get("status") == "win"]
    unk = [k for k in lost if true.get(k, {}).get("status") not in ("win", "no")]
    moved = [k for k in bw & xw if json.dumps(base[k]["move"], sort_keys=True) != json.dumps(x[k]["move"], sort_keys=True)]
    to = {}
    for k in lost:
        to[x[k]["status"]] = to.get(x[k]["status"], 0) + 1
    new_true = {}
    for k in new:
        st = true.get(k, {}).get("status")
        new_true[st] = new_true.get(st, 0) + 1
    out[v] = {"new_true": new_true, "points": len(x), "proofs": len(xw), "lost": len(lost), "lost_target_true_no": len(tgt), "target_pts": [":".join(map(str, k)) for k in tgt],
              "lost_true_win": len(loss), "lost_true_unknown": len(unk), "lost_to": to, "new": len(new), "new_pts": [":".join(map(str, k)) for k in new][:20],
              "move_changed": len(moved), "nodes": sum(r.get("nodes") or 0 for r in x.values()),
              "nodes_base": sum(r.get("nodes") or 0 for r in base.values())}
    print(v, json.dumps(out[v], ensure_ascii=False))
json.dump(out, open(f"{O}/variants.json", "w"), ensure_ascii=False, indent=1)
