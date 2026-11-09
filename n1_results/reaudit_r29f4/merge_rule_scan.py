"""判定した木の点（kind=main）で、出荷の読み（1,280 回 × 世界 4 本・同じ search_seed）を回し直し、
世界ごとの根の統計（per_world）から「世界ごとの値の平均で選ぶ」形の手を出して、打った手（訪問の合算で選んだ束）と比べる。
読み取りだけ（エンジンは変えない・選び方の比べは Python 側で per_world を読むだけ）。
平均の形: 束ごとに世界ごとの (訪問, 訪問で重みをつけた Q) を作り、合算の訪問が全体の FLOOR 以上の束だけを候補にして、
その束の訪問が 1 以上の世界の Q の平均が最大の束を選ぶ。"""
import json, os, sys
from multiprocessing import get_context
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tests", "scripts")); sys.path.insert(0, ROOT); os.chdir(ROOT)
import blunder_audit as BA
from opcg_sim.loop import engine as E, decks as D
FLOOR = float(os.environ.get("FLOOR", "0.05"))
G = {}

def init():
    E.engine(); G["net"] = E.load_net("opcg_sim/data/learned/nrel_r29f4.npz"); BA._G["db"] = D.load_db()
    G["games"] = {json.loads(l)["seed"]: json.loads(l) for l in open("n1_results/reaudit_r29f4/blunder/games.jsonl")}

def one(p):
    g = G["games"][p["seed"]]
    real = BA.replay(g, p["step"]); who = p["seat"]
    o = {"net": G["net"], "search_seed": E.search_seed(p["seed"], p["turn"], who), "sims": E.PLAY_SIMS, "budget": E.PLAY_BUDGET,
         "worlds": E.PLAY_WORLDS, "lethal_proof": E.PLAY_LETHAL_PROOF}
    out = json.loads(real.fork().decide(who, json.dumps(o)))
    if out.get("kind") != "main" or not out.get("per_world"):
        return {"seed": p["seed"], "step": p["step"], "kind": out.get("kind"), "skip": True}
    legal = out["stats"]["legal"]; pw = out["per_world"]
    tot = sum(sum(w["N"]) for w in pw) or 1
    groups = []
    for gr in out["groups"]:
        per = []
        for w in pw:
            n = sum(w["N"][i] for i in gr["idxs"]); 
            q = sum(w["N"][i] * w["Q"][i] for i in gr["idxs"]) / n if n > 0 else None
            per.append((n, q))
        mean = [q for n, q in per if n >= 1 and q is not None]
        groups.append({"rep": gr["rep"], "n": gr["n"], "q": gr["q"], "per": per, "mean_q": sum(mean) / len(mean) if mean else None})
    chosen = max(groups, key=lambda x: (x["n"], x["q"]))
    elig = [x for x in groups if x["n"] >= FLOOR * tot and x["mean_q"] is not None]
    alt = max(elig, key=lambda x: x["mean_q"]) if elig else chosen
    pk = lambda mv: BA.pkey(mv)
    return {"seed": p["seed"], "step": p["step"], "dtype": p["dtype"],
            "same_as_record": pk(legal[chosen["rep"]]) == pk(p["a"]),
            "alt_differs": alt["rep"] != chosen["rep"],
            "alt_is_b": pk(legal[alt["rep"]]) == pk(p["b"]),
            "alt_desc": BA.desc(real, legal[alt["rep"]])[:100], "chosen_desc": BA.desc(real, legal[chosen["rep"]])[:100],
            "chosen_mean_q": chosen["mean_q"], "alt_mean_q": alt["mean_q"],
            "d": p["ref"]["r13d"]["diff"], "ci": p["ref"]["r13d"]["ci"], "cand": bool(p.get("cand"))}

if __name__ == "__main__":
    pts = [json.loads(l) for l in open("n1_results/reaudit_r29f4/blunder/points.jsonl")]
    pts = [p for p in pts if p["ok"] and p["kind"] == "main" and p["dtype"] != "mulligan"]
    outp = "n1_results/reaudit_r29f4/merge_rule_scan.jsonl"
    done = {(json.loads(l)["seed"], json.loads(l)["step"]) for l in open(outp)} if os.path.exists(outp) else set()
    pts = [p for p in pts if (p["seed"], p["step"]) not in done]
    print("todo", len(pts), flush=True)
    with get_context("fork").Pool(4, initializer=init) as pool, open(outp, "a") as f:
        for i, r in enumerate(pool.imap_unordered(one, pts, chunksize=1)):
            f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()
            if i % 20 == 0:
                print(i, json.dumps(r, ensure_ascii=False)[:200], flush=True)
