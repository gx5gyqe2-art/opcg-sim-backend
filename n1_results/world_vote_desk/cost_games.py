"""1 手あたりの費用の分母: §7.84 の局を数局当て直し、全部の判断点で出荷の読み（持ち越しは使わず＝木の点は世界 4 本の木）を回して
決め方の種類別の CPU 秒（Rust の世界スレッドを含む）を測る（読み取りだけ）。木の点（kind=main）の CPU が全体のどれだけかを出す。
続きで決まった点は実局では木を回していない（費用ほぼ 0）ので、記録の kind で分けて数える。"""
import json, os, sys, time
from multiprocessing import get_context
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "n1_results", "world_vote_desk")); os.chdir(ROOT)
import wvd_scan as W  # noqa: E402
SEEDS = [int(s) for s in os.environ.get("SEEDS", "4461003,4461020,4461040,4461060,4461080").split(",")]

def run(seed):
    W.init()
    g = W.G["games"][seed]
    real = W.BA.replay(g, 0)
    rows = []
    for s, pt in enumerate(g["points"]):
        who = json.loads(real.pending_json())["player_id"]
        if pt["kind"] in ("main", "window", "lethal"):
            o = {"net": W.G["net"], "search_seed": W.E.search_seed(seed, pt["turn"], who), "sims": W.E.PLAY_SIMS, "budget": W.E.PLAY_BUDGET,
                 "worlds": W.E.PLAY_WORLDS, "lethal_proof": W.E.PLAY_LETHAL_PROOF}
            c0 = W.cpu(); out = json.loads(real.fork().decide(who, json.dumps(o))); c = W.cpu() - c0
            rows.append({"step": s, "kind": pt["kind"], "kind_now": out.get("kind"), "cpu": round(c, 3)})
        else:
            rows.append({"step": s, "kind": pt["kind"], "cpu": 0.0})
        W.BA.apply(real, who, g["moves"][s])
    return {"seed": seed, "rows": rows}

if __name__ == "__main__":
    with get_context("fork").Pool(4) as pool, open("n1_results/world_vote_desk/cost_games.jsonl", "w") as f:
        for r in pool.imap_unordered(run, SEEDS):
            f.write(json.dumps(r) + "\n"); f.flush(); print(r["seed"], len(r["rows"]), flush=True)
    print("ALL_DONE", flush=True)
