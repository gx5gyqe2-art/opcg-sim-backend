"""損の点（木の点）で、世界 1 本の読みを seed 8 本で回し、a と b の根の (訪問の割合, Q) を世界ごとに並べる。
「少数の世界で確実に勝つ手が、訪問の合算で大半の世界で良い手に勝つ」形かを見る。読み取りだけ。
python n1_results/reaudit_r29f4/worlds1_scan.py SEED:STEP ..."""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tests", "scripts")); sys.path.insert(0, ROOT); os.chdir(ROOT)
import blunder_audit as BA
from opcg_sim.loop import engine as E, decks as D
E.engine(); net = E.load_net("opcg_sim/data/learned/nrel_r29f4.npz"); BA._G["db"] = D.load_db()
G = {}
for l in open("n1_results/reaudit_r29f4/blunder/games.jsonl"):
    g = json.loads(l); G[g["seed"]] = g
P = {}
for l in open("n1_results/reaudit_r29f4/blunder/points.jsonl"):
    p = json.loads(l); P[(p["seed"], p["step"])] = p
res = {}
for arg in sys.argv[1:]:
    seed, step = map(int, arg.split(":"))
    p = P[(seed, step)]; real = BA.replay(G[seed], step); who = p["seat"]
    ss0 = E.search_seed(seed, p["turn"], who)
    rows_out = []
    for i in range(8):
        o = {"net": net, "search_seed": (ss0 ^ (i * 0x9E3779B97F4A7C15)) & 0xFFFFFFFFFFFFFFFF, "sims": E.PLAY_SIMS, "budget": 0, "worlds": 1}
        out = json.loads(real.fork().decide(who, json.dumps(o)))
        rows = BA.ranked(out); tot = sum(r["n"] for r in rows) or 1
        def find(mv):
            c = [r for r in rows if BA.key(r["mv"]) == BA.key(mv)] or [r for r in rows if BA.pkey(r["mv"]) == BA.pkey(mv)]
            r = max(c, key=lambda r: r["n"]) if c else None
            return (round(r["n"] / tot, 3), round(r["q"], 3)) if r else (0.0, None)
        fa, fb = find(p["a"]), find(p["b"])
        rows_out.append({"a": fa, "b": fb, "top": BA.desc(real, rows[0]["mv"])[:80]})
        print(f"{seed}:{step} w#{i} a(n,q)={fa} b(n,q)={fb} top={rows_out[-1]['top']}", flush=True)
    res[arg] = rows_out
json.dump(res, open("n1_results/reaudit_r29f4/worlds1_scan.json", "w"), ensure_ascii=False, indent=1)
