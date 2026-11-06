"""損の点を当て直して、出荷の読み（seed を替えて数回）・証明 off・深い読みの根の候補を並べる（読み取りだけ）。
python n1_results/reaudit_r29f4/point_probe.py SEED STEP [n_seeds]"""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tests", "scripts")); sys.path.insert(0, ROOT); os.chdir(ROOT)
import blunder_audit as BA
from opcg_sim.loop import engine as E, decks as D
seed, step = int(sys.argv[1]), int(sys.argv[2]); ns = int(sys.argv[3]) if len(sys.argv) > 3 else 3
E.engine(); net = E.load_net("opcg_sim/data/learned/nrel_r29f4.npz")
BA._G["db"] = D.load_db()
g = next(json.loads(l) for l in open("n1_results/reaudit_r29f4/blunder/games.jsonl") if json.loads(l)["seed"] == seed)
p = next(x for x in g["points"] if x["step"] == step)
game = BA.replay(g, step)
who = p["seat"]
h = json.loads(game.hidden_json())
opp = "p2" if who == "p1" else "p1"
print("who", who, "turn", p["turn"], "pending", json.loads(game.pending_json()).get("action"))
print("opp hand", [c.get("card_id") or c.get("id") for c in h["players"][opp]["hand"]][:12])
print("my hand", [c.get("card_id") or c.get("id") for c in h["players"][who]["hand"]][:14])
ss0 = E.search_seed(seed, p["turn"], who)
def run(tag, **kw):
    for i in range(ns):
        o = {"net": net, "search_seed": (ss0 ^ (i * 0x9E3779B97F4A7C15)) & 0xFFFFFFFFFFFFFFFF, **kw}
        out = json.loads(game.fork().decide(who, json.dumps(o)))
        rows = BA.ranked(out); tot = sum(r["n"] for r in rows) or 1
        print(f"[{tag} #{i}] kind={out.get('kind')} move={BA.desc(game, out.get('move'))[:110]}")
        for r in rows[:4]:
            print(f"    n={r['n'] / tot:.3f} q={r['q']:+.3f} {BA.desc(game, r['mv'])[:120]}")
X = json.loads(os.environ.get("PROBE_OPTS", "{}"))
if X:
    run("ship+" + json.dumps(X), sims=E.PLAY_SIMS, budget=E.PLAY_BUDGET, worlds=E.PLAY_WORLDS, lethal_proof=E.PLAY_LETHAL_PROOF, **X)
    sys.exit(0)
run("ship", sims=E.PLAY_SIMS, budget=E.PLAY_BUDGET, worlds=E.PLAY_WORLDS, lethal_proof=E.PLAY_LETHAL_PROOF)
run("ship_noproof", sims=E.PLAY_SIMS, budget=E.PLAY_BUDGET, worlds=E.PLAY_WORLDS)
run("deep", **BA.DEEP)
