"""4461051・163: 世界 1 本の読みを seed 8 本で。箱（ドン!! 6 枚）の根の値が世界（相手の手札の配り方）で割れるかを見る。
同じ seed で、箱を当てた後の p1 の読み（世界 1 本）の最善の値も並べる。読み取りだけ。"""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tests", "scripts")); sys.path.insert(0, ROOT); os.chdir(ROOT)
import blunder_audit as BA
from opcg_sim.loop import engine as E, decks as D
E.engine(); net = E.load_net("opcg_sim/data/learned/nrel_r29f4.npz"); BA._G["db"] = D.load_db()
g = next(json.loads(l) for l in open("n1_results/reaudit_r29f4/blunder/games.jsonl") if json.loads(l)["seed"] == 4461051)
p = next(json.loads(l) for l in open("n1_results/reaudit_r29f4/blunder/points.jsonl") if json.loads(l)["seed"] == 4461051 and json.loads(l)["step"] == 163)
real = BA.replay(g, 163)
ss0 = E.search_seed(4461051, 11, "p1")
for i in range(8):
    o = {"net": net, "search_seed": (ss0 ^ (i * 0x9E3779B97F4A7C15)) & 0xFFFFFFFFFFFFFFFF, "sims": E.PLAY_SIMS, "budget": 0, "worlds": 1}
    out = json.loads(real.fork().decide("p1", json.dumps(o)))
    rows = BA.ranked(out); tot = sum(r["n"] for r in rows) or 1
    box = next((r for r in rows if BA.key(r["mv"]) == BA.key(p["a"])), None)
    g2 = real.fork(); g2.apply_search_move("p1", json.dumps(p["a"]))
    o2 = dict(o); out2 = json.loads(g2.decide("p1", json.dumps(o2)))
    r2 = BA.ranked(out2)
    print(f"seed#{i}: box n={box['n'] / tot if box else 0:.3f} q={box['q'] if box else float('nan'):+.3f} | top={BA.desc(real, rows[0]['mv'])[:70]} q={rows[0]['q']:+.3f}"
          f" || after box: top={BA.desc(g2, r2[0]['mv'])[:70]} q={r2[0]['q']:+.3f}")
