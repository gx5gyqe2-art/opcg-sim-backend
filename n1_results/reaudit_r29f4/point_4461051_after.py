"""4461051・163: 打った箱（オオロンブスにドン!! 6 枚・攻撃なし）を本当の盤面に当てた後の p1 の読み（出荷の設定・seed 3 本）と、
同じ盤面を相手から見た読み。根の木の値（箱 +0.98）と、当てた後の読みの値が食い違う場所を探す。読み取りだけ。"""
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
for mode in ("real", "fair-world"):
    for i in range(3):
        game = real.fork() if mode == "real" else real.fork_world("p1", "fair", 1000 + i)
        game.apply_search_move("p1", json.dumps(p["a"]))
        o = {"net": net, "search_seed": (ss0 ^ (i * 0x9E3779B97F4A7C15)) & 0xFFFFFFFFFFFFFFFF, "sims": E.PLAY_SIMS, "budget": 0,
             "worlds": E.PLAY_WORLDS, "lethal_proof": E.PLAY_LETHAL_PROOF}
        out = json.loads(game.fork().decide("p1", json.dumps(o)))
        rows = BA.ranked(out); tot = sum(r["n"] for r in rows) or 1
        h = json.loads(game.hidden_json())
        oh = [c["card_id"] for c in h["players"]["p2"]["hand"]]
        print(f"[{mode} #{i}] opp hand={oh} kind={out.get('kind')} move={BA.desc(game, out.get('move'))[:90]}")
        for r in rows[:4]:
            print(f"    n={r['n'] / tot:.3f} q={r['q']:+.3f} {BA.desc(game, r['mv'])[:110]}")
