"""4461051・169（相手 p2 のカウンターの窓・ライフ 0）で、根の窓の決め方（今の形＝短い木）と、木の中と同じ畳み方
（世界 1 本・静的な値で 1 段だけ・以降は方策＝旧い 3 つの旗の形）で各候補の値を並べる。読み取りだけ。"""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tests", "scripts")); sys.path.insert(0, ROOT); os.chdir(ROOT)
import blunder_audit as BA
from opcg_sim.loop import engine as E, decks as D
E.engine(); net = E.load_net("opcg_sim/data/learned/nrel_r29f4.npz"); BA._G["db"] = D.load_db()
g = next(json.loads(l) for l in open("n1_results/reaudit_r29f4/blunder/games.jsonl") if json.loads(l)["seed"] == 4461051)
for step in (169, 170, 171, 172):
    game = BA.replay(g, step)
    who = json.loads(game.pending_json())["player_id"]
    ss = E.search_seed(4461051, 11, who)
    for tag, kw in (("now", {}), ("tree-style(legacy3)", {"window_tree_legacy": True, "window_worlds_legacy": True, "window_carry_legacy": True})):
        o = {"net": net, "search_seed": ss, "sims": E.PLAY_SIMS, "budget": 0, "worlds": E.PLAY_WORLDS, **kw}
        out = json.loads(game.fork().decide(who, json.dumps(o)))
        st = out.get("stats") or {}
        rows = sorted(zip(st.get("Q") or [], [BA.desc(game, m) for m in st.get("legal") or []]), key=lambda x: -x[0])
        print(step, who, tag, "->", BA.desc(game, out.get("move"))[:80], "| window", json.dumps(out.get("window"))[:80])
        for q, d in rows[:6]:
            print(f"     {q:+.3f} {d[:90]}")
    print("   actual:", json.dumps(g["moves"][step])[:120])
