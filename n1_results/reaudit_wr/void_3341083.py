import json, sys
sys.path.insert(0, "tests/scripts")
import blunder_audit as BA
from opcg_sim.loop import driver as DR, engine as E
BA._init_judge(1)
g = next(json.loads(l) for l in open("n1_results/reaudit_wr/blunder/games.jsonl") if json.loads(l)["seed"] == 3341083)
print("moves", len(g["moves"]))
for p in g["points"][-4:]:
    print(p["step"], p["seat"], p["turn"], p["act"], p["kind"], p["dtype"], json.dumps(p["move"])[:200], "commit_left", p["commit_left"])
game = BA.replay(g, 43)
pend = json.loads(game.pending_json()); print("pending", json.dumps(pend)[:600])
print("legal", game.legal_json(pend["player_id"])[:1500])
p = g["points"][-1]
opts = json.loads(BA._G["ship"].decide_opts(3341083, p["turn"], "p2", None))
print("opts", opts)
for extra in ({}, {"window_carry_legacy": True}, {"window_carry_legacy": True, "window_worlds_legacy": True, "window_tree_legacy": True}, {"lethal_proof": "off"}):
    o = dict(opts, **extra)
    d = json.loads(game.fork().decide("p2", json.dumps(o)))
    print(extra, d.get("kind"), json.dumps(d.get("move"))[:200], "commit", json.dumps(d.get("commit"))[:300], {k: v for k, v in d.items() if k not in ("move", "stats", "commit")})
h = json.loads(game.hidden_json())
print("keys", list(h.keys()))
for k in ("active_battle", "phase", "pending_request", "interaction"):
    print(k, json.dumps(h.get(k))[:400])
m = h["manager"]
print("mkeys", list(m.keys()))
for k, v in m.items():
    if k not in ("players",): print(k, json.dumps(v)[:300])
def where(u):
    for pn, P in h["players"].items():
        for z, cs in P.items():
            if isinstance(cs, list):
                for c in cs:
                    if isinstance(c, dict) and c.get("uuid") == u: return pn, z, c.get("name") or c.get("card_id"), c.get("is_rest")
            elif isinstance(cs, dict) and cs.get("uuid") == u: return pn, z, cs.get("name"), cs.get("is_rest")
for u in ("5e825d32-0167-4be7-8267-eb225de4c150", "5a31ecc5-450c-4252-920b-b93398fbc070"):
    print(u[:8], where(u))
print(list(h["players"]["p1"].keys()))
ev = []
g2 = BA.replay(g, 40)
nm = json.loads(g2.pending_json())["player_id"]
for s in (40, 41, 42):
    BA.apply(g2, json.loads(g2.pending_json())["player_id"], g["moves"][s])
    print(s, json.dumps(json.loads(g2.events_json()), ensure_ascii=False)[:900])
    print("  pending", json.dumps(json.loads(g2.pending_json()), ensure_ascii=False)[:300])
