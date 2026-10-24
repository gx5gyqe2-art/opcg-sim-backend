import json, sys
sys.path.insert(0, "tests/scripts")
import opcg_engine; print("engine", opcg_engine.__file__)
import blunder_audit as BA
BA._init_judge(1)
g = next(json.loads(l) for l in open("n1_results/reaudit_wr/blunder/games.jsonl") if json.loads(l)["seed"] == 3341083)
g2 = BA.replay(g, 41)
BA.apply(g2, "p2", g["moves"][41]); BA.apply(g2, "p2", g["moves"][42])
print("pending", json.dumps(json.loads(g2.pending_json()), ensure_ascii=False)[:200])
print("phase", json.loads(g2.hidden_json())["manager"]["phase"])
