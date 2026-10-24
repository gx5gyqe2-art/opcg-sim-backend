import json, sys
sys.path.insert(0, "tests/scripts")
import blunder_audit as BA
S = "n1_results/reaudit_wr/tree_budget"
BA._init_judge(1)
games = {g["seed"]: g for g in BA.read_jsonl(f"{S}/games.jsonl")}
out = []
for r in BA.read_jsonl(f"{S}/points.jsonl"):
    game = BA.replay(games[r["seed"]], r["step"])
    opts = json.loads(BA._G["ship"].decide_opts(r["seed"], r["turn"], r["seat"], None))
    for m in (1, 2, 4, 8):
        o = dict(opts, sims=1280 * m)
        d = json.loads(game.fork().decide(r["seat"], json.dumps(o)))
        legal = [BA.desc(game, x) for x in d["stats"].get("moves", [])] if "moves" in d["stats"] else None
        row = {"seed": r["seed"], "step": r["step"], "x": m, "pick": BA.desc(game, d["move"]),
               "Q": [round(x, 3) for x in d["stats"]["Q"]], "N": d["stats"]["N"], "legal": legal, "window": d.get("window")}
        print(json.dumps(row, ensure_ascii=False), flush=True); out.append(row)
json.dump(out, open(f"{S}/root_values.json", "w"), ensure_ascii=False, indent=1)
