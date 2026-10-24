"""WP weak-points §2 の続き: 3301005・107／3301025・117 の素通し／ブロックを、**今のターンの残りを両席とも出荷の設定**
（r13d・1,280 回・世界 2 本・証明 root）が打ち、その先は審判 r13d（160 回）が打つ形で K 局ずつ比べる（1 回きり）。
審判の相手役の打ち損じ（相手のターンの詰みを打ち切れない）を除いた比べ。"""
import json, os, sys, time
sys.path.insert(0, "tests/scripts")
import blunder_audit as BA
import move_judge as MJ
from opcg_sim.loop import driver as DR

K = int(os.environ.get("K", "16"))
BA._init_judge(K)
S = "n1_results/reaudit_wr/tree_budget"
games = {g["seed"]: g for g in BA.read_jsonl(f"{S}/games.jsonl")}
pts = BA.read_jsonl(f"{S}/points.jsonl")
ship, ref = BA._G["ship"], BA._G["refs"]["r13d"]


def roll(game, who, mv, ws):
    g = game.fork_world(who, "fair", int(ws))
    t0 = int(game.turn_count)
    g.apply_search_move(who, json.dumps(mv, ensure_ascii=False))
    carries = {}
    steps = 0
    while g.winner() is None and steps < 400:
        pend = json.loads(g.pending_json())
        if not pend:
            break
        name = pend["player_id"]
        turn = int(g.turn_count)
        strong = turn == t0
        spec = ship if strong else ref
        c = carries.setdefault((name, strong), DR.Carry())
        out = json.loads(g.decide(name, spec.decide_opts(int(ws), turn, name, c.get(turn, name))))
        c.put(out)
        if out.get("move") is None:
            break
        BA.apply(g, name, out["move"])
        steps += 1
    w = g.winner()
    return (None if w is None else int(w == who)), (w is not None and w != who and int(g.turn_count) == t0)


rows = []
for p in pts:
    seed, step, me = p["seed"], p["step"], p["seat"]
    game = BA.replay(games[seed], step)
    r = {"seed": seed, "step": step}
    for tag, mv in (("a", p["a"]), ("b", p["b"])):
        t0 = time.time()
        res = [roll(game, me, mv, MJ.world_seed(seed, step, i)) for i in range(K)]
        ws = [x for x, _ in res if x is not None]
        r[tag] = {"move": BA.desc(game, mv), "win": round(sum(ws) / max(1, len(ws)), 3), "n": len(ws),
                  "lost_this_turn": sum(l for _, l in res), "w": [x for x, _ in res], "sec": round(time.time() - t0, 1)}
    r["paired"] = MJ.paired(r["a"]["w"], r["b"]["w"])
    print(json.dumps(r, ensure_ascii=False), flush=True)
    rows.append(r)
json.dump(rows, open("n1_results/weak_points/dbl_attack2.json", "w"), ensure_ascii=False, indent=1)
