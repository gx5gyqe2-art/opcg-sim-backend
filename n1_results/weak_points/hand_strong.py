"""WP weak-points §3: 3341039・84 の a（OP12-039 を捨てる）と b（OP12-037 を捨てる）を、今のターンと次の相手のターンを
両席とも出荷の設定（r13d）が打ち、その先は審判 r13d（160 回）が打つ形で K 局ずつ比べる（1 回きり）。"""
import json, os, sys, time
sys.path.insert(0, "tests/scripts")
import blunder_audit as BA
import move_judge as MJ
from opcg_sim.loop import driver as DR
K = int(os.environ.get("K", "16"))
STRONG_TURNS = int(os.environ.get("STRONG_TURNS", "2"))
BA._init_judge(K)
ship, ref = BA._G["ship"], BA._G["refs"]["r13d"]
seed, step, me = 3341039, 84, "p2"
g = next(json.loads(l) for l in open("n1_results/reaudit_wr/blunder/games.jsonl") if json.loads(l)["seed"] == seed)
game = BA.replay(g, step)
h = json.loads(game.hidden_json())
hand = {c["uuid"]: c["card_id"] for c in h["players"][me]["hand"]}
out = json.loads(game.fork().decide(me, ship.decide_opts(seed, int(game.turn_count), me, None)))
mv = {}
for m in out["stats"]["legal"]:
    cid = hand.get(m["payload"]["selected_uuids"][0])
    mv.setdefault(cid, m)


def roll(move, ws):
    g2 = game.fork_world(me, "fair", int(ws))
    t0 = int(game.turn_count)
    g2.apply_search_move(me, json.dumps(move))
    carries, steps = {}, 0
    while g2.winner() is None and steps < 400:
        pend = json.loads(g2.pending_json())
        if not pend:
            break
        name, turn = pend["player_id"], int(g2.turn_count)
        strong = turn < t0 + STRONG_TURNS
        spec = ship if strong else ref
        c = carries.setdefault((name, strong), DR.Carry())
        o = json.loads(g2.decide(name, spec.decide_opts(int(ws), turn, name, c.get(turn, name))))
        c.put(o)
        if o.get("move") is None:
            break
        BA.apply(g2, name, o["move"])
        steps += 1
    w = g2.winner()
    return None if w is None else int(w == me)


res = {}
for cid in ("OP12-039", "OP12-037"):
    t = time.time()
    res[cid] = [roll(mv[cid], MJ.world_seed(seed, step, i) ^ 0x3A3A) for i in range(K)]
    print(cid, res[cid], round(time.time() - t, 1), flush=True)
r = {"a_discard_OP12-039": res["OP12-039"], "b_discard_OP12-037": res["OP12-037"], "strong_turns": STRONG_TURNS,
     "paired_b_minus_a": MJ.paired(res["OP12-039"], res["OP12-037"])}
print(json.dumps(r))
json.dump(r, open(f"n1_results/weak_points/hand_strong_t{STRONG_TURNS}.json", "w"), indent=1)
