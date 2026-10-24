"""WP weak-points §3: 3341039・84（捨てる 1 枚）の候補ごとに、捨てた後の盤面の静的な値（窓が読むもの）と、
その盤面から今のターンの残りを木で読んだ根の値（出荷の設定 r13d・1,280 回・世界 2 本、と 4 倍）を並べる（1 回きり）。"""
import json, sys
sys.path.insert(0, "tests/scripts")
sys.path.insert(0, "n1_results/weak_points")
import blunder_audit as BA
from dbl_attack import static  # noqa: E402  （r13d／r22 の静的な値）
BA._init_judge(1)
g = next(json.loads(l) for l in open("n1_results/reaudit_wr/blunder/games.jsonl") if json.loads(l)["seed"] == 3341039)
game = BA.replay(g, 84)
me = "p2"
turn = int(game.turn_count)
h = json.loads(game.hidden_json())
hand = {c["uuid"]: c["card_id"] for c in h["players"][me]["hand"]}
out = json.loads(game.fork().decide(me, BA._G["ship"].decide_opts(3341039, turn, me, None)))
legal, Q = out["stats"]["legal"], out["stats"]["Q"]
rows, seen = [], set()
for mv, q in zip(legal, Q):
    cid = tuple(hand.get(u) for u in mv["payload"]["selected_uuids"])
    if cid in seen:
        continue
    seen.add(cid)
    c = game.fork()
    c.apply_search_move(me, json.dumps(mv))
    r = {"discard": cid, "window_q": round(q, 4), "static": static(c, me)}
    pend = json.loads(c.pending_json())
    for tag, mult in (("tree_x1", 1), ("tree_x4", 4)):
        o = json.loads(BA._G["ship"].decide_opts(3341039, turn, me, None))
        o["sims"] = o["sims"] * mult
        d = json.loads(c.fork().decide(pend["player_id"], json.dumps(o)))
        rs = BA.ranked(d)
        tt = sum(x["n"] for x in rs)
        v = sum(x["n"] * x["q"] for x in rs) / tt if tt else None
        r[tag] = {"v": None if v is None else round(v if pend["player_id"] == me else -v, 4), "best": BA.desc(c, d.get("move"))[:90], "kind": d.get("kind")}
    print(json.dumps(r, ensure_ascii=False), flush=True)
    rows.append(r)
json.dump(rows, open("n1_results/weak_points/hand_tree.json", "w"), ensure_ascii=False, indent=1)
