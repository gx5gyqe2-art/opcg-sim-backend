"""WP weak-points §4: 詰みの証明の点を今の本線で 1 点ずつ読む（1 回きり）。記録の手を当て直し、
既定の配り方・本当の配り方（節点の上限を上げて）・守りの配り方の中身（trace）を書く。
使い方: python lethal_points.py SEED:SEAT:TURN:STEP ... [--max-nodes N]"""
import gzip, json, os, sys
sys.path.insert(0, "tests/scripts")
import lethal_speed_check as LS
from opcg_sim.loop import decks as D, driver as DR, engine as E

args = [x for x in sys.argv[1:] if not x.startswith("--")]
mx = int(sys.argv[sys.argv.index("--max-nodes") + 1]) if "--max-nodes" in sys.argv else 20_000_000
args = [x for x in args if x != str(mx)]
want = {}
for a in args:
    s, seat, t, st = a.split(":")
    want.setdefault(int(s), []).append((seat, int(t), int(st)))
E.engine(); db = D.load_db()
out = []
for line in gzip.open("n1_results/lethal_speed/record_3012000_300.jsonl.gz", "rt"):
    g = json.loads(line)
    if g["seed"] not in want:
        continue
    seed = g["seed"]
    p1, p2 = LS._decks(db, seed)
    game = DR.new_game(seed, p1, p2)
    steps = {st: (seat, t) for seat, t, st in want[seed]}
    for step, m in enumerate(g["moves"]):
        if step in steps:
            seat, t = steps[step]
            h = json.loads(game.hidden_json())
            opp = "p2" if seat == "p1" else "p1"
            r = {"pt": f"{seed}:{seat}:{t}:{step}",
                 "default": LS._judge(game, seat, seed, step, None),
                 "true_big": LS._judge(game, seat, seed, step, mx, True),
                 "opp_life": [c["card_id"] for c in h["players"][opp]["life"]],
                 "opp_hand": [c["card_id"] for c in h["players"][opp]["hand"]],
                 "opp_don_active": len((h["players"][opp].get("don") or {}).get("active") or [])}
            o = json.loads(game.lethal_json(seat, json.dumps({"search_seed": seed * 1000 + step, "force": True, "trace": True})))
            r["trace_deals"] = [{k: d.get(k) for k in ("deal", "opp_hand", "opp_life", "result", "win")} for d in (o.get("trace") or [])][:8] if isinstance(o.get("trace"), list) else str(o.get("trace"))[:2000]
            for k in ("default", "true_big"):
                r[k] = {x: r[k].get(x) for x in ("status", "move", "nodes", "deals", "line_len")}
            print(json.dumps(r, ensure_ascii=False), flush=True)
            out.append(r)
        name, move = m["name"], m["move"]
        if move.get("kind") == "battle":
            game.apply_battle_action(name, move["action_type"], move.get("card_uuid"))
        else:
            game.apply_game_action(name, move["action_type"], json.dumps(move.get("payload") or {}, ensure_ascii=False))
        if step >= max(steps):
            break
json.dump(out, open("n1_results/weak_points/lethal/points_read.json", "w"), ensure_ascii=False, indent=1)
