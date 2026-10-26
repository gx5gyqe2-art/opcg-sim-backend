"""a1 帯（修正後の wheel・読み 32 回）のトリガーの発動確認の点を 1 件ずつ読む（1 回きり・WP rs-overflow-trigger）。
golden_new.json の手の列を当て直し、確認の点で要求・札・ライフ・窓の値（発動する／しない）を書く。"""
import json, os, sys
REPO = "/home/user/opcg-sim-backend"
sys.path.insert(1, REPO); sys.path.insert(1, os.path.join(REPO, "tests"))
os.chdir(REPO)
import _bootstrap  # noqa
from opcg_sim.loop import decks as D, driver as DR, engine as E
E.engine(); db = D.load_db()
cards = {c["card_id"] if isinstance(c, dict) and "card_id" in c else None: c for c in []}
new = json.load(open("n1_results/overflow_trigger/golden_new.json"))
spec = E.SeatSpec(None, sims=32)
rows = []
for s, g in sorted(new.items(), key=lambda x: int(x[0])):
    seed = int(s)
    la, lb = D.leader_pair(db, seed, "random")
    p1, p2 = D.build_pair(db, la, lb, seed, "singleton")
    game = DR.new_game(seed, p1, p2)
    carry = {"p1": DR.Carry(), "p2": DR.Carry()}
    for i, (act, mv) in enumerate(zip(g["acts"], g["moves"])):
        pend = json.loads(game.pending_json()); name = pend["player_id"]; turn = int(game.turn_count or 0)
        if act == "CONFIRM_TRIGGER":
            o = json.loads(game.fork().decide(name, spec.decide_opts(seed, turn, name, None)))
            h = json.loads(game.hidden_json())
            me = h["players"][name]; op = h["players"]["p2" if name == "p1" else "p1"]
            src = pend.get("source_card_uuid") or pend.get("uuid")
            cid = None
            for P in h["players"].values():
                for z, cs in P.items():
                    for c in (cs if isinstance(cs, list) else [cs] if isinstance(cs, dict) else []):
                        if isinstance(c, dict) and c.get("uuid") == src:
                            cid = (c.get("card_id"), z)
            st = o.get("stats") or {}
            rows.append({"seed": seed, "step": i, "seat": name, "turn_player": h["manager"]["turn_player"], "msg": pend.get("message"),
                         "src": src, "card": cid, "my_life": len(me["life"]), "opp_life": len(op["life"]), "my_hand": len(me["hand"]),
                         "battle": h["manager"]["active_battle"] is not None,
                         "accepted": mv["payload"]["accepted"],
                         "q": dict(zip([str(m["payload"]["accepted"]) for m in st.get("legal") or []], [round(x, 3) for x in st.get("Q") or []])),
                         "pend_keys": sorted(pend.keys())})
        if mv.get("kind") == "battle":
            game.apply_battle_action(name, mv["action_type"], mv.get("card_uuid"))
        else:
            game.apply_game_action(name, mv["action_type"], json.dumps(mv.get("payload") or {}, ensure_ascii=False))
for r in rows:
    print(json.dumps(r, ensure_ascii=False))
json.dump(rows, open("n1_results/overflow_trigger/trigger_points.json", "w"), ensure_ascii=False, indent=1)
