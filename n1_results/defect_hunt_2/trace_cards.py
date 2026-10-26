"""hunt_play と同じ局を打ち直し、指定の札（card_id）の uuid ごとの状態（レスト・フラグ）を手ごとに出す（1 回きり）。
使い方: python trace_cards.py SEED CID [--deck IDX ...] [--sims 160]"""
import argparse, json, os, sys, random
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
os.environ.setdefault("OPCG_LOG_SILENT", "1")
from opcg_sim.loop import engine as E, driver as DR, decks as D
ap = argparse.ArgumentParser(); ap.add_argument("seed", type=int); ap.add_argument("cid")
ap.add_argument("--deck", type=int, nargs="*", default=[]); ap.add_argument("--sims", type=int, default=160)
a = ap.parse_args()
DS = D.user_decks() + D.official_decks(); E.engine(); spec = E.SeatSpec(sims=a.sims)
seed = a.seed; rng = random.Random(seed * 31 + 7)
if a.deck:
    x = a.deck[seed % len(a.deck)]; y = rng.randrange(len(DS)); pair = (x, y) if seed % 2 == 0 else (y, x)
else:
    pair = (rng.randrange(len(DS)), rng.randrange(len(DS)))
last = {}
def post(game, name, turn, step, move, events):
    b = json.loads(game.board_json())
    ph = b["turn_info"].get("current_phase")
    for seat, p in b["players"].items():
        for zn in ("field",):
            for c in p["zones"][zn]:
                if c["card_id"] == a.cid:
                    st = (c["is_rest"], tuple(c.get("flags") or ()), c.get("is_frozen"))
                    if last.get(c["uuid"]) != st:
                        mv = move and (move.get("action_type"), (move.get("payload") or {}).get("uuid", move.get("card_uuid")))
                        print(f"t{turn} s{step} {ph} {seat} {c['uuid'][:8]} rest={c['is_rest']} frozen={c.get('is_frozen')} by {name} {mv}")
                        last[c["uuid"]] = st
r = DR.run_game(seed, {"p1": spec, "p2": spec}, DS[pair[0]], DS[pair[1]], post=post)
print(r["winner"], r["turns"])
