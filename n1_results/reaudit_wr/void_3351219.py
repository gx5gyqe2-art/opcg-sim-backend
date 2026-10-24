"""守りの監査の決着せず 1 局（3351219・154 手目・DON_BOX）が 3341083 と同じ型かを見る（1 回きり）。"""
import argparse, json, sys
sys.path.insert(0, "tests/scripts")
import defense_audit as DA
from opcg_sim.loop import engine as E, driver as DR, decks as D
a = argparse.Namespace(net=None, sims=E.PLAY_SIMS, budget=E.PLAY_BUDGET, worlds=1, lethal_proof=None, window_worlds=None)
E.engine(); db = D.load_db()
spec = E.SeatSpec(None, sims=a.sims, budget=a.budget)
la, lb = D.leader_pair(db, 3351219, "random")
p1, p2 = D.build_pair(db, la, lb, 3351219, "synth_roles")
hist = []
def obs(game, name, turn, st, out, move):
    if st >= 150:
        h = json.loads(game.hidden_json())["manager"]
        hist.append({"step": st, "seat": name, "pending": json.loads(game.pending_json()).get("action"), "msg": json.loads(game.pending_json()).get("message"), "kind": out.get("kind"),
                     "move": move.get("action_type"), "phase": h["phase"], "battle": h["active_battle"] is not None})
try:
    DR.run_game(3351219, {"p1": spec, "p2": spec}, p1, p2, observer=obs)
except DR.GameAborted as exc:
    print("aborted:", exc)
for x in hist: print(x)
