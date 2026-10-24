"""修正前の wheel（本線 80f8f9c）で守りの監査の 3351219 を打ち直し、決着しなかった手の列（154 手）を取る（1 回きり）。
PYTHONPATH に修正前の wheel を先に置いて走らせる。設定は reaudit-wr の void_3351219.py と同じ（r13d・世界 1 本・1,280 回・予算なし）。"""
import json, sys
sys.path.insert(1, "."); sys.path.insert(1, "tests/scripts")
import opcg_engine
from opcg_sim.loop import engine as E, driver as DR, decks as D
print("engine", opcg_engine.__file__)
E.engine(); db = D.load_db()
spec = E.SeatSpec("opcg_sim/data/learned/nrel_r13d.npz", sims=E.PLAY_SIMS, budget=E.PLAY_BUDGET)
la, lb = D.leader_pair(db, 3351219, "random")
p1, p2 = D.build_pair(db, la, lb, 3351219, "synth_roles")
moves = []; seen = []
def obs(game, name, turn, st, out, move):
    moves.append(move)
    if st >= 150:
        p = json.loads(game.pending_json()); h = json.loads(game.hidden_json())["manager"]
        seen.append((st, name, p.get("action"), (p.get("message") or "")[:40], out.get("kind"), move.get("action_type"), h["phase"], h["active_battle"] is not None))
try:
    DR.run_game(3351219, {"p1": spec, "p2": spec}, p1, p2, observer=obs)
except DR.GameAborted as exc:
    print("aborted:", exc)
for x in seen: print(x)
json.dump({"seed": 3351219, "leaders": [la, lb], "decks": "synth_roles", "moves": moves[:154]},
          open("n1_results/overflow_trigger/void_3351219_old.json", "w"), ensure_ascii=False)
