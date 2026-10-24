"""reaudit-wr の決着せず 2 局（3341083・3351219）を修正後の wheel で当て直す（1 回きり・WP rs-overflow-trigger）。
3341083: 記録の手の列（43 手）を当てて押し出しの後の要求を見る→出荷の設定（r13d・記録時と同じ）で最後まで打つ。
3351219: 守りの監査と同じ設定（r13d・世界 1 本・1,280 回・予算なし）＋ confirm_trigger_legacy（記録時の打ち方）で
最初から打ち、押し出しの後の要求と局の終わりを見る。両方とも手の列を tests/fixtures へ写すために出力する。"""
import json, sys, time
sys.path.insert(0, "tests/scripts"); sys.path.insert(0, ".")
from opcg_sim.loop import engine as E, driver as DR, decks as D
import blunder_audit as BA
E.engine(); db = D.load_db()
R13D = "opcg_sim/data/learned/nrel_r13d.npz"
out = {}
# --- 3341083 ---
g = next(json.loads(l) for l in open("n1_results/reaudit_wr/blunder/games.jsonl") if json.loads(l)["seed"] == 3341083)
p1, p2 = D.build_pair(db, g["leaders"][0], g["leaders"][1], g["seed"], "real")
game = DR.new_game(g["seed"], p1, p2)
for s, mv in enumerate(g["moves"][:43]):
    BA.apply(game, json.loads(game.pending_json())["player_id"], mv)
pend = json.loads(game.pending_json()); h = json.loads(game.hidden_json())["manager"]
print("3341083 after 43 moves:", pend["player_id"], pend["action"], "phase", h["phase"], "battle", h["active_battle"] is not None)
spec = E.SeatSpec(R13D, sims=E.PLAY_SIMS, budget=E.PLAY_BUDGET, worlds=E.PLAY_WORLDS, lethal_proof="root")
t = time.time(); steps = 43
carries = {"p1": DR.Carry(), "p2": DR.Carry()}
while game.winner() is None and steps < 2000:
    pend = json.loads(game.pending_json())
    if not pend: break
    name = pend["player_id"]; turn = int(game.turn_count or 0)
    o = json.loads(game.decide(name, spec.decide_opts(g["seed"], turn, name, carries[name].get(turn, name))))
    carries[name].put(o)
    BA.apply(game, name, o["move"]); steps += 1
print("3341083 finished: winner", game.winner(), "steps", steps, "sec", round(time.time() - t, 1))
out["3341083"] = {"seed": 3341083, "leaders": g["leaders"], "decks": "real", "moves": g["moves"][:43],
                  "after": {"player_id": json.loads(game.pending_json() or "{}").get("player_id")}}
# --- 3351219 ---
la, lb = D.leader_pair(db, 3351219, "random")
q1, q2 = D.build_pair(db, la, lb, 3351219, "synth_roles")
spec2 = E.SeatSpec(R13D, sims=E.PLAY_SIMS, budget=E.PLAY_BUDGET, confirm_trigger_legacy=True)
moves = []; seen = []
def obs(game, name, turn, st, o, move):
    moves.append(move)
    p = json.loads(game.pending_json())
    if 150 <= st <= 156:
        h = json.loads(game.hidden_json())["manager"]
        seen.append((st, name, p.get("action"), (p.get("message") or "")[:40], o.get("kind"), move.get("action_type"), h["phase"], h["active_battle"] is not None))
t = time.time()
r = DR.run_game(3351219, {"p1": spec2, "p2": spec2}, q1, q2, observer=obs)
for x in seen: print("3351219", x)
print("3351219 finished: winner", r["winner"], "steps", r["steps"], "sec", round(time.time() - t, 1))
out["3351219"] = {"seed": 3351219, "leaders": [la, lb], "decks": "synth_roles", "moves": moves[:154]}
json.dump(out, open("n1_results/overflow_trigger/void_replay.json", "w"), ensure_ascii=False)
