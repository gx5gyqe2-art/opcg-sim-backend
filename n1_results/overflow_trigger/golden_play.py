"""a1 帯の再生 golden（seed 5000150〜5000199・読み 32 回）の手の列だけを打ち直して書き出す（1 回きり・WP rs-overflow-trigger）。

    python golden_play.py OUT.json [EXTRA_OPTS_JSON]

wheel は PYTHONPATH で替える（修正前＝本線 80f8f9c の wheel・修正後＝今の wheel）。EXTRA は SeatSpec に足す opts
（例 '{"confirm_trigger_legacy": true}'＝(A) だけ入れた形）。手を決めた点の要求（action）も書く。
"""
import json, os, sys
REPO = "/home/user/opcg-sim-backend"
sys.path.insert(1, REPO); sys.path.insert(1, os.path.join(REPO, "tests")); sys.path.insert(1, os.path.join(REPO, "tests", "scripts"))
os.chdir(REPO)
import _bootstrap  # noqa
import opcg_engine
from opcg_sim.loop import decks as D, driver as DR, engine as E
out_path = sys.argv[1]
extra = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
E.engine()
print("engine", opcg_engine.__file__, extra, flush=True)
spec = E.SeatSpec(None, sims=32, **extra)
db = D.load_db()
rows = {}
for seed in range(5000150, 5000200):
    la, lb = D.leader_pair(db, seed, "random")
    p1, p2 = D.build_pair(db, la, lb, seed, "singleton")
    game = DR.new_game(seed, p1, p2)
    carry = {"p1": DR.Carry(), "p2": DR.Carry()}
    moves, acts, kinds = [], [], []
    while game.winner() is None and len(moves) < 400:
        pend = json.loads(game.pending_json())
        if not pend:
            break
        name = pend["player_id"]
        turn = int(game.turn_count or 0)
        o = json.loads(game.decide(name, spec.decide_opts(seed, turn, name, carry[name].get(turn, name))))
        carry[name].put(o)
        mv = o.get("move")
        if mv is None:
            break
        DR_apply = game.apply_battle_action if mv.get("kind") == "battle" else None
        if mv.get("kind") == "battle":
            game.apply_battle_action(name, mv["action_type"], mv.get("card_uuid"))
        else:
            game.apply_game_action(name, mv["action_type"], json.dumps(mv.get("payload") or {}, ensure_ascii=False))
        moves.append(mv); acts.append(pend.get("action")); kinds.append(o.get("kind"))
    rows[seed] = {"moves": moves, "acts": acts, "kinds": kinds, "winner": game.winner(), "n_legal": None}
    print(seed, len(moves), game.winner(), flush=True)
json.dump(rows, open(out_path, "w"), ensure_ascii=False)
