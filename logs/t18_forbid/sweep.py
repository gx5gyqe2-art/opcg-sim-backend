"""10 局（主条件）を介入なしで打ち、行ごとに「選んだ手が外れる損の幅」を記録する。"""
import json, sys, os
sys.path.insert(0, "/home/user/opcg-sim-backend"); sys.path.insert(0, "/home/user/opcg-sim-backend/tests/scripts")
import t18_arena as TA
import shadow_forbid as SF, live_theory as LT, theory_rs as TR
from opcg_sim.loop import decks as D, driver as DR, engine as E
from opcg_sim.learned.train import plan_labels as PL
base, n, decks, leaders, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
cards, idx2cid = PL.Cards(), TR.idx2cid(); E.engine()
spec = E.SeatSpec(None, sims=64, dirichlet_eps=0.25, temp_turns=4, prune_futile=E.GEN_PRUNE_FUTILE, worlds=4)
db = D.load_db(); rows = []; games = []
for seed in range(base, base + n):
    la, lb = D.leader_pair(db, seed, leaders); p1, p2 = D.build_pair(db, la, lb, seed, decks)
    def swap(game, name, turn, step, o, move, _s=seed):
        if o.get("kind") != "main": return move
        sc, tok, ci = LT.raw_row(game, name); cands = LT.raw_candidates(game, name, o)
        if not cands: return move
        r = SF.shadow_row(sc, tok, ci, cards, idx2cid, cands, o, move)
        if r is None: return move
        pr = r["prices"]; bv = max(v for v in pr if v is not None)
        fam = [SF.move_family(c["sig"]) for c in cands]
        elig = [i for i in range(len(cands)) if pr[i] is not None and fam[i] not in TA.EXEMPT_FAMILIES]
        rows.append({"seed": _s, "who": name, "turn": turn, "chosen": r["chosen_index"], "fam": fam,
                     "gaps": {i: bv - pr[i] for i in elig}})
        return move
    res = DR.run_game(seed, {"p1": spec, "p2": spec}, p1, p2, swap=swap)
    games.append({"seed": seed, "winner": res.get("winner"), "turns": res.get("turns")})
    print(seed, res.get("winner"), len(rows), flush=True)
json.dump({"rows": rows, "games": games}, open(out, "w"))
