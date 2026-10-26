"""守りの証明の単体テストの盤面（3351102・99）を、局の手の列として fixture に書く（WP `rs-defense-proof`・1 回きり）。

  python n1_results/defense_proof/make_fixture.py   # → tests/fixtures/defense_proof_points.json
"""
import json
import os
import sys
import types

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "tests", "scripts"))
os.environ.setdefault("OPCG_LOG_SILENT", "1")
from opcg_sim.loop import decks as D, driver as DR, engine as E  # noqa: E402

NET = "opcg_sim/data/learned/nrel_r13d.npz"
SEED, SEAT, STEP = 3351102, "p2", 99


class _Stop(Exception):
    pass


def main():
    E.engine()
    db = D.load_db()
    spec = E.SeatSpec(NET, sims=E.PLAY_SIMS, budget=E.PLAY_BUDGET)
    la, lb = D.leader_pair(db, SEED, "random")
    p1, p2 = D.build_pair(db, la, lb, SEED, "synth_roles")
    moves, got = [], {}

    def obs(game, name, turn, step, out, move):
        if step == STEP:
            got.update(turn=turn, seat=name, move=move, q=out["stats"]["Q"], legal=out["stats"]["legal"],
                       action=json.loads(game.pending_json())["action"])
            raise _Stop()
        moves.append(move)

    try:
        DR.run_game(SEED, {"p1": spec, "p2": spec}, p1, p2, observer=obs)
    except _Stop:
        pass
    assert got["seat"] == SEAT and len(moves) == STEP, (got, len(moves))
    point = {"seed": SEED, "step": STEP, "seat": SEAT, "turn": got["turn"], "action": got["action"],
             "decks": {"p1": [p1[0], list(p1[1])], "p2": [p2[0], list(p2[1])]},
             "cpu_move": got["move"], "legal": got["legal"], "q": got["q"], "moves": moves}
    out = {"_note": "守りの再監査（2026-10-24_overflow_trigger.md §3）の乙の見逃し 3351102・99（自分のライフ 1・カウンターの窓・"
                    "両席 r13d・§7.24 の設定）。局の手の列（step まで）を当て直して盤面を作る。tests/test_rs_defense_proof.py が読む。",
           "points": [point]}
    path = os.path.join(_ROOT, "tests", "fixtures", "defense_proof_points.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(path, len(moves), got["move"])


if __name__ == "__main__":
    main()
