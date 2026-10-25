"""乙の 2 点（3351102・99／3351191・114）で、各候補を当てた後の盤面を相手の側から証明する（WP `rs-defense-proof`・1 回きり）。

本当の盤面（相手の手札も本物）で: (1) 本当の配り方 1 通り（`true_deal`）・(2) 自分の札を相手の視点で公平に配る 8 通り
（決定の守りの証明と同じ配り方・`Game.lethal_json` の `prove`）・(3) 自分の視点で配る 8 通り（相手の札を配り直し、
自分の手札は本物のまま・`view="other"`）。決定の守りの証明（`defense_proof`）の出力も並べる。
  python n1_results/defense_proof/points_probe.py > n1_results/defense_proof/points_probe.json
"""
import json
import os
import sys
import types

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "tests", "scripts"))
os.environ.setdefault("OPCG_LOG_SILENT", "1")
import defense_audit as DA  # noqa: E402
from opcg_sim.loop import engine as E  # noqa: E402

NET = "opcg_sim/data/learned/nrel_r13d.npz"
POINTS = ((3351102, "p2", 99), (3351191, "p1", 114))


def main():
    a = types.SimpleNamespace(budget=E.PLAY_BUDGET, worlds=1, lethal_proof=None, net=NET, sims=E.PLAY_SIMS)
    out = []
    for seed, seat, step in POINTS:
        game, move, got = DA.replay_to(seed, a, seat, step)
        opp = "p1" if seat == "p2" else "p2"
        spec = got["spec"]
        opts = json.loads(spec.decide_opts(seed, got["turn"], seat, None))
        base = json.loads(game.fork().decide(seat, json.dumps(opts)))
        new = json.loads(game.fork().decide(seat, json.dumps(dict(opts, defense_proof=True))))
        rows = []
        for i, mv in enumerate(base["stats"]["legal"]):
            g = game.fork()
            g.apply_search_move(seat, json.dumps(mv))
            true = json.loads(g.fork().lethal_json(opp, json.dumps({"prove": True, "true_deal": True, "max_nodes": 200000})))
            fair = json.loads(g.fork().lethal_json(opp, json.dumps({"prove": True, "search_seed": seed, "max_nodes": 200000})))
            mine = json.loads(g.fork().lethal_json(opp, json.dumps({"prove": True, "search_seed": seed, "max_nodes": 200000,
                                                                     "view": "other"})))
            rows.append({"i": i, "move": mv, "q": base["stats"]["Q"][i], "true": true["per"], "fair": fair["per"],
                         "nodes_fair": fair["nodes"], "mine": mine["per"], "nodes_mine": mine["nodes"]})
        out.append({"seed": seed, "seat": seat, "step": step, "cpu_move": move, "base_move": base["move"],
                    "new_move": new["move"], "defense_proof": new.get("defense_proof"), "rows": rows})
        print(json.dumps(out[-1], ensure_ascii=False), file=sys.stderr, flush=True)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
