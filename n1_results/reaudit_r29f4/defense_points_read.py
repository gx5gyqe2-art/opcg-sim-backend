"""守りの見逃し（確かめ後・甲）の 7 点を読み直す（§7.84 の 2（reaudit-r29f4・12 点）・1 回きり・器は defense_audit の replay_to をそのまま使う）。
実対局の decide の出力（窓の世界の本数・短い木か・候補の値）と、短い木の対象を広げた／読みを増やした打ち直しの手を書く。"""
import argparse, json, sys
sys.path.insert(0, "tests/scripts")
import defense_audit as DA
import lethal_audit as LA
from opcg_sim.loop import engine as E

PTS = [(4462001, "p1", 131), (4462038, "p2", 81), (4462057, "p2", 120), (4462075, "p2", 140), (4462102, "p1", 160), (4462106, "p1", 93),
       (4462121, "p1", 78), (4462203, "p2", 133), (4462251, "p1", 172), (4462260, "p1", 79), (4462283, "p2", 38), (4462285, "p1", 97)]
a = argparse.Namespace(net=None, sims=E.PLAY_SIMS, budget=E.PLAY_BUDGET, worlds=1, lethal_proof=None, window_worlds=None)
rows = []
for seed, seat, step in PTS:
    game, move, got = DA.replay_to(seed, a, seat, step)
    out = got["out"]
    opts = json.loads(got["spec"].decide_opts(seed, got["turn"], seat, None))
    legal = json.loads(game.legal_json(seat))
    r = {"seed": seed, "step": step, "pending": json.loads(game.pending_json()).get("action"), "pending_full": json.loads(game.pending_json()),
         "live": LA.describe(game, move), "kind": out.get("kind"), "window": out.get("window"),
         "Q": [round(x, 3) for x in (out.get("stats") or {}).get("Q", [])], "legal": [LA.describe(game, m) for m in legal]}
    for tag, extra in (("tree_life2", {"window_tree_life": 2}), ("sims_x4", {"sims": 4 * E.PLAY_SIMS}),
                       ("tree_life2_x4", {"window_tree_life": 2, "sims": 4 * E.PLAY_SIMS})):
        d = json.loads(game.fork().decide(seat, json.dumps(dict(opts, **extra))))
        r[tag] = {"move": LA.describe(game, d["move"]), "window": d.get("window"),
                  "Q": [round(x, 3) for x in d["stats"].get("Q", [])]}
    print(json.dumps(r, ensure_ascii=False), flush=True)
    rows.append(r)
json.dump(rows, open("n1_results/reaudit_r29f4/defense_points_read.json", "w"), ensure_ascii=False, indent=1)
