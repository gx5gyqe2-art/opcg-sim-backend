"""守りの見逃し（確かめ後・甲）の 7 点を読み直す（§7.43 の 2・1 回きり・器は defense_audit の replay_to をそのまま使う）。
実対局の decide の出力（窓の世界の本数・短い木か・候補の値）と、短い木の対象を広げた／読みを増やした打ち直しの手を書く。"""
import argparse, json, sys
sys.path.insert(0, "tests/scripts")
import defense_audit as DA
import lethal_audit as LA
from opcg_sim.loop import engine as E

PTS = [(3351039, "p1", 164), (3351102, "p2", 135), (3351124, "p2", 134), (3351191, "p1", 114),
       (3351233, "p1", 81), (3351284, "p1", 146), (3351284, "p1", 158)]
a = argparse.Namespace(net=None, sims=E.PLAY_SIMS, budget=E.PLAY_BUDGET, worlds=1, lethal_proof=None, window_worlds=None)
rows = []
for seed, seat, step in PTS:
    game, move, got = DA.replay_to(seed, a, seat, step)
    out = got["out"]
    opts = json.loads(got["spec"].decide_opts(seed, got["turn"], seat, None))
    legal = json.loads(game.legal_json(seat))
    r = {"seed": seed, "step": step, "pending": json.loads(game.pending_json()).get("action"),
         "live": LA.describe(game, move), "kind": out.get("kind"), "window": out.get("window"),
         "Q": [round(x, 3) for x in (out.get("stats") or {}).get("Q", [])], "legal": [LA.describe(game, m) for m in legal]}
    for tag, extra in (("tree_life2", {"window_tree_life": 2}), ("sims_x4", {"sims": 4 * E.PLAY_SIMS}),
                       ("tree_life2_x4", {"window_tree_life": 2, "sims": 4 * E.PLAY_SIMS})):
        d = json.loads(game.fork().decide(seat, json.dumps(dict(opts, **extra))))
        r[tag] = {"move": LA.describe(game, d["move"]), "window": d.get("window"),
                  "Q": [round(x, 3) for x in d["stats"].get("Q", [])]}
    print(json.dumps(r, ensure_ascii=False), flush=True)
    rows.append(r)
json.dump(rows, open("n1_results/reaudit_wr/defense_points_read.json", "w"), ensure_ascii=False, indent=1)
