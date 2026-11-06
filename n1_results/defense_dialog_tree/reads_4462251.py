"""4462251・172 の窓を旗ごとに seed 8 通りで読む（新しい既定＝畳まない／畳む形 defense_dialog_fold）。1 回きり。"""
import argparse, json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tests", "scripts")); sys.path.insert(0, ROOT); os.chdir(ROOT)
import blunder_audit as BA
import defense_audit as DA
from opcg_sim.loop import engine as E
E.engine()
a = argparse.Namespace(net=None, sims=E.PLAY_SIMS, budget=E.PLAY_BUDGET, worlds=1, lethal_proof=None, window_worlds=None)
game, move, got = DA.replay_to(4462251, a, "p1", 172)
base = json.loads(got["spec"].decide_opts(4462251, got["turn"], "p1", None))
for tag, extra in (("off", {}), ("life1", {"defense_dialog_tree": "life1"}),
                   ("life1_fold", {"defense_dialog_tree": "life1", "defense_dialog_fold": True})):
    for i in range(8):
        o = dict(base, **extra)
        if i:
            o["search_seed"] = (o["search_seed"] ^ (i * 0x9E3779B97F4A7C15)) & 0xFFFFFFFFFFFFFFFF
        d = json.loads(game.fork().decide("p1", json.dumps(o)))
        print(tag, i, "accepted" if d["move"]["payload"].get("accepted", True) else "declined", d.get("window"),
              [round(x, 3) for x in d["stats"]["Q"]], flush=True)
