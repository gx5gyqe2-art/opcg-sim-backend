"""a1 golden を defense_dialog_tree=off（旧い形）で打ち直し、別ディレクトリへ書く（差分レビュー用）。"""
import json, os, sys
sys.path.insert(0, "tests/scripts"); sys.path.insert(0, "tests")
import rs_golden_make as G
from opcg_sim.loop import engine as E
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
E.engine()
spec = E.SeatSpec(None, sims=32)
spec.opts["defense_dialog_tree"] = "off"
for i in range(50):
    seed = 5000150 + i
    g = G._record_game(seed, "a1", spec, "singleton")
    json.dump(g, open(os.path.join(out, f"a1_{seed}.json"), "w"), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    print(seed, flush=True)
