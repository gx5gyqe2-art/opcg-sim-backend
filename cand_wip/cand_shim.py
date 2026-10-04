"""usage: CAND="slope_take=life,slope_block=on,kappa_sigma=match" python cand_wip/cand_shim.py TOOL args...
Sets the named candidate switches (setters of the repo) and then runs tests/scripts/TOOL.py main(args). TOOL in
{crossing_bridge, win_calib, relative_ledger}. Empty CAND = baseline (the shipped default)."""
import importlib, os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path[:0] = [ROOT + "/tests/scripts", ROOT + "/tests", ROOT]
import crossing_bridge as CB
import theory_order as TO
SETTERS = {"slope_take": CB.set_slope_take_mode, "slope_block": CB.set_slope_block_mode,
           "kappa_sigma": TO.set_kappa_sigma_mode}
for kv in filter(None, os.environ.get("CAND", "").split(",")):
    k, v = kv.split("=")
    SETTERS[k](v)
tool = importlib.import_module(sys.argv[1])
rc = tool.main(sys.argv[2:])
print("CAND:", os.environ.get("CAND", ""), "slope_take=", CB.SLOPE_TAKE_MODE, "slope_block=", CB.SLOPE_BLOCK_MODE,
      "kappa_sigma=", TO.KAPPA_SIGMA_MODE, file=sys.stderr)
raise SystemExit(rc)
