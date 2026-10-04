"""usage: with_hand.py MODE module_name args... — crossing_bridge の手札の形を切り替えてから他の器の main を呼ぶ。"""
import sys, importlib, os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT + "/tests/scripts")
sys.path.insert(0, _ROOT + "/tests")
sys.path.insert(0, _ROOT)
import crossing_bridge as CB
sys.path.insert(0, _TOOLS)
import plan_cache  # noqa: F401  (H-4e: shared plans)
mode, modname = sys.argv[1], sys.argv[2]
CB.set_theta_hand_mode(mode)
m = importlib.import_module(modname)
import os as _os
if _os.environ.get("MIRROR") == "0":
    import theory_bridge as _TB
    _TB.MIRROR_ME = False
assert CB.THETA_HAND_MODE == mode
rc = m.main(sys.argv[3:])
print("THETA_HAND_MODE at end:", CB.THETA_HAND_MODE, file=sys.stderr)
raise SystemExit(rc)
