"""crossing_bridge を「第 1 段だけの rs」（守る側の動的計画・段の数え方は Rust、計画の列挙・歩き・試行のループは Python）で
回す——第 2a 段の速さを第 1 段と比べるためだけ（`rule_don_solve` の包みを Python の原文に戻す。値は同じ）。

    OPCG_RD_KERNEL=rs python rust2a/cb_stage1.py --in <dir> --limit-games 5 --out <json>
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tests"))
sys.path.insert(0, os.path.join(ROOT, "tests", "scripts"))
import crossing_bridge as CB  # noqa: E402
import rd_kernel as RK  # noqa: E402

assert RK.MODE == "rs"
vars(CB)["rule_don_solve"] = RK._ORIG[id(vars(CB))]["rule_don_solve"]
raise SystemExit(CB.main(sys.argv[1:]))
