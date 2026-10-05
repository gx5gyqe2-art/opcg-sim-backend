"""第 2a 段の速さ: 8 局面（`tests/fixtures/rd_speed_frames.json`）を予算 300000 と 150 で py と rs の冷たい解きで測る
（各 7 回の最良・覚え書きを毎回消す・同じ機械で 1 本だけ）。結果の repr が同じことも確かめる。

    OPCG_LOG_SILENT=1 python rust2a/speed_frames.py [--out rust2a/speed_frames.json]
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tests"))
for p in ("scripts", "harness"):
    sys.path.insert(0, os.path.join(ROOT, "tests", p))
import _bootstrap  # noqa: F401,E402
import crossing_bridge as CB  # noqa: E402
import rd_kernel as RK  # noqa: E402
import test_rd_speed as TS  # noqa: E402


def one(p, mode, reps=7):
    """`mode`: py／rs（第 2a 段）／rs1（第 1 段だけ: `rule_don_solve` の包みを Python の原文に戻した rs）。"""
    best, out = None, None
    wrapped = vars(CB)["rule_don_solve"]
    if mode == "rs1":
        vars(CB)["rule_don_solve"] = RK._ORIG[id(vars(CB))]["rule_don_solve"]
    try:
        return _one(p, "rs" if mode == "rs1" else mode, reps)
    finally:
        vars(CB)["rule_don_solve"] = wrapped


def _one(p, mode, reps):
    best, out = None, None
    for _ in range(reps):
        RK.set_mode(mode)
        TS._clear_new()
        RK.reset_global()
        t = time.perf_counter()
        out = TS._solve_new(p)
        dt = time.perf_counter() - t
        best = dt if best is None else min(best, dt)
    return best, repr(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "rust2a", "speed_frames.json"))
    a = ap.parse_args()
    rows = []
    for budget in (300000, 150):
        CB.EX_STATE_BUDGET = budget
        for src, p in TS._frames():
            tp, rp = one(p, "py")
            t1, r1 = one(p, "rs1")
            tr, rr = one(p, "rs")
            assert rp == rr == r1, (src, budget)
            rows.append({"frame": src, "budget": budget, "py_s": round(tp, 4), "rs1_s": round(t1, 4),
                         "rs_s": round(tr, 4), "speedup_rs1": round(tp / t1, 2), "speedup": round(tp / tr, 2)})
            print(rows[-1], flush=True)
    tot = {}
    for budget in (300000, 150):
        r = [x for x in rows if x["budget"] == budget]
        py, rs1, rs = (sum(x[k] for x in r) for k in ("py_s", "rs1_s", "rs_s"))
        tot[str(budget)] = {"py_s": round(py, 4), "rs1_s": round(rs1, 4), "rs_s": round(rs, 4),
                            "speedup_rs1": round(py / rs1, 2), "speedup": round(py / rs, 2)}
    with open(a.out, "w") as fh:
        json.dump({"rows": rows, "total": tot}, fh, indent=1)
    print(tot)


if __name__ == "__main__":
    main()
