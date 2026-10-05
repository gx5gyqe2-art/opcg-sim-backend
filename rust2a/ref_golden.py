"""L4 の補い: 記録した 152 解（`tests/fixtures/rd_kernel_golden.jsonl.gz`）を予算 300000 と 150 で
rs（第 2a 段）と速くする前の原文（`rule_don_ref`）で解き、`repr` と記録のビットの 3 者が一致することを確かめる
（遅い・`make test` には入れない・1 回だけ回して報告に書く）。

    OPCG_LOG_SILENT=1 python rust2a/ref_golden.py [--out rust2a/ref_golden.json]
"""
import argparse
import gzip
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
import cut_price as CP  # noqa: E402
import rd_kernel as RK  # noqa: E402
import rule_don_ref as REF  # noqa: E402
import rd_kernel_golden as G  # noqa: E402
import test_rd_speed as TS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "rust2a", "ref_golden.json"))
    a = ap.parse_args()
    with gzip.open(G.GOLDEN, "rt", encoding="utf-8") as fh:
        recs = [json.loads(line) for line in fh if line.strip()]
    n = {"real": 0, "syn": 0}
    bad = []
    t0 = time.time()
    for r in recs:
        cards, don, blk, life, ax, turns, lt, dt, arr = G.args_of(r)
        for budget, expect in [(r["budget"], r["expect"])] + [(e["budget"], e["expect"]) for e in r["extra"]]:
            CB.EX_STATE_BUDGET = budget
            TS._clear_new()
            RK.reset_global()
            RK.set_mode("rs")
            with CP.defending(TS._view_of(ax)):
                rs = CB.rule_don_solve(cards, don, blk, life, dict(ax), turns, lt, dt, arr)
                REF.clear()
                ref = REF.solve(cards, don, blk, life, dict(ax), turns, lt, dt, arr)
            ok = repr(rs) == repr(ref) and RK.enc(rs) == expect
            if not ok:
                bad.append((r["src"], budget))
            n[r["src"].split("#")[0]] += 1
        print(r["src"], "ok" if not bad else bad[-3:], flush=True)
    out = {"solves": n, "mismatch": bad, "seconds": round(time.time() - t0, 1)}
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(out)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
