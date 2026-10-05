"""**Rust 化・第 1 段**: 記録した解（ビットの値）の golden を作る／作り直す。

形式（`rd_kernel.enc`／`dec`）: 浮動小数は 16 桁の 16 進（ビット）・組と辞書は型を残す。
2 つのファイル:

* `tests/fixtures/rd_kernel_golden.jsonl.gz` — 1 行 = 実／合成の記録の 1 つの `rule_don_solve`（入力の全部・予算・解）。
  `tests/test_rd_kernel.py` が Rust の核で解き直してビットで比べる。**小さな予算の解**（`extra`）も添える
  （地平の縮めの道を通すため）。
* `rust/opcg_engine/tests/fixtures/rd_dp_golden.jsonl` — 1 行 = 同じ解の中の守る側の動的計画の 1 呼び出し
  （採った計画の守り）。`cargo test` が Python 無しで Rust 単体で再生する。

使い方:

    # 1) 記録から取り込む（実 w41・合成 w39+w42。測定の通しの途中で `OPCG_RD_CAPTURE` を付ける）
    OPCG_RD_CAPTURE=cap/real.jsonl OPCG_RD_CAPTURE_SRC=real OPCG_RD_CAPTURE_EVERY=4 \\
        python tests/scripts/crossing_bridge.py --in <w41> --limit-games 12
    # 2) golden にまとめる（解は Python の解き方で検算してから書く）
    python tests/scripts/rd_kernel_golden.py build cap/real.jsonl cap/syn.jsonl
    # 3) 挙動を意図して変えたときだけ作り直す（**差分は必ずレビューする**）
    python tests/scripts/rd_kernel_golden.py remake [--ref]

golden は「その時点の Python の出力」であって、正しさの独立した証拠ではない（CLAUDE.md の golden の作法と同じ）。
`--ref` は速くする前の原文（`rule_don_ref`）で解き直す（遅い・独立した検算）。
"""
import argparse
import gzip
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
import _bootstrap  # noqa: F401,E402
for _p in (_HERE, os.path.join(os.path.dirname(_HERE), "harness")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import crossing_bridge as CB  # noqa: E402
import cut_price as CP  # noqa: E402
import rd_kernel as RK  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(_HERE))
GOLDEN = os.path.join(ROOT, "tests", "fixtures", "rd_kernel_golden.jsonl.gz")
DP_GOLDEN = os.path.join(ROOT, "rust", "opcg_engine", "tests", "fixtures", "rd_dp_golden.jsonl")
#: 小さな予算（地平の縮めを通す）
EXTRA_BUDGETS = (150,)


class _AvgView:
    """`cut_price.CutView`（`avg`）と同じ値段の窓（`tests/test_rd_speed.py` と同じ）。"""
    kind = "avg"

    def __init__(self, g):
        self.curve = type("C", (), {"gbar": float(g)})()

    def price(self, k, mu=None):
        k = float(k)
        return 0.0 if k <= 0.0 else float(k * self.curve.gbar)


def _tup(v):
    return tuple(_tup(x) for x in v) if isinstance(v, (list, tuple)) else v


def args_of(rec):
    cards, don, blk, life, ax, turns, lt, dt, arr = RK.dec(rec["args"])
    ax = dict(ax)
    for k in ("key", "cp", "rest_blk"):
        ax[k] = _tup(ax[k])
    return cards, don, blk, life, ax, turns, lt, dt, arr


def solve(rec, budget, ref=False):
    cards, don, blk, life, ax, turns, lt, dt, arr = args_of(rec)
    view = _AvgView(ax["cp"][1]) if ax["cp"][0] is not None else None
    old = CB.EX_STATE_BUDGET
    CB.EX_STATE_BUDGET = budget
    for d in (CB._RULE_DON_CACHE, CB._RULE_EX_CACHE, CB._RULE_EX_MEMO, CB._RULE_EX_SETS, CB._RULE_EX_CTX):
        d.clear()
    try:
        with CP.defending(view):
            if ref:
                import rule_don_ref as REF
                REF.clear()
                return REF.solve(cards, don, blk, life, dict(ax), turns, lt, dt, arr)
            RK.set_mode("py")
            return CB.rule_don_solve(cards, don, blk, life, dict(ax), turns, lt, dt, arr)
    finally:
        CB.EX_STATE_BUDGET = old


def dp_expect(rec):
    """動的計画の 1 呼び出し（採った計画の守り）を Python の解き方で解き直した辞書（符号化済み）。"""
    dp = rec["dp"]
    cards = [tuple(c) for c in RK.dec(dp["cards"])]
    pr = dict(zip(("lam", "lam_net", "mu", "olp", "mlp"), RK.dec(dp["prices"])))
    ax = args_of(rec)[4]
    view = _AvgView(ax["cp"][1]) if ax["cp"][0] is not None else None
    RK.set_mode("py")
    for d in (CB._RULE_EX_CACHE, CB._RULE_EX_MEMO, CB._RULE_EX_SETS, CB._RULE_EX_CTX):
        d.clear()
    with CP.defending(view):
        res = CB.rule_guard_plan_ex(cards, RK.dec(dp["don"]), RK.dec(dp["xs_first"]), None, RK.dec(dp["blk"]),
                                    RK.dec(dp["life"]), dp["turns"], _tup(RK.dec(dp["life_types"])), pr,
                                    later_seq=_tup(RK.dec(dp["seq"])), rest_blk=_tup(RK.dec(dp["rest"])),
                                    arrive_blk=_tup(RK.dec(dp["arrive"])), draw_types=_tup(RK.dec(dp["draw_types"])))
    return RK.enc(res)


def write(recs):
    os.makedirs(os.path.dirname(GOLDEN), exist_ok=True)
    with gzip.open(GOLDEN, "wt", encoding="utf-8", compresslevel=9) as fh:
        for r in recs:
            fh.write(json.dumps({k: r[k] for k in ("src", "budget", "args", "expect", "extra")}, separators=(",", ":")) + "\n")
    os.makedirs(os.path.dirname(DP_GOLDEN), exist_ok=True)
    with open(DP_GOLDEN, "w", encoding="utf-8") as fh:
        for r in recs:
            if r.get("dp"):
                fh.write(json.dumps({"src": r["src"], "dp": r["dp"]}, separators=(",", ":")) + "\n")
    print("wrote %s (%d bytes) and %s (%d bytes)" % (GOLDEN, os.path.getsize(GOLDEN), DP_GOLDEN,
                                                     os.path.getsize(DP_GOLDEN)))


def cmd_build(paths):
    recs = []
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    recs.append(json.loads(line))
    out = []
    for r in recs:
        if "dp" not in r:
            continue
        got = solve(r, r["budget"])
        if RK.enc(got) != r["expect"]:
            raise SystemExit("取り込んだ解が Python の解き直しと違う: " + r["src"])
        if dp_expect(r) != r["dp"]["expect"]:
            raise SystemExit("取り込んだ動的計画の呼び出しが Python の解き直しと違う: " + r["src"])
        extra = [{"budget": b, "expect": RK.enc(solve(r, b))} for b in EXTRA_BUDGETS]
        r["extra"] = extra
        out.append(r)
    nr = sum(r["src"].startswith("real") for r in out)
    ns = sum(r["src"].startswith("syn") for r in out)
    print("records: real=%d syn=%d" % (nr, ns))
    write(out)


def cmd_remake(ref):
    with gzip.open(GOLDEN, "rt", encoding="utf-8") as fh:
        recs = [json.loads(line) for line in fh if line.strip()]
    dps = {}
    if os.path.exists(DP_GOLDEN):
        with open(DP_GOLDEN, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    d = json.loads(line)
                    dps[d["src"]] = d["dp"]
    changed = 0
    for r in recs:
        new = RK.enc(solve(r, r["budget"], ref))
        changed += int(new != r["expect"])
        r["expect"] = new
        for e in r["extra"]:
            new = RK.enc(solve(r, e["budget"], ref))
            changed += int(new != e["expect"])
            e["expect"] = new
        r["dp"] = dps.get(r["src"])
        if r["dp"]:
            new = dp_expect(r)
            changed += int(new != r["dp"]["expect"])
            r["dp"]["expect"] = new
    print("changed expectations: %d (差分をレビューする)" % changed)
    write(recs)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("paths", nargs="+")
    m = sub.add_parser("remake")
    m.add_argument("--ref", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "build":
        cmd_build(a.paths)
    else:
        cmd_remake(a.ref)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
