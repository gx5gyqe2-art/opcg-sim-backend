"""**Rust 化・第 1 段**: 記録した解（ビットの値）の golden を作る／作り直す。

形式（`rd_kernel.enc`／`dec`）: 浮動小数は 16 桁の 16 進（ビット）・組と辞書は型を残す。
2 つのファイル:

* `tests/fixtures/rd_kernel_golden.jsonl.gz` — 1 行 = 実／合成の記録の 1 つの `rule_don_solve`（入力の全部・予算・解）。
  `tests/test_rd_kernel.py` が Rust の核で解き直してビットで比べる。**小さな予算の解**（`extra`）も添える
  （地平の縮めの道を通すため）。
* `rust/opcg_engine/tests/fixtures/rd_dp_golden.jsonl` — 1 行 = 同じ解の中の守る側の動的計画の 1 呼び出し
  （採った計画の守り）。`cargo test` が Python 無しで Rust 単体で再生する。
* **第 2a 段** `rust/opcg_engine/tests/fixtures/rd_solve_golden.jsonl.gz` — 1 行 = 1 つの `rule_don_solve` の
  `opcg_engine.rd_solve` への入力の全部（Python の `_rule_don_masks`＝`rules_steps` の出力・`model_horizon` の地平・
  ν の表を含む）と、予算／地平ごとの**Python の解き方の答え**（採った組・付与・支払い・値打ち・τ・歩きの列・守る側の
  結果・試行の開示）。浮動小数は 16 桁の 16 進。`cargo test` が Python 無しで計画の列挙・歩き・試行のループまで再生する。
  `python tests/scripts/rd_kernel_golden.py solve-golden` で作る（記録の 152 解 × 予算 2 と 8 局面 × 予算 2 ＋地平 2）。

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
SOLVE_GOLDEN = os.path.join(ROOT, "rust", "opcg_engine", "tests", "fixtures", "rd_solve_golden.jsonl.gz")
FRAMES = os.path.join(ROOT, "tests", "fixtures", "rd_speed_frames.json")
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


# ----------------------------------------------------------------------------------------------
# 第 2a 段: `rd_solve` の金型（Rust 単体の再生用）

_SPEED_KEYS = ("attempt_fail", "attempt_skipped", "count_calls", "count_fallback")


def _hx(v):
    import struct
    return "%016x" % struct.unpack("<Q", struct.pack("<d", float(v)))[0]


def _hxs(seq):
    return [_hx(x) for x in seq]


def _enc_solve_args(a):
    (cards, don, blk, life, _turns, lt, dt, arr, rest, att1_x, budget, flow, no_now, pr, nu, eps, feq, ds, a_tab,
     ar_tab, e_tab, floor, cap, masks, h0, _lim, layer_count) = a
    return {"cards": [_hxs(c) for c in cards], "don": _hx(don), "blk": _hxs(blk), "life": _hx(life),
            "lt": [_hxs(t) for t in lt], "dt": [_hxs(t) for t in dt], "arrive": _hxs(arr), "rest": _hxs(rest),
            "att1_x": _hxs(att1_x), "budget": int(budget), "flow": _hxs(flow), "no_now": bool(no_now),
            "prices": _hxs(pr), "nu": [_hxs(kv) for kv in nu], "eps": _hx(eps), "feq": _hx(feq), "ds": _hxs(ds),
            "a_tab": _hxs(a_tab), "ar_tab": _hxs(ar_tab), "e_tab": _hxs(e_tab), "slope_floor": _hx(floor),
            "race_cap": _hx(cap), "h0": int(h0), "layer_count": bool(layer_count),
            "masks": [[int(c), int(b), [_hxs(x) for x in ls], _hxs(h1), _hx(pa), _hx(pe), [int(k) for k in caps],
                       [_hxs(st) for st in steps]] for c, b, ls, h1, pa, pe, caps, steps in masks]}


def _solve_case(p, budget, turns):
    """Python の解き方（`py`）で 1 つ解き、`rd_solve` の入力と、Rust の答えの形にした期待値を返す。"""
    cards, don, blk, life, ax, _t, lt, dt, arr = p
    ax = dict(ax)
    view = _AvgView(ax["cp"][1]) if ax["cp"][0] is not None else None
    old = CB.EX_STATE_BUDGET
    CB.EX_STATE_BUDGET = budget
    for d in (CB._RULE_DON_CACHE, CB._RULE_EX_CACHE, CB._RULE_EX_MEMO, CB._RULE_EX_SETS, CB._RULE_EX_CTX):
        d.clear()
    RK.set_mode("py")
    try:
        with CP.defending(view):
            before = [CB.EX_SPEED_STATS[k] for k in _SPEED_KEYS]
            cut, st, plan = CB.rule_don_solve(cards, don, blk, life, ax, turns, lt, dt, arr)
            stats = [CB.EX_SPEED_STATS[k] - b for k, b in zip(_SPEED_KEYS, before)]
            masks = CB._rule_don_masks(cards, blk, life, ax, lt)
            h0 = None if turns is not None else CB.model_horizon(ax, blk, int(max(0, round(float(life)))), arr)
            args = RK.solve_args(cards, don, blk, life, ax, turns, lt, dt, arr, masks, h0, vars(CB))
            h = plan.get("horizon", turns)
            res = CB.rule_guard_plan_ex(cards, don, plan["xs_first"], None, blk, life, h, lt, CB._prices_of(ax),
                                        later_seq=plan["later_seq"], rest_blk=tuple(ax.get("rest_blk") or ()),
                                        arrive_blk=arr, draw_types=dt)
    finally:
        CB.EX_STATE_BUDGET = old
    assert tuple(res["harms"]) == plan["harm_steps"] and res["theta"] == plan["theta"] and res["cut"] == cut
    mi = [i for i, m in enumerate(masks) if tuple(m["play"]) == plan["play"]]
    assert len(mi) == 1
    exp = {"h": None if turns is not None else int(h), "mask": mi[0], "ks": [int(k) for k in plan["k"]],
           "paid": int(plan["paid"]), "incr": _hx(plan["incr"]), "val": _hx(plan["value"]), "tau": _hx(plan["tau"]),
           "sched": _hxs(plan["sched"]),
           "res": [_hx(cut), _hx(st), _hx(res["alive"]), _hx(res["prevented"]), _hxs(res["harms"]),
                   _hx(res["theta"]), _hx(res["nu_all"])],
           "stats": stats if turns is None else [0, 0, 0, 0]}
    return args, exp


def _frame_problems():
    with open(FRAMES, encoding="utf-8") as fh:
        raw = json.load(fh)
    out = []
    for f in raw:
        ax = dict(f["actx"])
        for k in ("key", "cp", "rest_blk"):
            ax[k] = _tup(ax[k])
        out.append((f["src"], (f["cards"], f["don"], f["blk"], f["life"], ax, None, _tup(f["life_types"]),
                               _tup(f["draw_types"]), _tup(f["arrive"]))))
    return out


def cmd_solve_golden():
    with gzip.open(GOLDEN, "rt", encoding="utf-8") as fh:
        recs = [json.loads(line) for line in fh if line.strip()]
    cases = [(r["src"], args_of(r), [(r["budget"], None)] + [(e["budget"], None) for e in r["extra"]]) for r in recs]
    cases += [("frame:" + src, p, [(300000, None), (150, None), (300000, 2)]) for src, p in _frame_problems()]
    lines = []
    for src, p, runs in cases:
        enc_in, out_runs = None, []
        for budget, turns in runs:
            args, exp = _solve_case(p, budget, turns)
            e_in = _enc_solve_args(args)
            h0 = e_in.pop("h0")
            if enc_in is None:
                enc_in = e_in
            assert e_in == enc_in, src                         # 入力は予算・地平に依らない
            out_runs.append({"limit": budget, "turns": turns, "h0": h0, "expect": exp})
        lines.append(json.dumps({"src": src, "in": enc_in, "runs": out_runs}, separators=(",", ":")))
    with gzip.open(SOLVE_GOLDEN, "wt", encoding="utf-8", compresslevel=9) as fh:
        for ln in lines:
            fh.write(ln + "\n")
    print("wrote %s (%d records, %d bytes)" % (SOLVE_GOLDEN, len(lines), os.path.getsize(SOLVE_GOLDEN)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("paths", nargs="+")
    m = sub.add_parser("remake")
    m.add_argument("--ref", action="store_true")
    sub.add_parser("solve-golden")
    a = ap.parse_args(argv)
    if a.cmd == "solve-golden":
        cmd_solve_golden()
    elif a.cmd == "build":
        cmd_build(a.paths)
    else:
        cmd_remake(a.ref)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
