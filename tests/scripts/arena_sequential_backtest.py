"""アリーナの逐次判定の後ろ向きの検証（`docs/reports/2026-10-17_arena_sequential.md` §1.4・1 回きり・テストなし）。

r11 の台帳（`n1_results/r11/`）を記録の順（シャードのファイル名の順 × ファイルの中の seed の順）で 1 ペアずつ積み、
`arena.sequential_stop` が何ペアで止まるか・止めた結論が全ペアの今の規約の結論と同じかを出す。
シャードの並べ方を `--perms` 回入れ替えた分布も出す（ファイルの中の seed の順は保つ）。新しい対局は回さない。

    python tests/scripts/arena_sequential_backtest.py --out /tmp/seq.json
"""
import argparse
import glob
import json
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from opcg_sim.loop import arena as A                    # noqa: E402
from opcg_sim.loop import arena_merge as AM             # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R11 = os.path.join(ROOT, "n1_results", "r11")
MEASURES = [("r11a", "arena_r11a"), ("r11b", "arena_r11b"), ("r11c", "arena_r11c"), ("r11d", "arena_r11d"),
            ("r11c_vs_r11a", "arena_r11c_vs_r11a"), ("r11a_hires", "hires_r11a")]


def full_conclusion(xs, rule):
    line, strict = A.SEQ_LINES[rule]
    lo = A.pair_level_ci(xs)["lo"]
    return "above" if ((lo > line) if strict else (lo >= line)) else "below_or_open"


def one(per_file, rule, planned):
    xs = AM.ledger_order_scores(per_file)
    line, strict = A.SEQ_LINES[rule]
    seq = A.sequential_stop(xs, planned, line, strict)
    full = full_conclusion(xs, rule)
    if seq["stopped"] is None:
        return {"stopped": None, "frac": 1.0, "side": None, "full": full, "changed": False}
    changed = (seq["side"] == "above") != (full == "above")
    return {"stopped": seq["stopped"], "frac": seq["stopped"] / len(xs), "side": seq["side"], "full": full,
            "changed": changed, "mean_at_stop": round(seq["mean"], 4), "lo_at_stop": round(seq["lo"], 4),
            "hi_at_stop": round(seq["hi"], 4)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--perms", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rng = random.Random(a.seed)
    out = {"measures": []}
    for name, d in MEASURES:
        for rule in ("main", "mirror"):
            paths = sorted(glob.glob(os.path.join(R11, d, "%s_w*.jsonl" % rule)))
            if not paths:
                continue
            _m, dups, per_file = AM.merge_ledgers(paths)
            xs = AM.ledger_order_scores(per_file)
            planned = len(xs)
            ci = A.pair_level_ci(xs)
            rec = one(per_file, rule, planned)
            perm = []
            for _ in range(a.perms):
                pf = list(per_file)
                rng.shuffle(pf)
                perm.append(one(pf, rule, planned))
            fr = [p["frac"] for p in perm]
            out["measures"].append({
                "name": name, "rule": rule, "pairs": planned, "void": sum(1 for _p, dd in per_file for s in dd if dd[s] is None),
                "shards": len(paths), "dup_seeds": len(dups), "rho": round(A.cs_rho(planned), 4),
                "full": {"wr": round(ci["win_rate"], 4), "ci95": [round(ci["lo"], 4), round(ci["hi"], 4)],
                         "conclusion": rec["full"]},
                "halfwidth_at_planned": round(A.cs_halfwidth(planned, A.cs_rho(planned)), 4),
                "ledger_order": rec,
                "perm": {"n": a.perms, "stopped_share": round(sum(p["stopped"] is not None for p in perm) / a.perms, 3),
                         "frac_mean": round(statistics.mean(fr), 4), "frac_median": round(statistics.median(fr), 4),
                         "frac_min": round(min(fr), 4), "frac_max": round(max(fr), 4),
                         "changed_share": round(sum(p["changed"] for p in perm) / a.perms, 3)}})
    ms = out["measures"]
    out["overall"] = {
        "measures": len(ms),
        "ledger_order": {"stopped": sum(m["ledger_order"]["stopped"] is not None for m in ms),
                         "changed": sum(m["ledger_order"]["changed"] for m in ms),
                         "frac_mean": round(statistics.mean(m["ledger_order"]["frac"] for m in ms), 4),
                         "pairs_used": sum(round(m["ledger_order"]["frac"] * m["pairs"]) for m in ms),
                         "pairs_total": sum(m["pairs"] for m in ms)},
        "perm": {"frac_mean": round(statistics.mean(m["perm"]["frac_mean"] for m in ms), 4),
                 "changed_share_mean": round(statistics.mean(m["perm"]["changed_share"] for m in ms), 4)}}
    json.dump(out, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(out["overall"], ensure_ascii=False))
    for m in ms:
        lo = m["ledger_order"]
        print("%-14s %-6s N=%4d wr=%.4f ci=%s full=%-13s stop=%s side=%s changed=%s | perm stop=%.2f frac=%.3f changed=%.2f" % (
            m["name"], m["rule"], m["pairs"], m["full"]["wr"], m["full"]["ci95"], m["full"]["conclusion"], lo["stopped"],
            lo["side"], lo["changed"], m["perm"]["stopped_share"], m["perm"]["frac_mean"], m["perm"]["changed_share"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
