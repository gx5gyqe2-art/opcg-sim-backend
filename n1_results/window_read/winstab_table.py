"""窓の揺れ（blunder-audit の窓の 226 点・探索の乱数だけ替えて 8 回）を旗・K ごとに並べる（WP rs-window-read・1 回きり）。"""
import json
import math
import os
import sys

B = "n1_results/window_read"
SRC = {"orig": "n1_results/blunder_audit/window_stability.jsonl"}
for f in sorted(os.listdir(B)):
    if f.startswith("window_stability_") and f.endswith(".jsonl"):
        SRC[f[len("window_stability_"):-6]] = os.path.join(B, f)
LOSS = [(3301005, 107), (3301025, 117), (3301004, 31)]


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 3), round(c + h, 3)]


def main():
    out = {}
    for tag, path in SRC.items():
        rows = [json.loads(x) for x in open(path)]
        var = [r for r in rows if len(r["picks"]) > 1]
        by = {}
        for r in rows:
            d = by.setdefault(r["dtype"], [0, 0])
            d[0] += 1
            d[1] += len(r["picks"]) > 1
        loss = {}
        for r in rows:
            if (r["seed"], r["step"]) in LOSS:
                loss[f"{r['seed']}_{r['step']}"] = {"a_share": r["a_share"], "b_share": r["b_share"], "picks": r["picks"]}
        # 揺れの「多数決からの外れ」の率（8 回のうち最多の手以外を選んだ割合）
        minority = sum(8 - max(r["picks"].values()) for r in rows) / (8 * len(rows))
        out[tag] = {"points": len(rows), "varying": len(var), "rate": round(len(var) / len(rows), 4),
                    "ci": wilson(len(var), len(rows)), "minority_share": round(minority, 4),
                    "by_dtype": {k: {"n": v[0], "varying": v[1]} for k, v in sorted(by.items())},
                    "loss_points": loss}
    json.dump(out, open(os.path.join(B, "winstab_table.json"), "w"), ensure_ascii=False, indent=1)
    for tag, o in out.items():
        print(f"{tag:10s} {o['varying']:3d}/{o['points']} {o['rate']:.3f} {o['ci']} minority={o['minority_share']}"
              f" loss={ {k: v['b_share'] for k, v in o['loss_points'].items()} }")


if __name__ == "__main__":
    sys.exit(main())
