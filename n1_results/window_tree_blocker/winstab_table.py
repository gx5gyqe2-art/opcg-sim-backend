"""窓の揺れ（blunder-audit の窓の 226 点・探索の乱数だけ替えて 8 回）を修正前後で並べる（WP rs-window-tree-blocker・1 回きり）。

old_default = rs-window-read の既定（K=16・ライフ ≤ 1 のブロッカー／カウンターの窓は木・`n1_results/window_read/window_stability_fixed.jsonl`）
counter_legacy = この WP の wheel ＋ window_tree_counter_legacy（old_default と 1 点ずつ同じになるはず）
blocker = この WP の既定（木はブロッカーの窓だけ）
"""
import json
import math

SRC = {"old_default": "n1_results/window_read/window_stability_fixed.jsonl",
       "counter_legacy": "n1_results/window_tree_blocker/window_stability_counter_legacy.jsonl",
       "blocker": "n1_results/window_tree_blocker/window_stability_blocker.jsonl"}
LOSS = [(3301005, 107), (3301025, 117), (3301004, 31)]


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 3), round(c + h, 3)]


def group(r):
    if r["my_life"] <= 1 and r["dtype"] in ("counter", "blocker"):
        return "low_" + r["dtype"]
    return "rest"


def main():
    rows = {t: {(r["seed"], r["step"]): r for r in map(json.loads, open(p))} for t, p in SRC.items()}
    out = {}
    for t, rs in rows.items():
        var = [r for r in rs.values() if len(r["picks"]) > 1]
        by = {}
        for r in rs.values():
            g = by.setdefault(group(r), [0, 0])
            g[0] += 1
            g[1] += len(r["picks"]) > 1
        minority = sum(8 - max(r["picks"].values()) for r in rs.values()) / (8 * len(rs))
        out[t] = {"points": len(rs), "varying": len(var), "rate": round(len(var) / len(rs), 4),
                  "ci": wilson(len(var), len(rs)), "minority_share": round(minority, 4),
                  "by_group": {k: {"n": v[0], "varying": v[1]} for k, v in sorted(by.items())},
                  "loss_points": {f"{s}_{st}": {"b_share": rs[(s, st)]["b_share"], "picks": rs[(s, st)]["picks"]}
                                  for s, st in LOSS if (s, st) in rs}}
    od, cl, bl = rows["old_default"], rows["counter_legacy"], rows["blocker"]
    out["counter_legacy_equals_old_default"] = sum(od[k]["picks"] == cl[k]["picks"] for k in od)
    # 修正前後で多数の手が変わった点（ライフ ≤ 1 のカウンターの窓のはず）
    maj = lambda r: max(r["picks"], key=r["picks"].get)
    changed = [{"seed": k[0], "step": k[1], "dtype": od[k]["dtype"], "my_life": od[k]["my_life"],
                "old": od[k]["picks"], "new": bl[k]["picks"], "a": od[k]["a"], "b": od[k]["b"]}
               for k in od if maj(od[k]) != maj(bl[k]) or od[k]["picks"] != bl[k]["picks"]]
    out["changed_points"] = changed
    out["changed_by_group"] = {}
    for c in changed:
        g = group(c)
        out["changed_by_group"][g] = out["changed_by_group"].get(g, 0) + 1
    json.dump(out, open("n1_results/window_tree_blocker/winstab_table.json", "w"), ensure_ascii=False, indent=1)
    for t in SRC:
        o = out[t]
        print(t, o["varying"], o["rate"], o["ci"], o["minority_share"], o["by_group"],
              {k: v["b_share"] for k, v in o["loss_points"].items()})
    print("counter_legacy == old_default:", out["counter_legacy_equals_old_default"], "/", len(od))
    print("changed:", out["changed_by_group"])


if __name__ == "__main__":
    main()
