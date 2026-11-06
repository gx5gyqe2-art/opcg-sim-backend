"""merge_rule_scan.jsonl を集計する（読み取りだけ）。"""
import json, math
R = [json.loads(l) for l in open("n1_results/reaudit_r29f4/merge_rule_scan.jsonl")]
R = [r for r in R if not r.get("skip")]
def ms(x):
    n = len(x)
    if n < 2: return (round(sum(x) / n, 3) if n else None, None, n)
    m = sum(x) / n; sd = math.sqrt(sum((v - m) ** 2 for v in x) / (n - 1)); h = 1.96 * sd / math.sqrt(n)
    return (round(m, 4), [round(m - h, 4), round(m + h, 4)], n)
rep = sum(r["same_as_record"] for r in R)
dif = [r for r in R if r["alt_differs"]]
print("points", len(R), "reproduces_record", rep, "alt_differs", len(dif), round(len(dif) / len(R), 3))
print("alt differs & alt==b (judged): diff b-a", ms([r["d"] for r in dif if r["alt_is_b"] and r["same_as_record"]]))
print("alt same as chosen: diff b-a (runner-up)", ms([r["d"] for r in R if not r["alt_differs"] and r["same_as_record"]]))
print("alt differs but alt != b (not judged)", sum(1 for r in dif if not r["alt_is_b"]))
print("cand among alt==b", sum(1 for r in dif if r["alt_is_b"] and r["cand"]))
for r in dif:
    if r["alt_is_b"] and r["d"] is not None and abs(r["d"]) >= 0.15:
        print("  ", r["seed"], r["step"], r["dtype"], "d", r["d"], "chosen", r["chosen_desc"][:60], round(r["chosen_mean_q"], 3), "alt", r["alt_desc"][:60], round(r["alt_mean_q"], 3))
print("== 平均の値の差（選び直す手 − 打った手）で絞る")
for th in (0.0, 0.03, 0.05, 0.1, 0.2):
    sel = [r for r in dif if r["alt_is_b"] and (r["alt_mean_q"] - r["chosen_mean_q"]) >= th]
    allsel = [r for r in dif if (r["alt_mean_q"] - r["chosen_mean_q"]) >= th]
    print(f"  gap>={th}: alt differs {len(allsel)}/{len(R)} ({len(allsel)/len(R):.3f}) | judged(alt==b) {ms([r['d'] for r in sel])}")
