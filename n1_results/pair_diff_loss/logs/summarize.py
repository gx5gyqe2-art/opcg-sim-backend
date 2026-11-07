# 試験台の出力（~/pdl/beds）を summary.json と表にまとめる（WP pair-diff-loss・1 回きり）
import json, os, re, glob, sys
B = os.path.expanduser("~/pdl/beds")
O = "n1_results/pair_diff_loss"
out = {"tvval": {}, "jitter": {}, "band74c": {}, "rocks": {}, "pairs": {}}
for f in glob.glob(f"{B}/tvval_*.json"):
    out["tvval"].update(json.load(open(f))["nets"])
for f in glob.glob(f"{B}/jitter_rows_*.json"):
    d = json.load(open(f)); out["jitter"]["z"] = d["outcome_z"]
    out["jitter"].update(d["nets"])
for f in glob.glob(f"{B}/band_n74c_*.json"):
    d = json.load(open(f))
    for n, v in d["nets"].items():
        out["band74c"][n] = {"v_mse": v["all"]["v_mse"], "v_sign": v["all"]["v_sign"],
                             "calib_gap_weighted": v["all"]["calib"]["gap_weighted"], "calib_gap_mean": v["all"]["calib"]["gap_mean"]}
for f in glob.glob(f"{B}/rocks_*.log"):
    for line in open(f):
        m = re.match(r"(\S+\.npz) 点 (\d+) 戻すを選ぶ (\d+) 根の評価の差（戻す−訪問最多）(\S+)（(\d+) 点）戻すの訪問の割合 (\S+)", line.strip())
        if m:
            out["rocks"][m.group(1)] = {"points": int(m.group(2)), "pick_return": int(m.group(3)), "root_gap": float(m.group(4)), "visit_share": float(m.group(6))}
for f in glob.glob(f"{B}/beds_pairs*.json"):
    d = json.load(open(f)); out["pairs"]["groups"] = d["groups"]
    out["pairs"].update(d["nets"])
json.dump(out, open(f"{O}/summary.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: list(v) for k, v in out.items()}, ensure_ascii=False))
