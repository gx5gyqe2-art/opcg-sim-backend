"""n1_results/theory_input/ の出力を summary.json に束ねる（WP theory-input）。"""
import glob, json, os
D = os.path.dirname(os.path.abspath(__file__)); B = os.path.join(D, "beds")
out = {"bench": {k: v for k, v in json.load(open(os.path.join(D, "bench.json"))).items() if k != "cols"},
       "bench_cols": json.load(open(os.path.join(D, "bench.json")))["cols"], "tvval": {}, "jitter": {}, "band": {}, "calib": {}, "weights": {}, "train": {}}
for k, v in json.load(open(os.path.join(B, "tvval_final.json")))["nets"].items():
    out["tvval"][k] = {t: v[t]["mean"] for t in v} | {t + "_ratio": v[t]["ratio"] for t in v}
jr = json.load(open(os.path.join(B, "jitter_rows_final.json")))
out["jitter"] = {"outcome_z": {g: jr["outcome_z"][g]["diff"] for g in jr["outcome_z"]},
                 "nets": {k: {g: v[g]["diff"] for g in v} for k, v in jr["nets"].items()}}
for w in ("n74c", "n70f"):
    p = os.path.join(B, f"band_{w}.json")
    if os.path.exists(p):
        out["band"][w] = {k: {"v_mse": round(v["all"]["v_mse"], 4), "v_sign": round(v["all"]["v_sign"], 4)} for k, v in json.load(open(p))["nets"].items()}
    p = os.path.join(B, f"calib_{w}.json")
    if os.path.exists(p):
        out["calib"][w] = {k: {"gap_mean_abs": v["gap_mean_abs"], "gap_weighted": v["gap_weighted"], "bands": v["bands"]} for k, v in json.load(open(p))["nets"].items()}
jc = os.path.join(B, "jitter_cols.json")
if os.path.exists(jc):
    j = json.load(open(jc))
    out["jitter_cols"] = {c: {"turn3_6_diff": v["turn3_6"]["diff"], "ci95": v["turn3_6"]["ci95"], "diff_over_std": v["turn3_6_diff_over_std"]} for c, v in j["cols"].items()}
for k, v in json.load(open(os.path.join(B, "weights_final.json"))).items():
    out["weights"][k] = {kk: vv for kk, vv in v.items() if kk != "rank"} if v.get("theory_in") else v
for f in sorted(glob.glob(os.path.join(D, "logs", "train_thin*.log"))):
    for line in open(f):
        if line.startswith("N_REL_TRAIN_EPOCH"):
            e = json.loads(line.split(" ", 1)[1])
            out["train"][f"{os.path.basename(f)[:-4]}.ep{e['ep']}"] = {k: round(e[k], 4) for k in ("train_mse", "val_vmse", "val_vsign", "val_pi_top1", "val_aux", "train_sec")}
json.dump(out, open(os.path.join(D, "summary.json"), "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: out[k] for k in ("band", "train")}, ensure_ascii=False, indent=1))
