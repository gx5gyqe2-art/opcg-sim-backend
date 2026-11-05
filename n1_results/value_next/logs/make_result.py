import json, os, glob
os.chdir("/home/user/opcg-sim-backend")
B = "n1_results/value_next/beds"
J = lambda p: json.load(open(p))
tv_c, tv_v = J(f"{B}/tvval_ctrl.json")["nets"], J(f"{B}/tvval_vn_e0.json")["nets"]
jr_c, jr_v = J(f"{B}/jitter_rows_ctrl.json")["nets"], J(f"{B}/jitter_rows_vn_e0.json")["nets"]
b74 = J(f"{B}/band_n74c_run1.json")["nets"]; b70 = J(f"{B}/band_n70f_run1.json")["nets"]; b74l = J(f"{B}/band_n74c_lim100k_e0.json")["nets"]
tr = J(f"{B}/teacher_rows.json")
def rocks(path, net):
    xs = [json.loads(l) for l in open(path)]; xs = [r["nets"][net] for r in xs if r["ok"] and net in r["nets"]]
    d = [x["q_b"] - x["q_top"] for x in xs if x["q_b"] is not None and x["q_top"] is not None]
    return [sum(x["chose_b"] for x in xs), round(sum(d) / len(d), 4), round(sum(x["share_b"] for x in xs) / len(xs), 3)]
def tv(d): return {k: [v["mean"], v["ratio"]] for k, v in d.items()}
def band(d): a = d["all"]; return [round(a["v_mse"], 4), round(a["v_sign"], 4), round(a["calib"]["gap_weighted"], 4), round(a["calib"]["gap_mean"], 4)]
names = {"r28x4": "nrel_r28x4.npz", "r29f4_ep0": "nrel_r29f4.ep0.npz", "r29f4_ep1": "nrel_r29f4.ep1.npz", "vn_run1_ep0": "nrel_vn_run1.ep0.npz"}
nets = {}
for k, f in names.items():
    src_tv = tv_v if k.startswith("vn") else tv_c; src_jr = jr_v if k.startswith("vn") else jr_c
    fj = "nrel_vn.ep0.npz" if k.startswith("vn") else f
    nets[k] = {"tvval": tv(src_tv[fj]), "jitter_t3_6_and_holdout": [src_jr[fj]["turn3_6"]["diff"], src_jr[fj]["turn3_6_holdout"]["diff"]],
               "jitter_t7_10": src_jr[fj]["turn7_10"]["diff"],
               "band74c_mse_sign_calibw_calibm": band(b74[f]), "band70f_mse_sign_calibw_calibm": band(b70[f]),
               "band74c_first100k_mse_sign_calibw_calibm": band(b74l[fj])}
nets["r29f4_ep0"]["rocks"] = rocks(f"{B}/rocks_run1.jsonl", os.path.expanduser("~/nets/nrel_r29f4.ep0.npz"))
nets["vn_run1_ep0"]["rocks"] = rocks(f"{B}/rocks_run1.jsonl", "n1_results/value_next/nrel_vn_run1.ep0.npz")
nets["vn_run1_ep0"]["file"] = "n1_results/value_next/nrel_vn_run1.ep0.npz"
res = {"job": "value-next", "section": "§7.81", "ledger": "cpu_theory_gap.md 2+521", "status": "done", "branch": "claude/value-next",
       "report": "docs/reports/2026-11-05_value_next.md", "results_dir": "n1_results/value_next",
       "verdict": "no effect: teacher = root value at the head of the next own turn (83.3% of value rows replaced, corr with outcome +0.730); epoch-1 DON ratio at window close 0.29 (control r29f4 ep0 0.34, r28x4 0.23), 82j group gap 0.296 (control 0.301); holdout error worse by +0.003 (within 0.005) but calibration gap 0.047 vs 0.026 (1.85x > 1.5x) = the net copies the teacher's optimism (+0.148). Diagnostic on the 82j jitter rows: the teacher itself carries the DON gap (0.487 vs outcome 0.542; own-row root value only 0.342; next-decision root value 0.365) while the net trained on it shows 0.296 = the teacher has the difference, the net does not learn it (fitting side). Epoch 2 and the secondary arm (next_v0_step) were not run (coordinator decision: epoch 2 lost twice to OOM and a worker restart; conclusion unchanged).",
       "teacher": {"mode": "next_v0", "rows": 10712760, "replaced": 8919161, "kept": 1793599, "replaced_frac": 0.8326, "corr_teacher_z": 0.730,
                   "teacher_mean": 0.148, "teacher_std": 0.766, "same_turn": 0, "later_turn": 8919161,
                   "by_kind": {"main": 3942896, "window": 4171328, "commit": 804937}},
       "train": {"from": "r28x4", "args": "train_r29f4.json common args + --z-mode next_v0", "epochs_completed": 1,
                 "ep0_val_vs_teacher": {"v_mse": 0.1071, "v_sign": 0.917},
                 "note": "run 1 OOM-killed in epoch 2 (74c band pack build run concurrently); run 2 killed at step 90000/92192 of epoch 1 by a session worker restart; run 3 stopped by the coordinator"},
       "nets": nets,
       "diagnostic_82j_teacher_rows": {"rows": tr["rows"], "rows_with_teacher": tr["rows_with_teacher"],
                                       "turn3_6": {k: v["turn3_6"] for k, v in tr["cols"].items()},
                                       "turn3_6_holdout": {k: v["turn3_6_holdout"] for k, v in tr["cols"].items()},
                                       "turn7_10": {k: v["turn7_10"] for k, v in tr["cols"].items()}, "corr": tr["corr"]},
       "predictions": {"effective_ratio_ge_0.45": False, "effective_gap_ge_0.40": False, "all_4_times_above_control": False,
                       "drift_holdout_mse_within_0.005": True, "drift_calibration_within_1.5x": False, "no_effect_gain_lt_0.05": True},
       "invariants": {"default_bit_identical": "tests/test_value_next.py", "controls_match_prior_reports": True, "corr_teacher_z_ge_0.5": True},
       "not_run": {"epoch2": "lost twice (OOM, worker restart); stopped by coordinator", "secondary_arm_next_v0_step": "not run (coordinator); tool and column kept"},
       "gates": {"make_test": "green with OPENBLAS_CORETYPE=Haswell", "audit_cross": "not needed (engine untouched)"}}
json.dump(res, open("n1_results/value_next/RESULT.json", "w"), ensure_ascii=False, indent=1)
json.dump(res, open("RESULT.json", "w"), ensure_ascii=False, indent=1)
print("RESULT.json written")
