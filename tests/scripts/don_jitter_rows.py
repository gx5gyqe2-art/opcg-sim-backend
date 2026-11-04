"""WP `don-jitter`（`docs/search_efficiency_plan.md` §7.75・台帳 2+508）の数え 2 つ。読み取り専用・1 回きりの計測＝テストは足さない。

- `rows`: 波 82j のドンを動かした行（その手番の手番の席の最初の行）で、勝敗と各ネットの価値を符号（+1／−1）の群ごとに平均する。
  符号は局の seed から無作為に決まる（`record_gen.DonJitter`）ので、群の差＝ドンの効き（勝敗なら教材の打ち方での効き・ネットならネットが見る効き）。
  手番 3〜6 はドンデッキが必ず残る＝振り替えが起きず本当に半々。取り置き（seed % 7 == 0）と訓練に使った局にも分ける。→ `jitter_rows.json`
- `summary`: `n1_results/don_jitter/` の試験台の出力（ドンの比・訓練のログ・6キング・評価帯・外れの行の数え）を `summary.json` に束ねる。

    python tests/scripts/don_jitter_rows.py rows --nets opcg_sim/data/learned/nrel_r28x4.npz ~/nrel_djc.ep0.npz ...
    python tests/scripts/don_jitter_rows.py summary
"""
import argparse
import glob
import json
import math
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import _bootstrap  # noqa: E402,F401
import value_teach as VT  # noqa: E402

D = os.path.join("n1_results", "don_jitter")


def cmd_rows(a):
    nets = a.nets
    sc, ci, tok, sign, early, z, hold = [], [], [], [], [], [], []
    for f in sorted(glob.glob(os.path.expanduser(a.wave))):
        d = np.load(f, allow_pickle=True)
        dj, seed, turn, who = d["don_jitter"], d["seed"], d["turn"], d["who"]
        for s in np.unique(seed):
            idx = np.flatnonzero(seed == s)
            t, seat, sg = dj[idx[0]]
            if sg == 0:
                continue
            hit = idx[(turn[idx] == t) & (who[idx] == seat)]
            if not len(hit):
                continue
            i = int(hit[np.argmin(d["step"][hit])])
            sc.append(d["scalars"][i]); ci.append(d["card_idx"][i]); tok.append(d["tokens"][i])
            sign.append(int(sg)); early.append(int(t) <= 6); z.append(float(d["z"][i])); hold.append(int(s) % 7 == 0)
    sc, ci, tok = np.array(sc), np.array(ci)[:, :22], np.array(tok)
    sign, early, z, hold = np.array(sign), np.array(early), np.array(z), np.array(hold)
    res = {"rows": int(len(sign))}
    def diff(v, m):
        a, b = v[m & (sign == 1)], v[m & (sign == -1)]
        se = math.hypot(a.std() / math.sqrt(len(a)), b.std() / math.sqrt(len(b)))
        return {"plus": round(float(a.mean()), 4), "minus": round(float(b.mean()), 4), "diff": round(float(a.mean() - b.mean()), 4),
                "ci95": round(1.96 * se, 4), "n": [int(len(a)), int(len(b))]}
    groups = {"turn3_6": early, "turn7_10": ~early, "all": np.ones(len(sign), bool), "turn3_6_holdout": early & hold, "turn3_6_trained": early & ~hold}
    res["outcome_z"] = {g: diff(z, m) for g, m in groups.items()}
    res["nets"] = {}
    for n in nets:
        v = VT.net_values(n, sc, ci, tok)
        res["nets"][os.path.basename(n)] = {g: diff(v, m) for g, m in groups.items()}
        print(os.path.basename(n), {g: res["nets"][os.path.basename(n)][g]["diff"] for g in groups}, flush=True)
    print("z", {g: res["outcome_z"][g] for g in groups})
    json.dump(res, open(os.path.join(D, "jitter_rows.json"), "w"), ensure_ascii=False, indent=1)


def cmd_summary(a):
    out = {"tvval": {}, "train": {}, "rocks": {}, "band": None, "don_variation": {}}
    for f in sorted(glob.glob(f"{D}/tvval_*.json")):
        for k, v in json.load(open(f))["nets"].items():
            out["tvval"][k] = {t: v[t]["mean"] for t in v} | {t + "_ratio": v[t]["ratio"] for t in v}
    for name in [n for n in ("djc", "djt", "dja82", "dja72") if os.path.exists(f"{D}/train_{n}.log")]:
        for line in open(f"{D}/train_{name}.log"):
            if line.startswith("N_REL_TRAIN_EPOCH"):
                e = json.loads(line.split(" ", 1)[1])
                out["train"][f"nrel_{name}.ep{e['ep']}.npz"] = {k: round(e[k], 4) for k in ("train_mse", "val_vmse", "val_vsign", "val_pi_top1", "val_p_loss")}
            m = re.match(r"value (\d+)行（val (\d+)）", line)
            if m:
                out["train"][f"{name}_rows"] = {"value": int(m[1]), "val": int(m[2])}
    rf = f"{D}/rocks_dj.jsonl"
    if os.path.exists(rf):
        rows = [json.loads(l) for l in open(rf)]
        ok = [r for r in rows if r["ok"]]
        for net in sorted({n for r in ok for n in r["nets"]}):
            xs = [r["nets"][net] for r in ok if net in r["nets"]]
            d = [x["q_b"] - x["q_top"] for x in xs if x["q_b"] is not None and x["q_top"] is not None]
            out["rocks"][os.path.basename(net)] = {"points": len(xs), "chose_b": sum(x["chose_b"] for x in xs),
                                                  "q_diff": round(float(np.mean(d)), 4), "q_n": len(d),
                                                  "share_b": round(float(np.mean([x["share_b"] for x in xs])), 4)}
    bf = f"{D}/band_n73f.json"
    if os.path.exists(f"{D}/band_appendix_n70f.json"):
        out["band_appendix_n70f"] = {k: round(v["all"]["v_mse"], 4) for k, v in json.load(open(f"{D}/band_appendix_n70f.json"))["nets"].items()}
    if os.path.exists(f"{D}/jitter_rows_appendix.json"):
        out["jitter_rows_appendix"] = {k: {g: v[g]["diff"] for g in v} for k, v in json.load(open(f"{D}/jitter_rows_appendix.json"))["nets"].items()}
    out["band_common_n70f"] = {k: round(v["all"]["v_mse"], 4) for k, v in json.load(open(f"{D}/band_common_n70f.json"))["nets"].items()}
    if os.path.exists(bf):
        out["band"] = {k: {"v_mse": round(v["all"]["v_mse"], 4), "v_sign": round(v["all"]["v_sign"], 4)} for k, v in json.load(open(bf))["nets"].items()}; out["band_policy"] = {k: round(v["top1_visits"], 4) for k, v in json.load(open(bf))["policy"]["nets"].items()}
    for f in glob.glob(f"{D}/don_variation_*.json"):
        d = json.load(open(f)); c = d["dev_me_by_color"]; pur = [k for k in c if "PURPLE" in k]
        agg = lambda ks: round(sum(c[k]["nonzero"] * c[k]["n"] for k in ks) / sum(c[k]["n"] for k in ks), 4)
        out["don_variation"][os.path.basename(f)] = {"rows": d["dev_me"]["all"]["n"], "nonzero": d["dev_me"]["all"]["nonzero"],
            "minus1": d["dev_me"]["all"]["-1"], "plus1": d["dev_me"]["all"]["1"], "purple": agg(pur),
            "non_purple": agg([k for k in c if k not in pur]), "corr_total_vs_turn": d["corr_total_vs_turn"]}
    json.dump(out, open(f"{D}/summary.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: out[k] for k in ("tvval", "train", "rocks", "don_variation")}, ensure_ascii=False, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["rows", "summary"])
    ap.add_argument("--wave", default="~/n82j/n_records/n82j_w*_p*/n_record_*.npz")
    ap.add_argument("--nets", nargs="*", default=[])
    a = ap.parse_args(argv)
    {"rows": cmd_rows, "summary": cmd_summary}[a.cmd](a)


if __name__ == "__main__":
    main()
