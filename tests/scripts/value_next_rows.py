"""WP `value-next`（`docs/search_efficiency_plan.md` §7.81）の診断: 波 82j のドンを動かした行で、**教師そのもの**の群の差を出す。

`don_jitter_rows.py rows` と同じ行（ドンを動かした手番の、手番の席の最初の行）を ±1 の群に分け、次の 3 つの平均の差を並べる:
- 勝敗 `z`（その行の手番の視点）
- その行自身の根の値 `pol_v0`（生成役の探索がドンを動かした盤面をその場で読んだ値）
- `--z-mode next_v0` の教師（同じ局・同じ席の次の手番の頭の根の値・`n_rel_train.load_next_v0` と同じ器）。教師の無い行（最後の手番）は除く。
読み取り専用・1 回きりの計測＝テストは足さない。

    python tests/scripts/value_next_rows.py --out n1_results/value_next/beds/teacher_rows.json
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import _bootstrap  # noqa: E402,F401

from opcg_sim.learned.train import n_rel_train as NRT  # noqa: E402


def diff(v, m, sign):
    a, b = v[m & (sign == 1)], v[m & (sign == -1)]
    if not len(a) or not len(b):
        return None
    se = math.hypot(a.std() / math.sqrt(len(a)), b.std() / math.sqrt(len(b)))
    return {"plus": round(float(a.mean()), 4), "minus": round(float(b.mean()), 4), "diff": round(float(a.mean() - b.mean()), 4),
            "ci95": round(1.96 * se, 4), "n": [int(len(a)), int(len(b))]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave", default="~/n82j/n_records/n82j_w*_p*/n_record_*.npz")
    ap.add_argument("--out", default="n1_results/value_next/beds/teacher_rows.json")
    a = ap.parse_args(argv)
    rows = {k: [] for k in ("sign", "early", "hold", "z", "v0", "teach", "has_teach", "v0_next_step")}
    for f in sorted(glob.glob(os.path.expanduser(a.wave))):
        d = np.load(f, allow_pickle=True)
        n = int(d["z"].shape[0])
        dj, seed, turn, who, step = d["don_jitter"], d["seed"][:n], d["turn"][:n], d["who"][:n], d["step"][:n]
        z = np.asarray(d["z"][:n], np.float32)
        v0 = np.asarray(d["pol_v0"][:n], np.float32)
        mask = NRT._root_mask(d, n)
        t_turn, m_turn, _ = NRT.next_v0_teacher(z, seed, who, step, v0, mask, turn=turn, by="turn")
        t_step, m_step, _ = NRT.next_v0_teacher(z, seed, who, step, v0, mask, turn=turn, by="step")
        for s in np.unique(seed):
            idx = np.flatnonzero(seed == s)
            t, seat, sg = dj[idx[0]]
            if sg == 0:
                continue
            hit = idx[(turn[idx] == t) & (who[idx] == seat)]
            if not len(hit):
                continue
            i = int(hit[np.argmin(step[hit])])
            rows["sign"].append(int(sg)); rows["early"].append(int(t) <= 6); rows["hold"].append(int(s) % 7 == 0)
            rows["z"].append(float(z[i])); rows["v0"].append(float(v0[i]) if mask[i] else float("nan"))
            rows["teach"].append(float(t_turn[i])); rows["has_teach"].append(bool(m_turn[i]))
            rows["v0_next_step"].append(float(t_step[i]) if m_step[i] else float("nan"))
    R = {k: np.array(v) for k, v in rows.items()}
    sign, early, hold = R["sign"], R["early"], R["hold"]
    groups = {"turn3_6": early, "turn7_10": ~early, "all": np.ones(len(sign), bool),
              "turn3_6_holdout": early & hold, "turn3_6_trained": early & ~hold}
    res = {"rows": int(len(sign)), "rows_with_teacher": int(R["has_teach"].sum()),
           "rows_with_own_root": int(np.isfinite(R["v0"]).sum()), "cols": {}}
    cols = {"outcome_z": (R["z"], np.ones(len(sign), bool)),
            "own_root_v0": (R["v0"], np.isfinite(R["v0"])),
            "teacher_next_turn_head": (R["teach"], R["has_teach"]),
            "next_decision_v0": (R["v0_next_step"], np.isfinite(R["v0_next_step"])),
            "outcome_z_rows_with_teacher": (R["z"], R["has_teach"])}
    for name, (v, ok) in cols.items():
        v = np.where(ok, v, 0.0)
        res["cols"][name] = {g: diff(v, m & ok, sign) for g, m in groups.items()}
        print(name, {g: (res["cols"][name][g] or {}).get("diff") for g in groups}, flush=True)
    # 教師と勝敗・自分の根の値との相関（同じ行）
    ok = R["has_teach"] & np.isfinite(R["v0"])
    res["corr"] = {"teacher_vs_z": round(float(np.corrcoef(R["teach"][R["has_teach"]], R["z"][R["has_teach"]])[0, 1]), 4),
                   "own_root_vs_z": round(float(np.corrcoef(R["v0"][np.isfinite(R["v0"])], R["z"][np.isfinite(R["v0"])])[0, 1]), 4),
                   "teacher_vs_own_root": round(float(np.corrcoef(R["teach"][ok], R["v0"][ok])[0, 1]), 4),
                   "teacher_mean": round(float(R["teach"][R["has_teach"]].mean()), 4),
                   "own_root_mean": round(float(R["v0"][np.isfinite(R["v0"])].mean()), 4),
                   "z_mean": round(float(R["z"].mean()), 4)}
    print("corr", res["corr"], flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"→ {a.out}")


if __name__ == "__main__":
    main()
