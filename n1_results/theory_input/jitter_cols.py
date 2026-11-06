"""82j のドンを動かした行（don_jitter_rows.cmd_rows と同じ選び方）で、足した 25 列そのものの ±1 の群の差。"""
import glob, json, math, os, sys
import numpy as np
sys.path.insert(0, "tests/scripts"); sys.path.insert(0, "tests"); sys.path.insert(0, ".")
import _bootstrap  # noqa
import theory_input as TI, theory_rs as TR
from opcg_sim.learned import n_rel as NL
wave = os.path.expanduser("~/n82j/n_records/n82j_w*_p*/n_record_*.npz")
sc, ci, tok, sign, early, z, hold, rseed, rturn, rwho = [], [], [], [], [], [], [], [], [], []
files = sorted(f for f in glob.glob(wave) if not f.endswith(".theory.npz"))
for f in files:
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
        rseed.append(int(s)); rturn.append(int(t)); rwho.append(int(seat))
sc, ci, tok, sign, early, z = np.array(sc), np.array(ci), np.array(tok), np.array(sign), np.array(early), np.array(z)
TR.ready()
mode = TI.decks_mode_of(os.path.dirname(files[0]))
th = TI.cols_for_rows(sc, tok, ci, rwho, rturn, rseed, mode)
def diff(v, m):
    a, b = v[m & (sign == 1)], v[m & (sign == -1)]
    se = math.hypot(a.std() / math.sqrt(len(a)), b.std() / math.sqrt(len(b)))
    return {"plus": round(float(a.mean()), 4), "minus": round(float(b.mean()), 4), "diff": round(float(a.mean() - b.mean()), 4),
            "ci95": round(1.96 * se, 4), "std": round(float(v[m].std()), 4), "n": [int(len(a)), int(len(b))]}
groups = {"turn3_6": early, "turn7_10": ~early, "all": np.ones(len(sign), bool)}
out = {"rows": int(len(sign)), "outcome_z": {g: diff(z, m) for g, m in groups.items()}, "cols": {}}
for j, c in enumerate(NL.TH_COLS):
    out["cols"][c] = {g: diff(th[:, j], m) for g, m in groups.items()}
    # 手番 3〜6 の差を列の散らばりで割った値（効きの大きさ・単位なし）
    out["cols"][c]["turn3_6_diff_over_std"] = round(out["cols"][c]["turn3_6"]["diff"] / max(1e-9, out["cols"][c]["turn3_6"]["std"]), 3)
# アクティブのドンの列（sc[2]）も同じ物差しで
out["sc_don_active"] = {g: diff(sc[:, 2].astype(np.float64), m) for g, m in groups.items()}
json.dump(out, open("n1_results/theory_input/beds/jitter_cols.json", "w"), ensure_ascii=False, indent=1)
for c in sorted(NL.TH_COLS, key=lambda c: -abs(out["cols"][c]["turn3_6"]["diff"])):
    r = out["cols"][c]["turn3_6"]
    print(f"{c:12s} diff {r['diff']:+.4f} ±{r['ci95']:.4f} std {r['std']:.4f} d/std {out['cols'][c]['turn3_6_diff_over_std']:+.3f}")
print("don_active", out["sc_don_active"]["turn3_6"])
