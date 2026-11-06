# 対にした差: 同じ 82j の行で (dw − djt) の群の差とその誤差（局をまたいだ揺れを消す）
import glob, os, sys, json, math, numpy as np
sys.path.insert(0, "tests/scripts"); sys.path.insert(0, "tests")
import _bootstrap, value_teach as VT
sc, ci, tok, sign, early, hold = [], [], [], [], [], []
for f in sorted(glob.glob(os.path.expanduser("~/n82j/n_records/n82j_w*_p*/n_record_*.npz"))):
    d = np.load(f, allow_pickle=True)
    dj, seed, turn, who = d["don_jitter"], d["seed"], d["turn"], d["who"]
    for s in np.unique(seed):
        idx = np.flatnonzero(seed == s); t, seat, sg = dj[idx[0]]
        if sg == 0: continue
        hit = idx[(turn[idx] == t) & (who[idx] == seat)]
        if not len(hit): continue
        i = int(hit[np.argmin(d["step"][hit])])
        sc.append(d["scalars"][i]); ci.append(d["card_idx"][i][:22]); tok.append(d["tokens"][i])
        sign.append(int(sg)); early.append(int(t) <= 6); hold.append(int(s) % 7 == 0)
sc, ci, tok = np.array(sc), np.array(ci), np.array(tok)
sign, early, hold = np.array(sign), np.array(early), np.array(hold)
H = os.path.expanduser("~/dw")
nets = {"djt0": f"{H}/ctrl/nrel_djt.ep0.npz", "djt1": f"{H}/ctrl/nrel_djt.ep1.npz", "dw0": f"{H}/nets/nrel_dw.ep0.npz", "dw1": f"{H}/nets/nrel_dw.ep1.npz"}
v = {k: VT.net_values(p, sc, ci, tok) for k, p in nets.items()}
def gap(x, m):
    a, b = x[m & (sign == 1)], x[m & (sign == -1)]
    return float(a.mean() - b.mean()), 1.96 * math.hypot(a.std() / math.sqrt(len(a)), b.std() / math.sqrt(len(b)))
out = {}
for g, m in {"turn3_6_holdout": early & hold, "turn3_6_trained": early & ~hold, "turn7_10": ~early}.items():
    for e in "01":
        dd = v["dw" + e] - v["djt" + e]
        out[f"{g}_ep{e}"] = [round(x, 4) for x in gap(dd, m)]
print(json.dumps(out)); json.dump(out, open("n1_results/don_weight/paired_gap.json", "w"), indent=1)
