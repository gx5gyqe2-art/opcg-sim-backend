"""度合いの下見: 同じ seed の 240 局を、散らし方（比例／平ら／一様）で対にして数える（gen_explore.summarize ＋ 低訪問の手）。"""
import sys, json, numpy as np
sys.path.insert(0, "tests/scripts"); sys.path.insert(0, "tests"); 
import _bootstrap  # noqa
import gen_explore as X

def low_visit(games, frac=0.10):
    """後半の最善でない手のうち、打った手の訪問の割合が frac 未満のもの（1 局あたり）。"""
    cnt = []; shares = []
    for s, g in games.items():
        ex = g.get("explore", np.zeros(len(g["z"]), np.int8)); c0 = 0
        for i in np.where(ex == 2)[0]:
            n, q = g["cand"][i]; c = int(g["pol_chosen"][i])
            if c >= 0 and len(n) and n.sum() > 0:
                sh = float(n[c] / n.sum()); shares.append(sh); c0 += int(sh < frac)
        cnt.append(c0)
    return float(np.mean(cnt)), (float(np.median(shares)) if shares else None), len(shares)

runs = json.loads(sys.argv[1]); lo, hi = 2851000, 2851239
out = {}
base = None
for name, pats in runs.items():
    g = X.load_parts(pats); g = {s: v for s, v in g.items() if lo <= s <= hi}
    if base is None: base = g
    r = X.summarize(g, base if g is not base else None)
    r["low_visit_per_game"], r["chosen_share_median"], r["late_moves"] = low_visit(g)
    out[name] = r
print(json.dumps(out, ensure_ascii=False, indent=1))
