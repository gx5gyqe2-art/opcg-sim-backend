import json, sys, collections, numpy as np
sys.path.insert(0, 'tests/scripts')
import theory_map as TM
S = sys.argv[1]
def strip(o):
    if isinstance(o, list): return [strip(x) for x in o if not (isinstance(x, dict) and "tempo" in x)]
    if isinstance(o, dict): return {k: strip(v) for k, v in o.items()}
    return o
# invariant: compare line by line after stripping tempo
diff = same = 0
with open(S + '/ts/dump_t.jsonl') as ft, open(S + '/eo/dump_a1n.jsonl') as f1:
    a = {json.loads(l)['seed']: l for l in f1}
    for l in ft:
        d = json.loads(l); s = d['seed']
        if json.dumps(strip(d), sort_keys=True) == json.dumps(json.loads(a[s]), sort_keys=True): same += 1
        else: diff += 1
dump = TM.load_dump(S + '/ts/dump_t.jsonl')
R = collections.defaultdict(list)
for seed, dj in dump.items():
    for dr in dj['rows']:
        b = dr[9][0]; tv = next(e['tempo'] for e in b[5:] if isinstance(e, dict) and 'tempo' in e)
        band = TM.turn_band(int(dr[0]))
        for k in (band, 'all'):
            R[k].append((b[2], 1 - tv[0], 1 - tv[3] if tv[3] is not None else np.nan, b[0], b[1], tv[1], tv[2], tv[4], tv[5]))
out = {"invariant": {"games_identical_after_strip": same, "games_differ": diff}, "side": {}}
for k, v in R.items():
    A = np.array(v, dtype=float)
    out["side"][k] = {"n": len(v), "W_self_mean": round(np.nanmean(A[:, 0]), 4), "one_minus_Wopp_norw_mean": round(np.nanmean(A[:, 1]), 4),
        "one_minus_Wopp_rw_mean": round(np.nanmean(A[:, 2]), 4),
        "norw_opp_tau_me_minus_my_tau_opp_mean": round(np.nanmean(A[:, 5] - A[:, 4]), 3), "norw_opp_tau_opp_minus_my_tau_me_mean": round(np.nanmean(A[:, 6] - A[:, 3]), 3),
        "rw_opp_tau_me_minus_my_tau_opp_mean": round(np.nanmean(A[:, 7] - A[:, 4]), 3), "rw_opp_tau_opp_minus_my_tau_me_mean": round(np.nanmean(A[:, 8] - A[:, 3]), 3),
        "norw_median_dtau_me": round(np.nanmedian(A[:, 5] - A[:, 4]), 3), "norw_median_dtau_opp": round(np.nanmedian(A[:, 6] - A[:, 3]), 3)}
json.dump(out, open(S + '/ts/extra.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
