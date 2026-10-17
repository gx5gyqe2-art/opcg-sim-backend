import json, sys, os, collections
import numpy as np
sys.path.insert(0, 'tests/scripts')
import theory_map as TM
S = sys.argv[1]; W = S + '/data/sigma_wip/data/w39'
import os; d0 = TM.load_dump(os.environ['D0']); d1 = TM.load_dump(os.environ['D1'])
plan = json.load(open(os.environ['PLAN']))
stub = {"cap": 99.0, "rows_out": [], "turn_harm": [], "settled": []}
rows_all, _al, *_ = TM.build_rows(W, d0, stub)
cnt = collections.Counter(); bad = collections.Counter()
step_of = {}
for seed in d0:
    keyed = collections.defaultdict(list)
    for k, x in enumerate(d0[seed]["rows"]):
        keyed[(int(x[0]), int(x[1]), json.dumps(x[8]))].append(k)
    for r in [r for r in rows_all if r['seed'] == seed]:
        ks = keyed.get((r["t"], r["w"], json.dumps(r["cands"])))
        if ks: step_of[(seed, ks.pop(0))] = r['step']
for seed in d0:
    a, b = d0[seed]['rows'], d1.get(seed, {}).get('rows')
    if b is None or len(a) != len(b): cnt['seed_mismatch'] += 1; continue
    for k, (x, y) in enumerate(zip(a, b)):
        cnt['rows'] += 1
        if x[:2] != y[:2] or [c[0] for c in x[8]] != [c[0] for c in y[8]]: bad['row_shape'] += 1; continue
        px, py = (x[9] if len(x) > 9 else None), (y[9] if len(y) > 9 else None)
        if px and py:
            if px[0][2] != py[0][2]: bad['W0'] += 1
            if px[0][3] != py[0][3]: bad['kappa0'] += 1
        else:
            cnt['no_parts'] += 1
        st = step_of.get((seed, k)); pl = plan.get(str(seed), {}).get(str(st), {}) if st is not None else {}
        for gi, ((fam, vx), (_f, vy)) in enumerate(zip(x[8], y[8])):
            rebuilt = str(gi) in pl
            kind = 'attack' if fam == 'attack' else 'pass' if fam == 'end' else ('rebuilt' if rebuilt else 'nonattack_kept') if fam in ('play', 'attach', 'effect') else 'other'
            cnt['cand_' + kind] += 1
            if vx != vy:
                bad['cand_' + kind] += 1
out = {"counts": dict(cnt), "differing": dict(bad), "rows_with_step": len(step_of)}
json.dump(out, open(os.environ['OUT'], 'w'), indent=1)
print(json.dumps(out))
