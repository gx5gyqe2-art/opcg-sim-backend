import json, sys, collections
import numpy as np
sys.path.insert(0, 'tests/scripts')
import theory_map as TM
S = sys.argv[1]; W = S + '/data/sigma_wip/data/w39'
dump = TM.load_dump(S + '/pg/dump.jsonl')
stub = {"cap": 99.0, "rows_out": [], "turn_harm": [], "settled": []}
rows_all, align, *_ = TM.build_rows(W, dump, stub)
rows = [r for r in rows_all if r["pv"] is not None and r["bv"] is not None and r["nsc"] >= 2]
meta = json.load(open(W + '/meta_games.json'))
lcol = TM.leader_colors()
col_of = {(int(g['seed']), w): lcol.get(g['leaders'][w], []) for g in meta['games'] for w in (0, 1)}
seeds = sorted(dump)
per = collections.defaultdict(lambda: collections.defaultdict(float))
for r in rows:
    per[(r['seed'], r['w'])][TM.turn_band(r['t'])] += (float(r['bv']) - float(r['pv'])) * float(r['kap'])
# calibration per unit from the merged-engine clocks (first rows, pre-settle)
clocks = json.load(open(S + '/rb/C_after_freshjson/clocks.json'))
crow, *_ = TM.build_rows(W, dump, clocks)
crow = [r for r in crow if r['pv'] is not None and r['nsc'] >= 2 and r['ptype'] in TM.TYPES and r['btype'] in TM.TYPES and r.get('W') is not None]
first = [r for r in crow if r['first_in_turn'] and not r['settled_after']]
cal_u = collections.defaultdict(list)
for r in first:
    cal_u[(r['seed'], r['w'])].append(r)
cal_unit = {k: (float(np.mean([x['W'] for x in v])), v[0]['z']) for k, v in cal_u.items()}
B = TM.TURN_BANDS
def gap_means(ss):
    m = {}
    for c in TM.COLORS:
        us = [(s, w) for s in ss for w in (0, 1) if c in col_of.get((s, w), [])]
        if not us: continue
        for b in B + ('all',):
            m[(c, b)] = float(np.mean([sum(per.get(u, {}).values()) if b == 'all' else per.get(u, {}).get(b, 0.0) for u in us]))
    return m
def cal_means(ss):
    m = {}
    for c in TM.COLORS:
        us = [(s, w) for s in ss for w in (0, 1) if c in col_of.get((s, w), []) and (s, w) in cal_unit]
        if us:
            m[c] = float(np.mean([cal_unit[u][1] for u in us]) - np.mean([cal_unit[u][0] for u in us]))
    return m
def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])
g0 = gap_means(seeds); c0 = cal_means(seeds)
xs = [g0[(c, 'all')] for c in TM.COLORS]; ys = [c0[c] for c in TM.COLORS]
rho = spearman(xs, ys)
rng = np.random.default_rng(TM.SEED); bg = collections.defaultdict(list); br = []
for _ in range(TM.BOOT):
    ss = [seeds[k] for k in rng.integers(0, len(seeds), len(seeds))]
    g = gap_means(ss); c = cal_means(ss)
    for k, v in g.items(): bg[k].append(v)
    br.append(spearman([g[(cc, 'all')] for cc in TM.COLORS], [c[cc] for cc in TM.COLORS]))
ci = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
units = {c: sum(1 for s in seeds for w in (0, 1) if c in col_of.get((s, w), [])) for c in TM.COLORS}
out = {"record": W, "align": dict(align), "rows_all": len(rows_all), "rows": len(rows), "games": len(seeds),
       "gap_by_color_band": {c: {b: {"mean": round(g0[(c, b)], 4), "ci": ci(bg[(c, b)])} for b in B + ('all',)} for c in TM.COLORS},
       "units_by_color": units,
       "calibration_by_color": {c: round(c0[c], 4) for c in TM.COLORS},
       "calibration_overall": round(float(np.mean([z for _, z in cal_unit.values()]) - np.mean([w for w, _ in cal_unit.values()])), 4),
       "gap_overall_per_unit": round(float(np.mean([sum(per.get((s, w), {}).values()) for s in seeds for w in (0, 1)])), 4),
       "spearman_gap_vs_calibration": round(rho, 4), "spearman_boot_ci": ci(br),
       "rank_gap": dict(zip(TM.COLORS, [int(x) for x in np.argsort(np.argsort(xs))])),
       "rank_cal": dict(zip(TM.COLORS, [int(x) for x in np.argsort(np.argsort(ys))]))}
json.dump(out, open('logs/theory_map/syn_w39_playgap.json', 'w'), ensure_ascii=False, indent=1)
print(json.dumps({k: out[k] for k in ('rows', 'games', 'calibration_by_color', 'calibration_overall', 'spearman_gap_vs_calibration', 'spearman_boot_ci', 'gap_overall_per_unit')}, ensure_ascii=False))
print({c: out['gap_by_color_band'][c]['all']['mean'] for c in TM.COLORS})
