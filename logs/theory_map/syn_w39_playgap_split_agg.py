import json, sys, collections
import numpy as np
sys.path.insert(0, 'tests/scripts')
import theory_map as TM
S = sys.argv[1]; W = S + '/data/sigma_wip/data/w39'
dump = TM.load_dump(S + '/pg/dump.jsonl')
clocks = json.load(open(S + '/rb/C_after_freshjson/clocks.json'))
rows_all, align, *_ = TM.build_rows(W, dump, clocks)
rows = [r for r in rows_all if r["pv"] is not None and r["bv"] is not None and r["nsc"] >= 2]
meta = json.load(open(W + '/meta_games.json'))
lcol = TM.leader_colors()
col_of = {(int(g['seed']), w): lcol.get(g['leaders'][w], []) for g in meta['games'] for w in (0, 1)}
seeds = sorted(dump)
per = collections.defaultdict(lambda: collections.defaultdict(float))   # (s,w) -> (phase,band) -> v
nrows = collections.Counter()
for r in rows:
    ph = 'after' if r['settled_after'] else 'before'
    v = (float(r['bv']) - float(r['pv'])) * float(r['kap'])
    per[(r['seed'], r['w'])][(ph, TM.turn_band(r['t']))] += v
    nrows[ph] += 1
cal = json.load(open('logs/theory_map/syn_w39_playgap.json'))['calibration_by_color']
B = TM.TURN_BANDS; PH = ('before', 'after')
def means(ss):
    m = {}
    for c in TM.COLORS:
        us = [(s, w) for s in ss for w in (0, 1) if c in col_of.get((s, w), [])]
        for ph in PH + ('all',):
            for b in B + ('all',):
                def val(u):
                    d = per.get(u, {})
                    return sum(v for (p, bb), v in d.items() if (ph == 'all' or p == ph) and (b == 'all' or bb == b))
                m[(c, ph, b)] = float(np.mean([val(u) for u in us]))
    tot = lambda ph: sum(v for u in [(s, w) for s in ss for w in (0, 1)] for (p, _b), v in per.get(u, {}).items() if ph == 'all' or p == ph)
    m['share_after'] = tot('after') / tot('all')
    return m
def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y)); return float(np.corrcoef(rx, ry)[0, 1])
m0 = means(seeds)
ys = [cal[c] for c in TM.COLORS]
rho_b = spearman([m0[(c, 'before', 'all')] for c in TM.COLORS], ys)
rho_all = spearman([m0[(c, 'all', 'all')] for c in TM.COLORS], ys)
rng = np.random.default_rng(TM.SEED); bt = collections.defaultdict(list); brho = []; bshare = []
for _ in range(TM.BOOT):
    ss = [seeds[k] for k in rng.integers(0, len(seeds), len(seeds))]
    m = means(ss)
    for k, v in m.items():
        if k != 'share_after': bt[k].append(v)
    bshare.append(m['share_after'])
    brho.append(spearman([m[(c, 'before', 'all')] for c in TM.COLORS], ys))
ci = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
out = {"record": W, "rows": len(rows), "rows_by_phase": dict(nrows), "align": dict(align),
       "share_after_overall": round(m0['share_after'], 4), "share_after_ci": ci(bshare),
       "share_after_by_color": {c: round(m0[(c, 'after', 'all')] / m0[(c, 'all', 'all')], 4) for c in TM.COLORS},
       "gap": {c: {ph: {b: {"mean": round(m0[(c, ph, b)], 4), "ci": ci(bt[(c, ph, b)])} for b in B + ('all',)} for ph in PH + ('all',)} for c in TM.COLORS},
       "calibration_by_color": cal,
       "spearman_before_only": round(rho_b, 4), "spearman_before_only_ci": ci(brho),
       "spearman_all_recheck": round(rho_all, 4)}
json.dump(out, open('logs/theory_map/syn_w39_playgap_split.json', 'w'), ensure_ascii=False, indent=1)
print(json.dumps({k: out[k] for k in ('rows', 'rows_by_phase', 'share_after_overall', 'share_after_ci', 'share_after_by_color', 'spearman_before_only', 'spearman_before_only_ci', 'spearman_all_recheck')}, ensure_ascii=False))
for c in TM.COLORS: print(c, {ph: {b: out['gap'][c][ph][b]['mean'] for b in B + ('all',)} for ph in PH})
