import json, sys, collections
import numpy as np
sys.path.insert(0, 'tests/scripts')
import theory_map as TM
from theory_rs import MU
from opcg_sim.learned.train import plan_labels as PL
S = sys.argv[1]; W = S + '/data/sigma_wip/data/w39'
dump = TM.load_dump(S + '/pg/dump.jsonl')
cards = {c['number']: c for c in json.load(open('opcg_sim/data/opcg_cards.json', encoding='utf-8'))}
kind = lambda cid: (cards.get(cid) or {}).get('種類')
def cls(sj, u2c):
    at = sj[0]
    if at == 'ATTACK' or (at == 'DON_BOX' and sj[2]):
        tc = u2c.get(sj[2][0]) if sj[2] else None
        return 'atk_leader' if kind(tc) == 'リーダー' else 'atk_char'
    if at == 'PLAY':
        k = kind(u2c.get(sj[1]))
        return {'キャラクター': 'play_char', 'キャラ': 'play_char', 'ステージ': 'play_stage', 'イベント': 'play_event'}.get(k, 'other')
    if at in ('DON_BOX', 'ATTACH_DON'):
        tgt = sj[3][0] if (at == 'ATTACH_DON' and sj[3]) else sj[1]
        return 'don_leader' if kind(u2c.get(tgt)) == 'リーダー' else 'don_char'
    return {'ACTIVATE_MAIN': 'effect', 'TURN_END': 'pass'}.get(at, 'other')
cols = ("sig", "who", "turn", "seed", "z", "kind", "step", "pol_len", "pol_chosen")
rows = []; align = collections.Counter()
for rws, pol, _ex, L, ptr, idx in PL.iter_games([W], row_cols=cols, pol_cols=("pol_sig", "pol_cid", "pol_tcid")):
    seed = int(rws["seed"][idx[0]]); dj = dump.get(seed)
    if dj is None: align['game_missing'] += 1; continue
    u2c = PL.uuid_map(pol, L, ptr, idx)
    cand = []
    for i in idx:
        w = int(rws["who"][i]); t = int(rws["turn"][i])
        if t < 1 or not PL.is_own_turn(w, t) or int(rws["kind"][i]) != 0: continue
        k = int(L[i]); ch = int(rws["pol_chosen"][i])
        if k < 1: continue
        if json.loads(pol["pol_sig"][ptr[i]])[0] == "RESOLVE_EFFECT_SELECTION": continue
        if k < 2 or ch < 0 or ch >= k: continue
        cand.append((i, w, t, [json.loads(pol["pol_sig"][j]) for j in range(ptr[i], ptr[i] + k)], ch))
    drows = dj["rows"]
    if len(drows) != len(cand): align['game_len_mismatch'] += 1; continue
    for (i, w, t, sigs, ch), dr in zip(cand, drows):
        if int(dr[0]) != t or int(dr[1]) != w or [TM.family_of(s) for s in sigs] != [c[0] for c in dr[8]]:
            align['row_mismatch'] += 1; continue
        align['rows'] += 1
        pv, bv, nsc = dr[2], dr[3], dr[7]
        if pv is None or bv is None or nsc < 2: continue
        vals = [c[1] for c in dr[8]]
        bi = max((j for j in range(len(vals)) if vals[j] is not None), key=lambda j: vals[j])
        rows.append({'seed': seed, 'band': TM.turn_band(t), 'best': cls(sigs[bi], u2c), 'played': cls(sigs[ch], u2c),
                     'd': (float(bv) - float(pv)) / MU, 'bv_check': abs(float(vals[bi]) - float(bv)) < 1e-9})
isatk = lambda c: c.startswith('atk_')
def table(rs):
    g = collections.defaultdict(list)
    for r in rs: g[(r['best'], r['played'])].append(r['d'])
    return {"%s|%s" % k: {"n": len(v), "median": round(float(np.median(v)), 4), "sum": round(float(np.sum(v)), 2)} for k, v in sorted(g.items(), key=lambda kv: -sum(kv[1]))}
def share(rs):
    tot = sum(r['d'] for r in rs); a = sum(r['d'] for r in rs if isatk(r['best']) and not isatk(r['played']))
    return a / tot if tot > 0 else float('nan')
bys = collections.defaultdict(list)
for r in rows: bys[r['seed']].append(r)
ks = list(bys); rng = np.random.default_rng(TM.SEED)
def boot(sel):
    v = []
    for _ in range(TM.BOOT):
        rs = [r for k in rng.integers(0, len(ks), len(ks)) for r in bys[ks[k]] if sel(r)]
        v.append(share(rs))
    return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
out = {"record": W, "align": dict(align), "rows": len(rows), "best_value_matches_dump_bv": sum(r['bv_check'] for r in rows), "mu": MU, "bands": {}}
for b in TM.TURN_BANDS + ('all',):
    sel = (lambda r: True) if b == 'all' else (lambda r, b=b: r['band'] == b)
    rs = [r for r in rows if sel(r)]
    out["bands"][b] = {"rows": len(rs), "diff_sum": round(sum(r['d'] for r in rs), 2),
                       "share_best_attack_played_not": round(share(rs), 4), "share_ci": boot(sel),
                       "share_rows_best_attack_played_not": round(sum(1 for r in rs if isatk(r['best']) and not isatk(r['played'])) / len(rs), 4),
                       "table": table(rs)}
json.dump(out, open('logs/theory_map/syn_w39_ties_type.json', 'w'), ensure_ascii=False, indent=1)
print(json.dumps({k: out[k] for k in ('align', 'rows', 'best_value_matches_dump_bv')}))
for b, v in out['bands'].items():
    print(b, v['rows'], v['diff_sum'], v['share_best_attack_played_not'], v['share_ci'], v['share_rows_best_attack_played_not'])
    print('  ', list(v['table'].items())[:8])
