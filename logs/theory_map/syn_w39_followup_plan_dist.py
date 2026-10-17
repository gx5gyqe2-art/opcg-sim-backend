import json, sys, collections
import numpy as np
sys.path.insert(0, 'tests/scripts')
import theory_map as TM
from theory_rs import MU
from opcg_sim.learned.train import plan_labels as PL
S = sys.argv[1]; W = S + '/data/sigma_wip/data/w39'
import os; dump = TM.load_dump(os.environ['CP_DUMP'])
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
out = {"rows": len(rows), "align": dict(align), "bands": {}}
for b in TM.TURN_BANDS + ('all',):
    rs = [r for r in rows if b == 'all' or r['band'] == b]
    d = np.array([r['d'] for r in rs]); tot = d.sum()
    pair = sum(r['d'] for r in rs if isatk(r['best']) and not isatk(r['played']))
    out["bands"][b] = {"rows": len(rs), "pair_share": round(pair / tot, 4), "d_sum": round(float(tot), 2),
        "d_median": round(float(np.median(d)), 4), "d_q25": round(float(np.percentile(d, 25)), 4), "d_q75": round(float(np.percentile(d, 75)), 4),
        "d_mean": round(float(d.mean()), 4), "share_ge_1mu": round(float((d >= 1).mean()), 4), "share_zero": round(float((d <= 1e-12).mean()), 4),
        "best_attack_rows": round(float(np.mean([isatk(r['best']) for r in rs])), 4)}
json.dump(out, open(os.environ['CP_OUT'], 'w'), ensure_ascii=False, indent=1)
print(json.dumps({b: {k: v[k] for k in ('rows', 'pair_share', 'd_median', 'share_ge_1mu', 'best_attack_rows')} for b, v in out['bands'].items()}, ensure_ascii=False))
