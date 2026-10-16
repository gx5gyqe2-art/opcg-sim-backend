import json, sys, collections
import numpy as np
sys.path.insert(0, 'tests/scripts')
import theory_map as TM
from theory_rs import MU
from opcg_sim.learned.train import plan_labels as PL
S = sys.argv[1]; W = S + '/data/sigma_wip/data/w39'
import os; dump = TM.load_dump(os.environ.get('CP_DUMP', S + '/cp/dump.jsonl'))
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
        rec = {'seed': seed, 'band': TM.turn_band(t), 'best': cls(sigs[bi], u2c), 'played': cls(sigs[ch], u2c),
               'd': (float(bv) - float(pv)) / MU, 'bv_check': abs(float(vals[bi]) - float(bv)) < 1e-9}
        pt = dr[9] if len(dr) > 9 else None
        if pt and isinstance(pt, list) and len(pt) == 2:
            base, cp = pt
            if len(cp) == len(vals) and cp[bi] and cp[ch] and cp[bi][0] is not None and cp[ch][0] is not None:
                tot = cp[bi][2] - cp[ch][2]; opp = cp[bi][3] - cp[ch][3]
                rec.update({'w_gap': tot, 'w_opp': opp, 'w_own': tot - opp, 'k0': base[3]})
                Pb, Pc = (cp[bi][4] if len(cp[bi]) > 4 else None), (cp[ch][4] if len(cp[ch]) > 4 else None)
                if Pb and Pc:
                    rec['dP'] = [None if (Pb[k] is None or Pc[k] is None) else float(Pb[k]) - float(Pc[k]) for k in range(10)]
                    rec['price_ok'] = base[3] > 0 and abs(tot / base[3] - (float(bv) - float(pv))) < 1e-6
            else:
                align['parts_missing'] += 1
        else:
            align['parts_none'] += 1
        rows.append(rec)
isatk = lambda c: c.startswith('atk_')
def summ(rs):
    ys = [r for r in rs if 'w_gap' in r]
    t = sum(y['w_gap'] for y in ys)
    o = {"n": len(rs), "n_parts": len(ys), "price_ok": sum(1 for y in ys if y.get('price_ok')),
         "w_gap_sum": round(t, 4), "opp_share": round(sum(y['w_opp'] for y in ys) / t, 4) if t else None,
         "own_share": round(sum(y['w_own'] for y in ys) / t, 4) if t else None}
    zs = [y for y in ys if y.get('dP') and all(y['dP'][k] is not None for k in (0, 1, 2, 5, 6, 7))]
    for side, ks in (("own_clock_opp_durability", (0, 1, 2)), ("opp_clock_own_durability", (5, 6, 7))):
        sums = [sum(z['dP'][k] for z in zs) for k in ks]; tt = sum(sums)
        o[side] = {"n": len(zs), "sum_life": round(sums[0], 3), "sum_hand": round(sums[1], 3), "sum_body": round(sums[2], 3),
                   "share_life": round(sums[0] / tt, 4) if tt else None, "share_hand": round(sums[1] / tt, 4) if tt else None,
                   "share_body": round(sums[2] / tt, 4) if tt else None}
    for nm, k in (("opp_cards_cut", 3), ("opp_stopped", 4), ("own_cards_cut", 8), ("own_stopped", 9)):
        vv = [y['dP'][k] for y in ys if y.get('dP') and y['dP'][k] is not None]
        o["mean_d_" + nm] = round(float(np.mean(vv)), 4) if vv else None
        o["n_" + nm] = len(vv)
    return o
def hand_share(rs):
    zs = [r for r in rs if r.get('dP') and all(r['dP'][k] is not None for k in (0, 1, 2))]
    s = [sum(z['dP'][k] for z in zs) for k in (0, 1, 2)]; t = sum(s)
    return s[1] / t if t else float('nan')
bys = collections.defaultdict(list)
for r in rows: bys[r['seed']].append(r)
ks = list(bys); rng = np.random.default_rng(TM.SEED)
out = {"record": W, "align": dict(align), "rows": len(rows), "best_value_matches_dump_bv": sum(r['bv_check'] for r in rows), "mu": MU, "bands": {}}
for b in TM.TURN_BANDS + ('all',):
    sel = (lambda r: True) if b == 'all' else (lambda r, b=b: r['band'] == b)
    grp = [r for r in rows if sel(r) and isatk(r['best']) and not isatk(r['played'])]
    e = {"pair_best_attack_played_not": summ(grp), "all_rows": summ([r for r in rows if sel(r)])}
    v = []
    for _ in range(int(os.environ.get("CP_BOOT", TM.BOOT))):
        v.append(hand_share([r for k in rng.integers(0, len(ks), len(ks)) for r in bys[ks[k]] if sel(r) and isatk(r['best']) and not isatk(r['played'])]))
    e["pair_hand_share_ci"] = [round(float(np.nanpercentile(v, 2.5)), 4), round(float(np.nanpercentile(v, 97.5)), 4)]
    out["bands"][b] = e
json.dump(out, open(os.environ.get('CP_OUT', 'logs/theory_map/syn_w39_own_clock_parts.json'), 'w'), ensure_ascii=False, indent=1)
print(json.dumps(out["align"]), out["rows"])
for b, e in out["bands"].items():
    p = e["pair_best_attack_played_not"]
    print(b, p["n"], p["n_parts"], p["price_ok"], "own", p["own_share"], "opp", p["opp_share"], p["own_clock_opp_durability"], e["pair_hand_share_ci"])
