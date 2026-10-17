import json, sys, collections
import numpy as np
sys.path.insert(0, 'tests/scripts')
import theory_map as TM, theory_map_deckcal as DC
S = sys.argv[1]; W = S + '/data/sigma_wip/data/w39'
dump = TM.load_dump(S + '/pg/dump.jsonl'); out = {}
for t in ('def', 'sv'):
    clocks = json.load(open(f'{S}/hs/clocks_{t}.json'))
    rows_all, *_ = TM.build_rows(W, dump, clocks)
    rows = [r for r in rows_all if r['pv'] is not None and r['nsc'] >= 2 and r['ptype'] in TM.TYPES and r['btype'] in TM.TYPES and r.get('W') is not None]
    first = [r for r in rows if r['first_in_turn'] and not r['settled_after']]
    out[t] = {b: DC.cal_block(DC.unit_cal([r for r in first if TM.turn_band(r['t']) == b]), 2000) for b in TM.TURN_BANDS}
json.dump(out, open(S + '/hs/band.json', 'w'), ensure_ascii=False, indent=1)
for t in out: print(t, {b: (v['units'], v['W'], v['win'], v['diff'], v['diff_ci']) for b, v in out[t].items()})
