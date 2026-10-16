import json, sys, collections
A_PATH, B_PATH = sys.argv[1], sys.argv[2]
def strip(o):
    if isinstance(o, list): return [strip(x) for x in o if not (isinstance(x, dict) and "att" in x)]
    if isinstance(o, dict): return {k: strip(v) for k, v in o.items()}
    return o
A = {}
for l in open(A_PATH):
    d = json.loads(l); A[d['seed']] = d
c = collections.Counter(); extra = collections.Counter()
for l in open(B_PATH):
    d = strip(json.loads(l)); a = A[d['seed']]
    ok = d['z'] == a['z'] and d['guard'] == a['guard'] and len(d['rows']) == len(a['rows'])
    for x, y in zip(a['rows'], d['rows']):
        if not ok: break
        if x[:9] != y[:9] or len(x) != len(y): ok = False; break
        if len(x) > 9:
            bx, by = x[9][0], y[9][0]          # 打つ前の部品: 診断の守りの辿りが末尾に 1 つ足される
            if by[:len(bx)] != bx or len(by) - len(bx) > 1: ok = False; break
            extra[len(by) - len(bx)] += 1
            if x[9][1:] != y[9][1:]: ok = False; break
    c['same' if ok else 'diff'] += 1
print(json.dumps({"games": dict(c), "base_extra_len": dict(extra)}))
