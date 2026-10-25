"""下位デッキの局の、弱い側の席のリーダーの使い方を数える（1 回きり）。"""
import json, sys, collections
rows = [json.loads(l) for l in open(sys.argv[1])]
by = collections.defaultdict(list)
for r in rows:
    D = [21, 44, 18, 27, 48]
    t = D[r["seed"] % 5]
    seat = "p1" if r["seed"] % 2 == 0 else "p2"
    assert r["decks"][0 if seat == "p1" else 1] == t
    L = r["leaders"][0 if seat == "p1" else 1]
    c = collections.Counter()
    for (turn, name, at, cid, effs) in r["log"]:
        if name != seat: continue
        c[(at, cid == L)] += 1
    by[L].append((r["winner"] == seat, r["turns"], c))
for L, xs in by.items():
    n = len(xs); w = sum(x[0] for x in xs)
    tot = collections.Counter()
    for x in xs: tot.update(x[2])
    print(L, "games", n, "wins", w, "turns", sum(x[1] for x in xs) / n)
    print("   leader:", {k[0]: round(v / n, 2) for k, v in tot.items() if k[1]})
    print("   other :", {k[0]: round(v / n, 2) for k, v in tot.items() if not k[1] and k[0]})
