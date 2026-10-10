import json, sys
from collections import Counter
sys.path.insert(0, "/home/user/opcg-sim-backend/tests/scripts")
from theory_rs import MU
d = json.load(open(sys.argv[1])); rows = d["rows"]
print("MU", MU, "judged rows", len(rows), "games", len(d["games"]))
for m in (0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4):
    c = m * MU
    hit = [r for r in rows if str(r["chosen"]) in r["gaps"] and r["gaps"][str(r["chosen"])] > c]
    filt = sum(1 for r in rows if any(g > c for g in r["gaps"].values()))
    rem = sum(sum(1 for g in r["gaps"].values() if g > c) for r in rows)
    fams = Counter(r["fam"][r["chosen"]] for r in hit)
    print(f"{m:>5}xMU  intervene {len(hit)/len(rows):.3f} ({len(hit)})  filter-rows {filt/len(rows):.3f}  removed/row {rem/len(rows):.2f}  {dict(fams)}")
