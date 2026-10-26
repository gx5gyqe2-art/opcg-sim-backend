"""hunt_play の出力を集計する（不変量の違反・長い／短い局・打たれない札）。使い方: python hunt_agg.py A.jsonl [B.jsonl ...]"""
import json, sys, collections
rows = [json.loads(l) for f in sys.argv[1:] for l in open(f)]
print("games", len(rows), "err", sum(1 for r in rows if r["err"]), "void", sum(1 for r in rows if r["winner"] is None))
vk = collections.Counter(v["k"] for r in rows for v in r["viol"])
print("violations", dict(vk), "games with viol", sum(1 for r in rows if r["n_viol"]))
print("act twice same turn", sum(len(r["act_twice"]) for r in rows), collections.Counter(a["cid"] for r in rows for a in r["act_twice"]).most_common(10))
st = [r["steps"] for r in rows if r["steps"]]
tu = [r["turns"] for r in rows if r["turns"]]
print("steps max", max(st), ">600", sum(s > 600 for s in st), "turns min", min(tu), "<=6", sum(t <= 6 for t in tu), "<=3", sum(t <= 3 for t in tu))
agg = collections.defaultdict(lambda: [0] * 5)
for r in rows:
    for k, v in r["stats"].items():
        cid = k.split("|")[1]
        for i in range(5): agg[cid][i] += v[i]
never_play = sorted(((v[0], c, v) for c, v in agg.items() if v[0] >= 30 and v[2] == 0), reverse=True)
print("in hand at main >=30 but never played", len(never_play))
for x in never_play[:30]: print("  ", x[1], x[2])
never_act = sorted(((v[3], c, v) for c, v in agg.items() if v[3] >= 10 and v[4] == 0), reverse=True)
print("activatable >=10 but never activated", len(never_act))
for x in never_act[:30]: print("  ", x[1], x[2])
json.dump({c: v for c, v in agg.items()}, open("/tmp/hunt_agg_cards.json", "w"))
