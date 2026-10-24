"""WP weak-points §3: hand_points.jsonl（母集団）と hand_judge.jsonl（判定した標本）を数える（1 回きり）。
差＝b − a（a＝今の CPU の手・b＝次点の別の札・手を選ぶ席の勝率）。損の候補＝差 ≥ 0.20 かつ 95% 区間の下限 > 0。"""
import json, math, re, collections
P = [json.loads(l) for l in open("n1_results/weak_points/hand_points.jsonl")]
J = [json.loads(l) for l in open("n1_results/weak_points/hand_judge.jsonl")]
db = json.load(open("opcg_sim/data/opcg_cards.json"))
by_name = collections.defaultdict(list)
for c in db:
    by_name[c.get("name")].append(c)


def kind(msg):
    m = re.search(r"「(.+?)」の効果", msg or "")
    cs = by_name.get(m.group(1)) if m else None
    if not cs:
        return "不明"
    t = " ".join((c.get("効果(テキスト)") or "") for c in cs)
    if "手札" in t and "捨て" in t:
        return "捨てる"
    if "手札" in t and "公開" in t:
        return "公開する（コスト・条件）"
    if "手札" in t and ("デッキの下" in t or "デッキの上" in t):
        return "デッキへ戻す"
    if "手札" in t and "登場" in t:
        return "手札から登場"
    if "手札" in t and "ライフ" in t:
        return "ライフへ置く"
    return "その他"


def mci(xs):
    n = len(xs)
    if not n:
        return {"n": 0}
    m = sum(xs) / n
    v = sum((x - m) ** 2 for x in xs) / (n - 1) if n > 1 else 0
    h = 1.96 * math.sqrt(v / n) if n > 1 else float("nan")
    return {"n": n, "mean": round(m, 4), "ci": [round(m - h, 4), round(m + h, 4)]}


out = {"population": len(P), "games": len({p["seed"] for p in P}),
       "pop_by_kind": dict(collections.Counter(kind(p["msg"]) for p in P)),
       "pop_gap_le_0.05": sum(1 for p in P if len(p["top"]) > 1 and p["top"][0]["q"] - p["top"][1]["q"] <= 0.05)}
rows = [r for r in J if r.get("judge") and r["judge"].get("diff") is not None]
out["judged"] = len(J)
out["judged_with_b"] = len(rows)
out["now_same_as_record"] = sum(r["now_same"] for r in J)
out["r22_same_as_r13d"] = sum(sorted(r["r22_pick"]) == sorted(r["now_pick"]) for r in J)
out["diff"] = mci([r["judge"]["diff"] for r in rows])
cand = [r for r in rows if r["judge"]["diff"] >= 0.2 and r["judge"]["ci"][0] > 0]
rev = [r for r in rows if r["judge"]["diff"] <= -0.2 and r["judge"]["ci"][1] < 0]
out["loss_candidates"] = len(cand)
out["reverse"] = len(rev)
grp = collections.defaultdict(list)
for r in rows:
    grp[kind(r["msg"])].append(r)
out["by_kind"] = {k: {"n": len(v), "diff": mci([r["judge"]["diff"] for r in v]),
                      "loss": sum(r in cand for r in v), "reverse": sum(r in rev for r in v)} for k, v in grp.items()}
gg = collections.defaultdict(list)
for r in rows:
    gg["gap<=0.05" if (r.get("gap") or 0) <= 0.05 else "gap>0.05"].append(r)
out["by_gap"] = {k: {"n": len(v), "diff": mci([r["judge"]["diff"] for r in v]), "loss": sum(r in cand for r in v)} for k, v in gg.items()}
out["loss_points"] = [{k: r.get(k) for k in ("seed", "step", "msg", "now_pick", "b", "gap", "r22_pick", "my_life", "opp_life")} | {"kind": kind(r["msg"]), "diff": r["judge"]["diff"], "ci": r["judge"]["ci"]} for r in cand]
json.dump(out, open("n1_results/weak_points/hand_summary.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
