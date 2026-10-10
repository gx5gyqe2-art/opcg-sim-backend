"""T18-禁止だけの塊を集計する（腕ごとのペア水準勝率と 95% 区間・seed で対にした差・void・介入率）。

    python logs/t18_forbid/aggregate.py [--json out.json]
"""
import glob, json, math, os, sys
from collections import Counter
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
from opcg_sim.loop import arena as AR  # noqa: E402


def load(cond, arm):
    games = []
    for p in sorted(glob.glob(os.path.join(HERE, f"{cond}_{arm}_*.json"))):
        games += json.load(open(p))["games"]
    return games


def pairs(games):
    by = {}
    for g in games:
        by.setdefault(g["seed"], {})[g["intervened"]] = g
    out = {}
    for s, d in by.items():
        a, b = d.get("p1"), d.get("p2")
        out[s] = None if (a is None or b is None or a["score"] is None or b["score"] is None) \
            else (a["score"] + b["score"]) / 2.0
    return out


def arm_summary(games):
    ps = pairs(games)
    sc = [v for v in ps.values() if v is not None]
    seen = sum(g["n_seen"] for g in games)
    itv = sum(g["n_intervened"] for g in games)
    fam = Counter(f'{i["played_family"]}->{i["rep_family"]}' for g in games for i in g["interventions"])
    side = {s: [g["score"] for g in games if g["intervened"] == s and g["score"] is not None] for s in ("p1", "p2")}
    return {"n_games": len(games), "n_pairs": len(ps), "n_void_pairs": sum(1 for v in ps.values() if v is None),
            "void_games": sum(1 for g in games if g["score"] is None),
            "void_rate_games": round(sum(1 for g in games if g["score"] is None) / max(1, len(games)), 4),
            "ci": AR.pair_level_ci(sc) if sc else None,
            "n_seen": seen, "n_intervened": itv, "intervene_rate": round(itv / seen, 4) if seen else None,
            "filter_row_rate": round(sum(g["n_filter_rows"] for g in games) / seen, 4) if seen else None,
            "removed_per_row": round(sum(g["n_removed"] for g in games) / seen, 3) if seen else None,
            "intervened_per_game": round(itv / max(1, len(games)), 2),
            "win_as_p1": round(sum(side["p1"]) / len(side["p1"]), 4) if side["p1"] else None,
            "win_as_p2": round(sum(side["p2"]) / len(side["p2"]), 4) if side["p2"] else None,
            "turns_mean": round(sum(g["turns"] for g in games if g["turns"]) / max(1, sum(1 for g in games if g["turns"])), 2),
            "top_swaps": fam.most_common(8)}


def diff(th, pl):
    a, b = pairs(th), pairs(pl)
    d = [a[s] - b[s] for s in a if s in b and a[s] is not None and b[s] is not None]
    if not d:
        return None
    n, m = len(d), sum(d) / len(d)
    var = sum((x - m) ** 2 for x in d) / max(1, n - 1)
    h = 1.96 * math.sqrt(var / n)
    return {"n": n, "mean": m, "lo": m - h, "hi": m + h,
            "theory_better": sum(1 for x in d if x > 0), "tie": sum(1 for x in d if x == 0),
            "placebo_better": sum(1 for x in d if x < 0)}


def verdict(t, p, dd):
    if t is None or t["ci"] is None or dd is None:
        return "no_data"
    if max(t["void_rate_games"], p["void_rate_games"]) > 0.02:
        return "void>2%: 判定しない"
    if t["intervene_rate"] is not None and t["intervene_rate"] < 0.05:
        return "介入 5% 未満: 判定不能（弱い相手で読み分け）"
    if t["ci"]["lo"] > 0.5 and dd["lo"] > 0:
        return "合格"
    if t["ci"]["hi"] < 0.5:
        return "不合格"
    return "判定不能"


res = {}
for cond in ("main", "sub"):
    th, pl = load(cond, "theory"), load(cond, "placebo")
    if not th:
        continue
    t, p = arm_summary(th), (arm_summary(pl) if pl else None)
    dd = diff(th, pl) if pl else None
    res[cond] = {"theory": t, "placebo": p, "diff": dd, "verdict": verdict(t, p, dd)}
print(json.dumps(res, ensure_ascii=False, indent=1))
if "--json" in sys.argv:
    json.dump(res, open(sys.argv[sys.argv.index("--json") + 1], "w"), ensure_ascii=False, indent=1)
