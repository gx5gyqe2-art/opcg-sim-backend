"""WP world-vote-desk（§7.86 段 0）の集計（読み取りだけ）: scan.jsonl・judge.jsonl・§7.84 の points.jsonl → summary.json。"""
import json, math, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tests", "scripts")); sys.path.insert(0, ROOT); os.chdir(ROOT)
os.environ.setdefault("OPCG_LOG_SILENT", "1")
import blunder_audit as BA  # noqa: E402

D = "n1_results/world_vote_desk"
P = {(p["seed"], p["step"]): p for p in map(json.loads, open("n1_results/reaudit_r29f4/blunder/points.jsonl")) if p["ok"] and p["kind"] == "main"}
S = [r for r in map(json.loads, open(f"{D}/scan.jsonl")) if not r.get("skip")]
J = {(x["seed"], x["step"], x["key"]): x for x in map(json.loads, open(f"{D}/judge.jsonl"))} if os.path.exists(f"{D}/judge.jsonl") else {}
SKEW = [(4461051, 163), (4461063, 108)]


def ms(x):
    n = len(x)
    if n == 0:
        return {"n": 0, "mean": None, "ci": None}
    m = sum(x) / n
    if n < 2:
        return {"n": n, "mean": round(m, 4), "ci": None}
    sd = math.sqrt(sum((v - m) ** 2 for v in x) / (n - 1)); h = 1.96 * sd / math.sqrt(n)
    return {"n": n, "mean": round(m, 4), "ci": [round(m - h, 4), round(m + h, 4)]}


def diff_of(r, mv):
    """(手が替わったか, 差 c − a または None＝未判定)。"""
    p = P[(r["seed"], r["step"])]
    k = BA.key(mv)
    if k == BA.key(p["a"]):
        return False, 0.0
    if k == BA.key(p["b"]):
        return True, p["ref"]["r13d"]["diff"]
    j = J.get((r["seed"], r["step"], k))
    return True, (None if j is None else j["diff"])


def form_move(r, f):
    if f == "now":
        return r["now"]
    if f == "mean4":
        return r["mean4"]
    if not r["disagree"]:
        return r["now"]
    return r[f]


FORMS = {"now": "今（4 本の合算）", "mean4": "丸ごと平均（4 本）", "sum8": "食い違う点だけ 8 本の合算", "mean8": "食い違う点だけ 8 本の平均"}
out = {"points": len(S), "reproduce": sum(r["now_is_record"] for r in S), "disagree": sum(r["disagree"] for r in S)}
out["disagree_rate"] = round(out["disagree"] / len(S), 4)
# 食い違いの平均の差（平均の形の手 − 合算の手の、世界ごとの値の平均の差）で絞った割合
out["disagree_by_gap"] = {str(t): sum(1 for r in S if r["disagree"] and (r["gap4"] or 0) >= t) for t in (0.0, 0.01, 0.03, 0.05, 0.1, 0.2)}
tab = {}
for f in FORMS:
    ch, all_d, unj, big_gain, big_loss = [], [], 0, [], []
    for r in S:
        c, d = diff_of(r, form_move(r, f))
        if c:
            if d is None:
                unj += 1
                continue
            ch.append(d)
            if d >= 0.2: big_gain.append((r["seed"], r["step"], d))
            if d <= -0.2: big_loss.append((r["seed"], r["step"], d))
        if d is not None:
            all_d.append(d)
    skew = {}
    for s in SKEW:
        r = next(x for x in S if (x["seed"], x["step"]) == s)
        c, d = diff_of(r, form_move(r, f))
        skew[f"{s[0]}・{s[1]}"] = {"changed": c, "diff": d, "move": r["desc"].get(f if f != "now" else "now") if (f in ("now", "mean4") or r["disagree"]) else r["desc"]["now"]}
    tab[f] = {"label": FORMS[f], "changed": len(ch) + unj, "unjudged": unj, "changed_diff": ms(ch), "all_points_diff": ms(all_d),
              "sum_gain": round(sum(ch), 3), "big_gain(>=0.2)": big_gain, "big_loss(<=-0.2)": big_loss, "skew": skew}
out["forms"] = tab
# 費用（CPU 秒・Rust の世界スレッドを含む）
c4 = sum(r["cpu4"] for r in S); dis = [r for r in S if r["disagree"]]
c8 = sum(r["cpu8"] for r in dis); c4d = sum(r["cpu4"] for r in dis)
out["cost"] = {"cpu4_total": round(c4, 1), "cpu4_mean": round(c4 / len(S), 3),
               "disagree_cpu4_total": round(c4d, 1), "disagree_cpu8_total": round(c8, 1),
               "inc_reread8_over_tree": round(c8 / c4, 4),            # 食い違う点で 8 本を丸ごと読み直す
               "inc_add4_over_tree": round((c8 - c4d) / c4, 4),       # 最初の 4 本を使い回して 4 本だけ足す
               "inc_add4_worlds_count_over_tree": round(len(dis) / len(S), 4)}  # 世界の本数で数えた増分（読みの量）
json.dump(out, open(f"{D}/summary.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "forms"}, ensure_ascii=False, indent=1))
for f, t in tab.items():
    print(f, t["label"], "changed", t["changed"], "unj", t["unjudged"], "changed_diff", t["changed_diff"], "all", t["all_points_diff"],
          "gain", len(t["big_gain(>=0.2)"]), "loss", len(t["big_loss(<=-0.2)"]), t["skew"])
