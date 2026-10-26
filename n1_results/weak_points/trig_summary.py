"""WP weak-points §5: trig/points.jsonl を数える（1 回きり）。差＝断る − 発動する（審判 r22 で、確認する席の勝率）。
CPU の手の得＝CPU が断ったなら差・発動したなら −差。"""
import json, math, collections, sys
P = [json.loads(l) for l in open("n1_results/weak_points/trig/points.jsonl")]
G = [json.loads(l) for l in open("n1_results/weak_points/trig/games.jsonl")]
db = {c["number"]: c for c in json.load(open("opcg_sim/data/opcg_cards.json"))}


def ttype(cid):
    c = db.get(cid) or {}
    t = c.get("効果(トリガー)") or ""
    if not t:
        return "なし"
    if "このカードを登場" in t or ("登場させる" in t and "このキャラ" not in t and "このカード" in t):
        return "自身を登場"
    if "【メイン】効果を発動" in t or "このカードの【メイン】" in t:
        return "【メイン】を発動"
    if "相手" in t and any(w in t for w in ("KO", "手札に戻", "デッキの下", "トラッシュに置")):
        return "相手を除去"
    if "相手" in t and "レスト" in t:
        return "相手をレスト"
    if "パワー-" in t or "パワーー" in t:
        return "相手のパワー−"
    if "ライフ" in t and ("加える" in t or "置く" in t):
        return "ライフを足す"
    if "引く" in t or "手札に加える" in t:
        return "引く・加える"
    if "登場" in t:
        return "他を登場"
    if "パワー+" in t:
        return "パワー＋"
    return "その他"


def mci(xs):
    n = len(xs)
    if n == 0:
        return {"n": 0}
    m = sum(xs) / n
    v = sum((x - m) ** 2 for x in xs) / (n - 1) if n > 1 else 0
    h = 1.96 * math.sqrt(v / n) if n > 1 else float("nan")
    return {"n": n, "mean": round(m, 4), "ci": [round(m - h, 4), round(m + h, 4)]}


def row(p):
    j = p.get("judge") or {}
    if j.get("diff") is None:
        return None
    adv = j["diff"] if p["cpu"] is False else -j["diff"]
    lo, hi = j["ci"]
    sig = None
    if j["diff"] >= 0.2 and lo > 0:
        sig = "decline_better"
    elif j["diff"] <= -0.2 and hi < 0:
        sig = "accept_better"
    return {"adv": adv, "diff": j["diff"], "sig": sig, "flat": all(x == y for x, y in zip(j["a"], j["b"]))}


out = {"games": len(G), "void": sum(1 for g in G if g.get("winner") is None), "points": len(P)}
life = [p for p in P if p["life_trigger"]]
opt = [p for p in P if not p["life_trigger"]]
out["life_points"], out["optional_points"] = len(life), len(opt)
out["cand2"] = sum(p["n_legal"] == 2 for p in P)
for name, S in (("life", life), ("optional", opt)):
    rs = [(p, row(p)) for p in S]
    rs = [(p, r) for p, r in rs if r]
    d = {"n": len(rs), "cpu_decline": sum(p["cpu"] is False for p, _ in rs),
         "flat": sum(r["flat"] for _, r in rs),
         "adv": mci([r["adv"] for _, r in rs]),
         "diff_all": mci([r["diff"] for _, r in rs]),
         "sig_decline_better": sum(r["sig"] == "decline_better" for _, r in rs),
         "sig_accept_better": sum(r["sig"] == "accept_better" for _, r in rs),
         "cpu_wrong_sig": sum((r["sig"] == "decline_better" and p["cpu"] is True) or (r["sig"] == "accept_better" and p["cpu"] is False) for p, r in rs),
         "too_many_declines": sum(r["sig"] == "accept_better" and p["cpu"] is False for p, r in rs),
         "too_many_accepts": sum(r["sig"] == "decline_better" and p["cpu"] is True for p, r in rs)}
    for key, f in (("by_life", lambda p: str(min(p["my_life"], 3)) + ("+" if p["my_life"] >= 3 else "")),
                   ("by_type", lambda p: ttype(p["card"])),
                   ("by_hand", lambda p: "hand<=3" if p["my_hand"] <= 3 else "hand>=4"),
                   ("by_battle", lambda p: "battle" if p["battle"] else "no_battle")):
        grp = collections.defaultdict(list)
        for p, r in rs:
            grp[f(p)].append((p, r))
        d[key] = {k: {"n": len(v), "cpu_decline": sum(p["cpu"] is False for p, _ in v),
                      "adv": mci([r["adv"] for _, r in v])["mean"] if v else None,
                      "diff": mci([r["diff"] for _, r in v]),
                      "nonflat": sum(not r["flat"] for _, r in v),
                      "wrong_sig": sum((r["sig"] == "decline_better" and p["cpu"] is True) or (r["sig"] == "accept_better" and p["cpu"] is False) for p, r in v)}
                  for k, v in sorted(grp.items())}
    out[name] = d
out["wrong_points"] = [{k: p[k] for k in ("seed", "step", "card", "my_life", "opp_life", "my_hand", "cpu", "q")} | {"type": ttype(p["card"]), "diff": p["judge"]["diff"], "ci": p["judge"]["ci"]}
                       for p in P if (r := row(p)) and ((r["sig"] == "decline_better" and p["cpu"] is True) or (r["sig"] == "accept_better" and p["cpu"] is False))]
json.dump(out, open("n1_results/weak_points/trig_summary.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1)[:6000])
