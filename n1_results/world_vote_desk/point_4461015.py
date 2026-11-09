"""偏りの 3 点目 4461015・141（続き＝付与の枚数・§7.84 §2.2）: 持ち越し無しで読み直した木の根で 4 つの選び方を出す（読み取りだけ）。
続きの点は実局では木を回していないので、§7.84 の判定の b（持ち越し無しの読み直し）と同じ読みに 4 つの選び方を当てる。"""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "n1_results", "world_vote_desk")); os.chdir(ROOT)
import wvd_scan as W  # noqa: E402
W.init()
p = next(p for p in map(json.loads, open("n1_results/reaudit_r29f4/blunder/points.jsonl")) if (p["seed"], p["step"]) == (4461015, 141))
g = W.G["games"][p["seed"]]
real = W.BA.replay(g, p["step"]); who = p["seat"]
ss = W.E.search_seed(p["seed"], p["turn"], who)
res = {"a": p["a_desc"], "b(judged)": p["b_desc"], "diff_b_minus_a": p["ref"]["r13d"]["diff"]}
for nw in (4, 8):
    out, c, w = W.read(real, who, ss, nw)
    legal = out["stats"]["legal"]
    ch, al, gr = W.forms(out)
    res[f"sum{nw}"] = W.BA.desc(real, legal[ch["rep"]]); res[f"mean{nw}"] = W.BA.desc(real, legal[al["rep"]])
    res[f"top{nw}"] = sorted([{"d": W.BA.desc(real, legal[x["rep"]])[:90], "share": x["share"], "q": x["q"], "mean_q": x["mean_q"],
                               "per": x["per"]} for x in gr], key=lambda x: -x["share"])[:4]
    for f in ("sum", "mean"):
        mv = legal[(ch if f == "sum" else al)["rep"]]
        res[f"{f}{nw}_is_a"] = W.BA.key(mv) == W.BA.key(p["a"]); res[f"{f}{nw}_is_b"] = W.BA.key(mv) == W.BA.key(p["b"])
json.dump(res, open("n1_results/world_vote_desk/point_4461015.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1))
