"""WP world-vote-desk（§7.86 段 0）: 木の判断点で 4 つの選び方の手を出す（読み取りだけ・エンジンの既定は不変）。

点ごとに出荷の読み（1,280 回 × 世界 4 本・同じ search_seed＝打った手を再現）を回し、
- 今　　　＝訪問の合算で選んだ束（`max (n, q)`・エンジンの選び方）
- 丸ごと平均＝束ごとに「訪問が 1 以上の世界の値の平均」が最大の束（合算の訪問が全体の 5% 以上の束だけ・§7.84 §2.3 と同じ規則）
を出す。両者が食い違う点だけ、同じ search_seed で世界 8 本を読み直し（世界 i の seed は search_seed+i＝最初の 4 本は同じ世界）、
- 8 本の合算・8 本の平均（同じ規則）
を出す。費用は Rust の世界スレッドを含む CPU 時間（getrusage）で測る。

    python n1_results/world_vote_desk/wvd_scan.py            # 470 点（4 並列・再開可）
"""
import json, os, resource, sys, time
from multiprocessing import get_context
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tests", "scripts")); sys.path.insert(0, ROOT); os.chdir(ROOT)
os.environ.setdefault("OPCG_LOG_SILENT", "1")
import blunder_audit as BA  # noqa: E402
from opcg_sim.loop import engine as E, decks as D  # noqa: E402

NET = "opcg_sim/data/learned/nrel_r29f4.npz"
FLOOR = float(os.environ.get("FLOOR", "0.05"))
SRC = "n1_results/reaudit_r29f4/blunder"
OUT = "n1_results/world_vote_desk/scan.jsonl"
G = {}


def init():
    E.engine(); G["net"] = E.load_net(NET); BA._G["db"] = D.load_db()
    G["games"] = {json.loads(l)["seed"]: json.loads(l) for l in open(os.path.join(SRC, "games.jsonl"))}


def cpu():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def read(game, who, ss, worlds):
    o = {"net": G["net"], "search_seed": ss, "sims": E.PLAY_SIMS, "budget": E.PLAY_BUDGET,
         "worlds": worlds, "lethal_proof": E.PLAY_LETHAL_PROOF}
    c0, t0 = cpu(), time.time()
    out = json.loads(game.fork().decide(who, json.dumps(o)))
    return out, round(cpu() - c0, 3), round(time.time() - t0, 3)


def forms(out):
    """(合算の束, 平均の束, 束の表)。束は rep（合法手の番号）で返す。"""
    pw = out.get("per_world") or []
    tot = sum(sum(w["N"]) for w in pw) or 1
    groups = []
    for gr in out["groups"]:
        per = []
        for w in pw:
            n = sum(w["N"][i] for i in gr["idxs"])
            q = sum(w["N"][i] * w["Q"][i] for i in gr["idxs"]) / n if n > 0 else None
            per.append([n, None if q is None else round(q, 4)])
        mq = [q for n, q in per if n >= 1 and q is not None]
        groups.append({"rep": gr["rep"], "n": gr["n"], "q": round(gr["q"], 4), "per": per,
                       "mean_q": round(sum(mq) / len(mq), 4) if mq else None, "share": round(gr["n"] / tot, 4)})
    ch = max(groups, key=lambda x: (x["n"], x["q"]))
    el = [x for x in groups if x["n"] >= FLOOR * tot and x["mean_q"] is not None]
    al = max(el, key=lambda x: x["mean_q"]) if el else ch
    return ch, al, groups


def one(p):
    g = G["games"][p["seed"]]
    real = BA.replay(g, p["step"]); who = p["seat"]
    ss = E.search_seed(p["seed"], p["turn"], who)
    out4, c4, w4 = read(real, who, ss, E.PLAY_WORLDS)
    r = {"seed": p["seed"], "step": p["step"], "dtype": p["dtype"], "turn": p["turn"], "seat": who,
         "cpu4": c4, "wall4": w4, "kind4": out4.get("kind")}
    if out4.get("kind") != "main" or not out4.get("per_world"):
        r["skip"] = True
        return r
    legal = out4["stats"]["legal"]
    pk = BA.pkey
    ch, al, gr = forms(out4)
    mv = lambda rep: legal[rep]
    r.update({"now": mv(ch["rep"]), "mean4": mv(al["rep"]),
              "now_is_record": pk(mv(ch["rep"])) == pk(p["a"]),
              "disagree": al["rep"] != ch["rep"], "gap4": None if al["mean_q"] is None or ch["mean_q"] is None else round(al["mean_q"] - ch["mean_q"], 4),
              "groups4": [{k: x[k] for k in ("rep", "n", "q", "mean_q", "share", "per")} for x in gr],
              "desc": {}})
    r["desc"]["now"] = BA.desc(real, r["now"]); r["desc"]["mean4"] = BA.desc(real, r["mean4"])
    if r["disagree"]:
        out8, c8, w8 = read(real, who, ss, 8)
        r.update({"cpu8": c8, "wall8": w8, "kind8": out8.get("kind")})
        if out8.get("kind") == "main" and out8.get("per_world"):
            l8 = out8["stats"]["legal"]
            ch8, al8, gr8 = forms(out8)
            r.update({"sum8": l8[ch8["rep"]], "mean8": l8[al8["rep"]],
                      "engine8_is_sum8": pk(out8.get("move")) == pk(l8[ch8["rep"]]) if out8.get("move") else None,
                      "groups8": [{k: x[k] for k in ("rep", "n", "q", "mean_q", "share", "per")} for x in gr8]})
            r["desc"]["sum8"] = BA.desc(real, r["sum8"]); r["desc"]["mean8"] = BA.desc(real, r["mean8"])
    return r


if __name__ == "__main__":
    pts = [json.loads(l) for l in open(os.path.join(SRC, "points.jsonl"))]
    pts = [p for p in pts if p["ok"] and p["kind"] == "main"]
    if len(sys.argv) > 1 and sys.argv[1] == "--only":
        want = {tuple(map(int, s.split(":"))) for s in sys.argv[2].split(",")}
        pts = [p for p in pts if (p["seed"], p["step"]) in want]
    done = {(json.loads(l)["seed"], json.loads(l)["step"]) for l in open(OUT)} if os.path.exists(OUT) else set()
    pts = [p for p in pts if (p["seed"], p["step"]) not in done]
    nw = int(os.environ.get("WORKERS", "4"))
    print("todo", len(pts), "workers", nw, flush=True)
    t0 = time.time()
    with get_context("fork").Pool(nw, initializer=init) as pool, open(OUT, "a") as f:
        for i, r in enumerate(pool.imap_unordered(one, pts, chunksize=1)):
            f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()
            if i % 20 == 0:
                print(i, round(time.time() - t0), r["seed"], r["step"], r.get("disagree"), r.get("cpu4"), r.get("cpu8"), flush=True)
    print("ALL_DONE", round(time.time() - t0), flush=True)
