"""WP world-vote-desk（§7.86 段 0）の追加の確かめ（標本外・`PREREG_ext.md`）: 同じ 100 局の、判定に引かれなかった木の点で
「食い違う点だけ 8 本の合算」を測る（読み取りだけ・エンジンの既定は不変）。

    python n1_results/world_vote_desk/wvd_ext.py scan    # 木の点を世界 4 本で読み、食い違う点だけ 8 本（4 並列・再開可）
    python n1_results/world_vote_desk/wvd_ext.py judge   # 8 本の合算で手が替わった点だけ、打った手と替わった手を審判で 32 局ずつ（対）
"""
import json, os, sys, time
from multiprocessing import get_context
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "n1_results", "world_vote_desk")); os.chdir(ROOT)
import wvd_scan as W  # noqa: E402
import move_judge as MJ  # noqa: E402

SCAN = "n1_results/world_vote_desk/ext_scan.jsonl"
JUDGE = "n1_results/world_vote_desk/ext_judge.jsonl"
K = 32


def points():
    sampled = {(p["seed"], p["step"]) for p in map(json.loads, open(os.path.join(W.SRC, "points.jsonl")))}
    out = []
    for g in map(json.loads, open(os.path.join(W.SRC, "games.jsonl"))):
        for pt in g["points"]:
            if pt["kind"] != "main" or pt.get("n_cand", 2) < 2 or (g["seed"], pt["step"]) in sampled:
                continue
            out.append(dict(pt, seed=g["seed"], a=pt.get("a_box") or pt["move"]))
    return out


def scan_one(p):
    r = W.one(p)
    if not r.get("skip"):
        # 大きい表は落とす（標本外は数が多い）
        r.pop("groups4", None); r.pop("groups8", None)
    return r


def judge_one(r):
    W_init_once()
    t0 = time.time()
    g = W.G["games"][r["seed"]]
    game = W.BA.replay(g, r["step"])
    ws = [MJ.world_seed(r["seed"], r["step"], i) for i in range(K)]
    wa = [MJ.rollout(game, r["seat"], r["now"], s, W.G["ref"])[0] for s in ws]
    wc = [MJ.rollout(game, r["seat"], r["sum8"], s, W.G["ref"])[0] for s in ws]
    return {"seed": r["seed"], "step": r["step"], "dtype": r["dtype"], "gap4": r["gap4"], "now": r["desc"]["now"], "sum8": r["desc"]["sum8"],
            "a": wa, "c": wc, **MJ.paired(wa, wc), "sec": round(time.time() - t0, 1)}


def W_init_once():
    if "ref" not in W.G:
        W.G["ref"] = W.E.SeatSpec(W.NET, sims=MJ.REFEREE_SIMS)


def run(fn, jobs, outp, init):
    print("todo", len(jobs), flush=True)
    t0 = time.time()
    with get_context("fork").Pool(4, initializer=init) as pool, open(outp, "a") as f:
        for i, r in enumerate(pool.imap_unordered(fn, jobs, chunksize=1)):
            f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()
            if i % 50 == 0 or outp == JUDGE:
                print(i, round(time.time() - t0), r["seed"], r["step"], r.get("disagree"), r.get("diff"), flush=True)
    print("ALL_DONE", round(time.time() - t0), flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "scan":
        done = {(x["seed"], x["step"]) for x in map(json.loads, open(SCAN))} if os.path.exists(SCAN) else set()
        run(scan_one, [p for p in points() if (p["seed"], p["step"]) not in done], SCAN, W.init)
    elif cmd == "judge":
        done = {(x["seed"], x["step"]) for x in map(json.loads, open(JUDGE))} if os.path.exists(JUDGE) else set()
        R = [r for r in map(json.loads, open(SCAN)) if r.get("disagree") and r.get("sum8") is not None
             and W.BA.key(r["sum8"]) != W.BA.key(r["now"]) and (r["seed"], r["step"]) not in done]
        run(judge_one, R, JUDGE, W.init)
