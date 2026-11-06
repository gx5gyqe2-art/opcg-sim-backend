"""WP world-vote-desk（§7.86 段 0）: 4 つの選び方で打った手（a）と替わった手を、§7.84 と同じ審判で判定する（読み取りだけ）。

審判は §7.84 と同じ r29f4（160 回・世界 1 本・公平・証明 off）・K=32・配り方 `move_judge.world_seed(seed, step, i)`。
a の 32 局は §7.84 の記録（`points.jsonl` の `ref.r13d.a`＝器の鍵名のまま・審判 1＝r29f4）を使い、替わった手 c の 32 局だけを打つ
（同じ配り方と乱数で対の差 c − a。`--check` で記録の b を打ち直して決定論を確かめる）。c が §7.84 の b と同じ手なら記録の差をそのまま使う。

    python n1_results/world_vote_desk/wvd_judge.py --check 3   # 決定論の確かめ（b を 3 点打ち直す）
    python n1_results/world_vote_desk/wvd_judge.py             # 判定（4 並列・再開可）
"""
import json, os, sys, time
from multiprocessing import get_context
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tests", "scripts")); sys.path.insert(0, ROOT); os.chdir(ROOT)
os.environ.setdefault("OPCG_LOG_SILENT", "1")
import blunder_audit as BA  # noqa: E402
import move_judge as MJ  # noqa: E402
from opcg_sim.loop import engine as E, decks as D  # noqa: E402

NET = "opcg_sim/data/learned/nrel_r29f4.npz"
SRC = "n1_results/reaudit_r29f4/blunder"
SCAN = "n1_results/world_vote_desk/scan.jsonl"
OUT = "n1_results/world_vote_desk/judge.jsonl"
K = 32
FORMS = ("mean4", "sum8", "mean8")
G = {}


def init():
    E.engine(); BA._G["db"] = D.load_db()
    G["ref"] = E.SeatSpec(NET, sims=MJ.REFEREE_SIMS)
    G["games"] = {json.loads(l)["seed"]: json.loads(l) for l in open(os.path.join(SRC, "games.jsonl"))}


def job_one(job):
    p, mv, tag = job
    t0 = time.time()
    g = G["games"][p["seed"]]
    game = BA.replay(g, p["step"])
    wc = [MJ.rollout(game, p["seat"], mv, MJ.world_seed(p["seed"], p["step"], i), G["ref"])[0] for i in range(K)]
    wa = p["ref"]["r13d"]["a"]
    return {"seed": p["seed"], "step": p["step"], "key": BA.key(mv), "tag": tag, "mv": mv, "desc": BA.desc(game, mv),
            "c": wc, **MJ.paired(wa, wc), "sec": round(time.time() - t0, 1)}


def load_points():
    return {(p["seed"], p["step"]): p for p in map(json.loads, open(os.path.join(SRC, "points.jsonl"))) if p["ok"] and p["kind"] == "main"}


def jobs():
    P = load_points()
    seen, out = set(), []
    for r in map(json.loads, open(SCAN)):
        if r.get("skip"):
            continue
        p = P[(r["seed"], r["step"])]
        ka, kb = BA.key(p["a"]), BA.key(p["b"])
        for f in FORMS:
            mv = r.get(f)
            if mv is None:
                continue
            k = BA.key(mv)
            if k in (ka, kb) or (r["seed"], r["step"], k) in seen:
                continue
            seen.add((r["seed"], r["step"], k))
            out.append((p, mv, f))
    return out


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--check":
        n = int(sys.argv[2]); init()
        P = [p for p in load_points().values()][:n]
        for p in P:
            r = job_one((p, p["b"], "check"))
            print(p["seed"], p["step"], "recorded b == replayed b:", r["c"] == p["ref"]["r13d"]["b"], "diff", r["diff"], p["ref"]["r13d"]["diff"], r["sec"], flush=True)
        sys.exit(0)
    J = jobs()
    done = {(x["seed"], x["step"], x["key"]) for x in map(json.loads, open(OUT))} if os.path.exists(OUT) else set()
    J = [j for j in J if (j[0]["seed"], j[0]["step"], BA.key(j[1])) not in done]
    print("todo", len(J), flush=True)
    t0 = time.time()
    with get_context("fork").Pool(int(os.environ.get("WORKERS", "4")), initializer=init) as pool, open(OUT, "a") as f:
        for i, r in enumerate(pool.imap_unordered(job_one, J, chunksize=1)):
            f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()
            print(i, round(time.time() - t0), r["seed"], r["step"], r["tag"], r["diff"], r["sec"], flush=True)
    print("ALL_DONE", round(time.time() - t0), flush=True)
