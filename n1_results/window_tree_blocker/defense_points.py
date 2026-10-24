"""守りの再監査の確かめ済みの見逃しのうち、自分のライフ ≤ 1 のカウンターの窓の点で、修正前（カウンターの窓も木）と
修正後（ブロッカーの窓だけ木）が何を選ぶかを見る（WP rs-window-tree-blocker・1 回きり）。

点ごとに、その点が出た局（修正後の点は修正後の打ち方・修正前の点は旗を立てた打ち方）を打ち直し、点の盤面で
探索の乱数を 8 通りに替えて 2 つの形に選ばせる。守れた手＝監査の記録で相手の甲が False の応手（本当の手札で守れる）。
"""
import json
import os
import sys
from multiprocessing import get_context

REPO = "/home/user/opcg-sim-backend"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tests", "scripts"))
os.chdir(REPO)
os.environ.setdefault("OPCG_LOG_SILENT", "1")
import defense_audit as DA  # noqa: E402
import lethal_audit as LA  # noqa: E402
from opcg_sim.loop import engine as E  # noqa: E402

D = "n1_results/window_tree_blocker"
NET = "opcg_sim/data/learned/nrel_r13d.npz"
SETS = {"new": {}, "legacy": {"window_tree_counter_legacy": True}}
PATHS = {("before", 3013000): "n1_results/window_read/defense_new/points_3013000_300.jsonl.gz",
         ("before", 3351000): "n1_results/reaudit_wr/defense/points_3351000_300.jsonl",
         ("after", 3013000): f"{D}/defense_3013000/merged/points_3013000_300.jsonl.gz",
         ("after", 3351000): f"{D}/defense_3351000/merged/points_3351000_300.jsonl.gz"}


class A:
    budget, worlds, lethal_proof, net, sims = E.PLAY_BUDGET, 1, None, NET, E.PLAY_SIMS


def one(job):
    side, b, row, safe = job
    a = A()
    if side == "before":
        a.window_tree_counter_legacy = True
    game, move, got = DA.replay_to(row["seed"], a, row["seat"], row["step"])
    base = json.loads(got["spec"].decide_opts(row["seed"], got["turn"], row["seat"], None))
    base.pop("window_tree_counter_legacy", None)
    res = {"side": side, "band": b, "seed": row["seed"], "step": row["step"], "my_life": row["my_life"],
           "live": LA.describe(game, move), "safe": sorted(safe), "live_window": got["out"].get("window")}
    for tag, extra in SETS.items():
        picks = {}
        for j in range(8):
            o = dict(base, **extra, search_seed=(base["search_seed"] + 7919 * j) & 0xFFFFFFFFFFFFFFFF)
            d = json.loads(game.fork().decide(row["seat"], json.dumps(o)))
            k = LA.describe(game, d["move"])
            picks[k] = picks.get(k, 0) + 1
        res[tag] = picks
        res[tag + "_safe"] = sum(v for k, v in picks.items() if k in safe)
    return res


def main():
    cmp_ = json.load(open(f"{D}/compare.json"))
    jobs = []
    for b in (3013000, 3351000):
        for side in ("before", "after"):
            rows = [r for r in cmp_[str(b)][side]["verified_rows"]
                    if r["action"] == "SELECT_COUNTER" and r["my_life"] <= 1]
            if not rows:
                continue
            _g, pts = DA._load(PATHS[(side, b)])
            idx = {(p["seed"], p["seat"], p["step"]): p for p in pts if p["kind"] == "resp"}
            for r in rows:
                p = idx[(r["seed"], r["seat"], r["step"])]
                safe = {x["move"] for x in p["rows"] if (x.get("ko") or {}).get("res") is False}
                jobs.append((side, b, r, safe))
    out = []
    with get_context("fork").Pool(4) as pool:
        for r in pool.imap_unordered(one, jobs):
            print(json.dumps({k: r[k] for k in ("side", "seed", "step", "my_life", "new_safe", "legacy_safe")}), flush=True)
            out.append(r)
    out.sort(key=lambda r: (r["band"], r["side"], r["seed"]))
    json.dump(out, open(f"{D}/defense_points.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
