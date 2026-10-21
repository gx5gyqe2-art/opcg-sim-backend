"""1 手の決定時間の分布（`docs/search_efficiency_plan.md` §1・Phase 0）。

両席を同じ設定で打ち、観測者で `decide` の間隔を測る。1 コアに固定して測るときは
`taskset -c 0 python tests/scripts/rs_decide_latency.py --sims 1280 --worlds 1 --budget 0 --games 20`。
出力は JSON 1 行（条件・手数・平均／中央／90／99／最大）。
"""
import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ.setdefault("OPCG_LOG_SILENT", "1")

from opcg_sim.loop import decks as D, driver as DR, engine as E  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=E.PLAY_SIMS)
    ap.add_argument("--worlds", type=int, default=1)
    ap.add_argument("--budget", type=int, default=E.PLAY_BUDGET, help="戦闘箱の枝予算（0＝無制限・-1＝Rust 既定）")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--seed-base", type=int, default=2999900)
    ap.add_argument("--net", default=None)
    ap.add_argument("--decks", default="synth_roles")
    ap.add_argument("--omniscient", action="store_true",
                    help="両席とも全部見える探索（`docs/search_efficiency_plan.md` §7.10・測定専用）")
    a = ap.parse_args(argv)
    E.engine()
    db = D.load_db()
    kw = {}
    if a.budget >= 0:
        kw["budget"] = a.budget
    if a.worlds > 1:
        kw["worlds"] = a.worlds
    if a.omniscient:
        kw["omniscient"] = True
    spec = E.SeatSpec(a.net, sims=a.sims, **kw)
    ts = []
    last = [0.0]

    def obs(game, name, turn, step, out, move):
        now = time.perf_counter()
        ts.append(now - last[0])
        last[0] = now

    t0 = time.perf_counter()
    for g in range(a.games):
        seed = a.seed_base + g
        la, lb = D.leader_pair(db, seed, "random")
        p1, p2 = D.build_pair(db, la, lb, seed, a.decks)
        last[0] = time.perf_counter()
        DR.run_game(seed, {"p1": spec, "p2": spec}, p1, p2, observer=obs)
    ts.sort()
    n = len(ts)
    print(json.dumps({"sims": a.sims, "worlds": a.worlds, "budget": a.budget,
                      **({"omniscient": True} if a.omniscient else {}), "games": a.games, "moves": n,
                      "mean": round(statistics.mean(ts), 3), "p50": round(ts[n // 2], 3),
                      "p90": round(ts[int(n * 0.9)], 3), "p99": round(ts[int(n * 0.99)], 3),
                      "max": round(ts[-1], 2), "wall_sec": round(time.perf_counter() - t0, 1),
                      "affinity": sorted(os.sched_getaffinity(0))}))


if __name__ == "__main__":
    main()
