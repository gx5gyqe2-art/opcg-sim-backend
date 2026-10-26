"""迷う局面だけ世界 K 本で読み直すときの発動率・入れ替わり率・1 手の時間（`docs/search_efficiency_plan.md` §7.4）。

`rs_decide_latency.py` と同じ局面集（seed 2999900〜・random リーダー・synth_roles・両席同じ設定）を
**読み直さない席（1 本 × S 回）の手順**で打ち進め、各決定点で:

1. 1 本の `decide` を測る（時間 t1・根の比 r＝2 位の訪問数 ÷ 1 位・手）
2. r が線の最小値以上なら、**同じ opts の worlds だけを K 本にした** `decide` も測る（時間 tK・手）
   ＝`opcg_sim.loop.engine.decide_with_rethink` が読み直すときに呼ぶのと同じもの

線 τ ごとの 1 手の時間は「t1 ＋（r ≥ τ なら tK）」＝読み直しの席がその局面で払う時間。局面集を
共有するので、τ ごとの差は線の違いだけから来る（打ち進めの手順は τ に依らない）。

1 コアに固定して測る: 各対局を別プロセスで回し、プロセスごとに 1 コアへ固定する
（`--jobs` ≤ コア数。K 本の世界は 1 コアの上で順に回る＝1 CPU のサーバーと同じ）。

  python tests/scripts/rs_rethink_sweep.py --games 20 --jobs 4 --out /path/rows.jsonl
  python tests/scripts/rs_rethink_sweep.py --summarize /path/rows.jsonl --taus 0.3,0.5,0.7,0.9

出力は JSON 1 行（τ ごとの発動率・入れ替わり率・平均／中央／90／99／最大）。行の台帳は対局単位で
追記する（途中で止めても、済んだ対局は再実行で飛ばす）。
"""
import argparse
import json
import multiprocessing as mp
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ.setdefault("OPCG_LOG_SILENT", "1")

from opcg_sim.loop import decks as D, driver as DR, engine as E  # noqa: E402


def _key(move):
    return json.dumps(move, sort_keys=True, ensure_ascii=False)


def play_one(job):
    """1 局を 1 本の席の手順で打ち、決定点ごとの行を返す（このプロセスは 1 コアに固定）。"""
    g, a = job
    if a["pin"]:
        # ワーカーごとに別のコア（同じコアに 2 局が載らない）。
        wid = (mp.current_process()._identity or (1,))[0] - 1
        os.sched_setaffinity(0, {a["cores"][wid % len(a["cores"])]})
    E.engine()
    db = D.load_db()
    kw = {"budget": a["budget"]} if a["budget"] >= 0 else {}
    spec = E.SeatSpec(a["net"], sims=a["sims"], **kw)
    seed = a["seed_base"] + g
    la, lb = D.leader_pair(db, seed, "random")
    p1, p2 = D.build_pair(db, la, lb, seed, a["decks"])
    game = DR.new_game(seed, p1, p2)
    carries = {"p1": DR.Carry(), "p2": DR.Carry()}
    rows, steps = [], 0
    while game.winner() is None and steps < DR.DEFAULT_MAX_STEPS:
        pending = json.loads(game.pending_json())
        if not pending:
            break
        name = pending["player_id"]
        turn = int(game.turn_count or 0)
        opts_json = spec.decide_opts(seed, turn, name, carries[name].get(turn, name))
        t0 = time.perf_counter()
        out = json.loads(game.decide(name, opts_json))
        t1 = time.perf_counter() - t0
        move = out.get("move")
        if move is None:
            break
        r = E.rethink_ratio(out)
        row = {"game": g, "step": steps, "turn": turn, "kind": out.get("kind"), "r": r, "t1": t1,
               "tk": None, "swap": None}
        if r is not None and r >= a["tau_min"]:
            # `decide_with_rethink` の読み直しと同じ opts（worlds だけを K 本に）。
            o = json.loads(opts_json)
            o["worlds"] = a["worlds"]
            t0 = time.perf_counter()
            out_k = json.loads(game.decide(name, json.dumps(o, ensure_ascii=False)))
            row["tk"] = time.perf_counter() - t0
            row["swap"] = _key(out_k.get("move")) != _key(move)
        rows.append(row)
        carries[name].put(out)
        if move.get("kind") == "battle":
            game.apply_battle_action(name, move["action_type"], move.get("card_uuid"))
        else:
            game.apply_game_action(name, move["action_type"],
                                   json.dumps(move.get("payload") or {}, ensure_ascii=False))
        steps += 1
    return {"game": g, "winner": game.winner(), "steps": steps,
            "core": sorted(os.sched_getaffinity(0)), "rows": rows}


def _dist(ts):
    ts = sorted(ts)
    n = len(ts)
    return {"mean": round(statistics.mean(ts), 3), "p50": round(ts[n // 2], 3),
            "p90": round(ts[int(n * 0.9)], 3), "p99": round(ts[int(n * 0.99)], 3),
            "max": round(ts[-1], 2)}


def summarize(path, taus):
    games = [json.loads(line) for line in open(path) if line.strip()]
    rows = [r for gm in games for r in gm["rows"]]
    tree = [r for r in rows if r["r"] is not None]     # 木を回し候補が 2 つ以上の決定
    res = {"games": len(games), "moves": len(rows), "tree_moves": len(tree),
           "tree_share": round(len(tree) / len(rows), 4),
           "off": _dist([r["t1"] for r in rows]), "taus": {}}
    for tau in taus:
        fired = [r for r in tree if r["r"] >= tau]
        assert all(r["tk"] is not None for r in fired), f"τ={tau} は測った最小の線より下"
        swaps = sum(1 for r in fired if r["swap"])
        times = [r["t1"] + (r["tk"] if (r["r"] is not None and r["r"] >= tau) else 0.0) for r in rows]
        res["taus"][str(tau)] = {
            "fired": len(fired), "fire_rate": round(len(fired) / len(rows), 4),
            "fire_rate_tree": round(len(fired) / max(1, len(tree)), 4),
            "swaps": swaps, "swap_rate_of_fired": round(swaps / max(1, len(fired)), 4),
            "swap_rate": round(swaps / len(rows), 4),
            "compute_x": round(sum(times) / sum(r["t1"] for r in rows), 3),
            **_dist(times)}
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=E.PLAY_SIMS)
    ap.add_argument("--worlds", type=int, default=E.PLAY_RETHINK_WORLDS, help="読み直しの世界の本数 K")
    ap.add_argument("--budget", type=int, default=E.PLAY_BUDGET, help="戦闘箱の枝予算（0＝無制限・-1＝Rust 既定）")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--seed-base", type=int, default=2999900)
    ap.add_argument("--net", default=None)
    ap.add_argument("--decks", default="synth_roles")
    ap.add_argument("--taus", default="0.3,0.5,0.7,0.9")
    ap.add_argument("--jobs", type=int, default=1, help="並べる対局の数（各プロセスを 1 コアに固定）")
    ap.add_argument("--no-pin", action="store_true", help="1 コアに固定しない")
    ap.add_argument("--out", default=None, help="対局ごとの行の台帳（jsonl・追記）")
    ap.add_argument("--summarize", default=None, help="台帳を集計するだけ")
    a = ap.parse_args(argv)
    taus = [float(x) for x in a.taus.split(",")]
    if a.summarize:
        print(json.dumps(summarize(a.summarize, taus), ensure_ascii=False))
        return
    assert a.out, "--out が要る"
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(line)["game"] for line in open(a.out) if line.strip()}
    cfg = {"sims": a.sims, "worlds": a.worlds, "budget": a.budget, "seed_base": a.seed_base,
           "net": a.net, "decks": a.decks, "tau_min": min(taus), "pin": not a.no_pin,
           "cores": sorted(os.sched_getaffinity(0))}
    jobs = [(g, cfg) for g in range(a.games) if g not in done]
    with mp.get_context("spawn").Pool(a.jobs) as pool, open(a.out, "a") as f:
        for res in pool.imap_unordered(play_one, jobs):
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            f.flush()
            print(f"game {res['game']} steps {res['steps']} core {res['core']}", flush=True)
    print(json.dumps(summarize(a.out, taus), ensure_ascii=False))


if __name__ == "__main__":
    main()
