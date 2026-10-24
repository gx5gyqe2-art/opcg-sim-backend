"""守りの再監査の局のうち、修正（§7.45）で手が変わる局を探す（WP `rs-window-tree-blocker`）。

defense_audit.py の `run` と同じ設定（両席 r13d・世界 1 本 × 1,280 回・予算なし・証明 off・random × synth_roles）で
局を修正後の既定で打ち、自分のライフ ≤ 1 のカウンターの窓（候補 2 つ以上）で、同じ盤面・同じ opts に
`window_tree_counter_legacy` を足した決定（修正前）も出して手を比べる。最初に手が分かれた点で打ち切る。
分かれなかった局は修正前と同じ局（決定的）＝監査の記録をそのまま使える。分かれた局だけ監査を打ち直す。

  python n1_results/window_tree_blocker/diverge.py --seed-base 3013000 --games 300 --workers 4 --out <jsonl>
"""
import argparse
import json
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
os.environ.setdefault("OPCG_LOG_SILENT", "1")
from opcg_sim.loop import decks as D, driver as DR, engine as E  # noqa: E402

NET = "opcg_sim/data/learned/nrel_r13d.npz"


class Diverged(Exception):
    pass


def one(args):
    seed, stop = args
    E.engine()
    db = D.load_db()
    spec = E.SeatSpec(NET, sims=E.PLAY_SIMS, budget=E.PLAY_BUDGET)
    la, lb = D.leader_pair(db, seed, "random")
    p1, p2 = D.build_pair(db, la, lb, seed, "synth_roles")
    probes = []

    def obs(game, name, turn, step, out, move):
        pend = json.loads(game.pending_json())
        if pend.get("action") != "SELECT_COUNTER" or pend.get("player_id") != name:
            return
        n = len((out.get("stats") or {}).get("N") or [])
        if n < 2:
            return
        life = _life(game, name)
        if life > 1:
            return
        opts = json.loads(spec.decide_opts(seed, turn, name, None))
        again = json.loads(game.fork().decide(name, json.dumps(opts)))
        old = json.loads(game.fork().decide(name, json.dumps(dict(opts, window_tree_counter_legacy=True))))
        rec = {"step": step, "turn": turn, "seat": name, "life": life, "n": n, "move": move,
               "repro": again.get("move") == move, "legacy": old.get("move"),
               "legacy_window": old.get("window"), "new_window": out.get("window"),
               "same": old.get("move") == move}
        probes.append(rec)
        if stop and not rec["same"]:
            raise Diverged()

    t0 = time.perf_counter()
    diverged = False
    winner = turns = None
    try:
        r = DR.run_game(seed, {"p1": spec, "p2": spec}, p1, p2, observer=obs)
        winner, turns = r["winner"], r["turns"]
    except Diverged:
        diverged = True
    except DR.GameAborted as exc:
        winner = f"aborted: {exc}"
    diverged = diverged or any(not p["same"] for p in probes)
    return {"seed": seed, "diverged": diverged, "winner": winner, "turns": turns, "probes": probes,
            "sec": round(time.perf_counter() - t0, 1)}


def _life(game, name):
    return len(json.loads(game.hidden_json())["players"][name]["life"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-base", type=int, required=True)
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-stop", action="store_true", help="分かれても最後まで打つ")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(x)["seed"] for x in open(a.out)}
    seeds = [a.seed_base + i for i in range(a.games) if a.seed_base + i not in done]
    import multiprocessing as mp
    t0 = time.time()
    with open(a.out, "a") as f, mp.get_context("fork").Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(one, [(s, not a.no_stop) for s in seeds])):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i + 1}/{len(seeds)}] seed={r['seed']} div={r['diverged']} probes={len(r['probes'])} "
                  f"sec={r['sec']} ({time.time() - t0:.0f}s)", flush=True)
    print("DIVERGE_DONE", flush=True)


if __name__ == "__main__":
    main()
