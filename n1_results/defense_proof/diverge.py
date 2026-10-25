"""守りの再監査の局のうち、守りの証明（§7.51・`defense_proof`）で手が変わる局を探す（WP `rs-defense-proof`）。

`defense_audit.py run` と同じ設定（両席 r13d・世界 1 本 × 1,280 回・予算なし・詰みの証明 off・random × synth_roles）で
局を今の既定（守りの証明 off）で打ち、相手の手番の窓（自分のライフ ≤ 2・`kind="window"`）ごとに、同じ盤面・同じ opts に
`defense_proof` を足した決定も出して手を比べる。最初に手が分かれた点で打ち切る（`--no-stop` で最後まで）。
分かれなかった局は 2 つの形で同じ局（決定的）＝監査の結果も同じ。分かれた局だけ監査を打ち直す（`defense_partial.py`）。
`rs-window-tree-blocker` の `diverge.py` の形。

  python n1_results/defense_proof/diverge.py --seed-base 3351000 --games 300 --workers 4 --out <jsonl>
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


def _life(game, name):
    return len(json.loads(game.hidden_json())["players"][name]["life"])


def one(args):
    seed, stop = args
    E.engine()
    db = D.load_db()
    spec = E.SeatSpec(NET, sims=E.PLAY_SIMS, budget=E.PLAY_BUDGET)
    la, lb = D.leader_pair(db, seed, "random")
    p1, p2 = D.build_pair(db, la, lb, seed, "synth_roles")
    probes = []
    windows = [0]

    def obs(game, name, turn, step, out, move):
        if out.get("kind") != "window" or game.turn_player() == name:
            return
        windows[0] += 1
        life = _life(game, name)
        if life > 2 or len((out.get("stats") or {}).get("N") or []) < 2:
            return
        opts = json.loads(spec.decide_opts(seed, turn, name, None))
        t0 = time.perf_counter()
        new = json.loads(game.fork().decide(name, json.dumps(dict(opts, defense_proof=True))))
        dt = time.perf_counter() - t0
        info = new.get("defense_proof")
        if info is None:
            return
        pend = json.loads(game.pending_json())
        rec = {"step": step, "turn": turn, "seat": name, "life": life, "action": pend.get("action"),
               "n": len(out["stats"]["N"]), "q": out["stats"]["Q"], "move": move, "new": new.get("move"),
               "info": info, "same": new.get("move") == move, "sec": round(dt, 3)}
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
    return {"seed": seed, "diverged": diverged, "winner": winner, "turns": turns, "windows": windows[0],
            "probes": probes, "sec": round(time.perf_counter() - t0, 1)}


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
