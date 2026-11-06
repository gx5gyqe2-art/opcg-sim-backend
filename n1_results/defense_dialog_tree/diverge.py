"""守りの再監査の局のうち、守りの対話の木（§7.85・`defense_dialog_tree`）で手が変わる局を探す（WP `rs-defense-dialog-tree`）。

`defense_audit.py run` と同じ設定（両席 r29f4・世界 1 本 × 1,280 回・予算なし・詰みの証明 off・random × synth_roles）で
局を今の既定（off）で打ち、相手の手番の窓のうち戦闘の応手でないもの（`kind="window"`・候補 2 つ以上）ごとに、同じ盤面・
同じ opts に `defense_dialog_tree="all"` を足した決定も出して手を比べる（木が掛かった点だけ数える）。ライフ ≤ 1 の点では
`life1` も同じ計算（同じ木）なので、`all` の手をそのまま `life1` の手とみなす。両方の形で分かれたら打ち切る。
分かれなかった局は監査の結果も同じ（決定的）。分かれた局だけ監査を打ち直す（`defense_partial.py`）。
`n1_results/defense_proof/diverge.py` の形。

  python n1_results/defense_dialog_tree/diverge.py --seed-base 4462000 --games 300 --workers 4 --out <jsonl>
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

NET = "opcg_sim/data/learned/nrel_r29f4.npz"
BATTLE = ("SELECT_BLOCKER", "SELECT_COUNTER")


class Diverged(Exception):
    pass


def _life(game, name):
    return len(json.loads(game.hidden_json())["players"][name]["life"])


def one(args):
    seed, = args
    E.engine()
    db = D.load_db()
    spec = E.SeatSpec(NET, sims=E.PLAY_SIMS, budget=E.PLAY_BUDGET)
    la, lb = D.leader_pair(db, seed, "random")
    p1, p2 = D.build_pair(db, la, lb, seed, "synth_roles")
    probes = []
    first = {"life1": None, "all": None}
    windows = [0]

    def obs(game, name, turn, step, out, move):
        if out.get("kind") != "window" or game.turn_player() == name:
            return
        pend = json.loads(game.pending_json())
        if pend.get("action") in BATTLE or len((out.get("stats") or {}).get("N") or []) < 2:
            return
        windows[0] += 1
        opts = json.loads(spec.decide_opts(seed, turn, name, None))
        t0 = time.perf_counter()
        new = json.loads(game.fork().decide(name, json.dumps(dict(opts, defense_dialog_tree="all"))))
        dt = time.perf_counter() - t0
        if not (new.get("window") or {}).get("tree"):
            return
        life = _life(game, name)
        same = new.get("move") == move
        rec = {"step": step, "turn": turn, "seat": name, "life": life, "action": pend.get("action"),
               "message": (pend.get("message") or "")[:60], "n": len(out["stats"]["N"]), "q": out["stats"]["Q"],
               "q_new": new["stats"]["Q"], "move": move, "new": new.get("move"), "same": same, "sec": round(dt, 3)}
        probes.append(rec)
        if not same:
            if first["all"] is None:
                first["all"] = step
            if life <= 1 and first["life1"] is None:
                first["life1"] = step
        if first["all"] is not None and first["life1"] is not None:
            raise Diverged()

    t0 = time.perf_counter()
    winner = turns = None
    try:
        r = DR.run_game(seed, {"p1": spec, "p2": spec}, p1, p2, observer=obs)
        winner, turns = r["winner"], r["turns"]
    except Diverged:
        pass
    except DR.GameAborted as exc:
        winner = f"aborted: {exc}"
    return {"seed": seed, "first": first, "winner": winner, "turns": turns, "windows": windows[0],
            "probes": probes, "sec": round(time.perf_counter() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-base", type=int, required=True)
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(x)["seed"] for x in open(a.out)}
    seeds = [a.seed_base + i for i in range(a.games) if a.seed_base + i not in done]
    import multiprocessing as mp
    t0 = time.time()
    with open(a.out, "a") as f, mp.get_context("fork").Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(one, [(s,) for s in seeds])):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i + 1}/{len(seeds)}] seed={r['seed']} first={r['first']} probes={len(r['probes'])} "
                  f"sec={r['sec']} ({time.time() - t0:.0f}s)", flush=True)
    print("DIVERGE_DONE", flush=True)


if __name__ == "__main__":
    main()
