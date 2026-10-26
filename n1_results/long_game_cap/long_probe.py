"""手数の上限に届く長い局を集める（1 回きり・WP long-game-cap・§7.47）。

生成と同じ設定（gen_n52.json の引数・生成役 r22・デッキ real）で打ち、上限だけ EXT 手に上げる。
decide は seed・ターン・席だけで決まる（上限は打ち切るだけ）ので、最初の 400 手は生成と同じ局＝
「400 手で捨てたか」と「上げると決着するか」が 1 回の対局で両方分かる。

    python long_probe.py OUT_DIR SEED_LO SEED_HI [WORKERS]

OUT_DIR/games.jsonl に 1 局 1 行（再開可）・400 手を越えた局は OUT_DIR/moves/SEED.json に手の列。
"""
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time

sys.path.insert(1, ".")
from opcg_sim.loop import record_gen as RG, driver as DR, decks as D  # noqa: E402

EXT = int(os.environ.get("LONG_EXT", "1600"))
CAP = RG.MAX_STEPS  # 400
SEARCH = {"worlds": 1, "select_rule": "q_min_n", "q_min_frac": 0.125, "root_prior_temp": 2.0, "budget": 0}
NET = "opcg_sim/data/learned/nrel_r22.npz"
SKIP_KEYS = ("turn_events",)


def _init():
    RG._init_worker(1280, NET, 0.25, 4, "real", True, 0.03, 0.03, SEARCH, 0.0, None)


def _sig(game, with_turn=True):
    """盤面の指紋（ゾーンの中身・状態・ドン!!・フェイズ・保留中の要求）。turn_events などの記録は外す。"""
    h = json.loads(game.hidden_json())
    m = {k: v for k, v in h["manager"].items() if k not in SKIP_KEYS and (with_turn or k != "turn_count")}
    s = json.dumps([h["players"], m, game.pending_json()], sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(s.encode()).hexdigest()[:16], h


def _cid(h, u):
    if not u:
        return None
    for P in h["players"].values():
        for z, cs in P.items():
            for c in (cs if isinstance(cs, list) else [cs] if isinstance(cs, dict) else []):
                if isinstance(c, dict) and c.get("uuid") == u:
                    return c.get("card_id")
    return None


def play(seed):
    spec, db = RG._G["spec"], RG._G["db"]
    rec = RG._Recorder(seed=seed, db=db, eps_play=0.03, eps_hold=0.03, eps_target=0.0)
    t0 = time.time()
    moves, turn_rows, noops = [], [], {}
    st = {"prev": None, "turn": None, "turn_sigs": {}, "turn_repeat": 0}

    def obs(game, name, turn, step, out, move):
        rec(game, name, turn, step, out, move)
        sig, h = _sig(game)
        st["prev"] = (sig, name, turn)
        if turn != st["turn"]:
            st["turn"] = turn
            ts, _ = _sig(game, with_turn=False)
            if ts in st["turn_sigs"]:
                st["turn_repeat"] += 1
            st["turn_sigs"][ts] = turn
            turn_rows.append({"turn": turn, "step": step, "who": h["manager"]["turn_player"],
                              **{f"{p}_{z}": len(h["players"][p][z]) for p in ("p1", "p2")
                                 for z in ("life", "deck", "hand", "field", "trash")}})
        p = json.loads(game.pending_json())
        u = (move.get("payload") or {}).get("uuid") or move.get("card_uuid")
        moves.append([step, name, turn, p.get("action"), out.get("kind"), move.get("action_type"),
                      _cid(h, u), move.get("kind"), move.get("payload"), move.get("card_uuid")])

    def post(game, name, turn, step, move, events):
        rec.post(game, name, turn, step, move, events)
        if name is None or st["prev"] is None:
            return
        sig, _ = _sig(game)
        if sig == st["prev"][0]:
            k = f"{moves[-1][5]}:{moves[-1][6]}"
            noops[k] = noops.get(k, 0) + 1
            moves[-1].append("NOOP")

    la, lb = D.leader_pair(db, seed, "random")
    p1, p2, kinds = D.build_pair(db, la, lb, seed, "real", with_kinds=True)
    row = {"seed": seed, "leaders": [p1[0], p2[0]],
           "decks": [hashlib.sha1(json.dumps(sorted(x[1])).encode()).hexdigest()[:8] for x in (p1, p2)]}
    try:
        res = DR.run_game(seed, {"p1": spec, "p2": spec}, p1, p2, max_steps=EXT,
                          observer=obs, post=post, swap=rec.swap)
        winner, turns, steps, acts = res["winner"], res["turns"], res["steps"], res["acts"]
        row.update(winner=winner, turns=turns, steps=steps, acts=acts)
        # 400 手の上限での生成の判定（record_gen.play_one と同じ条件）
        reached = steps > CAP or (steps == CAP and winner is None)
        turn400 = moves[CAP - 1][2] if len(moves) >= CAP else turns
        drop400 = reached or winner is None or turns < 4 or min(acts.values()) == 0
        row.update(reached_cap=reached, drop400=drop400, turn_at_400=turn400 if reached else None,
                   ext_capped=steps >= EXT and winner is None)
        h = json.loads(res["game"].hidden_json())
        row["final"] = {p: {z: len(h["players"][p][z]) for z in ("life", "deck", "hand", "field", "trash")}
                        for p in ("p1", "p2")}
    except Exception as exc:  # noqa: BLE001
        row.update(error=f"{type(exc).__name__}: {exc}", steps=len(moves))
    row["turn_steps_max"] = 0
    if turn_rows:
        bounds = [r["step"] for r in turn_rows] + [len(moves)]
        per = [b - a for a, b in zip(bounds, bounds[1:])]
        row["turn_steps_max"] = max(per)
        row["turn_steps_max_at"] = turn_rows[per.index(max(per))]["turn"]
    row.update(noops=noops, turn_sig_repeat=st["turn_repeat"], sec=round(time.time() - t0, 1))
    row["turn_rows"] = turn_rows
    if len(moves) >= CAP or row.get("error"):
        row["moves_file"] = f"moves/{seed}.json"
        row["_moves"] = moves
    return row


def main():
    out, lo, hi = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    os.makedirs(os.path.join(out, "moves"), exist_ok=True)
    path = os.path.join(out, "games.jsonl")
    done = set()
    if os.path.exists(path):
        for line in open(path):
            done.add(json.loads(line)["seed"])
    pool_seeds = range(lo, hi + 1)
    if os.environ.get("SEEDS_FILE"):  # 絞った seed の列（例: ベガパンクの局だけ）から lo〜hi の範囲を取る
        pool_seeds = [s for s in json.load(open(os.environ["SEEDS_FILE"])) if lo <= s <= hi]
    seeds = [s for s in pool_seeds if s not in done]
    print(f"todo {len(seeds)} EXT {EXT}", flush=True)
    with mp.Pool(workers, initializer=_init, maxtasksperchild=20) as pool, open(path, "a") as f:
        for i, row in enumerate(pool.imap_unordered(play, seeds)):
            mv = row.pop("_moves", None)
            if mv is not None:
                json.dump(mv, open(os.path.join(out, row["moves_file"]), "w"), ensure_ascii=False)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(i + 1, row["seed"], row.get("winner"), row.get("turns"), row.get("steps"),
                  row.get("sec"), "CAP" if row.get("reached_cap") else "", row.get("error", ""), flush=True)
    print("PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
