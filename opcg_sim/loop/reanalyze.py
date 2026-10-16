"""記録の局面を深い探索で打ち直し、π の教材を作り直す（再分析・`docs/reports/2026-10-16_reanalysis_pilot.md` §1）。

記録（`record_gen` の dump）の局を、記録と同じ設定（sims・Dirichlet・温度・探索の設定・ε）で同じ seed から打ち直し、
選んだ main 行で `game.fork().decide(...)` を sims × `--mult`・雑音なし・温度なし・ε なしで回す。
記録の候補の並び（`pol` の順）に束ごとの訪問 n・値 q と根の価値 v0 を取り、**再分析した main 行だけ**を、
記録の全列をコピーして `pol_n`／`pol_q`／`pol_v0` だけ差し替えた dump に書く（`pol_p` は記録のまま）。
出力 dir は `n_rel_train.py --in` にそのまま渡せる（訓練コードは変えない・`--pi-teacher q_improved`）。

再生は**記録を作ったコミットの wheel**で回す（新しいエンジンでは手が分岐する）。この器は依存を `record_gen`・`driver`・
`decks`・`engine` の古いコミットにもある口だけに絞ってある＝その worktree へ写して、worktree を cwd にして回す。

    python opcg_sim/loop/reanalyze.py --in n_records/n29_w01 --out re/n29_w01_s0 --mult 4 --games 40 --shard 0/1 \
        --net opcg_sim/data/learned/nrel_r3.npz

点の選び方（局ごと・上限 `--per-game`・優先順・§1.1）: (1) ターン 1〜3 の main 行は全部、(2) 候補に「ターンを終える」と
「攻撃」が両方在る行、(3) エネル（OP15-058）がリーダーの席の行は全部、(4) 残りは乱数（種は局の seed）。
"""
import argparse
import inspect
import json
import os
import random
import sys
import time

import numpy as np

if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from opcg_sim.loop import decks as D            # noqa: E402
from opcg_sim.loop import driver as DR          # noqa: E402
from opcg_sim.loop import engine as E           # noqa: E402
from opcg_sim.loop import record_gen as RG      # noqa: E402

ENEL = "OP15-058"
RULES = ("early", "end_and_attack", "enel", "random")
#: 候補ごとの列（`pol_len` の区切りで並ぶ）。`pol_len`／`pol_chosen`／`pol_v0` は行の列。
ROW_POL_COLS = ("pol_len", "pol_chosen", "pol_v0")


def family(sig):
    at = sig[0] if sig else None
    if at == "ATTACK" or (at == "DON_BOX" and len(sig) > 2 and sig[2]):
        return "attack"
    return "end" if at == "TURN_END" else "other"


def load_record(src):
    """記録 dir → (局ごとの {行の列・候補の列}, meta, 局の meta)。局は seed で引く。"""
    meta = json.load(open(os.path.join(src, "meta_n_record.json"), encoding="utf-8"))
    gmeta = json.load(open(os.path.join(src, "meta_games.json"), encoding="utf-8"))
    games = {}
    for fn in sorted(f for f in os.listdir(src) if f.startswith("n_record_") and f.endswith(".npz")):
        z = np.load(os.path.join(src, fn))
        n = len(z["seed"])
        pol_cols = [k for k in z.files if k.startswith("pol_") and k not in ROW_POL_COLS]
        row_cols = [k for k in z.files if k not in pol_cols]
        lens = z["pol_len"].astype(np.int64)
        ptr = np.concatenate([[0], np.cumsum(lens)])
        cols = {k: z[k] for k in z.files}
        for k in pol_cols:
            if len(cols[k]) != ptr[-1]:
                raise ValueError("%s: 候補の列 %s の長さ %d が候補の数 %d と合わない" % (fn, k, len(cols[k]), ptr[-1]))
        for k in row_cols:
            if len(cols[k]) != n:
                raise ValueError("%s: 行の列 %s の長さ %d が行の数 %d と合わない" % (fn, k, len(cols[k]), n))
        for sd in np.unique(cols["seed"]):
            idx = np.nonzero(cols["seed"] == sd)[0]
            idx = idx[np.argsort(cols["step"][idx], kind="stable")]
            games[int(sd)] = {"cols": cols, "rows": idx, "ptr": ptr, "row_cols": row_cols, "pol_cols": pol_cols}
    return games, meta, gmeta


def select_points(g, leaders, seed, per_game):
    """局の main 行（候補 2 つ以上）から点を選ぶ → [(行の番号, 規則)]（step の昇順）。"""
    c, ptr = g["cols"], g["ptr"]
    by = {r: [] for r in RULES}
    for i in g["rows"]:
        if int(c["kind"][i]) != 0 or int(c["pol_len"][i]) < 2:
            continue
        fams = {family(json.loads(s)) for s in c["pol_sig"][ptr[i]:ptr[i + 1]]}
        if int(c["turn"][i]) <= 3:
            by["early"].append(int(i))
        elif "end" in fams and "attack" in fams:
            by["end_and_attack"].append(int(i))
        elif leaders[int(c["who"][i])] == ENEL:
            by["enel"].append(int(i))
        else:
            by["random"].append(int(i))
    rng = random.Random(seed)
    out = []
    for r in RULES:
        room = per_game - len(out)
        if room <= 0:
            break
        xs = by[r] if len(by[r]) <= room else rng.sample(by[r], room)
        out.extend((i, r) for i in xs)
    return sorted(out, key=lambda x: int(c["step"][x[0]]))


def specs(meta, net, mult):
    """記録の席（`record_gen._init_worker` と同じ組み方）と深い探索の席。"""
    search = dict(meta.get("search") or {})
    kw = {"prune_futile": E.GEN_PRUNE_FUTILE, **search}
    want = {"eps_play": meta.get("eps_play", 0.0), "eps_hold": meta.get("eps_hold", 0.0),
            "eps_target": meta.get("eps_target", 0.0)}
    params = inspect.signature(E.SeatSpec.__init__).parameters
    eps = {k: v for k, v in want.items() if k in params}
    lost = {k: v for k, v in want.items() if k not in params and float(v or 0.0) != 0.0}
    if lost:
        raise SystemExit("このコミットの SeatSpec は %s を受けない（記録は非 0）" % sorted(lost))
    rec = E.SeatSpec(net, sims=int(meta["sims"]), dirichlet_eps=float(meta["dirichlet_eps"]),
                     temp_turns=int(meta["temp_turns"]), **eps, **kw)
    deep = E.SeatSpec(net, sims=int(meta["sims"]) * int(mult), **kw)
    return rec, deep


def recorder(seed, db, spec):
    params = inspect.signature(RG._Recorder.__init__).parameters
    kw = {k: getattr(spec, k, 0.0) for k in ("eps_play", "eps_hold", "eps_target") if k in params}
    return RG._Recorder(seed=seed, db=db, **kw)


class _Stop(Exception):
    pass


def replay_game(seed, g, picks, meta, rec_spec, deep, db, leaders):
    """1 局を打ち直し、選んだ点で深い探索 → (点のリスト, 局の状態)。"""
    c, ptr = g["cols"], g["ptr"]
    by_step = {int(c["step"][i]): int(i) for i in g["rows"]}
    want = {int(c["step"][i]): (i, r) for i, r in picks}
    last = max(want) if want else -1
    rec = recorder(seed, db, rec_spec)
    st = {"bad_step": None, "pts": [], "t_deep": 0.0}

    def obs(game, name, turn, step, out, move):
        rec(game, name, turn, step, out, move)
        i = by_step.get(int(step))
        got_sig, got_len = rec.rows["sig"][-1], rec.rows["pol_len"][-1]
        if i is None or str(c["sig"][i]) != got_sig or int(c["pol_len"][i]) != int(got_len):
            st["bad_step"] = int(step)
            raise _Stop()
        if int(step) in want:
            i, rule = want[int(step)]
            t0 = time.time()
            # `decide` は盤面を変えない（`record_gen` の注記）。`fork` の無い古い wheel では生の局で読む
            # （その後の再生が記録と同じ手を打ち続けるかは、次の行の照合がそのまま確かめる）。
            g_read = game.fork() if hasattr(game, "fork") else game
            o2 = json.loads(g_read.decide(name, deep.decide_opts(seed, int(turn), name, None)))
            st["t_deep"] += time.time() - t0
            p = {"i": int(i), "rule": rule, "step": int(step), "turn": int(turn), "sec": time.time() - t0}
            legal = (out.get("stats") or {}).get("legal") or []
            legal2 = (o2.get("stats") or {}).get("legal") or []
            g2 = {gg["rep"]: gg for gg in (o2.get("groups") or [])}
            reps = [gg["rep"] for gg in (out.get("groups") or [])]
            if legal != legal2 or any(r not in g2 for r in reps) or len(reps) != int(c["pol_len"][i]):
                p["unmapped"] = True
            else:
                n2 = [float(g2[r]["n"]) for r in reps]
                q2 = [float(g2[r]["q"]) for r in reps]
                tot = sum(n2)
                p.update({"n": n2, "q": q2, "v0": (sum(a * b for a, b in zip(n2, q2)) / tot) if tot > 0 else 0.0})
                pr = (o2.get("stats") or {}).get("P")
                p["p_deep"] = [sum(float(pr[k]) for k in gg["idxs"]) for gg in out["groups"]] if pr else None
            st["pts"].append(p)
        if int(step) >= last:
            raise _Stop()

    t0 = time.time()
    p1, p2 = D.build_pair(db, leaders[0], leaders[1], seed, meta["decks"])
    try:
        DR.run_game(seed, {"p1": rec_spec, "p2": rec_spec}, p1, p2, max_steps=RG.MAX_STEPS,
                    observer=obs, swap=rec.swap)
    except _Stop:
        pass
    st["t_replay"] = time.time() - t0 - st["t_deep"]
    return st


def write_dump(out_dir, games, results, meta, gmeta, args, sel_counts):
    """再分析した main 行だけの dump（記録の全列をコピー・`pol_n`／`pol_q`／`pol_v0` だけ差し替え）と meta。"""
    os.makedirs(out_dir, exist_ok=True)
    buf = None
    n_rows = 0
    for seed, st in results:
        g = games[seed]
        c, ptr = g["cols"], g["ptr"]
        ok = [p for p in st["pts"] if not p.get("unmapped")]
        if not ok:
            continue
        if buf is None:
            buf = {k: [] for k in g["row_cols"] + g["pol_cols"] + (["pol_v0"] if "pol_v0" not in c else [])}
        for p in ok:
            i = p["i"]
            for k in g["row_cols"]:
                buf[k].append(c[k][i:i + 1])
            for k in g["pol_cols"]:
                v = c[k][ptr[i]:ptr[i + 1]]
                if k == "pol_n":
                    v = np.asarray(p["n"], c[k].dtype)
                elif k == "pol_q":
                    v = np.asarray(p["q"], c[k].dtype)
                buf[k].append(v)
            if "pol_v0" in c:
                buf["pol_v0"][-1] = np.asarray([p["v0"]], c["pol_v0"].dtype)
            else:
                buf["pol_v0"].append(np.asarray([p["v0"]], np.float16))
            n_rows += 1
    tag = args.shard.replace("/", "of") if args.shard else "0of1"
    if buf:
        tmp = os.path.join(out_dir, ".n_record_00000.tmp.npz")
        np.savez_compressed(tmp, **{k: np.concatenate(v) for k, v in buf.items()})
        os.replace(tmp, os.path.join(out_dir, "n_record_00000.npz"))
    seeds = {s for s, st in results if any(not p.get("unmapped") for p in st["pts"])}
    with open(os.path.join(out_dir, "meta_games.json"), "w", encoding="utf-8") as f:
        json.dump({**{k: v for k, v in gmeta.items() if k != "games"},
                   "games": [x for x in gmeta["games"] if int(x["seed"]) in seeds]}, f, ensure_ascii=False)
    m2 = dict(meta)
    m2.update({"games": len(seeds), "rows": n_rows, "main_rows": n_rows})
    m2["reanalyzed"] = {"mult": args.mult, "points": n_rows, "games": len(results),
                        "diverged": sum(1 for _s, st in results if st["bad_step"] is not None),
                        "unmapped": sum(1 for _s, st in results for p in st["pts"] if p.get("unmapped")),
                        "selection": sel_counts, "per_game": args.per_game, "shard": tag,
                        "record_dir": os.path.abspath(args.src), "net": args.net or meta.get("net"),
                        "engine_commit": args.engine_commit}
    with open(os.path.join(out_dir, "meta_n_record.json"), "w", encoding="utf-8") as f:
        json.dump(m2, f, ensure_ascii=False)


def summarize(dirs, c_scale=0.3, n_min=1.0, c_visit=50.0):
    """パイロットの物差し（§1.3）: 再生率・diverged／unmapped・秒・深い最善 ≠ 記録の最善・π' のエントロピー。"""
    from opcg_sim.learned.train.n_rel_train import q_improved_pi, seg_entropy, visits_pi
    games, pts = [], []
    for d in dirs:
        for line in open(os.path.join(d, "points.jsonl"), encoding="utf-8"):
            x = json.loads(line)
            (games if "game" in x else pts).append(x)
    ok = [p for p in pts if not p.get("unmapped")]

    def ent(xs, which):
        if not xs:
            return None
        lens = np.array([len(p["n"]) for p in xs])
        if which == "rec_visits":
            pi, ln = visits_pi(lens, np.concatenate([p["rec_n"] for p in xs])), lens
        elif which == "deep_visits":
            pi, ln = visits_pi(lens, np.concatenate([p["n"] for p in xs])), lens
        else:
            key_n, key_q = ("rec_n", "rec_q") if which == "rec_pi" else ("n", "q")
            pr = np.concatenate([p["rec_p"] if p.get("rec_p") else [1.0] * len(p["n"]) for p in xs])
            pi = q_improved_pi(lens, np.concatenate([p[key_n] for p in xs]), np.concatenate([p[key_q] for p in xs]), pr,
                               v0=(np.array([p["v0"] for p in xs]) if which == "deep_pi" else None),
                               n_min=n_min, c_visit=c_visit, c_scale=c_scale)
            ln = lens
        return round(float(np.mean(seg_entropy(ln, pi))), 4)

    def blk(xs):
        if not xs:
            return {"n": 0}
        diff = [int(np.argmax(p["n"])) != int(np.argmax(p["rec_n"])) for p in xs]
        return {"n": len(xs), "deep_best_differs": round(float(np.mean(diff)), 4),
                "entropy": {k: ent(xs, k) for k in ("rec_visits", "deep_visits", "rec_pi", "deep_pi")},
                "sec_per_point": round(float(np.mean([p["sec"] for p in xs])), 3),
                "cands_mean": round(float(np.mean([len(p["n"]) for p in xs])), 2)}
    replayed = [g for g in games if g["bad_step"] is None]
    out = {"dirs": list(dirs), "games": len(games), "replay_rate": round(len(replayed) / len(games), 4) if games else None,
           "diverged_games": len(games) - len(replayed), "picked": sum(g["picks"] for g in games),
           "points": len(pts), "unmapped": len(pts) - len(ok), "lost_to_divergence": sum(g["picks"] - g["pts"] for g in games),
           "sec_replay_per_game": round(float(np.mean([g["t_replay"] for g in games])), 2) if games else None,
           "sec_deep_per_game": round(float(np.mean([g["t_deep"] for g in games])), 2) if games else None,
           "all": blk(ok), "by_rule": {r: blk([p for p in ok if p["rule"] == r]) for r in RULES},
           "c_scale": c_scale}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="src", help="記録の dir（record_gen の出力）")
    ap.add_argument("--out", help="再分析の dump を書く dir")
    ap.add_argument("--summarize", nargs="+", default=None, help="再分析の出力 dir 群を集計して json を印字（§1.3）")
    ap.add_argument("--mult", type=int, default=4, help="深い探索の sims の倍率")
    ap.add_argument("--games", type=int, default=0, help="先頭から何局（0＝全部）")
    ap.add_argument("--shard", default="", help="i/n＝局を n 分割した i 番目")
    ap.add_argument("--per-game", type=int, default=12)
    ap.add_argument("--net", default=None, help="meta のネットの置き場が手元に無いときの上書き")
    ap.add_argument("--engine-commit", default=None, help="再生に使った wheel のコミット（meta に記録するだけ）")
    a = ap.parse_args(argv)
    if a.summarize:
        print(json.dumps(summarize(a.summarize), ensure_ascii=False, indent=1))
        return 0
    if not a.src or not a.out:
        ap.error("--in と --out が要る")
    games, meta, gmeta = load_record(a.src)
    seeds = sorted(games)
    if a.games:
        seeds = seeds[:a.games]
    if a.shard:
        i, n = (int(x) for x in a.shard.split("/"))
        seeds = [s for k, s in enumerate(seeds) if k % n == i]
    E.engine()
    net = a.net or meta.get("net")
    rec_spec, deep = specs(meta, net, a.mult)
    db = D.load_db()
    lead = {int(x["seed"]): x["leaders"] for x in gmeta["games"]}
    os.makedirs(a.out, exist_ok=True)
    sel = {r: 0 for r in RULES}
    results = []
    t0 = time.time()
    with open(os.path.join(a.out, "points.jsonl"), "w", encoding="utf-8") as fp:
        for seed in seeds:
            g = games[seed]
            leaders = lead[seed]
            picks = select_points(g, leaders, seed, a.per_game)
            for _i, r in picks:
                sel[r] += 1
            st = replay_game(seed, g, picks, meta, rec_spec, deep, db, leaders)
            results.append((seed, st))
            c, ptr = g["cols"], g["ptr"]
            for p in st["pts"]:
                i = p["i"]
                p.update({"seed": seed, "who": int(c["who"][i]), "leader": leaders[int(c["who"][i])],
                          "rec_n": [float(x) for x in c["pol_n"][ptr[i]:ptr[i + 1]]],
                          "rec_q": [float(x) for x in c["pol_q"][ptr[i]:ptr[i + 1]]],
                          "rec_p": ([float(x) for x in c["pol_p"][ptr[i]:ptr[i + 1]]] if "pol_p" in c else None),
                          "fams": [family(json.loads(s)) for s in c["pol_sig"][ptr[i]:ptr[i + 1]]]})
                fp.write(json.dumps(p) + "\n")
            lost = len(picks) - len(st["pts"])
            print("REANALYZE seed=%d picks=%d pts=%d lost=%d bad_step=%s replay=%.1fs deep=%.1fs (total %.0fs)"
                  % (seed, len(picks), len(st["pts"]), lost, st["bad_step"], st["t_replay"], st["t_deep"],
                     time.time() - t0), flush=True)
            fp.write(json.dumps({"game": seed, "picks": len(picks), "pts": len(st["pts"]), "bad_step": st["bad_step"],
                                 "last_pick_step": max([int(c["step"][i]) for i, _r in picks], default=-1),
                                 "t_replay": st["t_replay"], "t_deep": st["t_deep"]}) + "\n")
    write_dump(a.out, games, results, meta, gmeta, a, sel)
    print("REANALYZE_DONE " + json.dumps({"games": len(results), "selection": sel}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
