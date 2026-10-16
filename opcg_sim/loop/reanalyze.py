"""記録の局面を深い探索で打ち直し、π の教材を作り直す（再分析・`docs/reports/2026-10-16_reanalysis_pilot.md` §1）。

記録（`record_gen` の dump）の局を、記録と同じ設定（sims・Dirichlet・温度・探索の設定・ε）で同じ seed から打ち直し、
選んだ main 行で `game.fork().decide(...)` を sims × `--mult`・雑音なし・温度なし・ε なしで回す。
記録の候補の並び（`pol` の順）に束ごとの訪問 n・値 q と根の価値 v0 を取り、**再分析した main 行だけ**を、
記録の全列をコピーして `pol_n`／`pol_q`／`pol_v0` だけ差し替えた dump に書く（`pol_p` は記録のまま）。
出力 dir は `n_rel_train.py --in` にそのまま渡せる（訓練コードは変えない・`--pi-teacher q_improved`）。

再生は**記録を作ったコミットの wheel**で回す（新しいエンジンでは手が分岐する）。この器は依存を `record_gen`・`driver`・
`decks`・`engine` の古いコミットにもある口だけに絞ってある＝その worktree へ写して、worktree を cwd にして回す。

    python opcg_sim/loop/reanalyze.py --in n_records/n29_w01 --out re/n29_w01 --mult 4 --games 200 --quota 4,4,2,2 \
        --net opcg_sim/data/learned/nrel_r3.npz

点の選び方（局ごと・規則ごとの枠 `--quota`・既定 4,4,2,2・司令塔の決定 2026-10-16）: (1) ターン 1〜3 の main 行、
(2) 候補に「ターンを終える」と「攻撃」が両方在る行、(3) エネル（OP15-058）がリーダーの席の行、(4) 乱数。
規則の順に、まだ取っていない行から枠の数だけ取る（候補が枠より多ければ乱数・種は局の seed）。エネルの席が無い局の (3) の枠と、
(1)〜(3) で候補が足りずに余った枠は (4) に回す。(4) はまだ取っていない全部の main 行から引く。

**再開できる**: 局ごとに `n_record_<seed>.npz`（点が 1 つも読めなかった局は書かない）と `points.jsonl` の局の行を書き、
`meta_n_record.json`／`meta_games.json` をその都度書き直す。`points.jsonl` に局の行が在る seed は飛ばす（同じコマンドで再開）。
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


def select_points(g, leaders, seed, quota):
    """局の main 行（候補 2 つ以上）から規則ごとの枠で点を選ぶ → [(行の番号, 規則)]（step の昇順）。"""
    c, ptr = g["cols"], g["ptr"]
    pool = {r: [] for r in RULES}
    for i in g["rows"]:
        if int(c["kind"][i]) != 0 or int(c["pol_len"][i]) < 2:
            continue
        fams = {family(json.loads(s)) for s in c["pol_sig"][ptr[i]:ptr[i + 1]]}
        if int(c["turn"][i]) <= 3:
            pool["early"].append(int(i))
        if "end" in fams and "attack" in fams:
            pool["end_and_attack"].append(int(i))
        if leaders[int(c["who"][i])] == ENEL:
            pool["enel"].append(int(i))
        pool["random"].append(int(i))
    rng = random.Random(seed)
    q = dict(zip(RULES, quota))
    taken, out = set(), []
    for r in RULES:
        k = q[r]
        xs = [i for i in pool[r] if i not in taken]
        pick = xs if len(xs) <= k else rng.sample(xs, k)
        out.extend((i, r) for i in pick)
        taken.update(pick)
        if r != "random":
            q["random"] += k - len(pick)
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


def write_game(out_dir, g, st):
    """1 局ぶんの dump（再分析した main 行だけ・記録の全列をコピー・`pol_n`／`pol_q`／`pol_v0` だけ差し替え）→ 行の数。"""
    c, ptr = g["cols"], g["ptr"]
    ok = [p for p in st["pts"] if not p.get("unmapped")]
    if not ok:
        return 0
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
    seed = int(c["seed"][ok[0]["i"]])
    tmp = os.path.join(out_dir, ".n_record_%d.tmp.npz" % seed)
    np.savez_compressed(tmp, **{k: np.concatenate(v) for k, v in buf.items()})
    os.replace(tmp, os.path.join(out_dir, "n_record_%d.npz" % seed))
    return len(ok)


def read_progress(out_dir):
    """`points.jsonl` → (局の行, 点の行)。局の行が無い局（途中で止まった局）の点は捨てる。"""
    path = os.path.join(out_dir, "points.jsonl")
    if not os.path.exists(path):
        return [], []
    games, pts = [], []
    for line in open(path, encoding="utf-8"):
        try:
            x = json.loads(line)
        except json.JSONDecodeError:
            continue                                 # 書きかけの最後の行
        (games if "game" in x else pts).append(x)
    done = {int(x["game"]) for x in games}
    return games, [p for p in pts if int(p["seed"]) in done]


def write_meta(out_dir, meta, gmeta, args, games_done, pts_done):
    """`meta_n_record.json`／`meta_games.json` を、ここまでに終えた局から書き直す（原子的に）。"""
    seeds = {int(p["seed"]) for p in pts_done if not p.get("unmapped")}
    sel = {r: sum(int((x.get("sel") or {}).get(r, 0)) for x in games_done) for r in RULES}
    pts_rule = {r: sum(1 for p in pts_done if p["rule"] == r and not p.get("unmapped")) for r in RULES}
    n_rows = sum(1 for p in pts_done if not p.get("unmapped"))
    m2 = dict(meta)
    m2.update({"games": len(seeds), "rows": n_rows, "main_rows": n_rows})
    m2["reanalyzed"] = {"mult": args.mult, "points": n_rows, "points_by_rule": pts_rule, "games": len(games_done),
                        "diverged": sum(1 for x in games_done if x["bad_step"] is not None),
                        "unmapped": sum(1 for p in pts_done if p.get("unmapped")),
                        "lost_to_divergence": sum(int(x["picks"]) - int(x["pts"]) for x in games_done),
                        "selection": sel, "quota": list(args.quota), "shard": args.shard or "0/1",
                        "record_dir": os.path.abspath(args.src), "net": args.net or meta.get("net"),
                        "engine_commit": args.engine_commit}
    for name, obj in (("meta_n_record.json", m2),
                      ("meta_games.json", {**{k: v for k, v in gmeta.items() if k != "games"},
                                           "games": [x for x in gmeta["games"] if int(x["seed"]) in seeds]})):
        tmp = os.path.join(out_dir, "." + name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False)
        os.replace(tmp, os.path.join(out_dir, name))


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


def merge(orig, rean_dirs, out):
    """元の波の dir（`record_gen` の出力）の全行を写し、再分析した main 行だけ `pol_n`／`pol_q`（と、元に列が在れば `pol_v0`）を
    再分析の値に差し替えた dump を `out` に書く（r11b の教材・`docs/n_loop_ops.md` §7.6）。照合は (seed, step) と、候補の数・
    候補の並び（`pol_sig`）。差し替えない行・列はビットのまま。元に `pol_v0` が無い波（波 29）は列を足さない
    （訓練は訪問で重み付けた q の平均で補う＝再分析の v0 と同じ式）。"""
    rep = {}
    for d in rean_dirs:
        for fn_ in sorted(f for f in os.listdir(d) if f.startswith("n_record_") and f.endswith(".npz")):
            z = np.load(os.path.join(d, fn_))
            ptr = np.concatenate([[0], np.cumsum(z["pol_len"].astype(np.int64))])
            for i in range(len(z["seed"])):
                rep[(int(z["seed"][i]), int(z["step"][i]))] = (
                    z["pol_n"][ptr[i]:ptr[i + 1]], z["pol_q"][ptr[i]:ptr[i + 1]], z["pol_sig"][ptr[i]:ptr[i + 1]],
                    float(z["pol_v0"][i]))
    os.makedirs(out, exist_ok=True)
    n_rep = n_rows = 0
    used = set()
    for fn_ in sorted(f for f in os.listdir(orig) if f.startswith("n_record_") and f.endswith(".npz")):
        z = np.load(os.path.join(orig, fn_))
        cols = {k: z[k].copy() for k in z.files}
        ptr = np.concatenate([[0], np.cumsum(cols["pol_len"].astype(np.int64))])
        for i in range(len(cols["seed"])):
            key = (int(cols["seed"][i]), int(cols["step"][i]))
            if key not in rep:
                continue
            n2, q2, sig2, v0 = rep[key]
            a, b = ptr[i], ptr[i + 1]
            if b - a != len(n2) or not np.array_equal(cols["pol_sig"][a:b], sig2):
                raise ValueError("%s: seed=%d step=%d の候補が再分析と合わない" % (fn_, key[0], key[1]))
            cols["pol_n"][a:b] = n2
            cols["pol_q"][a:b] = q2
            if "pol_v0" in cols:
                cols["pol_v0"][i] = v0
            used.add(key)
            n_rep += 1
        n_rows += len(cols["seed"])
        tmp = os.path.join(out, "." + fn_ + ".tmp.npz")
        np.savez_compressed(tmp, **cols)
        os.replace(tmp, os.path.join(out, fn_))
    if len(used) != len(rep):
        raise ValueError("再分析の行 %d のうち %d 行が元の波に無い（波・シャードの取り違え）" % (len(rep), len(rep) - len(used)))
    m = json.load(open(os.path.join(orig, "meta_n_record.json"), encoding="utf-8"))
    m["reanalysis_merged"] = {"rean_dirs": [os.path.abspath(d) for d in rean_dirs], "rows_replaced": n_rep, "rows": n_rows}
    with open(os.path.join(out, "meta_n_record.json"), "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False)
    for extra in ("meta_games.json",):
        if os.path.exists(os.path.join(orig, extra)):
            with open(os.path.join(orig, extra), "rb") as fi, open(os.path.join(out, extra), "wb") as fo:
                fo.write(fi.read())
    return {"rows": n_rows, "rows_replaced": n_rep}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="src", help="記録の dir（record_gen の出力）")
    ap.add_argument("--out", help="再分析の dump を書く dir")
    ap.add_argument("--summarize", nargs="+", default=None, help="再分析の出力 dir 群を集計して json を印字（§1.3）")
    ap.add_argument("--result", default=None, help="再分析の出力 dir 1 つから RESULT.json（`docs/n_loop_ops.md` §5・§7.5）を印字")
    ap.add_argument("--job", default=None, help="--result の job 名（rean-n{W}-w{XX}）")
    ap.add_argument("--record", default=None, help="--result に書く記録の出どころ（ブランチ:パス）")
    ap.add_argument("--target-games", type=int, default=200, help="--result: これだけ終えたら status=done")
    ap.add_argument("--merge", nargs="+", default=None,
                    help="再分析の出力 dir 群を、--in の元の波の dir の全行へ差し込んで --out に書く（r11b の教材・§7.6）")
    ap.add_argument("--mult", type=int, default=4, help="深い探索の sims の倍率")
    ap.add_argument("--games", type=int, default=0, help="先頭から何局（0＝全部）")
    ap.add_argument("--shard", default="", help="i/n＝局を n 分割した i 番目")
    ap.add_argument("--quota", default="4,4,2,2",
                    help="規則ごとの枠（ターン 1〜3／終える手と攻撃／エネルの席／乱数）。合計が 1 局の点の数")
    ap.add_argument("--net", default=None, help="meta のネットの置き場が手元に無いときの上書き")
    ap.add_argument("--engine-commit", default=None, help="再生に使った wheel のコミット（meta に記録するだけ）")
    a = ap.parse_args(argv)
    if a.summarize:
        print(json.dumps(summarize(a.summarize), ensure_ascii=False, indent=1))
        return 0
    if a.result:
        sm = summarize([a.result])
        m = json.load(open(os.path.join(a.result, "meta_n_record.json"), encoding="utf-8"))["reanalyzed"]
        print(json.dumps({"job": a.job, "role": "reanalyze", "status": "done" if sm["games"] >= a.target_games else "partial",
                          "inputs": {"record": a.record, "engine_commit": a.engine_commit, "net": a.net, "mult": m["mult"],
                                     "games": a.target_games, "quota": m.get("quota")},
                          "outputs": {"dir": os.path.relpath(a.result, os.getcwd())},
                          "metrics": {"games": sm["games"], "points": m["points"], "diverged": m["diverged"],
                                      "unmapped": m["unmapped"], "lost_to_divergence": m.get("lost_to_divergence"),
                                      "points_by_rule": m.get("points_by_rule"), "replay_rate": sm["replay_rate"],
                                      "deep_best_differs": {r: o.get("deep_best_differs") for r, o in sm["by_rule"].items()},
                                      "deep_best_differs_all": sm["all"].get("deep_best_differs"),
                                      "sec_replay_per_game": sm["sec_replay_per_game"],
                                      "sec_deep_per_game": sm["sec_deep_per_game"]},
                          "commit": a.engine_commit, "notes": ""}, ensure_ascii=False, indent=1))
        return 0
    if not a.src or not a.out:
        ap.error("--in と --out が要る")
    if a.merge:
        print("REANALYZE_MERGE " + json.dumps(merge(a.src, a.merge, a.out)), flush=True)
        return 0
    a.quota = [int(x) for x in a.quota.split(",")]
    if len(a.quota) != len(RULES) or min(a.quota) < 0:
        ap.error("--quota は %d 個の 0 以上の整数" % len(RULES))
    games, meta, gmeta = load_record(a.src)
    seeds = sorted(games)
    if a.games:
        seeds = seeds[:a.games]
    if a.shard:
        i, n = (int(x) for x in a.shard.split("/"))
        seeds = [s for k, s in enumerate(seeds) if k % n == i]
    os.makedirs(a.out, exist_ok=True)
    games_done, pts_done = read_progress(a.out)
    done = {int(x["game"]) for x in games_done}
    # 途中で止まった局の点の行を捨てて書き直す（再開）
    with open(os.path.join(a.out, ".points.jsonl.tmp"), "w", encoding="utf-8") as f:
        for x in pts_done + games_done:
            f.write(json.dumps(x) + "\n")
    os.replace(os.path.join(a.out, ".points.jsonl.tmp"), os.path.join(a.out, "points.jsonl"))
    E.engine()
    net = a.net or meta.get("net")
    rec_spec, deep = specs(meta, net, a.mult)
    db = D.load_db()
    lead = {int(x["seed"]): x["leaders"] for x in gmeta["games"]}
    t0 = time.time()
    todo = [s for s in seeds if s not in done]
    print("REANALYZE_START " + json.dumps({"games": len(seeds), "done": len(seeds) - len(todo), "todo": len(todo)}), flush=True)
    for seed in todo:
        g = games[seed]
        leaders = lead[seed]
        picks = select_points(g, leaders, seed, a.quota)
        sel = {r: sum(1 for _i, rr in picks if rr == r) for r in RULES}
        st = replay_game(seed, g, picks, meta, rec_spec, deep, db, leaders)
        c, ptr = g["cols"], g["ptr"]
        for p in st["pts"]:
            i = p["i"]
            p.update({"seed": seed, "who": int(c["who"][i]), "leader": leaders[int(c["who"][i])],
                      "rec_n": [float(x) for x in c["pol_n"][ptr[i]:ptr[i + 1]]],
                      "rec_q": [float(x) for x in c["pol_q"][ptr[i]:ptr[i + 1]]],
                      "rec_p": ([float(x) for x in c["pol_p"][ptr[i]:ptr[i + 1]]] if "pol_p" in c else None),
                      "fams": [family(json.loads(s)) for s in c["pol_sig"][ptr[i]:ptr[i + 1]]]})
        write_game(a.out, g, st)
        gl = {"game": seed, "picks": len(picks), "pts": len(st["pts"]), "bad_step": st["bad_step"], "sel": sel,
              "last_pick_step": max([int(c["step"][i]) for i, _r in picks], default=-1),
              "t_replay": st["t_replay"], "t_deep": st["t_deep"]}
        with open(os.path.join(a.out, "points.jsonl"), "a", encoding="utf-8") as fp:
            for p in st["pts"]:
                fp.write(json.dumps(p) + "\n")
            fp.write(json.dumps(gl) + "\n")
            fp.flush()
            os.fsync(fp.fileno())
        games_done.append(gl)
        pts_done.extend(st["pts"])
        write_meta(a.out, meta, gmeta, a, games_done, pts_done)
        print("REANALYZE seed=%d picks=%d pts=%d lost=%d bad_step=%s replay=%.1fs deep=%.1fs (done %d/%d total %.0fs)"
              % (seed, len(picks), len(st["pts"]), len(picks) - len(st["pts"]), st["bad_step"], st["t_replay"],
                 st["t_deep"], len(games_done), len(seeds), time.time() - t0), flush=True)
    write_meta(a.out, meta, gmeta, a, games_done, pts_done)
    sel = {r: sum(int((x.get("sel") or {}).get(r, 0)) for x in games_done) for r in RULES}
    print("REANALYZE_DONE " + json.dumps({"games": len(games_done), "selection": sel}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
