"""形勢判断が良くなると、探索の選ぶ手は変わるのか（WP `value-to-move`・`docs/search_efficiency_plan.md` §7.22）。

読み取りだけの計測（エンジン・ネット・既定の挙動は変えない）。3 段:

1. ``points``: 波の記録（`record_gen` の dump）の holdout 局（seed % 7 == 0）を、記録と同じ設定（r13d・記録の探索の設定・
   Dirichlet・温度・ε）で同じ seed から打ち直す（`reanalyze` の再生と同じ照合＝各手の `sig` と候補の数が記録と一致するか）。
   選んだメインの判断点（候補 2 つ以上）で、盤面の写し（`fork`）に対して各ネットの
   (i) 160 回・世界 1 本（アリーナの席と同じ組み方）、(ii) 出荷の設定（1,280 回・世界 2 本・枝予算なし・詰みの証明は外す）の
   `Game.decide`（公平・同じ `search_seed`）を回し、選んだ候補（束の代表＝合法手の番号）と方策の最大の候補（根の事前分布を
   束で合算した最大）を書く。局ごとに、そこまでに打った手の列も書く（段 2 が探索なしで盤面を作り直すため）。
2. ``deep``: 段 1 の手の列を探索なしで適用して同じ局面を作り直し（合法手の並びで照合）、深い読み（r13d・出荷の 4 倍の
   5,120 回・世界 4 本・枝予算なし）の根の束ごとの訪問 n・値 q を書く。
3. ``summary``: 一致率・方策と探索の一致・変わった手の良し悪し（深い読みの訪問・値でどちらが良いか）を集計する。

    python tests/scripts/value_to_move.py points --band real --dirs 'vm_data/n_records/n50_w*_p*' \\
        --games-per-shard 32 --pts-per-game 6 --workers 4 --out vm_out/real
    python tests/scripts/value_to_move.py deep --out vm_out/real --workers 1
    python tests/scripts/value_to_move.py summary --out vm_out/real vm_out/synth

ネットは ``--nets name=path,...``（先頭が基準＝r13d）。再開できる（終えた局・点は飛ばす）。
"""
import argparse
import glob
import json
import math
import os
import random
import sys
import time
from multiprocessing import Pool

import numpy as np

if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from opcg_sim.loop import decks as D            # noqa: E402
from opcg_sim.loop import driver as DR          # noqa: E402
from opcg_sim.loop import engine as E           # noqa: E402
from opcg_sim.loop import reanalyze as RA       # noqa: E402
from opcg_sim.loop import record_gen as RG      # noqa: E402

DEFAULT_NETS = ("r13d=opcg_sim/data/learned/nrel_r13d.npz,r19=vm_nets/nrel_r19.npz,"
                "r20=vm_nets/nrel_r20.npz,r20s=vm_nets/nrel_r20s.npz")
#: 測る設定（名前 → opts の追加分）。160＝アリーナの席（`SeatSpec(net, sims=160)`）・prod＝出荷（詰みの証明は外す）。
SETTINGS = {"s160": {"sims": 160}, "prod": {"sims": 1280, "worlds": 2, "budget": 0}}
DEEP = {"sims": 5120, "worlds": 4, "budget": 0}
_G = {}


def parse_nets(s):
    out = []
    for part in s.split(","):
        k, v = part.split("=", 1)
        out.append((k, v))
    return out


def chosen_rep(out):
    """decide の出力 → 選んだ束の代表（合法手の番号）。`record_gen._Recorder` の照合と同じ (sig, don_k)。"""
    legal = (out.get("stats") or {}).get("legal") or []
    sig, k = out.get("sig"), out.get("k")
    if sig is None:
        sig = RG.move_sig(out.get("move") or {})
    for g in out.get("groups") or []:
        rep = legal[g["rep"]]
        if RG.move_sig(rep) == sig and RG._don_k(rep) == k:
            return int(g["rep"])
    return None


def prior_rep(out):
    """根の事前分布を束で合算した最大の束の代表（方策の最大の手・探索なし）。"""
    pr = (out.get("stats") or {}).get("P")
    gs = out.get("groups") or []
    if not pr or not gs:
        return None
    best = max(gs, key=lambda g: sum(float(pr[i]) for i in g["idxs"]))
    return int(best["rep"])


def groups_brief(out):
    return {int(g["rep"]): [float(g["n"]), float(g["q"])] for g in out.get("groups") or []}


def legal_key(out):
    legal = (out.get("stats") or {}).get("legal") or []
    return [json.dumps(RG.move_sig(m), ensure_ascii=False) + "|" + str(RG._don_k(m)) for m in legal]


# ---------------------------------------------------------------- 段 1

def _init_points(nets):
    E.engine()
    _G["nets"] = [(k, E.load_net(p)) for k, p in nets]
    _G["db"] = D.load_db()


def measure(game, name, seed, turn):
    ss = E.search_seed(seed, int(turn), name)
    res = {}
    for key, net in _G["nets"]:
        r = {}
        for sk, extra in SETTINGS.items():
            opts = {"net": net, "search_seed": ss, **extra}
            t0 = time.time()
            o = json.loads(game.fork().decide(name, json.dumps(opts)))
            r[sk] = {"rep": chosen_rep(o), "kind": o.get("kind"), "sec": round(time.time() - t0, 3),
                     "v0": _v0(o)}
            if sk == "s160":
                r["prior"] = prior_rep(o)
                res.setdefault("_legal", legal_key(o))
                res.setdefault("_groups", sorted(int(g["rep"]) for g in o.get("groups") or []))
        res[key] = r
    return res


def _v0(o):
    gs = o.get("groups") or []
    tot = sum(float(g["n"]) for g in gs)
    return round(sum(float(g["n"]) * float(g["q"]) for g in gs) / tot, 4) if tot > 0 else None


class _Stop(Exception):
    pass


def points_game(job):
    """1 局を打ち直し、選んだ点で測る → (局の行, 点の行のリスト, 手の列)。"""
    src, seed, ppg = job
    games, meta, gmeta = _load(src)
    g = games[seed]
    leaders = {int(x["seed"]): x["leaders"] for x in gmeta["games"]}[seed]
    c = g["cols"]
    cand = [int(i) for i in g["rows"] if int(c["kind"][i]) == 0 and int(c["pol_len"][i]) >= 2]
    rng = random.Random(seed * 7919 + 17)
    picks = sorted(rng.sample(cand, min(ppg, len(cand))), key=lambda i: int(c["step"][i]))
    want = {int(c["step"][i]): i for i in picks}
    last = max(want) if want else -1
    by_step = {int(c["step"][i]): int(i) for i in g["rows"]}
    rec_spec, _deep = RA.specs(meta, meta["net"], 1)
    db = _G["db"]
    rec = RA.recorder(seed, db, rec_spec)
    moves, pts = [], []
    st = {"bad_step": None}
    t0 = time.time()

    def obs(game, name, turn, step, out, move):
        rec(game, name, turn, step, out, move)
        i = by_step.get(int(step))
        if i is None or str(c["sig"][i]) != rec.rows["sig"][-1] or int(c["pol_len"][i]) != int(rec.rows["pol_len"][-1]):
            st["bad_step"] = int(step)
            raise _Stop()
        if int(step) in want:
            t1 = time.time()
            m = measure(game, name, seed, turn)
            p = {"seed": seed, "step": int(step), "turn": int(turn), "who": name, "src": src,
                 "leader": leaders[0 if name == "p1" else 1], "n_groups": int(c["pol_len"][i]),
                 "legal": m.pop("_legal"), "groups": m.pop("_groups"), "res": m,
                 "sec": round(time.time() - t1, 2)}
            pts.append(p)

    def post(game, name, turn, step, move, events):
        if step >= 0:
            moves.append(move)
            if int(step) >= last:
                raise _Stop()

    p1, p2 = D.build_pair(db, leaders[0], leaders[1], seed, meta["decks"])
    try:
        DR.run_game(seed, {"p1": rec_spec, "p2": rec_spec}, p1, p2, max_steps=RG.MAX_STEPS,
                    observer=obs, post=post, swap=rec.swap)
    except _Stop:
        pass
    gl = {"game": seed, "src": src, "leaders": leaders, "picks": len(picks), "pts": len(pts),
          "bad_step": st["bad_step"], "sec": round(time.time() - t0, 1)}
    return gl, pts, moves


_LOADED = {}


def _load(src):
    if src not in _LOADED:
        _LOADED.clear()
        _LOADED[src] = RA.load_record(src)
    return _LOADED[src]


def holdout_seeds(src, k):
    gm = json.load(open(os.path.join(src, "meta_games.json"), encoding="utf-8"))
    s = sorted(int(x["seed"]) for x in gm["games"] if int(x["seed"]) % 7 == 0)
    return s[:k]


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    out = []
    for line in open(path, encoding="utf-8"):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def cmd_points(a):
    os.makedirs(a.out, exist_ok=True)
    dirs = sorted(glob.glob(a.dirs))
    per_part = int(math.ceil(a.games_per_shard / 4.0))
    jobs = []
    for d in dirs:
        for sd in holdout_seeds(d, per_part):
            jobs.append((d, sd, a.pts_per_game))
    done = {int(x["game"]) for x in read_jsonl(os.path.join(a.out, "games.jsonl"))}
    # 点の行は局の行より先に書く＝局の行が無い点（書きかけ）は捨てる
    pts = [p for p in read_jsonl(os.path.join(a.out, "points.jsonl")) if int(p["seed"]) in done]
    with open(os.path.join(a.out, "points.jsonl"), "w", encoding="utf-8") as f:
        for p in pts:
            f.write(json.dumps(p) + "\n")
    todo = [j for j in jobs if j[1] not in done]
    if a.limit:
        todo = todo[:a.limit]
    json.dump({"band": a.band, "dirs": dirs, "nets": a.nets, "settings": SETTINGS, "games_per_shard": a.games_per_shard,
               "pts_per_game": a.pts_per_game, "jobs": len(jobs)},
              open(os.path.join(a.out, "meta.json"), "w"), ensure_ascii=False, indent=1)
    print("POINTS_START", json.dumps({"jobs": len(jobs), "done": len(done), "todo": len(todo)}), flush=True)
    t0 = time.time()
    n = 0
    with Pool(a.workers, initializer=_init_points, initargs=(parse_nets(a.nets),)) as pool:
        for gl, ps, moves in pool.imap_unordered(points_game, todo, chunksize=1):
            with open(os.path.join(a.out, "points.jsonl"), "a", encoding="utf-8") as f:
                for p in ps:
                    f.write(json.dumps(p) + "\n")
            with open(os.path.join(a.out, "moves.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps({"game": gl["game"], "src": gl["src"], "leaders": gl["leaders"], "moves": moves}) + "\n")
            with open(os.path.join(a.out, "games.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(gl) + "\n")
            n += 1
            print("POINTS game=%d pts=%d bad_step=%s sec=%.1f (%d/%d %.0fs)"
                  % (gl["game"], gl["pts"], gl["bad_step"], gl["sec"], n, len(todo), time.time() - t0), flush=True)
    print("POINTS_DONE", flush=True)


# ---------------------------------------------------------------- 段 2

def _init_deep(net):
    E.engine()
    _G["deep_net"] = E.load_net(net)
    _G["db"] = D.load_db()


def replay_points(mv, pts, decks, fn):
    """手の列を探索なしで適用して局面を作り直し、点で `fn(game, name, seed, turn, p)` → 行のリスト。"""
    db = _G["db"]
    seed = int(mv["game"])
    leaders = mv["leaders"]
    p1, p2 = D.build_pair(db, leaders[0], leaders[1], seed, decks)
    game = DR.new_game(seed, p1, p2)
    want = {int(p["step"]): p for p in pts}
    out = []
    for step, move in enumerate(mv["moves"]):
        if step in want:
            p = want[step]
            name = json.loads(game.pending_json())["player_id"]
            turn = int(game.turn_count or 0)
            t0 = time.time()
            row = fn(game, name, seed, turn, p)
            # 記録の行と盤面の符号化が同じか（合成デッキ v2 はデッキが PYTHONHASHSEED で変わりうる＝別プロセスでは
            # 同じ seed でも別の局になる・報告 §1）。合法手の並びだけでは同じ uuid の別の札を見逃すので、ここで足す。
            row["ok"] = bool(row.get("ok")) and same_as_record(game, name, p)
            row.update({"seed": seed, "step": step, "sec": round(time.time() - t0, 2)})
            out.append(row)
        name = json.loads(game.pending_json())["player_id"]
        if move.get("kind") == "battle":
            game.apply_battle_action(name, move["action_type"], move.get("card_uuid"))
        else:
            game.apply_game_action(name, move["action_type"], json.dumps(move.get("payload") or {}, ensure_ascii=False))
    return out


def same_as_record(game, name, p):
    games, _meta, _gm = _load(p["src"])
    g = games[int(p["seed"])]
    c = g["cols"]
    i = [int(j) for j in g["rows"] if int(c["step"][j]) == int(p["step"])]
    if len(i) != 1:
        return False
    enc = json.loads(game.encode(name))
    ci = np.zeros(RG.MAX_CI, np.int64)
    src = np.asarray(enc["card_idx"], np.int64)[:RG.MAX_CI]
    ci[:len(src)] = src
    return bool(np.array_equal(ci, c["card_idx"][i[0]])
                and np.allclose(np.asarray(enc["scalars"], np.float32), c["scalars"][i[0]], atol=1e-3))


def _deep_fn(game, name, seed, turn, p):
    opts = {"net": _G["deep_net"], "search_seed": E.search_seed(seed, turn, name), **DEEP}
    o = json.loads(game.fork().decide(name, json.dumps(opts)))
    ok = (name == p["who"] and turn == p["turn"] and legal_key(o) == p["legal"])
    return {"ok": ok, "rep": chosen_rep(o), "groups": groups_brief(o), "v0": _v0(o)}


#: 探索の乱数だけを替えた読み（同じネット・同じ設定）の seed のずらし（世界 i は seed + i なので重ならない大きさ）。
RESEED_XOR = 0x5EED_0F_F5E7_0000


def _reseed_fn(game, name, seed, turn, p):
    ss = E.search_seed(seed, turn, name) ^ RESEED_XOR
    res = {}
    for key, net in _G["nets"]:
        r = {}
        for sk, extra in SETTINGS.items():
            o = json.loads(game.fork().decide(name, json.dumps({"net": net, "search_seed": ss, **extra})))
            r[sk] = {"rep": chosen_rep(o)}
            if sk == "s160":
                r["ok"] = legal_key(o) == p["legal"]
        res[key] = r
    return {"ok": all(v["ok"] for v in res.values()) and name == p["who"], "res": res}


def deep_game(job):
    mv, pts, decks, mode = job
    return replay_points(mv, pts, decks, _deep_fn if mode == "deep" else _reseed_fn)


def _init_any(mode, arg):
    if mode == "deep":
        _init_deep(arg)
    else:
        _init_points(parse_nets(arg))


def cmd_deep(a):
    meta = json.load(open(os.path.join(a.out, "meta.json"), encoding="utf-8"))
    decks = json.load(open(os.path.join(meta["dirs"][0], "meta_n_record.json"), encoding="utf-8"))["decks"]
    pts = read_jsonl(os.path.join(a.out, "points.jsonl"))
    dpath = os.path.join(a.out, a.name + ".jsonl")
    done = {(int(x["seed"]), int(x["step"])) for x in read_jsonl(dpath)}
    by_game = {}
    base = parse_nets(meta["nets"])[0][0]

    def changed(p):
        r = p["res"]
        return any(r[k][sk]["rep"] != r[base][sk]["rep"] for k in r for sk in SETTINGS) or \
            any(r[k]["prior"] != r[base]["prior"] for k in r)
    for p in pts:
        if a.changed_only and not changed(p):
            continue
        if (int(p["seed"]), int(p["step"])) not in done:
            by_game.setdefault(int(p["seed"]), []).append(p)
    mvs = {int(m["game"]): m for m in read_jsonl(os.path.join(a.out, "moves.jsonl"))}
    mode = "reseed" if a.reseed_nets else "deep"
    jobs = [(mvs[s], ps, decks, mode) for s, ps in sorted(by_game.items()) if s in mvs]
    print("DEEP_START", json.dumps({"games": len(jobs), "points": sum(len(j[1]) for j in jobs)}), flush=True)
    t0 = time.time()
    n = 0
    with Pool(a.workers, initializer=_init_any,
              initargs=(mode, a.reseed_nets if mode == "reseed" else a.deep_net)) as pool:
        for rows in pool.imap_unordered(deep_game, jobs, chunksize=1):
            with open(dpath, "a", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            n += 1
            if n % 10 == 0 or n == len(jobs):
                print("DEEP %d/%d games %.0fs" % (n, len(jobs), time.time() - t0), flush=True)
    print("DEEP_DONE", flush=True)


# ---------------------------------------------------------------- 段 3

def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def rate(k, n):
    return {"k": k, "n": n, "rate": round(k / n, 4) if n else None, "ci95": wilson(k, n)}


def summarize_band(out_dir):
    meta = json.load(open(os.path.join(out_dir, "meta.json"), encoding="utf-8"))
    nets = [k for k, _ in parse_nets(meta["nets"])]
    base = nets[0]
    games = read_jsonl(os.path.join(out_dir, "games.jsonl"))
    # 打ち直しが記録から外れた局の点は捨てる（外れる前の点も、記録と同じデッキだった保証が無い）
    bad = {int(g["game"]) for g in games if g["bad_step"] is not None}
    pts = [p for p in read_jsonl(os.path.join(out_dir, "points.jsonl")) if int(p["seed"]) not in bad]
    judges = {}
    reseed = {(int(x["seed"]), int(x["step"])): x for x in read_jsonl(os.path.join(out_dir, "reseed.jsonl"))}
    for fn in sorted(glob.glob(os.path.join(out_dir, "deep*.jsonl"))):
        judges[os.path.basename(fn)[:-6]] = {(int(x["seed"]), int(x["step"])): x for x in read_jsonl(fn)}
    s = {"band": meta["band"], "games": len(games), "diverged_games": sum(1 for g in games if g["bad_step"] is not None),
         "points": len(pts), "deep_bad": {jn: sum(1 for x in jd.values() if not x["ok"]) for jn, jd in judges.items()},
         "n_groups_mean": round(sum(p["n_groups"] for p in pts) / max(1, len(pts)), 2)}
    # 2. 一致率（各ネット × 設定・基準 r13d の同じ設定と）・方策の最大と探索の一致
    agree, pol, self_set = {}, {}, {}
    for k in nets:
        for sk in SETTINGS:
            ok = [p for p in pts if p["res"][k][sk]["rep"] is not None and p["res"][base][sk]["rep"] is not None]
            agree[f"{k}/{sk}"] = rate(sum(p["res"][k][sk]["rep"] == p["res"][base][sk]["rep"] for p in ok), len(ok))
            ok2 = [p for p in ok if p["res"][k]["prior"] is not None]
            pol[f"{k}/{sk}"] = rate(sum(p["res"][k]["prior"] == p["res"][k][sk]["rep"] for p in ok2), len(ok2))
        ok = [p for p in pts if p["res"][k]["prior"] is not None and p["res"][base]["prior"] is not None]
        agree[f"{k}/prior"] = rate(sum(p["res"][k]["prior"] == p["res"][base]["prior"] for p in ok), len(ok))
        ok = [p for p in pts if p["res"][k]["s160"]["rep"] is not None and p["res"][k]["prod"]["rep"] is not None]
        self_set[k] = rate(sum(p["res"][k]["s160"]["rep"] == p["res"][k]["prod"]["rep"] for p in ok), len(ok))
    s["agree_with_base"] = agree
    # 揺らぎの対照: 同じネット・同じ設定で探索の乱数だけを替えた読みとの一致
    noise = {}
    rs = [(p, reseed.get((int(p["seed"]), int(p["step"])))) for p in pts]
    rs = [(p, r) for p, r in rs if r and r["ok"]]
    for k in (rs[0][1]["res"] if rs else {}):
        for sk in SETTINGS:
            ok = [(p, r) for p, r in rs if p["res"][k][sk]["rep"] is not None]
            noise[f"{k}/{sk}"] = rate(sum(p["res"][k][sk]["rep"] == r["res"][k][sk]["rep"] for p, r in ok), len(ok))
            if k != base and base in rs[0][1]["res"]:
                # 乱数を替えた基準と、乱数を替えたこのネット（同じ乱数同士）の一致＝別の seed での agree_with_base
                noise[f"{k}_vs_{base}/{sk}@reseed"] = rate(
                    sum(r["res"][k][sk]["rep"] == r["res"][base][sk]["rep"] for p, r in ok), len(ok))
    s["reseed_agree"] = noise
    s["reseed_points"] = len(rs)
    # 方策の最大が基準と同じ点・違う点に分けた探索の一致（価値と方策のどちらから手が変わるか）
    cond = {}
    for k in nets[1:]:
        for sk in SETTINGS:
            same = [p for p in pts if p["res"][k]["prior"] == p["res"][base]["prior"]]
            diff = [p for p in pts if p["res"][k]["prior"] != p["res"][base]["prior"]]
            cond[f"{k}/{sk}"] = {
                "prior_same": rate(sum(p["res"][k][sk]["rep"] == p["res"][base][sk]["rep"] for p in same), len(same)),
                "prior_diff": rate(sum(p["res"][k][sk]["rep"] == p["res"][base][sk]["rep"] for p in diff), len(diff))}
    s["agree_by_prior"] = cond
    s["prior_eq_search"] = pol
    s["s160_eq_prod"] = self_set
    # 4. 変わった手の良し悪し（深い読み: 訪問の多い方／値の高い方）・深い読みの最善との一致
    for jn, jd in judges.items():
        s["judge_" + jn] = judge(pts, jd, nets, base)
    s["judge_points"] = {jn: len(jd) for jn, jd in judges.items()}
    return s


def judge(pts, deep, nets, base):
    s = {}
    better, deep_eq = {}, {}
    for k in nets:
        for sk in list(SETTINGS) + ["prior"]:
            def rep_of(p, kk):
                return p["res"][kk]["prior"] if sk == "prior" else p["res"][kk][sk]["rep"]
            ds = [(p, deep.get((int(p["seed"]), int(p["step"])))) for p in pts]
            ds = [(p, d) for p, d in ds if d and d["ok"] and rep_of(p, k) is not None]
            deep_eq[f"{k}/{sk}"] = rate(sum(rep_of(p, k) == d["rep"] for p, d in ds), len(ds))
            if k == base:
                continue
            diff = [(p, d) for p, d in ds if rep_of(p, base) is not None and rep_of(p, k) != rep_of(p, base)]
            wn = wq = tie_n = 0
            dq = []
            for p, d in diff:
                gb, gk = d["groups"].get(str(rep_of(p, base))), d["groups"].get(str(rep_of(p, k)))
                if gb is None or gk is None:
                    continue
                if gk[0] > gb[0]:
                    wn += 1
                elif gk[0] == gb[0]:
                    tie_n += 1
                if gk[1] > gb[1]:
                    wq += 1
                dq.append(gk[1] - gb[1])
            m = len(dq)
            better[f"{k}/{sk}"] = {"changed": len(diff), "net_more_visits": rate(wn, m), "visit_ties": tie_n,
                                   "net_higher_q": rate(wq, m),
                                   "dq_mean": round(sum(dq) / m, 4) if m else None,
                                   "dq_median": round(sorted(dq)[m // 2], 4) if m else None}
    s["deep_best_eq"] = deep_eq
    s["changed_move_quality"] = better
    return s


def cmd_summary(a):
    out = {"bands": [summarize_band(d) for d in a.out]}
    txt = json.dumps(out, ensure_ascii=False, indent=1)
    if a.write:
        with open(a.write, "w", encoding="utf-8") as f:
            f.write(txt + "\n")
    print(txt)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("points")
    p.add_argument("--band", required=True)
    p.add_argument("--dirs", required=True, help="記録の dir の glob")
    p.add_argument("--games-per-shard", type=int, default=32, help="シャード（4 本の dir）あたりの holdout 局")
    p.add_argument("--pts-per-game", type=int, default=6)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=0, help="今回回す局の上限（0＝全部）")
    p.add_argument("--nets", default=DEFAULT_NETS)
    p.add_argument("--out", required=True)
    p = sub.add_parser("deep")
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--deep-net", default="opcg_sim/data/learned/nrel_r13d.npz")
    p.add_argument("--name", default="deep", help="書く台帳の名前（<name>.jsonl・判定役のネットを替えるとき）")
    p.add_argument("--reseed-nets", default=None,
                   help="深い読みの代わりに、探索の乱数だけを替えて（RESEED_XOR）これらのネットの 2 設定を読み直す（揺らぎの対照）")
    p.add_argument("--changed-only", action="store_true",
                   help="基準と手が違った点（どれかのネット・どれかの設定）だけ読む")
    p = sub.add_parser("summary")
    p.add_argument("--out", nargs="+", required=True)
    p.add_argument("--write", default=None)
    a = ap.parse_args(argv)
    {"points": cmd_points, "deep": cmd_deep, "summary": cmd_summary}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
