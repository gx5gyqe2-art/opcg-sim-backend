"""メインの手でも「先を読めていない損」があるか（WP `main-read`・`docs/search_efficiency_plan.md` §7.68・台帳 2+479）。

読み取りだけの計測（エンジン・ネット・既定の挙動は変えない）。1 回きりの計測＝テストは足さない（CLAUDE.md の理論の道具の規約）。
`window_read_all.py`（§7.67 段 1）・`don_read.py`（§7.66 段 1）をメインの手に一般化したもの。

記録（両席 r27・出荷の設定）は `window_read_all` と同じ 3 つ。メインの判断点（木で決まる点＝記録の `kind == "main"`・続き・詰みの証明・
マリガンは除く・候補 2 つ以上・勝負の決まった局面は除く）で、打った手 a と根の次点 b の**手元の資源の減り方が違う**点を種類ごとに抜く:

- ``play``＝キャラを出す 対 出さない／``don``＝ドンを付ける（多く付ける）対 付けない（少なく付ける）／
  ``activate``＝起動メインを使う 対 使わない／``discard``＝手札を捨てる効果のある札を使う 対 使わない／``event``＝イベントを使う 対 使わない

1. ``points``: 種類ごとに局の記録だけで決まる順位で抜く（hold_bias とマッチアップを交互に）。``points.jsonl``。
2. ``eval``: (a) 記録の根の値と手（a と b）、(b) 4 倍の回数（5,120 回・世界 4 本）の木の手、(c) a と b それぞれを当て、
   次の自分のターンの頭まで両席とも出荷の設定で進めた盤面の値（世界 8 本・審判 160 回の根の値とネットの生の値）。``evals*.jsonl``。
3. ``judge``: 今の手 a と、(b) の手（a と違えば）・(c) が選ぶ手（b を選べば）を 1 点 16 通りで打ち比べる（§7.60 と同じ判定）。
   差は「別の読みの手 − 今の手」。``judge*.jsonl``・``--shard i/n``。
4. ``summary``: 種類ごとの表。``summary.json``。
"""
import argparse
import glob
import json
import os
import sys
import time
import zlib
from multiprocessing import get_context

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blunder_audit as BA  # noqa: E402
import don_read as DRd  # noqa: E402
import matchup_kr as KR  # noqa: E402
import move_judge as MJ  # noqa: E402
import window_read_all as W  # noqa: E402

from opcg_sim.loop import driver as DR, engine as E  # noqa: E402

OUT_DIR = os.path.join("n1_results", "main_read")
NET = KR.NET
#: 配り方の番号のずらし（… don_read 9000・window_read_all 10000／11000 と重ねない）。
OFFSET_NEXT = 12000
OFFSET_JUDGE = 13000
TYPES = ["play", "don", "activate", "discard", "event"]
N_PER_TYPE = 50
BIG_SIMS = E.PLAY_SIMS * 4
NEXT_WORLDS = 8
DECIDED = 0.9
_G = {}


def _cards():
    if "cards" not in _G:
        db = json.load(open(os.path.join("opcg_sim", "data", "opcg_cards.json"), encoding="utf-8"))
        _G["cards"] = {c["number"]: c for c in db}
    return _G["cards"]


def use_of(game, mv):
    """手 → 使う資源の種類（None＝資源を使わない手）と記述。"""
    d = json.loads(BA.desc(game, mv) or "{}")
    at = d.get("action_type")
    c = _cards().get(d.get("card") or "") or {}
    text = c.get("効果(テキスト)") or ""
    kind = None
    if at == "PLAY":
        kind = "event" if c.get("種類") == "イベント" else ("play" if c.get("種類") == "キャラクター" else "stage")
        if "捨て" in text and "手札" in text and kind != "event":
            kind = "discard"
    elif at == "ACTIVATE_MAIN":
        kind = "discard" if ("捨て" in text and "手札" in text) else "activate"
    return kind, int(d.get("don_k") or 0), d


def pair_type(game, a, b):
    ka, da, xa = use_of(game, a)
    kb, db, xb = use_of(game, b)
    for k in ("event", "discard", "activate", "play"):
        if (ka == k) != (kb == k):
            return k, ("a" if ka == k else "b"), xa, xb
    if ka is None and kb is None and da != db:
        return "don", ("a" if da > db else "b"), xa, xb
    return None, None, xa, xb


# ---------------------------------------------------------------- 1. points

def cmd_points(a):
    os.makedirs(a.out, exist_ok=True)
    W._boot()
    games = W.load_all()
    pools = {"hb": [], "mu": []}
    for k, g in games.items():
        for p in g["points"]:
            if p.get("kind") != "main" or p.get("dtype") == "mulligan" or int(p.get("n_cand") or 0) < 2:
                continue
            top = p.get("top") or []
            if len(top) < 2:
                continue
            q = W.chosen_q(p)
            if q is None or abs(q) >= DECIDED:
                continue
            r = E.search_seed(zlib.crc32(json.dumps(list(k)).encode()) ^ 0x3A1A_0EAD, int(p["step"]), p["seat"])
            pools["hb" if k[0] == "hb" else "mu"].append((r, k, p))
    for s in pools:
        pools[s].sort(key=lambda x: x[0])
    print("CANDS", {s: len(v) for s, v in pools.items()}, flush=True)
    seen = {t: 0 for t in TYPES}
    out = []
    pos = {"hb": 0, "mu": 0}
    side = "hb"
    tries = 0
    while min(seen.values()) < a.n and tries < 60000:
        tries += 1
        if pos[side] >= len(pools[side]):
            side = "mu" if side == "hb" else "hb"
            if pos[side] >= len(pools[side]):
                break
        _, k, p = pools[side][pos[side]]
        pos[side] += 1
        side = "mu" if side == "hb" else "hb"
        top = p["top"]
        a_mv = p.get("a_box") or p["move"]
        bs = [t for t in top if BA.key(t["mv"]) != BA.key(a_mv)]
        if not bs:
            continue
        b_mv = bs[0]["mv"]
        pt = {"g": list(k), "step": p["step"], "seat": p["seat"], "turn": p["turn"]}
        try:
            game = W.replay(pt, games[k])
        except Exception:  # noqa: BLE001
            continue
        if json.loads(game.pending_json()).get("player_id") != p["seat"]:
            continue
        t, user, xa, xb = pair_type(game, a_mv, b_mv)
        if t is None or seen[t] >= a.n:
            continue
        seen[t] += 1
        ra = next((x for x in top if BA.key(x["mv"]) == BA.key(a_mv)), None)
        pt.update(type=t, src=k[0], user=user, a=a_mv, b=b_mv, a_desc=xa, b_desc=xb,
                  root_a={"n": ra["n"], "q": ra["q"]} if ra else None, root_b={"n": bs[0]["n"], "q": bs[0]["q"]},
                  n_cand=p["n_cand"], own_turn=None)
        out.append(pt)
    for n, pt in enumerate(out):
        pt["pid"] = n
    with open(os.path.join(a.out, "points.jsonl"), "w", encoding="utf-8") as f:
        for pt in out:
            f.write(json.dumps(pt, ensure_ascii=False) + "\n")
    print("POINTS", len(out), seen, "tries", tries, flush=True)


# ---------------------------------------------------------------- 2. eval

def _init_eval():
    W._init_eval()
    _G["big"] = E.SeatSpec(NET, sims=BIG_SIMS, budget=E.PLAY_BUDGET, worlds=E.PLAY_WORLDS,
                           lethal_proof=E.PLAY_LETHAL_PROOF)


def _which(out, a, b):
    mv = out.get("move")
    sig, k = out.get("sig"), out.get("k")
    for nm, x in (("a", a), ("b", b)):
        if sig is not None and json.dumps(BA.RG.move_sig(x)) == json.dumps(sig) and BA.RG._don_k(x) == k:
            return nm
        if BA.key(x) == BA.key(mv) or BA.pkey(x) == BA.pkey(mv):
            return nm
    return "other"


def eval_point(p):
    t0 = time.time()
    who = p["seat"]
    row = {"pid": p["pid"], "type": p["type"], "ok": False}
    try:
        game = W.replay(p, W._G["games"][W.gkey(p)])
    except Exception as exc:  # noqa: BLE001
        row["fail"] = "replay: %s" % type(exc).__name__
        return row
    seed = W.game_seed(p)
    turn = int(game.turn_count or 0)
    t1 = time.time()
    out = json.loads(game.fork().decide(who, _G["big"].decide_opts(seed, turn, who, DR.Carry().get(turn, who))))
    row["big"] = {"which": _which(out, p["a"], p["b"]), "kind": out.get("kind"), "move": out.get("move"),
                  "sig": out.get("sig"), "k": out.get("k"), "desc": BA.desc(game, out.get("move"))}
    rows = BA.ranked(out)
    for nm in ("a", "b"):
        hit = [r for r in rows if BA.key(r["mv"]) == BA.key(p[nm])] or [r for r in rows if BA.pkey(r["mv"]) == BA.pkey(p[nm])]
        row["big"]["q_" + nm] = hit[0]["q"] if hit else None
        row["big"]["n_" + nm] = hit[0]["n"] if hit else None
    # 4 倍の木が選んだ手の、原始手でなく束の代表（打ち比べに使う）
    ch = BA.chosen(out, rows) if rows else None
    row["big"]["box"] = ch["mv"] if ch else out.get("move")
    row["sec_big"] = round(time.time() - t1, 2)
    t2 = time.time()
    nx = {"a": [], "b": []}
    for w in range(NEXT_WORLDS):
        ws = MJ.world_seed(seed, p["step"], OFFSET_NEXT + w)
        for nm in ("a", "b"):
            try:
                nx[nm].append(W.next_head(game, who, p[nm], ws))
            except Exception as exc:  # noqa: BLE001
                nx[nm].append({"err": "%s: %s" % (type(exc).__name__, str(exc)[:100])})
    row["next"] = nx
    row["sec_next"] = round(time.time() - t2, 2)
    row["ok"] = True
    row["sec"] = round(time.time() - t0, 1)
    return row


def cmd_eval(a):
    pts = BA.read_jsonl(os.path.join(a.out, "points.jsonl"))
    si, sn = KR.shard_of(a)
    jobs = [p for p in pts if p["pid"] % sn == si]
    if a.limit:
        jobs = jobs[:a.limit]
    rpath = os.path.join(a.out, f"evals_shard{si}.jsonl" if a.shard else "evals.jsonl")
    done = W._done(a.out, "evals")
    jobs = [p for p in jobs if p["pid"] not in done]
    print(f"EVAL shard={a.shard} todo {len(jobs)} (done {len(done)}) -> {rpath}", flush=True)
    t0 = time.time()
    with get_context("fork").Pool(a.workers, initializer=_init_eval) as pool:
        for i, r in enumerate(pool.imap_unordered(eval_point, jobs, chunksize=1)):
            with open(rpath, "a", encoding="utf-8") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"[{i + 1}/{len(jobs)}] pid={r['pid']} type={r['type']} ok={r['ok']} big={(r.get('big') or {}).get('which')} "
                  f"sec={r.get('sec')} ({time.time() - t0:.0f}s)", flush=True)


# ---------------------------------------------------------------- 3. judge

def _next_diff(e, key="v"):
    ds = []
    for x, y in zip(e["next"]["a"], e["next"]["b"]):
        if "err" in x or "err" in y or x.get(key) is None or y.get(key) is None:
            continue
        ds.append(y[key] - x[key])
    return sum(ds) / len(ds) if ds else None


def judge_jobs(out_dir):
    pts = {p["pid"]: p for p in BA.read_jsonl(os.path.join(out_dir, "points.jsonl"))}
    ev = W.load_evals(out_dir)
    jobs = []
    for pid, e in sorted(ev.items()):
        p = pts[pid]
        alts = {}
        if e["big"]["which"] != "a" and e["big"].get("box") is not None:
            alts["big"] = e["big"]["box"] if e["big"]["which"] == "other" else p["b"]
        dn = _next_diff(e)
        if dn is not None and dn > 0:
            alts["next"] = p["b"]
        if alts:
            jobs.append((p, e, alts))
    return jobs


def judge_point(job):
    p, e, alts = job
    t0 = time.time()
    who = p["seat"]
    row = {"pid": p["pid"], "type": p["type"], "ok": False, "alts": sorted(alts)}
    try:
        game = W.replay(p, W._G["games"][W.gkey(p)])
    except Exception as exc:  # noqa: BLE001
        row["fail"] = "replay: %s" % type(exc).__name__
        return row
    seed = W.game_seed(p)
    # 同じ手は 1 回だけ打つ
    moves = {"a": p["a"]}
    for nm, mv in alts.items():
        hit = next((k for k, v in moves.items() if BA.key(v) == BA.key(mv)), None)
        if hit is None:
            moves[nm] = mv
        row["same_" + nm] = hit
    res = {k: [] for k in moves}
    errs = {}
    for i in range(_G["k"]):
        ws = MJ.world_seed(seed, p["step"], OFFSET_JUDGE + i)
        for k, mv in moves.items():
            r, er = BA.rollout_ship(game, who, mv, ws, W._G["ref"], W._G["ship"], both=True)
            res[k].append(r)
            if er:
                errs[er] = errs.get(er, 0) + 1
    row["res"] = res
    for nm in alts:
        src = row.get("same_" + nm) or nm
        row[nm] = MJ.paired(res["a"], res[src])  # 差は「別の読みの手 − 今の手」
    row["errs"] = errs or None
    row["ok"] = True
    row["sec"] = round(time.time() - t0, 1)
    return row


def _init_judge(k):
    _init_eval()
    _G["k"] = int(k)


def cmd_judge(a):
    jobs = judge_jobs(a.out)
    si, sn = KR.shard_of(a)
    jobs = [j for j in jobs if j[0]["pid"] % sn == si]
    rpath = os.path.join(a.out, f"judge_shard{si}.jsonl" if a.shard else "judge.jsonl")
    done = W._done(a.out, "judge")
    jobs = [j for j in jobs if j[0]["pid"] not in done]
    if a.limit:
        jobs = jobs[:a.limit]
    print(f"JUDGE shard={a.shard} todo {len(jobs)} (done {len(done)}) k={a.k} -> {rpath}", flush=True)
    t0 = time.time()
    with get_context("fork").Pool(a.workers, initializer=_init_judge, initargs=(a.k,)) as pool:
        for i, r in enumerate(pool.imap_unordered(judge_point, jobs, chunksize=1)):
            with open(rpath, "a", encoding="utf-8") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"[{i + 1}/{len(jobs)}] pid={r['pid']} type={r['type']} ok={r['ok']} "
                  + " ".join(f"{nm}={(r.get(nm) or {}).get('diff')}" for nm in r.get("alts", []))
                  + f" sec={r.get('sec')} ({time.time() - t0:.0f}s)", flush=True)


# ---------------------------------------------------------------- 4. summary

def cmd_summary(a):
    pts = {p["pid"]: p for p in BA.read_jsonl(os.path.join(a.out, "points.jsonl"))}
    ev = W.load_evals(a.out)
    jd = {}
    for path in sorted(glob.glob(os.path.join(a.out, "judge*.jsonl"))):
        for x in BA.read_jsonl(path):
            if x.get("ok"):
                jd[x["pid"]] = x
    rows = []
    for pid, e in sorted(ev.items()):
        p = pts[pid]
        j = jd.get(pid) or {}
        qa, qb = (p["root_a"] or {}).get("q"), (p["root_b"] or {}).get("q")
        rows.append({"pid": pid, "type": p["type"], "src": p["src"], "user": p["user"],
                     "d_now": None if qa is None or qb is None else round(qb - qa, 4),
                     "big": e["big"]["which"],
                     "d_big": None if e["big"]["q_a"] is None or e["big"]["q_b"] is None else round(e["big"]["q_b"] - e["big"]["q_a"], 4),
                     "d_next": W._r(_next_diff(e, "v"), 4), "d_next_net": W._r(_next_diff(e, "net"), 4),
                     "diff_big": (j.get("big") or {}).get("diff"), "ci_big": (j.get("big") or {}).get("ci"),
                     "diff_next": (j.get("next") or {}).get("diff"), "ci_next": (j.get("next") or {}).get("ci"),
                     "a": p["a_desc"], "b": p["b_desc"], "sec_big": e["sec_big"], "sec_next": e["sec_next"]})
    out = {}
    for t in TYPES:
        rs = [r for r in rows if r["type"] == t]
        if not rs:
            continue
        g = {"n": len(rs), "a_uses": sum(1 for r in rs if r["user"] == "a"),
             "big_changes": W._r(sum(1 for r in rs if r["big"] != "a") / len(rs)),
             "big_to_b": sum(1 for r in rs if r["big"] == "b"), "big_to_other": sum(1 for r in rs if r["big"] == "other"),
             "next_prefers_b": sum(1 for r in rs if r["d_next"] is not None and r["d_next"] > 0),
             "next_prefers_b_rate": W._r(sum(1 for r in rs if r["d_next"] is not None and r["d_next"] > 0) / len(rs)),
             # 次のターンの頭が「資源を使わない側」を選ぶ数
             "next_prefers_saving": sum(1 for r in rs if r["d_next"] is not None and
                                        ((r["d_next"] > 0) == (r["user"] == "a"))),
             "judged_big": W._ci([r["diff_big"] for r in rs]),
             "judged_next": W._ci([r["diff_next"] for r in rs]),
             "gain_big_per_point": W._r(sum(r["diff_big"] for r in rs if r["diff_big"] is not None) / len(rs)),
             "gain_next_per_point": W._r(sum(r["diff_next"] for r in rs if r["diff_next"] is not None) / len(rs)),
             "corr_next_vs_diff": W._r(DRd._corr([r["d_next"] for r in rs if r["diff_next"] is not None],
                                                 [r["diff_next"] for r in rs if r["diff_next"] is not None])),
             "corr_big_vs_diff": W._r(DRd._corr([r["d_big"] for r in rs if r["diff_big"] is not None],
                                                [r["diff_big"] for r in rs if r["diff_big"] is not None])),
             "sec_big_mean": W._r(sum(r["sec_big"] for r in rs) / len(rs), 2),
             "sec_next_mean": W._r(sum(r["sec_next"] for r in rs) / len(rs), 1)}
        g["sig_big"] = [sum(1 for r in rs if r["ci_big"] and r["ci_big"][0] is not None and r["ci_big"][0] > 0),
                        sum(1 for r in rs if r["ci_big"] and r["ci_big"][1] is not None and r["ci_big"][1] < 0)]
        g["sig_next"] = [sum(1 for r in rs if r["ci_next"] and r["ci_next"][0] is not None and r["ci_next"][0] > 0),
                         sum(1 for r in rs if r["ci_next"] and r["ci_next"][1] is not None and r["ci_next"][1] < 0)]
        out[t] = g
    allr = {"judged_big": W._ci([r["diff_big"] for r in rows]), "judged_next": W._ci([r["diff_next"] for r in rows])}
    res = {"types": out, "all": allr, "rows": rows}
    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    for t, g in out.items():
        print(t, json.dumps(g, ensure_ascii=False))
    print("ALL", json.dumps(allr))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["points", "eval", "judge", "summary"])
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--n", type=int, default=N_PER_TYPE)
    ap.add_argument("--shard", default=None)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    {"points": cmd_points, "eval": cmd_eval, "judge": cmd_judge, "summary": cmd_summary}[a.cmd](a)


if __name__ == "__main__":
    main()
