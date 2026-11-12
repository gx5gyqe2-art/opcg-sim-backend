"""探索の天井の測定（WP `search-ceiling`・`docs/search_efficiency_plan.md` §7.90）。

読み取りだけの計測（既定の挙動は変えない）。1 回きりの計測＝テストは足さない（CLAUDE.md の理論の道具の規約）。
局面は §7.84 の出荷の設定の 100 局（`n1_results/reaudit_r29f4/blunder/games.jsonl`）を記録の手で当て直して作る。

1. ``points``: 100 局から局ごとに 1 点、メインの判断点（木が決めた・候補が 2 つ以上・続きの持ち越し無し）を
   手番 2〜12 に散らして選ぶ（`points.json`）。段 0 の 20 点はそのうち手番の順に 5 つおき。
2. ``one``: 1 点 × 1 つの倍率 × 世界の本数で `decide`（出荷の設定の読む回数だけを倍にする・同じ探索の種・
   深さの読み出し on）を 1 回回し、1 行の JSON を出す（時間・CPU 時間・メモリの山・選んだ手・根の値・深さ）。
   メモリの山を点ごとに測り、落ちても親が生き残るように**別プロセス**で回す（``run`` がこれを呼ぶ）。
3. ``run``: 点 × 倍率 × 世界の本数の組を、終えていないものだけ別プロセスで回して `runs.jsonl` に足す（再開できる）。
4. ``judge``: 1 倍と手が替わった点で、審判（`blunder_audit` の recheck と同じ作り・160 回の固定の審判が両方の手の先を
   同じ配り方で最後まで打つ）で「高い倍率の手 − 1 倍の手」の勝率の差を出す（`judge.jsonl`）。
5. ``summary``: 段 0／段 1 の表（`summary.json`）。

    python tests/scripts/search_ceiling.py points
    python tests/scripts/search_ceiling.py run --stage0 --mults 1,10,100 --worlds 8,1
    python tests/scripts/search_ceiling.py run --mults 1,10,100,1000 --worlds 8 --shard 1/8
    python tests/scripts/search_ceiling.py judge --workers 4
    python tests/scripts/search_ceiling.py summary
"""
import argparse
import json
import os
import random
import resource
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("OPCG_LOG_SILENT", "1")

GAMES = os.path.join("n1_results", "reaudit_r29f4", "blunder", "games.jsonl")
OUT_DIR = os.path.join("n1_results", "search_ceiling")
N_POINTS = 100
TURNS = list(range(2, 13))
#: 子プロセスのアドレス空間の上限（コンテナの cgroup は約 14 GB＝越える前に子だけが落ちるようにする）。
MEM_LIMIT_GB = 12.0
#: 木の降下は再帰＝深い木でスタックを使い切らないように、世界のスレッドのスタックを広げる。
THREAD_STACK = 1 << 28


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def load_games() -> List[Dict[str, Any]]:
    return read_jsonl(os.path.join(_ROOT, GAMES))


# ---------------------------------------------------------------------------------------------------------
# points
# ---------------------------------------------------------------------------------------------------------
def eligible(p: Dict[str, Any]) -> bool:
    return (p.get("kind") == "main" and p.get("act") == "MAIN_ACTION" and int(p.get("n_cand") or 0) >= 2
            and not p.get("commit_left"))


def cmd_points(a):
    games = load_games()
    rng = random.Random(7_900_001)
    order = list(range(len(games)))
    rng.shuffle(order)
    pts = []
    for j, gi in enumerate(order[:N_POINTS]):
        g = games[gi]
        want = TURNS[j % len(TURNS)]
        cand = [p for p in g["points"] if eligible(p)]
        if not cand:
            continue
        # 狙った手番に最も近い手番の点から 1 つ（同じ近さは無作為）
        best = min(abs(int(p["turn"]) - want) for p in cand)
        pool = [p for p in cand if abs(int(p["turn"]) - want) == best]
        p = rng.choice(pool)
        pts.append({"gi": gi, "seed": g["seed"], "step": p["step"], "turn": p["turn"], "seat": p["seat"],
                    "n_cand": p["n_cand"], "dtype": p.get("dtype"), "want_turn": want,
                    "rec_move": p.get("a_box") or p.get("move")})
    pts.sort(key=lambda r: (r["turn"], r["gi"]))
    for i, r in enumerate(pts):
        r["pid"] = i
        r["stage0"] = (i % 5 == 2)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "points.json"), "w") as f:
        json.dump(pts, f, ensure_ascii=False, indent=1)
    hist: Dict[int, int] = {}
    for r in pts:
        hist[r["turn"]] = hist.get(r["turn"], 0) + 1
    print(json.dumps({"points": len(pts), "stage0": sum(r["stage0"] for r in pts), "turn_hist": hist}))


def load_points(out: str) -> List[Dict[str, Any]]:
    with open(os.path.join(out, "points.json")) as f:
        return json.load(f)


# ---------------------------------------------------------------------------------------------------------
# one（別プロセスで 1 回の decide）
# ---------------------------------------------------------------------------------------------------------
def _summ_depth(d: Optional[Dict[str, Any]], sims: int) -> Optional[Dict[str, Any]]:
    """深さの読み出し → 世界ごとの要約と合算。

    合算の決まり: 「木の最大の深さ」「越えた境目の最大」は世界の**最大**と**平均**の両方、主な読み筋は
    **選んだ手の訪問が最も多い世界**（`pv_world`＝エンジンが読み筋とコミットの続きに使う世界）の値と世界の平均、
    節の総数と訪問は世界の**和**、「読みの何割が境目を t 回以上越えたか」は世界の和どうしの比。
    """
    if not d:
        return None
    per = d["per_world"]
    k = len(per)
    pvw = d.get("pv_world") or 0

    def pv_n(w, floor):
        """主な読み筋のうち、訪問が floor 以上の手だけを数えた長さと境目。"""
        ln, tn = 0, 0
        for n, t in w["pv"]:
            if n < floor:
                break
            ln, tn = ln + 1, t
        return ln, tn

    def tot(key):
        m = max(len(w[key]) for w in per)
        return [sum((w[key][i] if i < len(w[key]) else 0) for w in per) for i in range(m)]

    rv = sum(w["root_visits"] for w in per) or 1.0
    reach_t = tot("reach_turns")
    reach_d = tot("reach_depth")
    # 読み 1 回あたりの平均の深さ・平均の越えた境目（＝届いた回数の和 ÷ 根の訪問）
    mean_depth = sum(reach_d[1:]) / rv
    mean_turns = sum(reach_t[1:]) / rv
    floor10 = max(10.0, 0.001 * sims)
    return {
        "worlds": k, "pv_world": pvw,
        "nodes": sum(w["nodes"] for w in per), "expanded": sum(w["expanded"] for w in per),
        "root_visits": sum(w["root_visits"] for w in per),
        "max_depth_max": max(w["max_depth"] for w in per), "max_depth_mean": sum(w["max_depth"] for w in per) / k,
        "max_turns_max": max(w["max_turns"] for w in per), "max_turns_mean": sum(w["max_turns"] for w in per) / k,
        "pv_len": per[pvw]["pv_len"], "pv_turns": per[pvw]["pv_turns"],
        "pv_len_mean": sum(w["pv_len"] for w in per) / k, "pv_turns_mean": sum(w["pv_turns"] for w in per) / k,
        # 訪問が 10 回（かつ読む回数の 0.1%）以上の手だけで数えた主な読み筋（末端の 1 回きりの枝を除く）
        "pv10_len": pv_n(per[pvw], floor10)[0], "pv10_turns": pv_n(per[pvw], floor10)[1],
        "pv10_len_mean": sum(pv_n(w, floor10)[0] for w in per) / k,
        "pv10_turns_mean": sum(pv_n(w, floor10)[1] for w in per) / k,
        "mean_depth": round(mean_depth, 3), "mean_turns": round(mean_turns, 3),
        "reach_turns_frac": [round(x / rv, 5) for x in reach_t],
        "by_turns": tot("by_turns"), "by_depth": tot("by_depth"),
        "pv": per[pvw]["pv"][:400],
    }


def cmd_one(a):
    from opcg_sim.loop import decks as D, driver as DR, engine as E
    import blunder_audit as BA
    games = load_games()
    pt = next(p for p in load_points(a.out) if p["pid"] == a.pid)
    g = games[pt["gi"]]
    E.engine()
    db = D.load_db()
    BA._G["db"] = db
    game = BA.replay(g, pt["step"])
    pend = json.loads(game.pending_json())
    name = pend["player_id"]
    assert name == pt["seat"], (name, pt["seat"])
    spec = BA.ship_spec(E.DEFAULT_NET)
    opts = dict(spec.opts)
    base_sims = int(opts["sims"])
    opts["sims"] = base_sims * a.mult
    opts["worlds"] = a.worlds
    opts["search_seed"] = E.search_seed(g["seed"], pt["turn"], name)
    opts["depth_readout"] = True
    if a.world_threads:
        opts["world_threads"] = a.world_threads
    if a.lean:
        # 世界を順に回し、1 本ごとに深さを読み出して木を捨てる（メモリの山は世界 1 本ぶん・打つ手と根の統計は同じ）
        opts["depth_lean"] = True
        opts["world_threads"] = 1
    t0 = time.time()
    c0 = time.process_time()
    out = json.loads(game.decide(name, json.dumps(opts)))
    sec = time.time() - t0
    cpu = time.process_time() - c0
    rows = BA.ranked(out)
    ch = BA.chosen(out, rows)
    tot = sum(r["n"] for r in rows) or 1.0
    root_v = sum(r["n"] * r["q"] for r in rows) / tot
    rec = pt.get("rec_move")
    res = {
        "pid": pt["pid"], "gi": pt["gi"], "seed": g["seed"], "step": pt["step"], "turn": pt["turn"], "seat": name,
        "mult": a.mult, "worlds": a.worlds, "sims": opts["sims"], "sec": round(sec, 2), "cpu": round(cpu, 2),
        "rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1),
        "kind": out.get("kind"), "move": out.get("move"),
        "box": ch["mv"] if ch else out.get("move"),
        "key": BA.key(ch["mv"]) if ch else BA.key(out.get("move")),
        "pkey": BA.pkey(out.get("move")), "desc": BA.desc(game, ch["mv"] if ch else out.get("move")),
        "rec_same": (BA.key(rec) == (BA.key(ch["mv"]) if ch else None)) if rec else None,
        "root_v": round(root_v, 5),
        "n_share": round(ch["n"] / tot, 4) if ch else None, "q": round(ch["q"], 5) if ch else None,
        "top": [{"key": BA.key(r["mv"]), "desc": BA.desc(game, r["mv"]), "n": round(r["n"] / tot, 4), "q": round(r["q"], 5)}
                for r in rows[:5]],
        "n_groups": len(rows),
        "depth": _summ_depth(out.get("depth"), opts["sims"]),
    }
    print("ROW " + json.dumps(res, ensure_ascii=False))


# ---------------------------------------------------------------------------------------------------------
# run（終えていない組を別プロセスで）
# ---------------------------------------------------------------------------------------------------------
def _child_limits(mem_gb: float):
    def f():
        lim = int(mem_gb * (1 << 30))
        resource.setrlimit(resource.RLIMIT_AS, (lim, lim))
        try:
            resource.setrlimit(resource.RLIMIT_STACK, (THREAD_STACK, THREAD_STACK))
        except (ValueError, OSError):
            pass
    return f


def run_one(out: str, pid: int, mult: int, worlds: int, timeout: float, mem_gb: float,
            world_threads: int = 0, lean: bool = True) -> Dict[str, Any]:
    env = dict(os.environ)
    env["RUST_MIN_STACK"] = str(THREAD_STACK)
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    cmd = [sys.executable, os.path.abspath(__file__), "one", "--pid", str(pid), "--mult", str(mult),
           "--worlds", str(worlds), "--out", out, "--world-threads", str(world_threads)]
    if lean:
        cmd.append("--lean")
    t0 = time.time()
    base = {"pid": pid, "mult": mult, "worlds": worlds}
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env, cwd=_ROOT,
                           preexec_fn=_child_limits(mem_gb))
    except subprocess.TimeoutExpired:
        return {**base, "fail": "timeout", "sec": round(time.time() - t0, 1)}
    for line in r.stdout.splitlines():
        if line.startswith("ROW "):
            return json.loads(line[4:])
    return {**base, "fail": f"exit {r.returncode}", "sec": round(time.time() - t0, 1),
            "err": (r.stderr or "")[-600:]}


def cmd_run(a):
    pts = load_points(a.out)
    if a.stage0:
        pts = [p for p in pts if p["stage0"]]
    if a.rest:
        pts = [p for p in pts if not p["stage0"]]
    if a.pids:
        want = {int(x) for x in a.pids.split(",")}
        pts = [p for p in pts if p["pid"] in want]
    if a.shard:
        k, n = (int(x) for x in a.shard.split("/"))
        pts = [p for i, p in enumerate(pts) if i % n == k - 1]
    mults = [int(x) for x in a.mults.split(",")]
    worlds = [int(x) for x in a.worlds.split(",")]
    path = os.path.join(a.out, a.file)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    done = {(r["pid"], r["mult"], r["worlds"]) for r in read_jsonl(path) if not r.get("fail") or a.keep_fail}
    jobs = [(p["pid"], m, w) for m in mults for w in worlds for p in pts if (p["pid"], m, w) not in done]
    print(f"jobs {len(jobs)} (done {len(done)})", flush=True)
    from concurrent.futures import ThreadPoolExecutor
    t0 = time.time()

    def work(job):
        pid, m, w = job
        return run_one(a.out, pid, m, w, a.timeout, a.mem_gb, a.world_threads, not a.no_lean)

    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex, open(path, "a") as f:
        for i, row in enumerate(ex.map(work, jobs)):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            d = row.get("depth") or {}
            print(f"[{i + 1}/{len(jobs)} {time.time() - t0:.0f}s] pid {row['pid']} x{row['mult']} w{row['worlds']} "
                  f"{row.get('fail') or ''} sec {row.get('sec')} cpu {row.get('cpu')} rss {row.get('rss_mb')} "
                  f"nodes {d.get('nodes')} pv {d.get('pv_len')}/{d.get('pv_turns')} max {d.get('max_depth_max')}/"
                  f"{d.get('max_turns_max')}", flush=True)
    print("RUN_DONE", flush=True)


# ---------------------------------------------------------------------------------------------------------
# judge（手が替わった点を固定の審判で打ち比べる）
# ---------------------------------------------------------------------------------------------------------
#: 審判の配り方の番号のずらし（`blunder_audit` の judge 0〜・recheck 1000〜・verify 2000〜 と重ねない）。
JUDGE_OFFSET = 7900
_J: Dict[str, Any] = {}


def _init_judge(k: int):
    from opcg_sim.loop import decks as D, engine as E
    import blunder_audit as BA
    import move_judge as MJ
    E.engine()
    BA._G["db"] = D.load_db()
    _J.update(BA=BA, MJ=MJ, k=int(k), ref=E.SeatSpec(E.DEFAULT_NET, sims=MJ.REFEREE_SIMS))


def judge_pair(job):
    """a＝1 倍の手・b＝高い倍率の手。審判（出荷のネット・160 回・世界 1 本）が同じ配り方 K 通りで両方の先を最後まで打つ。"""
    g, row = job
    BA, MJ = _J["BA"], _J["MJ"]
    t0 = time.time()
    out = {"pid": row["pid"], "key_b": row["key_b"], "mults": row["mults"], "seed": g["seed"], "step": row["step"]}
    try:
        game = BA.replay(g, row["step"])
    except Exception as exc:  # noqa: BLE001
        out["error"] = "replay: %s: %s" % (type(exc).__name__, str(exc)[:120])
        return out
    wa, wb, errs = [], [], {}
    for i in range(_J["k"]):
        ws = MJ.world_seed(g["seed"], row["step"], JUDGE_OFFSET + i)
        ra, ea = MJ.rollout(game, row["seat"], row["a"], ws, _J["ref"])
        rb, eb = MJ.rollout(game, row["seat"], row["b"], ws, _J["ref"])
        wa.append(ra)
        wb.append(rb)
        for e in (ea, eb):
            if e:
                errs[e] = errs.get(e, 0) + 1
    out.update(MJ.paired(wa, wb), a=wa, b=wb, errs=errs or None, k=_J["k"], sec=round(time.time() - t0, 1))
    return out


def changed_pairs(out: str, file: str, worlds: int) -> List[Dict[str, Any]]:
    """1 倍と手が替わった (点, 高い倍率の手) の組（同じ手を複数の倍率が選んだら 1 組にまとめる）。"""
    rows = [r for r in read_jsonl(os.path.join(out, file)) if not r.get("fail") and r["worlds"] == worlds]
    by: Dict[int, Dict[int, Dict[str, Any]]] = {}
    for r in rows:
        by.setdefault(r["pid"], {})[r["mult"]] = r
    pairs: Dict[Any, Dict[str, Any]] = {}
    for pid, d in sorted(by.items()):
        if 1 not in d:
            continue
        base = d[1]
        for m, r in sorted(d.items()):
            if m == 1 or r["key"] == base["key"]:
                continue
            k = (pid, r["key"])
            if k not in pairs:
                pairs[k] = {"pid": pid, "step": base["step"], "seat": base["seat"], "gi": base["gi"],
                            "a": base["box"], "b": r["box"], "key_b": r["key"], "mults": []}
            pairs[k]["mults"].append(m)
    return list(pairs.values())


def cmd_judge(a):
    from multiprocessing import get_context
    games = load_games()
    pairs = changed_pairs(a.out, a.file, a.worlds)
    jpath = os.path.join(a.out, a.judge_file)
    done = {(r["pid"], r["key_b"]) for r in read_jsonl(jpath)}
    jobs = [({k: v for k, v in games[p["gi"]].items() if k != "points"}, p) for p in pairs
            if (p["pid"], p["key_b"]) not in done]
    print(f"JUDGE todo {len(jobs)} (done {len(done)}) k={a.k}", flush=True)
    t0 = time.time()
    with get_context("fork").Pool(a.workers, initializer=_init_judge, initargs=(a.k,)) as pool:
        for i, r in enumerate(pool.imap_unordered(judge_pair, jobs, chunksize=1)):
            with open(jpath, "a", encoding="utf-8") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"JUDGE pid={r['pid']} mults={r['mults']} diff={r.get('diff')} {r.get('ci')} sec={r.get('sec')} "
                  f"({i + 1}/{len(jobs)} {time.time() - t0:.0f}s)", flush=True)
    print("JUDGE_DONE", flush=True)


# ---------------------------------------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------------------------------------
def _stat(xs: List[float]) -> Optional[Dict[str, float]]:
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    n = len(xs)
    return {"n": n, "mean": round(sum(xs) / n, 3), "med": round(xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2, 3),
            "min": round(xs[0], 3), "max": round(xs[-1], 3)}


def _mean_ci(xs: List[float]) -> Dict[str, Any]:
    n = len(xs)
    if n == 0:
        return {"n": 0, "mean": None, "ci": [None, None]}
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / (n - 1) if n > 1 else 0.0
    h = 1.96 * (var / n) ** 0.5 if n > 1 else float("inf")
    return {"n": n, "mean": round(m, 4), "ci": [round(m - h, 4), round(m + h, 4)]}


def _wilson(k: int, n: int, z: float = 1.96) -> List[Optional[float]]:
    if n == 0:
        return [None, None]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
    return [round(c - h, 4), round(c + h, 4)]


def cost_depth_table(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """(世界の本数, 倍率) ごとの費用と深さ。"""
    out: Dict[str, Any] = {}
    keys = sorted({(r["worlds"], r["mult"]) for r in rows})
    for w, m in keys:
        ok = [r for r in rows if r["worlds"] == w and r["mult"] == m and not r.get("fail")]
        bad = [r for r in rows if r["worlds"] == w and r["mult"] == m and r.get("fail")]
        d = [r["depth"] for r in ok if r.get("depth")]
        cell = {
            "n": len(ok), "fail": len(bad), "fail_why": sorted({str(r.get("fail")) for r in bad}) or None,
            "cpu_sec": _stat([r["cpu"] for r in ok]), "rss_mb": _stat([r["rss_mb"] for r in ok]),
            "nodes": _stat([x["nodes"] for x in d]),
            "visits_per_plan": _stat([x["root_visits"] / (r["sims"] * r["worlds"]) for r, x in
                                      ((r, r["depth"]) for r in ok if r.get("depth"))]),
        }
        for k in ("pv_len", "pv_turns", "pv10_len", "pv10_turns", "pv_turns_mean", "pv10_turns_mean", "max_depth_max",
                  "max_turns_max", "max_turns_mean", "mean_depth", "mean_turns"):
            cell[k] = _stat([x[k] for x in d])
        # 読みの何割が境目を t 回以上越えたか（点の平均）
        mx = max((len(x["reach_turns_frac"]) for x in d), default=0)
        cell["reach_turns_frac"] = [round(sum((x["reach_turns_frac"][t] if t < len(x["reach_turns_frac"]) else 0.0)
                                              for x in d) / len(d), 4) for t in range(mx)] if d else []
        out[f"w{w}_x{m}"] = cell
    return out


def choice_table(rows: List[Dict[str, Any]], worlds: int, judge: List[Dict[str, Any]]) -> Dict[str, Any]:
    """選ぶ手の比べ（1 倍との違い・隣の倍率との一致・根の値）と、替わった手の審判の差。"""
    by: Dict[int, Dict[int, Dict[str, Any]]] = {}
    for r in rows:
        if r["worlds"] == worlds and not r.get("fail"):
            by.setdefault(r["pid"], {})[r["mult"]] = r
    mults = sorted({m for d in by.values() for m in d})
    jd = {(j["pid"], j["key_b"]): j for j in judge if j.get("diff") is not None}
    res: Dict[str, Any] = {"worlds": worlds, "mults": mults, "points": len(by)}
    for m in mults:
        if m == 1:
            continue
        both = [(d[1], d[m]) for d in by.values() if 1 in d and m in d]
        ch = [(a, b) for a, b in both if a["key"] != b["key"]]
        cell: Dict[str, Any] = {"n": len(both), "changed": len(ch),
                                "changed_rate": round(len(ch) / len(both), 4) if both else None,
                                "changed_ci": _wilson(len(ch), len(both)),
                                "root_v_shift": _mean_ci([b["root_v"] - a["root_v"] for a, b in both]),
                                "abs_root_v_shift": _mean_ci([abs(b["root_v"] - a["root_v"]) for a, b in both]),
                                "top_share": _mean_ci([b["n_share"] for _, b in both if b.get("n_share") is not None])}
        # 審判: 替わった点の差（高い倍率の手 − 1 倍の手）。全体＝替わらなかった点を 0 として全点で平均。
        # 高い倍率の読みの中での値の差（新しい手 − 1 倍の手・勝率の単位＝q の差の半分）。1 倍の手が上位に無い点は数えない。
        # 選んだ手の値は選ばれたぶん上に偏る（勝者の呪い）＝上限の目安として読む。
        qd = []
        for a_, b_ in ch:
            qa = [t["q"] for t in b_["top"] if t.get("key") == a_["key"]]
            qb = [t["q"] for t in b_["top"] if t.get("key") == b_["key"]]
            if qa and qb:
                qd.append((qb[0] - qa[0]) / 2)
        cell["inner_gain_changed"] = _mean_ci(qd)
        cell["inner_gain_all"] = _mean_ci(qd + [0.0] * (len(both) - len(ch))) if len(qd) == len(ch) else None
        diffs = [jd[(b["pid"], b["key"])]["diff"] for _, b in ch if (b["pid"], b["key"]) in jd]
        cell["judged"] = len(diffs)
        cell["gain_changed"] = _mean_ci(diffs)
        if len(diffs) == len(ch):
            cell["gain_all"] = _mean_ci(diffs + [0.0] * (len(both) - len(ch)))
        sig_pos = sum(1 for _, b in ch if (b["pid"], b["key"]) in jd and jd[(b["pid"], b["key"])]["ci"][0] > 0)
        sig_neg = sum(1 for _, b in ch if (b["pid"], b["key"]) in jd and jd[(b["pid"], b["key"])]["ci"][1] < 0)
        cell["points_better"] = sig_pos
        cell["points_worse"] = sig_neg
        res[f"x{m}_vs_x1"] = cell
    for lo, hi in zip(mults, mults[1:]):
        both = [(d[lo], d[hi]) for d in by.values() if lo in d and hi in d]
        same = sum(1 for a, b in both if a["key"] == b["key"])
        res[f"x{lo}_same_x{hi}"] = {"n": len(both), "same": same, "rate": round(same / len(both), 4) if both else None,
                                    "ci": _wilson(same, len(both))}
    return res


def cmd_summary(a):
    rows = read_jsonl(os.path.join(a.out, a.file))
    # 同じ組が複数あれば最後の成功を採る
    last: Dict[Any, Dict[str, Any]] = {}
    for r in rows:
        k = (r["pid"], r["mult"], r["worlds"])
        if k not in last or not r.get("fail"):
            last[k] = r
    rows = list(last.values())
    pts = {p["pid"]: p for p in load_points(a.out)}
    s0 = [r for r in rows if pts[r["pid"]]["stage0"]]
    judge = read_jsonl(os.path.join(a.out, a.judge_file))
    summ = {
        "stage0": {"points": sorted({r["pid"] for r in s0}), "table": cost_depth_table(s0),
                   "choice_w8": choice_table(s0, 8, judge), "choice_w1": choice_table(s0, 1, judge)},
        "stage1": {"table": cost_depth_table([r for r in rows if r["worlds"] == 8]),
                   "choice_w8": choice_table(rows, 8, judge)},
        "judge": {"pairs": len(judge), "k": (judge[0].get("k") if judge else None),
                  "void": sum(sum(x is None for x in j.get("a", []) + j.get("b", [])) for j in judge)},
    }
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summ, f, ensure_ascii=False, indent=1)
    for name in ("stage0", "stage1"):
        print("==", name)
        for k, c in summ[name]["table"].items():
            if not c["n"]:
                print(k, "n 0 fail", c["fail"], c["fail_why"])
                continue
            g = lambda key, f="mean": (c[key] or {}).get(f)  # noqa: E731
            print(f"{k:10s} n {c['n']:3d} fail {c['fail']} cpu {g('cpu_sec')}/{g('cpu_sec', 'max')} rss {g('rss_mb')}/"
                  f"{g('rss_mb', 'max')} nodes {g('nodes')} vis {g('visits_per_plan')} pv {g('pv_len')}/{g('pv_turns')} "
                  f"pv10 {g('pv10_len')}/{g('pv10_turns')} max {g('max_depth_max')}/{g('max_turns_max')}"
                  f"(max {g('max_turns_max', 'max')}) mean {g('mean_depth')}/{g('mean_turns')} reach {c['reach_turns_frac']}")
        for ck in ("choice_w8", "choice_w1"):
            for k, c in (summ[name].get(ck) or {}).items():
                if isinstance(c, dict):
                    print(ck, k, json.dumps(c))


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("points")
    p.add_argument("--out", default=OUT_DIR)
    p = sub.add_parser("one")
    p.add_argument("--pid", type=int, required=True)
    p.add_argument("--mult", type=int, default=1)
    p.add_argument("--worlds", type=int, default=8)
    p.add_argument("--world-threads", type=int, default=0)
    p.add_argument("--lean", action="store_true")
    p.add_argument("--out", default=OUT_DIR)
    p = sub.add_parser("run")
    p.add_argument("--stage0", action="store_true")
    p.add_argument("--rest", action="store_true", help="段 0 の 20 点を除いた 80 点（シャード用）")
    p.add_argument("--pids", default=None)
    p.add_argument("--shard", default=None, help="k/N（点を N 個に割った k 番目・1 始まり）")
    p.add_argument("--mults", default="1,10,100")
    p.add_argument("--worlds", default="8")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--world-threads", type=int, default=0)
    p.add_argument("--timeout", type=float, default=6 * 3600)
    p.add_argument("--mem-gb", type=float, default=MEM_LIMIT_GB)
    p.add_argument("--file", default="runs.jsonl")
    p.add_argument("--no-lean", action="store_true", help="世界を同時に回し、木を最後まで持つ（既定は順に回して捨てる）")
    p.add_argument("--keep-fail", action="store_true", help="落ちた組を終えた扱いにして回し直さない")
    p.add_argument("--out", default=OUT_DIR)
    p = sub.add_parser("judge")
    p.add_argument("--k", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--worlds", type=int, default=8)
    p.add_argument("--file", default="runs.jsonl")
    p.add_argument("--judge-file", default="judge.jsonl")
    p.add_argument("--out", default=OUT_DIR)
    p = sub.add_parser("summary")
    p.add_argument("--file", default="runs.jsonl")
    p.add_argument("--judge-file", default="judge.jsonl")
    p.add_argument("--out", default=OUT_DIR)
    a = ap.parse_args(argv)
    {"points": cmd_points, "one": cmd_one, "run": cmd_run, "judge": cmd_judge, "summary": cmd_summary}[a.cmd](a)


if __name__ == "__main__":
    main()
