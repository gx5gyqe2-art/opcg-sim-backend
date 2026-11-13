"""同じ 100 倍の回数を「深く」使う形の比べ（WP `search-deep`・`docs/search_efficiency_plan.md` §7.91）。

読み取りだけの計測（既定の挙動は変えない）。1 回きりの計測＝テストは足さない（CLAUDE.md の理論の道具の規約）。
点・当て直し・深さの要約・審判は §7.90 の `search_ceiling.py` をそのまま使う。ここが足すのは「形」だけ:
出荷の設定の読む回数を倍にした上で、形ごとの opts（探索の広がりの係数・世界の本数・測定専用の旗
`tree_prior_temp`／`tree_topk`／`tree_topk_side`／`tree_fpu_red`）を重ねる。**世界の本数を減らす形は 1 本あたりの
回数を増やして総数を同じに保つ**（世界 8 本 × 128,000 回 ＝ 世界 2 本 × 512,000 回 ＝ 世界 1 本 × 1,024,000 回）。

基準（今の形の 100 倍・1 倍）は §7.90 の記録（`n1_results/search_ceiling/runs.jsonl`・`shards/*.jsonl`）を読む。

    python tests/scripts/search_deep.py run --stage0 --shapes c4,k3 --workers 4
    python tests/scripts/search_deep.py run --shapes k3,fpu0 --shard 1/8 --file shards/s01.jsonl
    python tests/scripts/search_deep.py judge --workers 4
    python tests/scripts/search_deep.py summary
"""
import argparse
import glob
import json
import os
import resource
import subprocess
import sys
import time
from typing import Any, Dict, List

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("OPCG_LOG_SILENT", "1")

import search_ceiling as SC  # noqa: E402

OUT_DIR = os.path.join("n1_results", "search_deep")
PTS_DIR = SC.OUT_DIR
BASE_WORLDS = 8
#: 審判の配り方の番号のずらし（§7.90 は 7900〜）。
JUDGE_OFFSET = 7910

#: 形の名前 → (世界の本数, 重ねる opts)。`base`＝今の形（§7.90 の記録と同じはず＝照合用）。
SHAPES: Dict[str, Any] = {
    "base": (8, {}),
    # (A) 今ある設定だけ
    "c2": (8, {"c_puct": 0.75}),
    "c4": (8, {"c_puct": 0.375}),
    "c8": (8, {"c_puct": 0.1875}),
    "w2": (2, {}),
    "w1": (1, {}),
    "w2c4": (2, {"c_puct": 0.375}),
    "w1c4": (1, {"c_puct": 0.375}),
    # (B1) 木の中の全節の打ち方の割合を尖らせる
    "t2": (8, {"tree_prior_temp": 0.5}),
    "t4": (8, {"tree_prior_temp": 0.25}),
    # (B2) 各節の上位 k 手だけ（根は全部）
    "k2": (8, {"tree_topk": 2}),
    "k3": (8, {"tree_topk": 3}),
    "k5": (8, {"tree_topk": 5}),
    # (B3) 相手の手番の節だけ上位 k 手
    "ok1": (8, {"tree_topk": 1, "tree_topk_side": "opp"}),
    "ok2": (8, {"tree_topk": 2, "tree_topk_side": "opp"}),
    "ok3": (8, {"tree_topk": 3, "tree_topk_side": "opp"}),
    # (B4・足した形) まだ読んでいない手の値を 0 ではなく「節の今の平均 − r」に置く
    "f0": (8, {"tree_fpu_red": 0.0}),
    "f2": (8, {"tree_fpu_red": 0.2}),
    # 組み合わせ
    "f2c4": (8, {"tree_fpu_red": 0.2, "c_puct": 0.375}),
    "k3c4": (8, {"tree_topk": 3, "c_puct": 0.375}),
    "k2c4": (8, {"tree_topk": 2, "c_puct": 0.375}),
    "k2f2": (8, {"tree_topk": 2, "tree_fpu_red": 0.2}),
    "k3f2": (8, {"tree_topk": 3, "tree_fpu_red": 0.2}),
    "k2f2c4": (8, {"tree_topk": 2, "tree_fpu_red": 0.2, "c_puct": 0.375}),
    "w2k2": (2, {"tree_topk": 2}),
    "w1k2": (1, {"tree_topk": 2}),
    "w1k2f2c4": (1, {"tree_topk": 2, "tree_fpu_red": 0.2, "c_puct": 0.375}),
    "w2k2f2c4": (2, {"tree_topk": 2, "tree_fpu_red": 0.2, "c_puct": 0.375}),
    "t4f2": (8, {"tree_prior_temp": 0.25, "tree_fpu_red": 0.2}),
    "t4f2c4": (8, {"tree_prior_temp": 0.25, "tree_fpu_red": 0.2, "c_puct": 0.375}),
}


def cmd_one(a):
    from opcg_sim.loop import decks as D, engine as E
    import blunder_audit as BA
    worlds, extra = SHAPES[a.shape]
    games = SC.load_games()
    pt = next(p for p in SC.load_points(PTS_DIR) if p["pid"] == a.pid)
    g = games[pt["gi"]]
    E.engine()
    BA._G["db"] = D.load_db()
    game = BA.replay(g, pt["step"])
    name = json.loads(game.pending_json())["player_id"]
    assert name == pt["seat"], (name, pt["seat"])
    opts = dict(BA.ship_spec(E.DEFAULT_NET).opts)
    assert int(opts.get("worlds", 1)) == BASE_WORLDS, opts
    total = int(opts["sims"]) * a.mult * BASE_WORLDS
    opts["sims"] = total // worlds
    opts["worlds"] = worlds
    opts["search_seed"] = E.search_seed(g["seed"], pt["turn"], name)
    opts["depth_readout"] = True
    opts["depth_lean"] = True
    opts["world_threads"] = 1
    opts.update(extra)
    t0 = time.time()
    c0 = time.process_time()
    out = json.loads(game.decide(name, json.dumps(opts)))
    sec = time.time() - t0
    cpu = time.process_time() - c0
    rows = BA.ranked(out)
    ch = BA.chosen(out, rows)
    tot = sum(r["n"] for r in rows) or 1.0
    res = {
        "pid": pt["pid"], "gi": pt["gi"], "seed": g["seed"], "step": pt["step"], "turn": pt["turn"], "seat": name,
        "shape": a.shape, "mult": a.mult, "worlds": worlds, "sims": opts["sims"], "extra": extra,
        "sec": round(sec, 2), "cpu": round(cpu, 2),
        "rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1),
        "kind": out.get("kind"), "move": out.get("move"),
        "box": ch["mv"] if ch else out.get("move"),
        "key": BA.key(ch["mv"]) if ch else BA.key(out.get("move")),
        "desc": BA.desc(game, ch["mv"] if ch else out.get("move")),
        "root_v": round(sum(r["n"] * r["q"] for r in rows) / tot, 5),
        "n_share": round(ch["n"] / tot, 4) if ch else None, "q": round(ch["q"], 5) if ch else None,
        "top": [{"key": BA.key(r["mv"]), "desc": BA.desc(game, r["mv"]), "n": round(r["n"] / tot, 4), "q": round(r["q"], 5)}
                for r in rows[:5]],
        "n_groups": len(rows),
        "depth": SC._summ_depth(out.get("depth"), opts["sims"]),
    }
    if res["depth"]:
        res["depth"].pop("pv", None)
        res["depth"].pop("by_depth", None)
    print("ROW " + json.dumps(res, ensure_ascii=False))


def run_one(pid: int, shape: str, mult: int, timeout: float, mem_gb: float) -> Dict[str, Any]:
    env = dict(os.environ)
    env["RUST_MIN_STACK"] = str(SC.THREAD_STACK)
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    cmd = [sys.executable, os.path.abspath(__file__), "one", "--pid", str(pid), "--shape", shape, "--mult", str(mult)]
    t0 = time.time()
    base = {"pid": pid, "shape": shape, "mult": mult}
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env, cwd=_ROOT,
                           preexec_fn=SC._child_limits(mem_gb))
    except subprocess.TimeoutExpired:
        return {**base, "fail": "timeout", "sec": round(time.time() - t0, 1)}
    for line in r.stdout.splitlines():
        if line.startswith("ROW "):
            return json.loads(line[4:])
    return {**base, "fail": f"exit {r.returncode}", "sec": round(time.time() - t0, 1), "err": (r.stderr or "")[-600:]}


def cmd_run(a):
    pts = SC.load_points(PTS_DIR)
    pts = [p for p in pts if p["pid"] != 56]  # §7.90: 当て直しがずれる点
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
    shapes = a.shapes.split(",")
    for s in shapes:
        assert s in SHAPES, s
    path = os.path.join(a.out, a.file)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    done = {(r["pid"], r["shape"], r["mult"]) for r in SC.read_jsonl(path) if not r.get("fail")}
    jobs = [(p["pid"], s) for s in shapes for p in pts if (p["pid"], s, a.mult) not in done]
    print(f"jobs {len(jobs)} (done {len(done)})", flush=True)
    from concurrent.futures import ThreadPoolExecutor
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex, open(path, "a") as f:
        for i, row in enumerate(ex.map(lambda j: run_one(j[0], j[1], a.mult, a.timeout, a.mem_gb), jobs)):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            d = row.get("depth") or {}
            print(f"[{i + 1}/{len(jobs)} {time.time() - t0:.0f}s] pid {row['pid']} {row['shape']} {row.get('fail') or ''} "
                  f"cpu {row.get('cpu')} rss {row.get('rss_mb')} nodes {d.get('nodes')} pv {d.get('pv_len_mean')}/"
                  f"{d.get('pv_turns_mean')} max {d.get('max_depth_max')}/{d.get('max_turns_max')}", flush=True)
    print("RUN_DONE", flush=True)


# ---------------------------------------------------------------------------------------------------------
# 記録の読み込み
# ---------------------------------------------------------------------------------------------------------
def load_rows(out: str, mult: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in [os.path.join(out, "runs.jsonl")] + sorted(glob.glob(os.path.join(out, "shards", "*.jsonl"))):
        rows += SC.read_jsonl(path)
    last: Dict[Any, Dict[str, Any]] = {}
    for r in rows:
        if r.get("fail") or r.get("mult") != mult:
            continue
        last[(r["pid"], r["shape"])] = r
    return list(last.values())


def load_ceiling(mult: int) -> Dict[int, Dict[str, Any]]:
    """§7.90 の記録から、今の形（世界 8 本）の `mult` 倍の行を点ごとに。"""
    rows: List[Dict[str, Any]] = []
    for path in [os.path.join(PTS_DIR, "runs.jsonl")] + sorted(glob.glob(os.path.join(PTS_DIR, "shards", "*.jsonl"))):
        rows += SC.read_jsonl(path)
    return {r["pid"]: r for r in rows if not r.get("fail") and r["worlds"] == 8 and r["mult"] == mult}


# ---------------------------------------------------------------------------------------------------------
# judge（今の形の 100 倍の手 a と、深い形の手 b を固定の審判で打ち比べる）
# ---------------------------------------------------------------------------------------------------------
def changed_pairs(out: str, mult: int, shapes: List[str]) -> List[Dict[str, Any]]:
    base = load_ceiling(mult)
    pairs: Dict[Any, Dict[str, Any]] = {}
    for r in sorted(load_rows(out, mult), key=lambda r: (r["pid"], r["shape"])):
        if r["shape"] not in shapes or r["shape"] == "base" or r["pid"] not in base:
            continue
        b0 = base[r["pid"]]
        if r["key"] == b0["key"]:
            continue
        k = (r["pid"], r["key"])
        if k not in pairs:
            pairs[k] = {"pid": r["pid"], "step": b0["step"], "seat": b0["seat"], "gi": b0["gi"],
                        "a": b0["box"], "b": r["box"], "key_b": r["key"], "mults": []}
        pairs[k]["mults"].append(r["shape"])
    return list(pairs.values())


def cmd_judge(a):
    from multiprocessing import get_context
    SC.JUDGE_OFFSET = JUDGE_OFFSET
    games = SC.load_games()
    shapes = a.shapes.split(",")
    pairs = changed_pairs(a.out, a.mult, shapes)
    if a.shard:
        k, n = (int(x) for x in a.shard.split("/"))
        pairs = [p for i, p in enumerate(pairs) if i % n == k - 1]
    jpath = os.path.join(a.out, a.judge_file)
    os.makedirs(os.path.dirname(jpath), exist_ok=True)
    done = {(r["pid"], r["key_b"]) for path in glob.glob(os.path.join(a.out, "judge*.jsonl")) +
            glob.glob(os.path.join(a.out, "shards", "judge*.jsonl")) for r in SC.read_jsonl(path)}
    jobs = [({k: v for k, v in games[p["gi"]].items() if k != "points"}, p) for p in pairs
            if (p["pid"], p["key_b"]) not in done]
    print(f"JUDGE todo {len(jobs)} (done {len(done)}) k={a.k}", flush=True)
    t0 = time.time()
    with get_context("fork").Pool(a.workers, initializer=SC._init_judge, initargs=(a.k,)) as pool:
        for i, r in enumerate(pool.imap_unordered(SC.judge_pair, jobs, chunksize=1)):
            with open(jpath, "a", encoding="utf-8") as f:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"JUDGE pid={r['pid']} shapes={r['mults']} diff={r.get('diff')} {r.get('ci')} sec={r.get('sec')} "
                  f"({i + 1}/{len(jobs)} {time.time() - t0:.0f}s)", flush=True)
    print("JUDGE_DONE", flush=True)


# ---------------------------------------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------------------------------------
def shape_cell(rows: List[Dict[str, Any]], base: Dict[int, Dict[str, Any]], one: Dict[int, Dict[str, Any]],
               jd: Dict[Any, Dict[str, Any]]) -> Dict[str, Any]:
    d = [r["depth"] for r in rows if r.get("depth")]
    cell: Dict[str, Any] = {"n": len(rows), "cpu_sec": SC._stat([r["cpu"] for r in rows]),
                            "rss_mb": SC._stat([r["rss_mb"] for r in rows]), "nodes": SC._stat([x["nodes"] for x in d])}
    for k in ("pv_turns_mean", "pv10_turns_mean", "pv_len_mean", "max_turns_max", "max_turns_mean", "max_depth_max",
              "mean_depth", "mean_turns"):
        cell[k] = SC._stat([x[k] for x in d])
    mx = max((len(x["reach_turns_frac"]) for x in d), default=0)
    cell["reach_turns_frac"] = [round(sum((x["reach_turns_frac"][t] if t < len(x["reach_turns_frac"]) else 0.0)
                                          for x in d) / len(d), 4) for t in range(mx)] if d else []
    both = [(base[r["pid"]], r) for r in rows if r["pid"] in base]
    ch = [(a, b) for a, b in both if a["key"] != b["key"]]
    cell["vs_base"] = {"n": len(both), "changed": len(ch), "rate": round(len(ch) / len(both), 4) if both else None,
                       "ci": SC._wilson(len(ch), len(both)),
                       "root_v_shift": SC._mean_ci([b["root_v"] - a["root_v"] for a, b in both]),
                       "abs_root_v_shift": SC._mean_ci([abs(b["root_v"] - a["root_v"]) for a, b in both]),
                       "top_share": SC._mean_ci([b["n_share"] for _, b in both if b.get("n_share") is not None])}
    b1 = [(one[r["pid"]], r) for r in rows if r["pid"] in one]
    cell["vs_x1_changed"] = {"n": len(b1), "changed": sum(1 for a, b in b1 if a["key"] != b["key"])}
    diffs = [jd[(b["pid"], b["key"])]["diff"] for _, b in ch if (b["pid"], b["key"]) in jd]
    cell["judged"] = len(diffs)
    cell["gain_changed"] = SC._mean_ci(diffs)
    if len(diffs) == len(ch):
        cell["gain_all"] = SC._mean_ci(diffs + [0.0] * (len(both) - len(ch)))
    cell["points_better"] = sum(1 for _, b in ch if (b["pid"], b["key"]) in jd and jd[(b["pid"], b["key"])]["ci"][0] > 0)
    cell["points_worse"] = sum(1 for _, b in ch if (b["pid"], b["key"]) in jd and jd[(b["pid"], b["key"])]["ci"][1] < 0)
    return cell


def cmd_summary(a):
    rows = load_rows(a.out, a.mult)
    base, one = load_ceiling(a.mult), load_ceiling(1)
    pts = {p["pid"]: p for p in SC.load_points(PTS_DIR)}
    judge = [j for path in sorted(glob.glob(os.path.join(a.out, "judge*.jsonl")) +
                                  glob.glob(os.path.join(a.out, "shards", "judge*.jsonl"))) for j in SC.read_jsonl(path)]
    jd = {(j["pid"], j["key_b"]): j for j in judge if j.get("diff") is not None}
    summ: Dict[str, Any] = {"mult": a.mult, "judge_pairs": len(jd)}
    for name, keep in (("stage0", lambda r: pts[r["pid"]]["stage0"]), ("all", lambda r: True)):
        # 基準（§7.90 の記録）を同じ器の形に直す
        brows = [dict(r, shape="ceiling") for r in base.values() if keep(r)]
        tab = {"ceiling": shape_cell(brows, base, one, jd)}
        for s in SHAPES:
            rs = [r for r in rows if r["shape"] == s and keep(r)]
            if rs:
                tab[s] = shape_cell(rs, base, one, jd)
        summ[name] = tab
    with open(os.path.join(a.out, f"summary_x{a.mult}.json"), "w") as f:
        json.dump(summ, f, ensure_ascii=False, indent=1)
    for name in ("stage0", "all"):
        print("==", name)
        for k, c in summ[name].items():
            g = lambda key, f="mean": (c[key] or {}).get(f)  # noqa: E731
            vb = c["vs_base"]
            print(f"{k:9s} n {c['n']:3d} cpu {g('cpu_sec')}/{g('cpu_sec', 'max')} rss {g('rss_mb')}/{g('rss_mb', 'max')} "
                  f"pv {g('pv_turns_mean')} pv10 {g('pv10_turns_mean')} max {g('max_turns_max')}({g('max_turns_max', 'max')}) "
                  f"mean {g('mean_turns')} len {g('pv_len_mean')}/{g('max_depth_max')} chg {vb['changed']}/{vb['n']} "
                  f"x1chg {c['vs_x1_changed']['changed']} |dv| {vb['abs_root_v_shift']['mean']} share {vb['top_share']['mean']} "
                  f"gain {c['gain_changed']['mean']} all {(c.get('gain_all') or {}).get('mean')} {(c.get('gain_all') or {}).get('ci')}"
                  f" reach {c['reach_turns_frac'][:8]}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("one")
    p.add_argument("--pid", type=int, required=True)
    p.add_argument("--shape", required=True)
    p.add_argument("--mult", type=int, default=100)
    p = sub.add_parser("run")
    p.add_argument("--stage0", action="store_true")
    p.add_argument("--rest", action="store_true", help="段 0 の 20 点を除いた残り（シャード用）")
    p.add_argument("--pids", default=None)
    p.add_argument("--shard", default=None, help="k/N（点を N 個に割った k 番目・1 始まり）")
    p.add_argument("--shapes", required=True)
    p.add_argument("--mult", type=int, default=100)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--timeout", type=float, default=6 * 3600)
    p.add_argument("--mem-gb", type=float, default=SC.MEM_LIMIT_GB)
    p.add_argument("--file", default="runs.jsonl")
    p.add_argument("--out", default=OUT_DIR)
    p = sub.add_parser("judge")
    p.add_argument("--shapes", required=True)
    p.add_argument("--mult", type=int, default=100)
    p.add_argument("--k", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--shard", default=None)
    p.add_argument("--judge-file", default="judge.jsonl")
    p.add_argument("--out", default=OUT_DIR)
    p = sub.add_parser("summary")
    p.add_argument("--mult", type=int, default=100)
    p.add_argument("--out", default=OUT_DIR)
    a = ap.parse_args(argv)
    {"one": cmd_one, "run": cmd_run, "judge": cmd_judge, "summary": cmd_summary}[a.cmd](a)


if __name__ == "__main__":
    main()
