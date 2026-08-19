"""原因分析プローブ D（P8検証）: #13 506006@101 と #14 500000@31 を worlds=24 で再測定。

段2（worlds=6・1/6刻み）の regret 0.333 が量子化ノイズなら、worlds=24 では 0 近傍へ
縮むはず。move_regret の機構を流用し、**世界インデックスをオフセットした 6世界×4チャンク**
で並列化して 24 世界ぶんを合算する（CRN の世界 seed は w に単調なので、チャンク間で
世界が重複しない）。

出力: /home/user/cause_d.log ＋ /home/user/cause_d.jsonl
"""
import os, sys, json, subprocess, collections
import multiprocessing as mp
os.environ.setdefault("OPCG_LOG_SILENT", "1")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
REPO = "/home/user/opcg-sim-backend"
sys.path.insert(0, f"{REPO}/tests")
sys.path.insert(0, f"{REPO}/tests/scripts")
sys.path.insert(0, f"{REPO}/tests/harness")
os.chdir(REPO)
import _bootstrap  # noqa

TARGETS = {(506006, 101), (500000, 31)}
CHUNK = 6
OFFSETS = (0, 6, 12, 18)


def _sh(c):
    return subprocess.run(c, shell=True, capture_output=True, text=True, cwd=REPO).stdout


def load_rows():
    uniq = {}
    for br in sorted(b for b in _sh(
            "git for-each-ref refs/remotes/origin --format='%(refname:short)'").split()
            if "moveaudit-shard" in b):
        for l in _sh(f"git show {br}:audit_results 2>/dev/null").splitlines():
            if not l.startswith("shard"):
                continue
            for line in _sh(f"git show {br}:audit_results/{l.strip('/')}/regret.jsonl 2>/dev/null").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (r["seed"], r["decision"]) in uniq:
                    continue
                if (r["seed"], r["decision"]) in TARGETS:
                    uniq[(r["seed"], r["decision"])] = r
    return list(uniq.values())


def _winit():
    import move_regret as MR
    MR._init("", "random", "synth", 160, CHUNK, 3)


def _wjob(job):
    """1ジョブ = (row, world_offset): 6世界ぶんを測って選択肢別の (win数, 測定世界数) を返す。"""
    row, off = job
    import move_regret as MR
    orig = MR._shuffle_decks
    MR._shuffle_decks = lambda m, w: orig(m, w + off)
    try:
        res = MR._run_seed((row["seed"], [row]))[0]
    finally:
        MR._shuffle_decks = orig
    if res.get("error"):
        return {"loc": f"{row['seed']}@{row['decision']}", "off": off, "error": res["error"]}
    opts = {}
    for o in res.get("options", []):
        key = json.dumps(o["move"], ensure_ascii=False, sort_keys=True)
        opts[key] = {"wins": o["wr"] * CHUNK, "n": CHUNK, "chosen": o["chosen"], "move": o["move"]}
    return {"loc": f"{row['seed']}@{row['decision']}", "off": off, "opts": opts}


def main():
    rows = load_rows()
    print(f"対象 {len(rows)} 点 × オフセット {OFFSETS} = {len(rows)*len(OFFSETS)} ジョブ", flush=True)
    jobs = [(r, off) for r in rows for off in OFFSETS]
    merged = collections.defaultdict(lambda: collections.defaultdict(
        lambda: {"wins": 0.0, "n": 0, "chosen": False, "move": None}))
    errs = []
    with mp.Pool(min(8, len(jobs)), initializer=_winit) as pool:
        for res in pool.imap_unordered(_wjob, jobs):
            if res.get("error"):
                errs.append(res)
                print(f"  {res['loc']} off={res['off']}: {res['error']}", flush=True)
                continue
            print(f"  {res['loc']} off={res['off']}: 完了", flush=True)
            for key, o in res["opts"].items():
                b = merged[res["loc"]][key]
                b["wins"] += o["wins"]; b["n"] += o["n"]
                b["chosen"] = b["chosen"] or o["chosen"]; b["move"] = o["move"]

    with open("/home/user/cause_d.jsonl", "w") as f:
        for loc, opts in merged.items():
            wrs = {k: v["wins"] / v["n"] for k, v in opts.items() if v["n"]}
            if not wrs:
                continue
            best = max(wrs, key=lambda k: (wrs[k], k))
            chosen = next((k for k, v in opts.items() if v["chosen"]), None)
            regret = round(wrs[best] - wrs[chosen], 4) if chosen else None
            rec = {"loc": loc, "worlds": sum(v["n"] for v in opts.values()) // max(1, len(opts)),
                   "regret24": regret,
                   "options": [{"wr": round(wrs[k], 4), "n": opts[k]["n"],
                                "chosen": opts[k]["chosen"], "move": opts[k]["move"]}
                               for k in sorted(wrs, key=lambda k: -wrs[k])]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"■ {loc}: worlds=24 regret={regret}", flush=True)
            for o in rec["options"]:
                print(f"   wr={o['wr']:.3f} (n={o['n']}) {o['move']} {'★打った手' if o['chosen'] else ''}",
                      flush=True)
    print("CAUSE_D_DONE", flush=True)


if __name__ == "__main__":
    main()
