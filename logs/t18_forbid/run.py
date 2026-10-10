"""T18-禁止だけの本番を小分けに回す（再開可能・済んだ塊は飛ばす）。

    python logs/t18_forbid/run.py [塊の最大数]

塊 = 25 ペア（50 局）。順番は 主条件（理論→プラセボを塊ごとに交互）→ 副条件。
seed 帯: 主条件 7200000–7200299・副条件 7300000–7300299（両腕で同じ seed＝CRN）。
線: 手札 1 枚の価値の 1 倍（--cutoff-mu 1・本番の前に固定・動かさない）。
"""
import os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CHUNK, PAIRS = 25, 300
CONDS = [("main", 7200000, ["--decks", "synth_roles", "--leaders", "random"]),
         ("sub", 7300000, ["--decks", "singleton", "--leaders", "fixed"])]
plan = []
for cond, band, extra in CONDS:
    for j in range(PAIRS // CHUNK):
        for arm in ("theory", "placebo"):
            plan.append((cond, arm, band + CHUNK * j, extra))
limit = int(sys.argv[1]) if len(sys.argv) > 1 else len(plan)
done = 0
for cond, arm, base, extra in plan:
    out = os.path.join(HERE, f"{cond}_{arm}_{base}.json")
    if os.path.exists(out):
        continue
    if done >= limit:
        break
    tmp = out + ".part"
    cmd = [sys.executable, os.path.join(ROOT, "tests/scripts/t18_arena.py"), "--pairs", str(CHUNK),
           "--seed-base", str(base), "--mode", "forbid", "--cutoff-mu", "1", "--arm", arm,
           "--json", tmp] + extra
    print("run", cond, arm, base, flush=True)
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, env=dict(os.environ, OPCG_LOG_SILENT="1"))
    os.replace(tmp, out)
    done += 1
print("batch done", done, flush=True)
