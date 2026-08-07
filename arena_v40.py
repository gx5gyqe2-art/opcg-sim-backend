"""v40 アリーナ分担: 指定 seed 帯のペア対局（cand=v40 vs best=gen12・席入替2局/ペア）。

使い方: OPCG_LOG_SILENT=1 PYTHONPATH=tests:tests/scripts python arena_v40.py <出力jsonl> <開始-終了> ...
判定規約は promotion_gate._play_pair と同一。途中終了しても再実行で続きから（jsonl 追記）。
"""
import json
import multiprocessing as mp
import os
import sys

sys.path.insert(0, "tests")
sys.path.insert(0, "tests/scripts")
import _bootstrap  # noqa: F401

out = sys.argv[1]
seeds = []
for spec in sys.argv[2:]:
    a, b = spec.split("-")
    seeds += list(range(int(a), int(b) + 1))
done = set()
if os.path.exists(out):
    done = {int(json.loads(l)["seed"]) for l in open(out) if l.strip()}
todo = [s for s in seeds if s not in done]
print(f"担当 {len(seeds)} ペア・残り {len(todo)}", flush=True)

from promotion_gate import _init_pool, _play_pair

CAND = ("tests/fixtures/candidates/v40_inj4_value.npz,"
        "opcg_sim/data/learned/gen12_policy.npz")
if todo:
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with mp.Pool(3, initializer=_init_pool, initargs=(CAND, "", None)) as pool:
        with open(out, "a") as f:
            n = 0
            for seed, score in zip(todo, pool.imap(_play_pair, todo)):
                f.write(json.dumps({"seed": seed, "score": score}) + "\n")
                f.flush()
                n += 1
                if n % 5 == 0:
                    print(f"{n}/{len(todo)} 完了", flush=True)
print("ARENA_SHARD_DONE", flush=True)
