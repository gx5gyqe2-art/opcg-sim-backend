"""再開可能アリーナ（v25・`arena_gate.py` の chunk 実行版）。

なぜ要るか: 実行環境（エフェメラルコンテナ）はフォアグラウンド1回あたり約10分で、
バックグラウンドプロセスはターン終了時に回収される。`arena_gate.py --pairs 400`（800局・
約85分）は一度に走り切れない。本スクリプトはペア単位のスコアを jsonl 台帳へ追記し、
再実行のたびに未消化 seed から `--max-pairs` ぶんだけ進める＝10分×N回で同一判定を積み上げる。

判定規約は arena_gate と同一: 帯設計は `arena_gate.plan_bands` を import（二重化しない）、
対局は `promotion_gate._play_pair`（席入替CRN）、集計は `arena_parallel._pair_level_ci`
（ペア水準95%CI・promoted は wr≥0.55 かつ CI下限>0.50）。全ペア消化後の実行が最終判定を出す。

実行例（消化しきるまで繰り返し実行）:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/arena_resume.py \
    --candidate /tmp/cand/value.npz,/tmp/cand/policy.npz \
    --pairs 400 --max-pairs 40 --out /tmp/arena_pairs.jsonl
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import multiprocessing as mp
import time

import os as _os, sys as _sys  # noqa: E402  test bootstrap
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap  # noqa: E402,F401


def load_ledger(path):
    """台帳 jsonl → {seed: score}（pure I/O 読み）。壊れた行は無視せず落とす＝黙って欠測にしない。"""
    done = {}
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            done[int(r["seed"])] = float(r["score"])
    return done


def remaining_seeds(planned, done):
    """計画 seed 列から消化済みを除いた残り（計画順を保つ・pure）。"""
    return [s for s in planned if s not in done]


def final_result(planned, done, frac=0.55):
    """全ペア消化後の最終判定（pure・arena_gate.final_decision と同規約）。未消化があれば None。"""
    if any(s not in done for s in planned):
        return None
    from arena_parallel import _pair_level_ci
    scores = [done[s] / 2.0 for s in planned]          # 勝ち数0..2 → ペア水準0/0.5/1
    ci = _pair_level_ci(scores)
    return {"pairs": len(planned), "games": 2 * len(planned),
            "wins": sum(done[s] for s in planned),
            "wr": round(ci["win_rate"], 4), "ci95": [round(ci["lo"], 4), round(ci["hi"], 4)],
            "elo": round(ci["elo"], 1),
            "promoted": bool(ci["win_rate"] >= frac and ci["lo"] > 0.50)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True, help="value.npz[,policy.npz]")
    ap.add_argument("--baseline", default="", help="空=出荷既定")
    ap.add_argument("--pairs", type=int, default=400)
    ap.add_argument("--bands", type=int, default=4)
    ap.add_argument("--seed-base", type=int, default=71000)
    ap.add_argument("--max-pairs", type=int, default=40, help="この実行で回す上限（≈10分/40ペア）")
    ap.add_argument("--frac", type=float, default=0.55)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True, help="ペアスコア jsonl（追記台帳・再開の正）")
    args = ap.parse_args()

    from arena_gate import plan_bands
    planned = [s for band in plan_bands(args.pairs, args.bands, args.seed_base) for s in band]
    done = load_ledger(args.out)
    todo = remaining_seeds(planned, done)
    print(f"消化済み {len(done)}/{args.pairs} ペア・残り {len(todo)}", flush=True)
    if todo:
        batch = todo[: args.max_pairs]
        from promotion_gate import _init_pool, _play_pair
        t0 = time.time()
        with mp.Pool(args.workers, initializer=_init_pool,
                     initargs=(args.candidate, args.baseline)) as pool:
            with open(args.out, "a") as f:
                for seed, score in zip(batch, pool.imap(_play_pair, batch)):  # imap=入力順を保存
                    f.write(json.dumps({"seed": seed, "score": score}) + "\n")
                    f.flush()                                    # ターン打切りでも書けた分は残す
        done = load_ledger(args.out)
        print(f"今回 {len(batch)} ペア（{time.time() - t0:.0f}s）・累計 {len(done)}/{args.pairs}",
              flush=True)
    res = final_result(planned, done, args.frac)
    if res is not None:
        res["candidate"] = args.candidate
        print(f"ARENA_RESUME_FINAL {json.dumps(res, ensure_ascii=False)}", flush=True)
    return 0


if __name__ == "__main__":
    _sys.exit(main())
