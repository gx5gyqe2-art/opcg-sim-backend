"""再開可能アリーナのシャード実行（旧 `tests/scripts/arena_resume.py`）。

なぜ要るか: 実行環境（エフェメラルコンテナ）はフォアグラウンド 1 回あたり約 10 分で、
バックグラウンドプロセスはターン終了時に回収される。400 ペア（800 局）は一度に走り切れない。
本器はペア単位のスコアを jsonl 台帳へ追記し、再実行のたびに未消化 seed から `--max-pairs`
ぶんだけ進める＝10 分×N 回で同一判定を積み上げる。

判定規約は [`opcg_sim.loop.arena`] と共有（帯設計 `plan_bands`・席入替 CRN `play_pair_detail`・
ペア水準 95% CI・void は母数から外して件数を載せる）＝集計規約を二重化しない。
全ペア消化後の実行が最終判定（`ARENA_RESUME_FINAL`）を出す。

実行例（消化しきるまで繰り返し実行）:
  OPCG_LOG_SILENT=1 python -m opcg_sim.loop.arena_shard \\
    --candidate /home/user/cand/nrel_c12.npz --pairs 400 --max-pairs 40 \\
    --leaders random --decks synth --out /tmp/arena_pairs.jsonl
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse                                                        # noqa: E402
import json                                                            # noqa: E402
import multiprocessing as mp                                           # noqa: E402
import sys                                                             # noqa: E402
import time                                                            # noqa: E402

from opcg_sim.loop import arena as A                                   # noqa: E402
from opcg_sim.loop import engine as E                                  # noqa: E402


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--candidate", required=True, help="候補ネットの npz（`neff:`/`n1:` 接頭辞可）")
    ap.add_argument("--baseline", default="", help="空=出荷既定")
    ap.add_argument("--pairs", type=int, default=400)
    ap.add_argument("--bands", type=int, default=4)
    ap.add_argument("--seed-base", type=int, default=71000)
    ap.add_argument("--max-pairs", type=int, default=40, help="この実行で回す上限")
    ap.add_argument("--frac", type=float, default=0.55)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--sims", type=int, default=E.SERVE_SIMS)
    ap.add_argument("--leaders", default="fixed", choices=("fixed", "random", "real", "purple"),
                    help="対面の選び方: fixed=既定リーダーミラー（歴代判定と地続き）／"
                         "random=全リーダーからペアごとに 2 枚引く（汎化）／"
                         "real=実デッキ 4 リーダー／purple=紫を含むリーダー（掘り実験の母集団）")
    ap.add_argument("--decks", default="singleton", choices=("singleton", "synth", "synth_dig"),
                    help="デッキの中身。singleton=従来（色が合う 50 枚・全部 1 枚ずつ）／"
                         "synth=リーダーに合わせて合成／synth_dig=合成に掘りカードを差し込む")
    ap.add_argument("--pair-timeout", type=int, default=900,
                    help="1 ペアの実時間上限（秒・0=無制限）。超過したペアは void として残す")
    ap.add_argument("--out", required=True, help="ペアスコア jsonl（追記台帳・再開の正）")
    A.add_cand_args(ap)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    planned = [s for band in A.plan_bands(args.pairs, args.bands, args.seed_base) for s in band]
    done = A.load_ledger(args.out)
    todo = A.remaining_seeds(planned, done)
    print(f"消化済み {len(done)}/{args.pairs} ペア・残り {len(todo)}", flush=True)
    if todo:
        batch = todo[: args.max_pairs]
        t0 = time.time()
        initargs = (args.candidate, args.baseline, A.cand_kw_from_args(args), args.leaders,
                    args.decks, args.pair_timeout, args.sims)
        with mp.get_context("spawn").Pool(args.workers, initializer=A.init_pool,
                                          initargs=initargs) as pool:
            with open(args.out, "a") as f:
                # imap（入力順）だと先頭のペアが詰まっている間、後続が完了しても台帳へ
                # flush されない＝1 局面でシャード全体が止まる。行は seed を自分で持つので
                # 順序は不要＝完了順に書き出す。
                for row in pool.imap_unordered(A.play_pair_detail, batch):
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f.flush()                       # ターン打切りでも書けた分は残す
        done = A.load_ledger(args.out)
        print(f"今回 {len(batch)} ペア（{time.time() - t0:.0f}s）・累計 {len(done)}/{args.pairs}",
              flush=True)
    res = A.final_result(planned, done, args.frac)
    if res is not None:
        res["candidate"] = args.candidate
        print(f"ARENA_RESUME_FINAL {json.dumps(res, ensure_ascii=False)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
