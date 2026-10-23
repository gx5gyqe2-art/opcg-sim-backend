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

**ネットを固定して探索の設定だけを比べる**こともできる（WP `rs-search-arena`・§20.5.3）:
`--candidate` に基準と同じ npz（または空の `--baseline`）を渡し、`--cand-*` で候補席の
つまみだけを変える。候補席の設定は台帳の各行（`cand_opts`）と `ARENA_RESUME_FINAL` に載る
＝判定が設定に紐づく。既定（`--cand-*` 無し）の記録は歴代の台帳と同じ形のまま。

  OPCG_LOG_SILENT=1 python -m opcg_sim.loop.arena_shard \\
    --candidate opcg_sim/data/learned/nrel_r3.npz --leaders random --decks synth \\
    --cand-worlds 4 --cand-select-rule q_min_n --cand-q-min-frac 0.125 \\
    --cand-root-prior-temp 2.0 --workers 2 --out /tmp/arena_r2.jsonl
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
    ap.add_argument("--budget", type=int, default=None,
                    help="基準席の戦闘箱の枝予算（既定 8000・0＝無制限）。候補席は --cand-budget")
    ap.add_argument("--rethink-ratio", type=float, default=None,
                    help="基準席の「迷う局面だけ世界 K 本で読み直す」の線 τ（§7.4・省略＝off）。"
                         "候補席は --cand-rethink-ratio")
    ap.add_argument("--rethink-worlds", type=int, default=None,
                    help="基準席の読み直しの世界の本数 K（省略＝4）")
    ap.add_argument("--omniscient", action="store_true",
                    help="基準席を全部見える探索で打たせる（`docs/search_efficiency_plan.md` §7.10・"
                         "測定専用）。候補席は --cand-omniscient")
    ap.add_argument("--reveal", default=None, choices=("fair", "peek", "opp_hand"),
                    help="基準席の探索の世界の見える範囲（`docs/search_efficiency_plan.md` §7.11・§7.15）。"
                         "省略＝既定の fair。peek＝2026-10-21 までの既定（自分の山札と裏向きライフを本物のまま"
                         "読む・測定用）／opp_hand（測定用）。候補席は --cand-reveal。--omniscient とは排他")
    ap.add_argument("--opp-blind", action="store_true",
                    help="基準席の木の中の相手にこちらの手札を見せない（`docs/search_efficiency_plan.md` §7.14・"
                         "測定専用）。候補席は --cand-opp-blind")
    ap.add_argument("--worlds", type=int, default=None,
                    help="基準席の世界サンプルの本数（§20.7.1・省略＝1）。両席を出荷の設定（世界 2 本）で打たせるとき"
                         "に --cand-worlds と組で使う（§7.21）。K 本ぶんスレッドを使うので --workers を下げること")
    ap.add_argument("--lethal-proof", default=None, choices=("off", "root", "root_reply"),
                    help="基準席の詰みの証明（`docs/search_efficiency_plan.md` §7.21・省略＝off）。"
                         "候補席は --cand-lethal-proof")
    ap.add_argument("--commit-sig-legacy", action="store_true",
                    help="基準席を旧い手の署名で打たせる（WP `rs-commit-sig`・測定専用＝箱の commit の照合で"
                         "カウンター・ブロッカーの札が落ちる 2026-10-22 までの形）。候補席は --cand-commit-sig-legacy")
    ap.add_argument("--lethal-deal-legacy", action="store_true",
                    help="基準席の詰みの証明の守りの強い配り方を旧い形にする（`docs/search_efficiency_plan.md` §7.33・"
                         "測定用・カウンターイベントを払えるかを見ずに積む）。候補席は --cand-lethal-deal-legacy")
    ap.add_argument("--commit-hidden-legacy", action="store_true",
                    help="基準席を、箱の commit の続きに見えていなかった札から選ぶ段も積む旧い形で打たせる"
                         "（`docs/search_efficiency_plan.md` §7.39・測定専用）。候補席は --cand-commit-hidden-legacy")
    ap.add_argument("--leaders", default="fixed", choices=("fixed", "random", "real", "purple"),
                    help="対面の選び方: fixed=既定リーダーミラー（歴代判定と地続き）／"
                         "random=全リーダーからペアごとに 2 枚引く（汎化）／"
                         "real=実デッキ 4 リーダー／purple=紫を含むリーダー（掘り実験の母集団）")
    ap.add_argument("--decks", default="singleton",
                    choices=("singleton", "synth", "synth_dig", "synth_roles", "synth_roles2", "user", "official", "real"),
                    help="デッキの中身。singleton=従来（色が合う 50 枚・全部 1 枚ずつ）／"
                         "synth=リーダーに合わせて合成／synth_dig=合成に掘りカードを差し込む／"
                         "synth_roles=合成に除去の型を色ごとに差し込む（§20.8.1）")
    ap.add_argument("--pair-timeout", type=int, default=900,
                    help="1 ペアの実時間上限（秒・0=無制限）。超過したペアは void として残す")
    ap.add_argument("--out", required=True, help="ペアスコア jsonl（追記台帳・再開の正）")
    A.add_cand_args(ap)
    return ap


def _best_kw(args):
    """基準席のつまみ（§7.4 の読み直し・§7.10 の全部見える探索・§7.11 の見える範囲・§7.14 の相手に手札を見せない・§7.21 の詰みの証明・WP `rs-commit-sig` の旧い手の署名・§7.33 の守りの強い配り方の旧い形・§7.39 の見えていなかった札を選ぶ続きの旧い形）。何も立っていなければ None＝基準席は今までどおり。"""
    kw = {k: v for k, v in (("rethink_ratio", args.rethink_ratio),
                            ("rethink_worlds", args.rethink_worlds),
                            ("omniscient", True if args.omniscient else None),
                            ("reveal", args.reveal),
                            ("opp_blind", True if args.opp_blind else None),
                            ("lethal_proof", args.lethal_proof),
                            ("commit_sig_legacy", True if args.commit_sig_legacy else None),
                            ("lethal_deal_legacy", True if args.lethal_deal_legacy else None),
                            ("commit_hidden_legacy", True if args.commit_hidden_legacy else None),
                            ("worlds", args.worlds if args.worlds and args.worlds > 1 else None)) if v is not None}
    return kw or None


def main(argv=None):
    args = build_parser().parse_args(argv)
    cand_kw = A.cand_kw_from_args(args)
    if cand_kw:
        print(f"候補席の設定 {json.dumps(cand_kw, ensure_ascii=False)}（基準席は既定）", flush=True)
    planned = [s for band in A.plan_bands(args.pairs, args.bands, args.seed_base) for s in band]
    done = A.load_ledger(args.out)
    todo = A.remaining_seeds(planned, done)
    print(f"消化済み {len(done)}/{args.pairs} ペア・残り {len(todo)}", flush=True)
    if todo:
        batch = todo[: args.max_pairs]
        t0 = time.time()
        initargs = (args.candidate, args.baseline, cand_kw, args.leaders,
                    args.decks, args.pair_timeout, args.sims, args.budget,
                    _best_kw(args))
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
        if cand_kw:
            res["cand_opts"] = cand_kw          # 判定は設定に紐づく（§20.5.3）
        if _best_kw(args):
            res["best_opts"] = _best_kw(args)   # 基準席の読み直しも判定に紐づく（§7.4）
        print(f"ARENA_RESUME_FINAL {json.dumps(res, ensure_ascii=False)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
