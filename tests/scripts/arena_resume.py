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
    """台帳 jsonl → {seed: score}（pure I/O 読み）。壊れた行は無視せず落とす＝黙って欠測にしない。

    score=None は **void**（対局がエンジン欠陥で成立しなかったペア）。消化済みとしては数えるが
    （決定論なので撃ち直しても同じ所で落ちる）、勝率の母数からは外す。
    """
    done = {}
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            sc = r.get("score")
            done[int(r["seed"])] = None if sc is None else float(sc)
    return done


def remaining_seeds(planned, done):
    """計画 seed 列から消化済みを除いた残り（計画順を保つ・pure）。"""
    return [s for s in planned if s not in done]


def final_result(planned, done, frac=0.55):
    """全ペア消化後の最終判定（pure・arena_gate.final_decision と同規約）。未消化があれば None。

    void（score=None）のペアは母数から外し、`void` 件数として結果に必ず載せる
    ＝落としたぶんを黙って隠さない。全ペアが void なら判定は出さない。
    """
    if any(s not in done for s in planned):
        return None
    from arena_parallel import _pair_level_ci
    valid = [s for s in planned if done[s] is not None]
    if not valid:
        return None
    scores = [done[s] / 2.0 for s in valid]            # 勝ち数0..2 → ペア水準0/0.5/1
    ci = _pair_level_ci(scores)
    return {"pairs": len(valid), "games": 2 * len(valid),
            "void": len(planned) - len(valid),
            "wins": sum(done[s] for s in valid),
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
    ap.add_argument("--leaders", default="fixed", choices=("fixed", "random", "real", "purple"),
                    help="対面の選び方（2026-08-15 ユーザ提案）: fixed=従来の既定リーダーミラー"
                         "（歴代判定と地続き）／random=全リーダーからペアごとに2枚引く（汎化）／"
                         "real=実デッキ4リーダーの総当たり（出荷先）。ペア内では席とリーダーを"
                         "入替＝リーダー相性は相殺され打ち回しの差だけが残る")
    ap.add_argument("--out", required=True, help="ペアスコア jsonl（追記台帳・再開の正）")
    # 補償層系の cand フラグ（--cand-box/--cand-tree-box/--cand-don-margin/--cand-don-box/
    # --cand-plan-readout/--cand-plan-box/--cand-guard-policy）は純正AZ化（2026-08-25）で削除。
    ap.add_argument("--cand-macro", action="store_true",
                    help="候補席だけマクロ手化 P1（配分箱＋戦闘の木内箱化・quiesce）を有効化"
                         "（2026-08-24・macro_p0_probe が示した読みの浪費の A/B）")
    ap.add_argument("--cand-defense-box", action="store_true",
                    help="候補席だけ防御箱 v1（P4-c・D1'/D2' 支配則の候補整形）を有効化"
                         "（2026-08-24・ネット不変の防御矯正の A/B）")
    ap.add_argument("--cand-dialog-box", action="store_true",
                    help="候補席だけ対話箱（P3/P5・効果対話窓を出口value最良で畳む）を有効化"
                         "（2026-08-25）")
    ap.add_argument("--cand-boxes-all", action="store_true",
                    help="候補席で箱化インフラ全部入り（P1配分箱+P2アタック箱+P4c防御箱+"
                         "P3/P5対話箱＝macro_moves+defense_box+box_dialog+戦闘箱設定）を有効化")
    ap.add_argument("--cand-box-commit", action="store_true",
                    help="候補席だけ箱コミット実行（2026-08-26・選んだ箱の中身を機械実行）を"
                         "有効化（config 既定 ON の明示上書き）")
    ap.add_argument("--cand-no-box-commit", action="store_true",
                    help="候補席だけ箱コミット実行を**無効化**（box_commit=False）＝"
                         "「既定(コミットON) vs OFF」の欠陥検出 A/B の OFF 側測定用")
    ap.add_argument("--cand-residual-dig", action="store_true",
                    help="候補席だけ残ドン掘り（2026-09-02・対照生成の腕A）: 木が TURN_END を"
                         "選んだ時にアクティブドンが残り、手札に「登場時ドン-Xでドロー」の"
                         "コスト1キャラがあれば代わりに出す。発火は台帳行 dig に記録")
    ap.add_argument("--cand-residual-activate", default=None, choices=("low", "high"),
                    help="候補席だけ残り起動（2026-09-02・対照生成の腕A2）: 木が TURN_END を選んだ"
                         "時に、リーダーのドン追加起動効果が未使用なら起動し（ドンデッキが空でも"
                         "レストのドンの付与が効く）、付与対話を方針で解く（low=攻撃できる最低"
                         "パワー／high=最高パワー）。"
                         "発火は台帳行 act に記録")
    ap.add_argument("--pair-timeout", type=int, default=900,
                    help="1ペアの実時間上限（秒・0=無制限）。超過したペアは void として台帳に"
                         "残し次へ進む。手数上限では捕まらない「1回の decide() から戻らない」"
                         "暴走（戦闘箱の組合せ爆発）を切るための保険")
    ap.add_argument("--decks", default="singleton", choices=("singleton", "synth", "synth_dig"),
                    help="デッキの中身。singleton=従来（色が合う50枚・全部1枚ずつ・イベント0）／"
                         "synth=リーダーに合わせて合成（deck_synth）／synth_dig=合成に掘りカード"
                         "（登場時ドン-Xドローのコスト1）を差し込む（deck_dig・--leaders purple と対）")
    args = ap.parse_args()
    cand_kw = None
    if args.cand_macro:
        cand_kw = dict(cand_kw or {}, macro_moves=True,
                       box_battle=True, quiesce=True)
    if args.cand_defense_box:
        cand_kw = dict(cand_kw or {}, defense_box=True)
    if args.cand_dialog_box:
        cand_kw = dict(cand_kw or {}, box_dialog=True)
    if args.cand_boxes_all:
        cand_kw = dict(cand_kw or {}, macro_moves=True, defense_box=True, box_dialog=True,
                       box_battle=True, quiesce=True)
    if args.cand_box_commit:
        cand_kw = dict(cand_kw or {}, box_commit=True)
    if args.cand_no_box_commit:
        cand_kw = dict(cand_kw or {}, box_commit=False)
    if args.cand_residual_dig:
        cand_kw = dict(cand_kw or {}, residual_dig=True)
    if args.cand_residual_activate:
        cand_kw = dict(cand_kw or {}, residual_activate=args.cand_residual_activate)

    from arena_gate import plan_bands
    planned = [s for band in plan_bands(args.pairs, args.bands, args.seed_base) for s in band]
    done = load_ledger(args.out)
    todo = remaining_seeds(planned, done)
    print(f"消化済み {len(done)}/{args.pairs} ペア・残り {len(todo)}", flush=True)
    if todo:
        batch = todo[: args.max_pairs]
        from promotion_gate import _init_pool, _play_pair_detail
        t0 = time.time()
        with mp.Pool(args.workers, initializer=_init_pool,
                     initargs=(args.candidate, args.baseline, cand_kw, args.leaders,
                               args.decks, args.pair_timeout)) as pool:
            with open(args.out, "a") as f:
                # imap（入力順）だと先頭のペアが詰まっている間、後続が完了しても台帳へ
                # flush されない＝1局面でシャード全体が止まる（2026-08-16 b_p04）。行は
                # seed を自分で持つので順序は不要＝完了順に書き出す。
                for row in pool.imap_unordered(_play_pair_detail, batch):
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
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
