"""生成デッキの実プレイ監査（2026-08-15）: 全リーダーで自己対戦し**ハングするカードを洗い出す**。

なぜ要るか: `deck_synth` の生成デッキで自己対戦したところ 6戦中3戦が上限手数まで終わらず、
原因はステージ「聖地マリージョア」の対象選択が解決せず `RESOLVE_EFFECT_SELECTION` が
無限ループすることだった。**固定ハンニャバルデッキ（ステージ0枚・イベント0枚）では一度も
通らない経路**で、歴代のアリーナ・ゲート・自己対戦では検出できなかった実バグ。
生成デッキはカードDBの広い範囲を実プレイに乗せるので、この種の欠陥を掘り当てられる。

各リーダーについて「そのリーダーのデッキ同士のミラー」を1局回し、
  ok        … 正常決着（ターン数も記録）
  hang      … 上限手数に到達（**繰り返している対話の source_card_name を記録＝犯人**）
  timeout   … `--timeout` の実時間上限に到達（手数上限では捕まらない「1手が終わらない」暴走）
  error     … 例外（種別とメッセージを記録）
を集計する。犯人カードの一覧がそのまま修正対象／暫定除外リストになる。

`--cross N`（2026-08-16）は**ミラーではなく交差対面**を N 件回す。アリーナをランダム対面へ
広げたときに出た void（終局しないペア・20件）は**交差対面でしか出ない**（ミラー監査は ok=137）。
対面の引き方はアリーナ（`promotion_gate._leader_pair`）と同じで、hang 時は繰り返していた
**対話の発生元**に加えて**繰り返していた手**も出す（対話を伴わないループは発生元が付かないため）。

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/deck_synth_audit.py \\
    --sims 8 --max-steps 700 --workers 4 --out /tmp/deck_audit.json
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/deck_synth_audit.py \\
    --cross 200 --sims 8 --max-steps 700 --workers 4 --out /tmp/cross_audit.json
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import collections
import json
import multiprocessing as mp

import os as _os, sys as _sys  # noqa: E402
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: E402,F401

_G = {}


class _GameTimeout(BaseException):
    """1局の実時間上限。**BaseException 派生**にするのが要点で、エンジン/探索側の
    広い `except Exception` に食われて握り潰されないようにする（Exception 派生だと
    アラームが消費されるだけで対局が続き、上限が効かない）。"""


def _init(sims, max_steps, timeout=0):
    from cpu_arena import _load_db
    _G["db"] = _load_db()
    _G["sims"] = sims
    _G["max_steps"] = max_steps
    _G["timeout"] = timeout


def _audit_one(job):
    """1件分: 生成デッキで1局回して結果を返す（子プロセスで実行）。

    job は `leader_id`（ミラー）または `(l1, l2, seed)`（交差対面）。アリーナの void
    （終局しないペア）は**交差対面でだけ**出たので、ミラーと同じ器で対面を非対称にできるようにする。
    """
    from cpu_arena import _arena_seat
    from game_driver import run_game
    from deck_synth import synth_deck_builder
    db, sims, max_steps = _G["db"], _G["sims"], _G["max_steps"]
    if isinstance(job, tuple):
        l1, l2, gseed = job
        dseed = gseed           # 交差対面はデッキ中身も件ごとに振る
        label = f"{l1}×{l2}"
    else:
        l1 = l2 = job
        gseed, dseed = 4242, 0  # ミラーは従来どおり（既存の監査結果と地続き）
        label = job
    seen = {}
    moves = {}

    def wrap(fn, who):
        def g(ctx):
            m = getattr(ctx, "manager", None)
            src = None
            if m is not None:
                ai = getattr(m, "active_interaction", None)
                if ai:
                    src = ai.get("source_card_name") if isinstance(ai, dict) else None
                    if src:
                        seen[src] = seen.get(src, 0) + 1
            mv = fn(ctx)
            # **繰り返している「手」**も数える。対話を伴わないループ（同じ起動メインの連打等）は
            # source_card_name が付かないため、対話の集計だけでは犯人が見えない。
            key = (who, src, str(mv)[:110])
            moves[key] = moves.get(key, 0) + 1
            return mv
        return g

    def _hot():
        return {"dialogs": sorted(seen.items(), key=lambda kv: -kv[1])[:2],
                "moves": [[f"{k[0]} {k[1] or '-'} {k[2]}", n]
                          for k, n in sorted(moves.items(), key=lambda kv: -kv[1])[:3]]}

    # 手数ではなく**実時間**の上限。手数上限では捕まらない「1手の中で終わらない」暴走
    # （探索がバトル箱の中で回り続ける等）を timeout として記録し、監査全体を止めない。
    if _G.get("timeout"):
        import signal

        def _alarm(_sig, _frm):
            raise _GameTimeout(f"game exceeded {_G['timeout']}s")
        signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(int(_G["timeout"]))
    try:
        res = run_game(gseed, db, seats={
            "p1": wrap(_arena_seat("learned", None, None, 1, None, None, None, sims), "p1"),
            "p2": wrap(_arena_seat("learned", None, None, 1, None, None, None, sims), "p2")},
            deck_builder=synth_deck_builder(l1, l2, seed=gseed),
            max_steps=max_steps, legal_moves="skip", invariants="raise")
        row = {"leader": label, "status": "ok",
               "turns": getattr(res, "turns", None), "steps": getattr(res, "steps", None)}
        if _G.get("timeout"):
            import signal as _sg
            _sg.alarm(0)
        return row
    except _GameTimeout as e:
        h = _hot()
        return {"leader": label, "status": "timeout", "error": str(e),
                "hot_dialogs": h["dialogs"], "hot_moves": h["moves"]}
    except Exception as e:
        kind = type(e).__name__
        h = _hot()
        # InvariantError は「上限手数」と「手の適用中に例外」の両方を運ぶ。前者だけが hang
        # （終わらないループ）で、後者は素の欠陥＝error。混ぜると原因の切り分けができない。
        status = "hang" if "MAX_STEPS" in str(e) else "error"
        return {"leader": label, "status": status, "error": f"{kind}: {str(e)[:80]}",
                "hot_dialogs": h["dialogs"], "hot_moves": h["moves"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=8, help="探索数（監査は軽くて良い）")
    ap.add_argument("--max-steps", type=int, default=700)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="先頭N体だけ（0=全部）")
    ap.add_argument("--timeout", type=int, default=0,
                    help="1局あたりの実時間上限（秒。0=無制限）")
    ap.add_argument("--cross", type=int, default=0,
                    help="ミラーではなく**交差対面**をN件監査する（アリーナの void はここでしか出ない）")
    ap.add_argument("--cross-seed", type=int, default=0, help="交差対面の引き直し基点")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from cpu_arena import _load_db
    db = _load_db()
    leaders = sorted(cid for cid in db.raw_db
                     if db.get_card(cid) is not None and db.get_card(cid).type.name == "LEADER")
    if args.limit:
        leaders = leaders[:args.limit]
    if args.cross:
        # アリーナと同じ引き方（`promotion_gate._leader_pair`）で対面を決める＝void の再現条件と
        # 揃える。件ごとに seed を変えるので、同じ対面でも別の展開を踏める。
        import random as _rnd
        jobs = []
        for i in range(args.cross):
            s = args.cross_seed + i
            rng = _rnd.Random(s * 7919 + 13)
            jobs.append((rng.choice(leaders), rng.choice(leaders), s))
        print(f"生成デッキ監査（交差対面）: {len(jobs)}件・sims={args.sims}・上限{args.max_steps}手",
              flush=True)
    else:
        jobs = leaders
        print(f"生成デッキ監査: {len(leaders)}リーダー・sims={args.sims}・上限{args.max_steps}手",
              flush=True)

    rows = []
    with mp.Pool(args.workers, initializer=_init, initargs=(args.sims, args.max_steps, args.timeout)) as pool:
        for r in pool.imap_unordered(_audit_one, jobs):
            rows.append(r)
            if r["status"] != "ok":
                print(f"  {r['leader']}: {r['status']} {r.get('error','')} "
                      f"{r.get('hot_dialogs','')} {r.get('hot_moves','')}", flush=True)
            if len(rows) % 20 == 0:
                print(f"  ...{len(rows)}/{len(jobs)}", flush=True)

    st = collections.Counter(r["status"] for r in rows)
    culprit = collections.Counter()
    for r in rows:
        for name, n in (r.get("hot_dialogs") or []):
            culprit[name] += 1
    turns = [r["turns"] for r in rows if r["status"] == "ok" and r.get("turns")]
    print(f"\n結果: ok={st['ok']} hang={st['hang']} timeout={st['timeout']} error={st['error']}")
    if turns:
        print(f"正常決着のターン数: 平均{sum(turns)/len(turns):.1f} 最小{min(turns)} 最大{max(turns)}")
    print("ハング時に繰り返していた対話の発生元（＝修正/除外の候補）:")
    for name, n in culprit.most_common(12):
        print(f"  {n:>3} リーダーで {name}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"rows": rows, "summary": dict(st),
                       "culprits": culprit.most_common()}, f, ensure_ascii=False, indent=1)
    print("DECK_SYNTH_AUDIT_DONE " + json.dumps(dict(st), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    _sys.exit(main())
