"""生成デッキの実プレイ監査（2026-08-15）: 全リーダーで自己対戦し**ハングするカードを洗い出す**。

なぜ要るか: `deck_synth` の生成デッキで自己対戦したところ 6戦中3戦が上限手数まで終わらず、
原因はステージ「聖地マリージョア」の対象選択が解決せず `RESOLVE_EFFECT_SELECTION` が
無限ループすることだった。**固定ハンニャバルデッキ（ステージ0枚・イベント0枚）では一度も
通らない経路**で、歴代のアリーナ・ゲート・自己対戦では検出できなかった実バグ。
生成デッキはカードDBの広い範囲を実プレイに乗せるので、この種の欠陥を掘り当てられる。

各リーダーについて「そのリーダーのデッキ同士のミラー」を1局回し、
  ok        … 正常決着（ターン数も記録）
  hang      … 上限手数に到達（**繰り返している対話の source_card_name を記録＝犯人**）
  error     … 例外（種別とメッセージを記録）
を集計する。犯人カードの一覧がそのまま修正対象／暫定除外リストになる。

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/deck_synth_audit.py \\
    --sims 8 --max-steps 700 --workers 4 --out /tmp/deck_audit.json
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


def _init(sims, max_steps):
    from cpu_arena import _load_db
    _G["db"] = _load_db()
    _G["sims"] = sims
    _G["max_steps"] = max_steps


def _audit_one(leader_id):
    """1リーダー分: 生成デッキのミラーを1局回して結果を返す（子プロセスで実行）。"""
    from cpu_arena import _arena_seat
    from game_driver import run_game
    from deck_synth import synth_deck_builder
    db, sims, max_steps = _G["db"], _G["sims"], _G["max_steps"]
    seen = {}

    def wrap(fn):
        def g(ctx):
            m = getattr(ctx, "manager", None)
            if m is not None:
                ai = getattr(m, "active_interaction", None)
                if ai:
                    src = ai.get("source_card_name") if isinstance(ai, dict) else None
                    if src:
                        seen[src] = seen.get(src, 0) + 1
            return fn(ctx)
        return g

    try:
        res = run_game(4242, db, seats={
            "p1": wrap(_arena_seat("learned", None, None, 1, None, None, None, sims)),
            "p2": wrap(_arena_seat("learned", None, None, 1, None, None, None, sims))},
            deck_builder=synth_deck_builder(leader_id, leader_id),
            max_steps=max_steps, legal_moves="skip", invariants="raise")
        return {"leader": leader_id, "status": "ok",
                "turns": getattr(res, "turns", None), "steps": getattr(res, "steps", None)}
    except Exception as e:
        kind = type(e).__name__
        hot = sorted(seen.items(), key=lambda kv: -kv[1])[:2]
        status = "hang" if ("MAX_STEPS" in str(e) or kind == "InvariantError") else "error"
        return {"leader": leader_id, "status": status, "error": f"{kind}: {str(e)[:80]}",
                "hot_dialogs": hot}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=8, help="探索数（監査は軽くて良い）")
    ap.add_argument("--max-steps", type=int, default=700)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="先頭N体だけ（0=全部）")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from cpu_arena import _load_db
    db = _load_db()
    leaders = sorted(cid for cid in db.raw_db
                     if db.get_card(cid) is not None and db.get_card(cid).type.name == "LEADER")
    if args.limit:
        leaders = leaders[:args.limit]
    print(f"生成デッキ監査: {len(leaders)}リーダー・sims={args.sims}・上限{args.max_steps}手",
          flush=True)

    rows = []
    with mp.Pool(args.workers, initializer=_init, initargs=(args.sims, args.max_steps)) as pool:
        for r in pool.imap_unordered(_audit_one, leaders):
            rows.append(r)
            if r["status"] != "ok":
                print(f"  {r['leader']}: {r['status']} {r.get('error','')} "
                      f"{r.get('hot_dialogs','')}", flush=True)
            if len(rows) % 20 == 0:
                print(f"  ...{len(rows)}/{len(leaders)}", flush=True)

    st = collections.Counter(r["status"] for r in rows)
    culprit = collections.Counter()
    for r in rows:
        for name, n in (r.get("hot_dialogs") or []):
            culprit[name] += 1
    turns = [r["turns"] for r in rows if r["status"] == "ok" and r.get("turns")]
    print(f"\n結果: ok={st['ok']} hang={st['hang']} error={st['error']}")
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
