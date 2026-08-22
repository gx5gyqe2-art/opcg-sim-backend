"""V8 フリー攻撃族のターン出口教師（系統3・π非依存・2026-08-22）。

段3裁定 504004@43（有利な攻撃「リーダー→ホーリー」を残して TURN_END）を一般化した
支配ペア。**リーダーで格下のレストキャラを取る攻撃は実質フリー**——リーダーは戦闘で
失われず、レスト状態に守備上の失点もない（防御にアクティブは不要・ブロッカーは
キャラのみ）。相手がカウンターで守れば手札を消費させ、守らなければキャラを失う。
どちらに転んでも素通し閉幕より悪くならない＝論理支配。

適用条件（公開情報のみ・pure 判定は free_kill_target）:
  - 自リーダーがアクティブ
  - 相手の場に「レスト済み・パワー ≤ リーダーパワー」のキャラがいる（最大パワーを選ぶ）
  - 相手の場に**アクティブなブロッカーがいない**（ブロックで空振りに変えられる盤面は
    「厳密に得」と言い切れないため対を立てない＝系統2の教訓: 厳密支配でないものは注入しない）

対: good=[("ATTACK", リーダー, 対象uuid)]→閉幕（+0.5） vs bad=素通し閉幕（−0.5）。
実現は `plan.scripted_plan`（ATTACK 意図の対象指定は本族のために追加した3要素形）・
出口は `plan.execute_plan`＝serve と同一規約。戦闘対話（カウンター等）は現行ネットで
解決されるがラベルは順位ルールなので π 汚染しない。

出力: `planatk_*.npz`（plandom 互換）。

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests:tests/harness python tests/scripts/plan_freeatk_gen.py \
    --games 24 --seed-base 740000 --workers 4 --out /tmp/planatk
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import multiprocessing as mp

import numpy as np

import os as _os, sys as _sys  # noqa: E402  test bootstrap
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: E402,F401

from opcg_sim.src.core import cpu_ai
from opcg_sim.src.learned import plan as PL
from plan_lethal_gen import _MainCap, _Done, cut_at_group  # 同型ハーネスを再利用

MAX_MAIN_PER_TURN = 2


def _power(c, attacking):
    try:
        return int(c.get_power(attacking))
    except Exception:
        return 0


def free_kill_target(manager, name):
    """(リーダー, 取れる最大の相手レストキャラ) か None（公開情報のみ・pure）。"""
    p = manager.p1 if getattr(manager.p1, "name", None) == name else manager.p2
    o = manager.p2 if p is manager.p1 else manager.p1
    lead = getattr(p, "leader", None)
    if lead is None or getattr(lead, "is_rest", False):
        return None
    lp = _power(lead, True)
    for c in (getattr(o, "field", None) or []):
        if not getattr(c, "is_rest", False) and c.has_keyword("ブロッカー") \
                and _power(c, False) > lp:
            return None      # リーダーで取れないアクティブブロッカー＝空振りに変えられる。
                             # 取れるブロッカーなら、ブロックされても獲物が変わるだけ＝支配は維持
                             # （504004@43 の実測 2026-08-22: ブロッカー1000 vs リーダー5000）
    best = None
    for c in (getattr(o, "field", None) or []):
        if not getattr(c, "is_rest", False):
            continue
        cp = _power(c, False)
        if cp <= lp and (best is None or cp > _power(best, False)):
            best = c
    return (lead, best) if best is not None else None


_G = {}


def _init(sims, enc_version):
    from cpu_arena import _load_db
    from opcg_sim.src.core import cpu_learned as CL
    _G["db"] = _load_db()
    eng = CL.LearnedEngine(sims=sims)
    _G["eng"] = eng
    _G["vf"] = CL._value_fn(eng.vnet, eng.vocab, eng.enc_version)
    _G["pf"] = CL._priors_fn(eng.pnet, eng.vocab, eng.enc_version)
    _G["enc_version"] = enc_version or eng.enc_version


def _run_game(job):
    seed, sims = job
    from game_driver import run_game, make_seat
    from promotion_gate import _leader_pair
    from deck_synth import synth_deck_builder
    from opcg_sim.src.learned import encoder as E
    db, eng = _G["db"], _G["eng"]
    la, lb = _leader_pair(db, seed, "random")
    cap = _MainCap(limit_decisions=200)   # ターン内上限は plan_lethal_gen 側の定数（=2）
    seat = make_seat(kind="learned", want_trace=False, sims=sims, engine=eng)
    try:
        run_game(seed, db, seats={"p1": seat, "p2": seat},
                 deck_builder=synth_deck_builder(la, lb, seed=seed),
                 observers=(cap,), max_steps=1500, legal_moves="skip",
                 invariants="raise", stop_after_decisions=200)
    except _Done:
        pass
    except BaseException as e:
        return {"seed": seed, "error": f"{type(e).__name__}: {str(e)[:80]}", "rows": []}

    rows = []
    gi = 0
    hits = 0
    for dec, turn, name, frame in cap.frames:
        try:
            fk = free_kill_target(frame, name)
            if fk is None:
                continue
            hits += 1
            lead, tgt = fk
            sg = PL.scripted_plan(eng.game, frame.clone(), name,
                                  [("ATTACK", lead.uuid, tgt.uuid)], _G["vf"], _G["pf"])
            if not sg:
                continue                     # 対象指定攻撃が列挙に無い＝実現不能
            good = PL.execute_plan(eng.game, frame.clone(), name, list(sg),
                                   _G["vf"], _G["pf"], battle_value_fn=None)
            bad = PL.execute_plan(eng.game, frame.clone(), name, [],
                                  _G["vf"], _G["pf"], battle_value_fn=None)
            eg = E.encode(good, name, eng.vocab, version=_G["enc_version"])
            eb = E.encode(bad, name, eng.vocab, version=_G["enc_version"])
        except BaseException:
            continue
        if np.array_equal(eg["scalars"], eb["scalars"]) and \
                np.array_equal(eg["field"], eb["field"]):
            continue
        rows.append((eg, +0.5, gi))
        rows.append((eb, -0.5, gi))
        gi += 1
    return {"seed": seed, "error": None, "rows": rows,
            "frames": len(cap.frames), "hits": hits}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=24)
    ap.add_argument("--seed-base", type=int, default=740000)
    ap.add_argument("--sims", type=int, default=160, help="局面採取の自己対戦（分布=本番仕様）")
    ap.add_argument("--enc-version", type=int, default=0, help="0=エンジンの符号化世代に従う")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--shard-size", type=int, default=64)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    jobs = [(args.seed_base + i, args.sims) for i in range(args.games)]
    buf, stats = [], {"pairs": 0, "hits": 0, "errors": 0}
    shard = [0]

    def _flush(chunk):
        if not chunk:
            return
        arrays = {
            "scalars": np.stack([r[0]["scalars"] for r in chunk]).astype(np.float32),
            "field": np.stack([r[0]["field"] for r in chunk]).astype(np.float32),
            "card_idx": np.stack([r[0]["card_idx"] for r in chunk]).astype(np.int32),
            "value": np.array([r[1] for r in chunk], dtype=np.float32),
            "group": np.array([r[2] for r in chunk], dtype=np.int64),
        }
        np.savez_compressed(os.path.join(args.out, f"planatk_{shard[0]:05d}.npz"), **arrays)
        shard[0] += 1

    with mp.Pool(args.workers, initializer=_init,
                 initargs=(args.sims, args.enc_version)) as pool:
        for res in pool.imap_unordered(_run_game, jobs):
            if res["error"]:
                stats["errors"] += 1
                print(f"  seed {res['seed']}: {res['error']}", flush=True)
                continue
            gbase = res["seed"] * 1000
            for enc, z, g in res["rows"]:
                buf.append((enc, z, gbase + g))
            stats["pairs"] += len(res["rows"]) // 2
            stats["hits"] += res.get("hits", 0)
            print(f"  seed {res['seed']}: 判断点{res.get('frames', 0)}"
                  f"・適用点{res.get('hits', 0)}・対 {len(res['rows']) // 2}", flush=True)
            while len(buf) >= args.shard_size:
                chunk, buf = cut_at_group(buf, args.shard_size)
                _flush(chunk)
    _flush(buf)
    print(f"PLAN_FREEATK_DONE pairs={stats['pairs']} hits={stats['hits']} "
          f"errors={stats['errors']} shards={shard[0]} -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    _sys.exit(main())
