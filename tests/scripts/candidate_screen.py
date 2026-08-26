"""候補ネットの一次スクリーニング（**レイテンシ関門つき**・2026-08-15）。

重い検証（コーチゲート・アリーナ）の**前段**に置く足切り。順序に意味がある:

  1. **decide レイテンシ**（本番 sims160・予算1秒）… 2026-08-15 の実害への処方。v11 候補は
     decide **13.5s＝予算の28倍**だったのに、ns2（事前計算した行列でのバッチ予測）でも
     コーチゲート（1点ずつの判定）でも遅さが表面化せず、**アリーナが 10分/ペアになって
     初めて発覚**した（33時間コース）。符号化コストは v9=1.3ms / v10=25.1ms（リーサル距離Δの
     台本再生）/ v12=1.3ms で、探索は1手で数百回符号化する＝**1盤面のコストが桁で効く**。
  2. **ns2 相関**（レフェリー24世界ラベル・接戦帯と全帯）… 盤面評価の質
  3. **裁定3点**（m1@3 展開／m1@14 入口は素通し／m1@15 払い切る）… 交換レートの要所

判定は出さない（数字を並べるだけ）。正式判定はコーチゲート（`coach_gate.py`）とアリーナ
（`arena_resume.py`）が行う。

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/candidate_screen.py \\
    --candidate /tmp/cand/value.npz,/tmp/cand/policy.npz --ns2 /tmp/ns2_v12.npz
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import time

import numpy as np

import os as _os, sys as _sys  # noqa: E402
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: E402,F401

import coach_gate as CG  # noqa: E402
import counterfactual_referee as CR  # noqa: E402
import mark_gate as MG  # noqa: E402
import replay_reeval as RE  # noqa: E402
from cpu_selfplay import _load_db  # noqa: E402

RULING_POINTS = (3, 14, 15)     # m1 の交換レート裁定（展開/入口素通し/払い切る）


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", default="", help="value.npz[,policy.npz]（空=出荷既定）")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--sims", type=int, default=160)
    ap.add_argument("--budget", type=float, default=1.0, help="decide の予算（秒）")
    ap.add_argument("--ns2", default="", help="ns2 評価行列 npz（空＝相関を測らない）")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    CG.ARGS = argparse.Namespace(seeds=args.seeds, sims=args.sims)
    CR.ARGS = argparse.Namespace(true_board=True)

    from opcg_sim.src.core.cpu_learned import LearnedEngine
    parts = [p for p in args.candidate.split(",") if p]
    eng = LearnedEngine(value_path=parts[0] if parts else None,
                        policy_path=parts[1] if len(parts) > 1 else None)
    db = _load_db()
    res = {"enc_version": eng.enc_version,
           "battle_head": bool(eng.vnet.has_exit_head("battle"))}
    print(f"enc_v={res['enc_version']} battle_head={res['battle_head']}", flush=True)

    replays = {**MG.REPLAYS, **CG.REPLAYS_V2, **CG.REPLAYS_V48, **CG.REPLAYS_HUMAN}
    CR.GAMES = {}
    raw = RE.load_replay_json(replays["m1"])
    rec = raw.get("replay", raw)
    CR.GAMES["m1"] = (rec, {f.get("action_index"): f for f in raw.get("frames") or []},
                      rec["actions"])

    def board(i):
        built = CR._restore_board(db, "m1", i)
        if isinstance(built, str):
            r, fbi, acts = CR.GAMES["m1"]
            built = MG._restore(db, r, fbi, acts, i)
        m0, who = built
        name = who if isinstance(who, str) else who.name
        return m0, (m0.p1 if m0.p1.name == name else m0.p2)

    # --- 関門1: レイテンシ
    m0, actor = board(RULING_POINTS[0])
    eng._world_seeds = {}
    eng.decide(m0, actor, sims=8, rng=np.random.default_rng(1))          # ウォーム
    t0 = time.time()
    for s in range(3):
        eng._world_seeds = {}
        getattr(eng, "_battle_plans", {}).clear()
        eng.decide(m0, actor, sims=args.sims, rng=np.random.default_rng(100 + s))
    lat = (time.time() - t0) / 3
    res["latency_s"] = round(lat, 3)
    res["latency_ok"] = bool(lat <= args.budget)
    print(f"LATENCY {lat:.2f} s/decide (sims{args.sims})  "
          f"{'OK' if res['latency_ok'] else f'OVER BUDGET({args.budget}s)'}", flush=True)

    # --- 関門2: ns2 相関
    if args.ns2 and os.path.exists(args.ns2):
        z = np.load(args.ns2)
        p = eng.vnet.predict({k: z[k] for k in ("scalars", "field", "card_idx")})
        ev = z["value"]
        mid = np.abs(ev) < 0.999
        res["ns2_all_r"] = round(float(np.corrcoef(p, ev)[0, 1]), 4)
        res["ns2_mid_r"] = round(float(np.corrcoef(p[mid], ev[mid])[0, 1]), 4)
        print(f"ns2: all n={len(ev)} r={res['ns2_all_r']:+.3f} / "
              f"mid n={int(mid.sum())} r={res['ns2_mid_r']:+.3f}", flush=True)

    # --- 関門3: 裁定点
    accepts = {i: a for (t, i, a) in CG.VERIFIED_V2 if t == "m1"}
    res["rulings"] = {}
    for i in RULING_POINTS:
        if i not in accepts:
            continue
        m0, actor = board(i)
        rate = CG.decide_rate(eng, m0, actor, accepts[i], args.seeds, args.sims)
        res["rulings"][f"m1@{i}"] = round(rate, 3)
        print(f"m1@{i}: rate={rate:.2f}", flush=True)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
    print("CANDIDATE_SCREEN_DONE " + json.dumps(res, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    _sys.exit(main())
