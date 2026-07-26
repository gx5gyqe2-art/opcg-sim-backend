"""固定N・帯層別の CRN アリーナ判定器（v16・一次判定の正本）。

背景（`docs/reports/cpu_v15_ensemble_power_20260726.md`）: 本セッションの判定は 24〜120局で行われ、
**検定力不足**だった（真の 0.55 を 0.5 と区別するには約800局が必要）。実際「60局で有望→確証で消える」
を2回（c_puct 0.567→0.458／アンサンブル 0.583→0.442）経験している。一方で実測コストは
**6.8 s/局＝800局で約1.5時間**であり、**必要な精度は買える**ことも判明した。

`promotion_gate.py` は昇格運用の逐次ゲート（stage1 24局で早期棄却）で、固定N・帯層別・ペア水準CI を
出さない。本スクリプトは**判定専用の計器**として分離する（CLAUDE.md「1トピック=1ファイル」）。
対局実行は promotion_gate の実績ある席入替CRN（`_init_pool`/`_play_pair`）を import して再利用する。

判定:
  - 一次スクリーン（既定 48ペア=96局・SE 5.1pp）: wr < --screen-floor で早期棄却（無駄な1.5時間を防ぐ）
  - 本判定（既定 400ペア=800局・SE 1.77pp）: wr ≥ --frac **かつ** ペア水準95%CI下限 > 0.50
  - **帯層別**（既定4帯）: 帯ごとに離れた seed 基点で回し帯間ばらつきを併記する
    ＝単一帯の偶然を人間が目視で検出できるようにする（v15 の帯間 0.442-0.583 の教訓）

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/arena_gate.py \
    --candidate /tmp/cand/value.npz,/tmp/cand/policy.npz --pairs 400 --workers 4
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


def plan_bands(pairs, bands, seed_base, stride=100000):
    """帯ごとに離れた seed 基点へ pairs を等分する（pure）。返り値 [[seed,...], ...]。

    帯間で seed 空間を大きく離すことで、デッキ/初期配置の偶然が帯をまたいで共有されないようにする
    （v15: 隣接 seed 帯でも 0.442-0.583 と振れた＝帯内相関を疑う根拠）。"""
    per = pairs // bands
    rem = pairs - per * bands
    out = []
    for b in range(bands):
        n = per + (1 if b < rem else 0)
        base = seed_base + b * stride
        out.append(list(range(base, base + n)))
    return out


def screen_decision(wins, games, floor):
    """一次スクリーン判定（pure）: 勝率が floor 未満なら reject。"""
    return "reject" if (wins / max(games, 1)) < floor else "continue"


def final_decision(pair_scores, frac=0.55):
    """本判定（pure）: 勝率 ≥ frac かつ **ペア水準95%CI下限 > 0.50**。

    2条件にするのは、点推定だけだと標本誤差で通ってしまうため（v15 の反省）。CI は
    `arena_parallel._pair_level_ci`（対照ペア設計の正しい区間）を使う。"""
    from arena_parallel import _pair_level_ci
    ci = _pair_level_ci(pair_scores)
    return bool(ci["win_rate"] >= frac and ci["lo"] > 0.50), ci


def _run(pool, seeds):
    from promotion_gate import _play_pair
    return [s for s in pool.imap_unordered(_play_pair, seeds)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True, help="value.npz[,policy.npz]")
    ap.add_argument("--baseline", default="", help="空=出荷既定（現 gen6）")
    ap.add_argument("--pairs", type=int, default=400, help="本判定のペア数（×2局）")
    ap.add_argument("--bands", type=int, default=4, help="seed 帯の数（帯間ばらつきの可視化）")
    ap.add_argument("--seed-base", type=int, default=71000)
    ap.add_argument("--screen-pairs", type=int, default=48,
                    help="一次スクリーンのペア数（0 で無効）")
    ap.add_argument("--screen-floor", type=float, default=0.48)
    ap.add_argument("--frac", type=float, default=0.55, help="本判定の勝率しきい")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=None, help="JSON 保存先")
    args = ap.parse_args()

    from promotion_gate import _init_pool
    t_all = time.time()
    res = {"candidate": args.candidate, "baseline": args.baseline or "gen6(default)"}
    with mp.Pool(args.workers, initializer=_init_pool,
                 initargs=(args.candidate, args.baseline)) as pool:
        # --- 一次スクリーン（無駄な本判定を避ける安全弁） ---
        if args.screen_pairs > 0:
            sseeds = list(range(args.seed_base - 500000, args.seed_base - 500000 + args.screen_pairs))
            t0 = time.time()
            sc = _run(pool, sseeds)
            swins, sgames = sum(sc), 2 * len(sc)
            verdict = screen_decision(swins, sgames, args.screen_floor)
            print(f"screen: {swins:.1f}/{sgames} wr={swins / sgames:.3f} → {verdict} "
                  f"({time.time() - t0:.0f}s)", flush=True)
            res["screen"] = {"wins": swins, "games": sgames, "wr": round(swins / sgames, 4),
                             "verdict": verdict}
            if verdict == "reject":
                res["promoted"] = False
                res["sec"] = int(time.time() - t_all)
                print(f"ARENA_GATE_RESULT {json.dumps(res, ensure_ascii=False)}", flush=True)
                if args.out:
                    json.dump(res, open(args.out, "w"), ensure_ascii=False)
                return 0
        # --- 本判定（帯層別） ---
        all_scores, band_rows = [], []
        for bi, seeds in enumerate(plan_bands(args.pairs, args.bands, args.seed_base)):
            t0 = time.time()
            sc = _run(pool, seeds)
            all_scores += sc
            w, g = sum(sc), 2 * len(sc)
            band_rows.append({"band": bi, "wins": w, "games": g, "wr": round(w / g, 4)})
            print(f"band{bi}: {w:.1f}/{g} wr={w / g:.3f} ({time.time() - t0:.0f}s)", flush=True)
    ok, ci = final_decision(all_scores, args.frac)
    games = 2 * len(all_scores)
    res.update({"promoted": ok, "games": games, "wins": sum(all_scores),
                "wr": round(ci["win_rate"], 4),
                "ci95": [round(ci["lo"], 4), round(ci["hi"], 4)],
                "elo": round(ci["elo"], 1), "bands": band_rows,
                "band_spread": round(max(b["wr"] for b in band_rows)
                                     - min(b["wr"] for b in band_rows), 4),
                "sec": int(time.time() - t_all)})
    print(f"総合: {sum(all_scores):.1f}/{games} wr={ci['win_rate']:.3f} "
          f"CI[{ci['lo']:.3f},{ci['hi']:.3f}] 帯間差={res['band_spread']:.3f} "
          f"→ {'PASS' if ok else 'FAIL'}", flush=True)
    print(f"ARENA_GATE_RESULT {json.dumps(res, ensure_ascii=False)}", flush=True)
    if args.out:
        json.dump(res, open(args.out, "w"), ensure_ascii=False)
    return 0


if __name__ == "__main__":
    _sys.exit(main())
