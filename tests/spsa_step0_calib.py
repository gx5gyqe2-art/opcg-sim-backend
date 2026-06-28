"""SPSA Step 0: ノイズフロア較正（dev・計画 §3.3）。

SPSA 本走の前に、現行（最終形）評価関数で「対局評価のブレ（σ）」と「高レバレッジ係数を ±c 振った信号」を
実測し、SNR から 摂動 c / games-per-eval の妥当性を判断する。推測でなく実測で c を決めるための足場。

- ノイズフロア σ: challenger==baseline（同一係数）の win_rate を別 seed で reps 回測り、その標準偏差。
  CRN＋席交互の効果込みの実効ノイズ。理想は ~0.5 まわりに分布。
- 信号: 高レバレッジ係数 V2_W_LIFE_PRECIOUS を ±c 振った A/B の |win_rate − 0.5|。SPSA が1評価で拾う
  勾配信号の目安。signal ≳ 2σ なら c/games は十分（埋もれない）。

実行: OPCG_LOG_SILENT=1 PYTHONPATH=tests pypy3 tests/spsa_step0_calib.py --games 40 --c 0.2 --reps 5
"""
import argparse
import statistics
import time

import conftest  # noqa: F401
from arena_parallel import paired_play
from opcg_sim.src.core import cpu_eval_v2 as V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=40, help="1評価の総局数（pairs=games//2）")
    ap.add_argument("--c", type=float, default=0.2, help="摂動の乗数振り幅（±c）")
    ap.add_argument("--reps", type=int, default=5, help="ノイズフロアの反復回数")
    ap.add_argument("--param", default="V2_W_LIFE_PRECIOUS", help="信号測定する高レバレッジ係数")
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()
    pairs = max(1, args.games // 2)
    w = args.workers or None
    t0 = time.time()

    # 1) ノイズフロア: 同一係数（差なし）で別 seed の win_rate を reps 回。
    floor = []
    for k in range(args.reps):
        r = paired_play(pairs, seed0=k * 1000, workers=w,
                        challenger_coeffs={}, baseline_coeffs={})
        floor.append(r["win_rate"])
        print(f"  noise rep{k}: win_rate={r['win_rate']:.3f}", flush=True)
    sigma = statistics.pstdev(floor) if len(floor) > 1 else 0.0
    mean = statistics.mean(floor)

    # 2) 信号: param を ±c 振った A/B（challenger=×(1+c) / baseline=×(1-c)）。
    base = float(getattr(V, args.param))
    sig = paired_play(pairs, seed0=7777, workers=w,
                      challenger_coeffs={args.param: base * (1 + args.c)},
                      baseline_coeffs={args.param: base * (1 - args.c)})
    signal = abs(sig["win_rate"] - 0.5)
    dt = time.time() - t0

    print("\n=== SPSA Step 0 ノイズフロア較正 ===")
    print(f"games/eval={args.games}（{pairs}ペア）・c={args.c}・reps={args.reps}・{dt:.0f}s")
    print(f"ノイズフロア: mean={mean:.3f} σ={sigma:.4f}（同一係数の win_rate ばらつき）")
    print(f"信号: {args.param} ±{args.c} → win_rate={sig['win_rate']:.3f}・|偏差|={signal:.4f}")
    snr = (signal / sigma) if sigma > 1e-9 else float('inf')
    print(f"SNR = signal/σ = {snr:.2f}  （≳2 で c/games 十分・<2 なら c↑ or games↑）")


if __name__ == "__main__":
    main()
