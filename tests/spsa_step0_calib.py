"""SPSA Step 0: ノイズフロア較正（dev・計画 §3.3）。

SPSA 本走の前に、高レバレッジ係数を ±c 振った A/B を**複数 seed で反復**し、その win_rate の
**平均（=信号）とばらつき σ（=ノイズフロア）**から SNR を出して、摂動 c / games-per-eval の妥当性を判断する。
推測でなく実測で c を決める足場。

注意（設計）: 「同一係数の A/B」は CRN＋席交互で対称＝ペアが必ず 0.5 になり σ=0（null のばらつきはゼロ）。
よってノイズは**係数差を入れた A/B を別 seed で反復**したときの win_rate のばらつきで測る（これが SPSA の
1評価が抱える実効ノイズ）。signal（平均偏差）≳ 2σ なら c/games は十分（勾配が埋もれない）。

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
    base = float(getattr(V, args.param))
    t0 = time.time()

    # 高レバレッジ係数 param を ±c 振った A/B（challenger=×(1+c) / baseline=×(1-c)）を別 seed で reps 回。
    # win_rate の平均=信号（偏差）、ばらつき σ=実効ノイズフロア。
    wrs = []
    for k in range(args.reps):
        r = paired_play(pairs, seed0=k * 1000, workers=w,
                        challenger_coeffs={args.param: base * (1 + args.c)},
                        baseline_coeffs={args.param: base * (1 - args.c)})
        wrs.append(r["win_rate"])
        print(f"  rep{k}: win_rate={r['win_rate']:.3f}", flush=True)
    mean = statistics.mean(wrs)
    sigma = statistics.pstdev(wrs) if len(wrs) > 1 else 0.0
    signal = abs(mean - 0.5)
    dt = time.time() - t0

    print("\n=== SPSA Step 0 ノイズフロア較正 ===")
    print(f"param={args.param}・games/eval={args.games}（{pairs}ペア）・c={args.c}・reps={args.reps}・{dt:.0f}s")
    print(f"win_rate 平均={mean:.3f}（信号 |偏差|={signal:.4f}）／σ={sigma:.4f}（実効ノイズフロア）")
    snr = (signal / sigma) if sigma > 1e-9 else float('inf')
    print(f"SNR = signal/σ = {snr:.2f}  （≳2 で c/games 十分・<2 なら c↑ or games↑）")
    # 参考: この pairs での片側評価の理論ノイズ（二項上界）
    import math
    print(f"参考: 二項上界 σ≈sqrt(0.25/{pairs})={math.sqrt(0.25/pairs):.4f}（CRN で実測σはこれ以下が期待）")


if __name__ == "__main__":
    main()
