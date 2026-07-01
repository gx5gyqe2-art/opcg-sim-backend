"""自己対戦スループット実測（GPUなしGO の可否を数字で詰めるための一回限りの計測・dev）。

3水準を1コアで測る:
  raw   = ランダム方策（純エンジンのロールアウト速度＝Rust移植が直接効く層）
  ai40  = 現CPU(budget=40,pimc=1)＝RL自己対戦に近い「探索付き意思決定」コスト
  ai_full = 現CPUの本走設定（既定budget・pimc=4）＝参考（最重）

出力: 1コアあたり games/sec・steps/sec・1手あたり秒。Rust係数とコア数を外挿して
ローカル日産局数を見積もる。実行:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/rl_throughput.py --games 8
"""
import argparse
import time

import conftest  # noqa: F401
from opcg_sim.src.core import cpu_ai
from cpu_selfplay import _load_db, run_one_game


def bench(db, label, games, seed0, policy, budget):
    if budget is not None:
        cpu_ai.set_budget_override(budget)
    steps = 0
    fin = 0
    t0 = time.perf_counter()
    try:
        for g in range(games):
            try:
                res = run_one_game(seed0 + g, db, policy=policy, difficulty="hard")
            except Exception as e:
                print(f"  [{label}] seed={seed0+g} 中断: {type(e).__name__}")
                continue
            steps += res["steps"]; fin += 1
    finally:
        if budget is not None:
            cpu_ai.set_budget_override(None)
    dt = time.perf_counter() - t0
    if fin == 0:
        print(f"{label:8s}: 完走0/{games}"); return None
    gps = fin / dt
    sps = steps / dt
    print(f"{label:8s}: {fin}/{games}完走  {dt:6.1f}s  "
          f"{gps:7.3f} games/s  {sps:8.1f} steps/s  "
          f"{steps/fin:5.0f} steps/game  {dt/steps*1000:6.2f} ms/step")
    return {"games_per_s": gps, "steps_per_s": sps, "steps_per_game": steps / fin}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--skip-full", action="store_true", help="ai_full(最重)を省く")
    args = ap.parse_args()
    db = _load_db()
    print(f"=== 自己対戦スループット 1コア実測 (games={args.games}) ===", flush=True)
    r_raw = bench(db, "raw", args.games, args.seed0, "random", None)
    r_ai = bench(db, "ai40", args.games, args.seed0, "ai", 40)
    r_full = None if args.skip_full else bench(db, "ai_full", max(2, args.games // 2), args.seed0, "ai", None)

    print("\n=== 外挿（GPUなしGO・ローカルCPU） ===")
    cores = 4
    for name, r in (("raw", r_raw), ("ai40", r_ai), ("ai_full", r_full)):
        if not r:
            continue
        g = r["games_per_s"]
        for rust in (1, 50, 100):
            day = g * rust * cores * 86400
            tag = "現Python" if rust == 1 else f"Rust×{rust}"
            print(f"  {name:8s} {tag:9s} {cores}コア: {day:12,.0f} 局/日")


if __name__ == "__main__":
    main()
