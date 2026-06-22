"""Phase 1（切り分け実験・docs/reports/cpu_strength_roadmap_20260622.md §4）。

「深く読めば fair が cheat に勝率回復するか」を **horizon 掃引**で測り、限界が**探索か情報欠損か**を
切り分ける。探索ノブ（OPCG_HARD_HORIZON 等）は `cpu_ai` の import 時に確定するので、設定ごとに
**別プロセス**で `cpu_arena.py arena-paired` を起動して掃引する（このスクリプトはその driver）。

判定:
  - fair の vs cheat 勝率が horizon とともに**単調増/上昇** → 深さが効く＝**探索が限界**
    （→ Phase 2A 探索路線 / Phase 4 TF を前倒し）。
  - **頭打ち/平坦** → 深く読んでも情報欠損を埋められない＝**情報/評価が限界**
    （→ Phase 2 PIMC＝決定化で隠れ情報を補う／Phase 3 評価）。

使い方（重い・実機本走は手動/定期）:
    OPCG_LOG_SILENT=1 python tests/phase1_sweep.py --pairs 40 --horizons 2,3,4,6 --seed 0

注: challenger=fair・baseline=cheat を同一プロセス（同一 global horizon）で対戦させるので、両者が
同じ H で深くなる。よって信号は「H を上げると fair-vs-cheat の**差が縮むか**」＝深さが fair に
相対的に効くか。完全な単独効果（fair の H だけ変える）には per-decider 探索設定が要る（将来）。
"""
import argparse
import os
import re
import subprocess
import sys
from typing import List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_LINE = re.compile(r"win_rate=([0-9.]+)\s+Elo=([+-]?\d+)\s+\[([+-]?\d+),\s*([+-]?\d+)\]\s+half=(\d+)")


def run_one(horizon: int, pairs: int, seed: int, max_steps: Optional[int]) -> Tuple[float, int, int]:
    """1 設定（global horizon=H）で fair[challenger] vs cheat[baseline] の arena-paired を別プロセス実行。

    戻り値 (win_rate, elo, half_width)。出力末尾の集計行を正規表現で回収する。
    """
    env = dict(os.environ)
    env["OPCG_HARD_HORIZON"] = str(horizon)
    env.setdefault("OPCG_LOG_SILENT", "1")
    cmd = [sys.executable, os.path.join(_HERE, "cpu_arena.py"), "arena-paired",
           "--challenger", "hard", "--baseline", "hard",
           "--challenger-policy", "fair", "--baseline-policy", "cheat",
           "--pairs", str(pairs), "--seed", str(seed)]
    if max_steps is not None:
        cmd += ["--max-steps", str(max_steps)]
    out = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=_HERE)
    m = None
    for line in out.stdout.splitlines():
        mm = _LINE.search(line)
        if mm:
            m = mm
    if m is None:
        raise RuntimeError(f"arena-paired 出力を解析できず (horizon={horizon}):\n{out.stdout}\n{out.stderr}")
    return float(m.group(1)), int(m.group(2)), int(m.group(5))


def sweep(horizons: List[int], pairs: int, seed: int, max_steps: Optional[int]) -> None:
    rows = []
    for h in horizons:
        wr, elo, half = run_one(h, pairs, seed, max_steps)
        rows.append((h, wr, elo, half))
        print(f"  horizon={h:>2}  fair-vs-cheat win_rate={wr:.3f}  Elo={elo:+d}  half={half}", flush=True)
    print("\n--- 切り分け判定 ---")
    elos = [r[2] for r in rows]
    rising = all(elos[i] <= elos[i + 1] + 5 for i in range(len(elos) - 1)) and (elos[-1] - elos[0] > 15)
    if rising:
        print("fair-vs-cheat Elo が horizon とともに上昇 → 深さが効く＝**探索が限界**寄り"
              "（Phase 2A 探索路線 / Phase 4 TT 前倒しを検討）。")
    else:
        print("fair-vs-cheat Elo は horizon を上げても頭打ち/平坦 → 深く読んでも情報欠損を埋められない"
              "＝**情報/評価が限界**寄り（Phase 2 PIMC＝決定化 / Phase 3 評価へ）。")
    print("注: CI 半幅が大きいと判定不能。pairs を増やして再走すること。")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Phase 1 切り分け実験: horizon 掃引で探索/情報の限界を切り分ける")
    ap.add_argument("--horizons", default="2,3,4,6", help="カンマ区切りの horizon 値")
    ap.add_argument("--pairs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=None)
    args = ap.parse_args(argv)
    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    print(f"Phase 1 horizon 掃引: horizons={horizons} pairs={args.pairs} seed={args.seed}", flush=True)
    sweep(horizons, args.pairs, args.seed, args.max_steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
