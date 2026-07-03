"""Gen K の学習エージェント(value+policy+MCTS) を 製品L1+α-β+PIMC と直接対戦（参考測定）。

計画外の寄り道（P3の損切りは世代間で回す）。だが「学習エージェントは出荷CPUより強いか」に
直球で答える。P2の必要条件チェック（SL-net+MCTS vs L1 = 0.450, sims160/pimc4）と同条件にして
直接比較可能にする。water-oil注意つき（α-β評価器 vs NN-MCTS）。

実行: OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/p3_vs_l1.py --gen 2 --pairs 20 --sims 160 --pimc 4
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import math
import subprocess
import time

import os as _os, sys as _sys  # noqa: E402  test bootstrap (sys.path + google stub)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap  # noqa: E402,F401
import rl_encoder as E
import rl_net as RN
from az_policy import PolicyScorer
from opcg_game import OPCGGame
from cpu_selfplay import _load_db
import p3_loop as P
import p2_gen0 as P2

WT, CK, BR = "/tmp/p3ckpt-wt", "/tmp/p3ckpt-wt/p3ckpt", "claude/p3-checkpoints"
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def ensure_wt():
    if not os.path.exists(WT + "/.git"):
        subprocess.run(["git", "-C", REPO, "worktree", "prune"], capture_output=True)
        subprocess.run(["git", "-C", REPO, "fetch", "origin", BR], capture_output=True)
        subprocess.run(["git", "-C", REPO, "worktree", "add", WT, BR], capture_output=True)
    subprocess.run(["git", "-C", WT, "fetch", "origin", BR], capture_output=True)
    subprocess.run(["git", "-C", WT, "reset", "--hard", "origin/" + BR], capture_output=True)


def wilson(p, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", type=int, default=2)
    ap.add_argument("--pairs", type=int, default=20)
    ap.add_argument("--sims", type=int, default=160)
    ap.add_argument("--pimc", type=int, default=4)
    ap.add_argument("--c-puct", type=float, default=1.5)
    args = ap.parse_args()

    ensure_wt()
    db = _load_db()
    vocab = E.build_vocab(db)
    game = OPCGGame()
    vnet = RN.ValueNet.load(CK + f"/gen{args.gen}_value.npz")
    pnet_path = CK + f"/gen{args.gen}_policy.npz"
    pnet = PolicyScorer.load(pnet_path) if os.path.exists(pnet_path) else None
    print(f"Gen{args.gen} net ロード（policy={'あり' if pnet else 'なし(uniform)'}）", flush=True)

    gen_act = P._agent(game, vnet, pnet, vocab, args.sims, args.c_puct)
    l1_factory = lambda: P2.l1_agent_factory("hard", args.pimc)
    print(f"=== Gen{args.gen}+MCTS(sims={args.sims}) vs 製品L1+α-β(pimc={args.pimc}) "
          f"CRN {args.pairs}ペア×2={args.pairs*2}戦 ===", flush=True)
    print(f"（比較基準: P2の SL/Gen0 vs L1 = 0.450）", flush=True)
    t0 = time.perf_counter()
    r = P2.match(game, db, gen_act, l1_factory, args.pairs)
    n = r["games"]
    p = (r["sl_win"] + 0.5 * r["draw"]) / n
    lo, hi = wilson(p, n)
    print(f"\nGen{args.gen} 勝率={p:.3f}  95%CI=[{lo:.3f},{hi:.3f}]  {r}  ({time.perf_counter()-t0:.0f}s)")
    print(f"→ {'製品CPUを上回る' if lo > 0.5 else ('互角圏' if p > 0.42 else '製品CPUに及ばず')}"
          f"（P2/Gen0の0.450と比較）。※参考値・water-oil注意。")
    return 0


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("fork", force=True)
    import sys
    sys.exit(main())
