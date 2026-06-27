"""効果価値（§S2・実シミュレーション）の強さ A/B 計測（dev専用）: challenger=効果価値ON / baseline=OFF を
同一難易度 hard で head-to-head 対戦し、勝率→Elo を出す。席交互（先手有利相殺）・別シード独立法。

challenger は `set_effect_override((eff_w, cost_w))` で効果価値項＋コスト項を有効化、baseline は (0,0)＝従来の
手書きJ値評価。両者とも同じ α-β/PIMC コアパス（cpu_arena.play_game）で進行。

実行例:
    OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/effect_arena.py --games 40 --eff 1.0 --cost 120
    OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/effect_arena.py --games 40 --eff 1.0 --cost 0    # 効果のみ
    OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/effect_arena.py --games 40 --eff 0   --cost 120  # コストのみ
"""
import argparse
import math
import multiprocessing as mp
import os
import sys

import conftest  # noqa: F401
import cpu_arena
from cpu_arena import _load_db, play_game, DEFAULT_MAX_STEPS, win_rate, elo_delta, elo_ci

_DB = None
_CFG = {}


def _init(cfg):
    global _DB, _CFG
    _DB = _load_db()
    _CFG = cfg


def _one(i):
    seed = _CFG["seed0"] + i
    chal_is_p1 = (i % 2 == 0)              # 席交互＝先手有利を相殺
    eff = (_CFG["eff"], _CFG["cost"])
    off = (0.0, 0.0)
    p1_effect, p2_effect = (eff, off) if chal_is_p1 else (off, eff)
    try:
        res = play_game(seed, _DB, "hard", "hard", max_steps=_CFG["max_steps"],
                        p1_pimc=_CFG["pimc"], p2_pimc=_CFG["pimc"],
                        p1_budget=_CFG["budget"], p2_budget=_CFG["budget"],
                        p1_effect=p1_effect, p2_effect=p2_effect)
    except Exception:
        return None
    if res is None or res.get("winner") is None:
        return None
    chal = "p1" if chal_is_p1 else "p2"
    return 1.0 if res["winner"] == chal else 0.0


def main(argv=None):
    ap = argparse.ArgumentParser(description="効果価値 A/B アリーナ（hard・challenger=効果ON vs baseline=OFF）")
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--eff", type=float, default=1.0, help="効果価値Δの重み（0=コストのみ）")
    ap.add_argument("--cost", type=float, default=0.0, help="コスト1あたりの重み（0=効果のみ）")
    ap.add_argument("--pimc", type=int, default=4)
    ap.add_argument("--budget", type=int, default=75)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    args = ap.parse_args(argv)

    cfg = {"seed0": args.seed0, "eff": args.eff, "cost": args.cost, "pimc": args.pimc,
           "budget": args.budget, "max_steps": args.max_steps}
    workers = args.workers or max(1, (os.cpu_count() or 2) - 1)

    wins = 0.0
    n = 0
    with mp.Pool(workers, initializer=_init, initargs=(cfg,)) as pool:
        for i, r in enumerate(pool.imap_unordered(_one, range(args.games)), 1):
            if r is None:
                continue
            n += 1
            wins += r
            if i % 10 == 0:
                print(f"  {i}/{args.games} … valid={n} chal_wins={wins:.0f}", flush=True)

    if n == 0:
        print("有効局なし")
        return 1
    wr = wins / n
    ci = elo_ci(wins, n)
    print(f"\n=== 効果価値 A/B（hard・{n}局・席交互） eff={args.eff} cost={args.cost} pimc={args.pimc} ===")
    print(f"challenger（効果ON）勝率 = {wr:.3f}  |  Elo = {elo_delta(wr):+.0f}  "
          f"(95% CI [{ci['elo_lo']:+.0f}, {ci['elo_hi']:+.0f}])")
    print("判定: " + ("効果ON が有意に強い" if ci["elo_lo"] > 0 else
                      "効果ON が有意に弱い" if ci["elo_hi"] < 0 else "互角（有意差なし）"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
