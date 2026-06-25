"""評価 v2（L1コア）の強さ A/B 計測ハーネス（dev専用・段階導入§9）。

**分散低減版（C・Phase0）**: 独立 N 局でなく **対照ペア（antithetic）** で測る＝`arena_paired` を使い、各 seed を
両席で1回ずつ（同一 game-seed・`separate_policy_rng=True`）戦わせて席順（先手有利）とデッキ非対称をペア内で
相殺する。さらに CI は**ペア単位スコア**（{0,0.5,1}）から算出する＝相関を無視する素朴な Bernoulli CI より狭い
（＝分散低減が数値に出る）。

challenger=評価v2 ON / baseline=評価v2 OFF（現行の手書きJ値評価）を同一難易度 hard で対戦。

実行例:
    OPCG_LOG_SILENT=1 python tests/eval_v2_arena.py --pairs 15
    OPCG_LOG_SILENT=1 python tests/eval_v2_arena.py --pairs 15 --compare   # 独立法と分散を比較
"""
import argparse
import math
import random
import sys

import conftest  # noqa: F401
import cpu_arena
from cpu_arena import _load_db, play_game, arena_paired, win_rate, elo_delta, elo_ci


def _pair_level_ci(pair_scores):
    """ペア単位スコア（{0,0.5,1}）の平均と 95% CI（正規近似）→ 勝率と Elo 区間。

    対照ペア設計の正しい CI＝ペアを独立試行として扱う（2×pairs 局を独立 Bernoulli 扱いする素朴 CI より狭い）。
    """
    n = len(pair_scores)
    mean = sum(pair_scores) / n
    var = sum((s - mean) ** 2 for s in pair_scores) / max(1, n - 1)
    half = 1.96 * math.sqrt(var / n)
    lo, hi = max(0.0, mean - half), min(1.0, mean + half)
    return {"win_rate": mean, "lo": lo, "hi": hi,
            "elo": elo_delta(mean), "elo_lo": elo_delta(lo), "elo_hi": elo_delta(hi)}


def run_paired(pairs, seed0, max_steps):
    from arena_parallel import paired_play
    res = paired_play(pairs, seed0=seed0, max_steps=max_steps)   # コア並列・現行係数
    pair_scores = res["pair_scores"]
    ci = _pair_level_ci(pair_scores)
    print(f"\n=== 評価v2 ON vs OFF（hard・{pairs}ペア={2*pairs}局・対照ペア） ===")
    print(f"v2 勝率 = {ci['win_rate']:.3f}  |  Elo = {ci['elo']:+.0f}")
    print(f"  ペア単位CI（分散低減・正）  : Elo95% [{ci['elo_lo']:+.0f}, {ci['elo_hi']:+.0f}]")
    naive = elo_ci(res["win_rate"] * res["games"], res["games"])
    print(f"  素朴Bernoulli CI（参考・広い）: Elo95% [{naive['elo_lo']:+.0f}, {naive['elo_hi']:+.0f}]")
    print("判定: " + ("v2 が有意に強い" if ci["elo_lo"] > 0 else
                      "v2 が有意に弱い" if ci["elo_hi"] < 0 else "互角（有意差なし）"))
    return ci


def run_independent(games, seed0, max_steps):
    """旧・独立法（席交互・別シード）。分散比較用に残す。"""
    db = _load_db()
    wins = 0.0
    for i in range(games):
        seed = seed0 + i
        chal_is_p1 = (i % 2 == 0)
        p1_v2, p2_v2 = (True, False) if chal_is_p1 else (False, True)
        res = play_game(seed, db, "hard", "hard", max_steps=max_steps, p1_eval_v2=p1_v2, p2_eval_v2=p2_v2)
        wins += 1.0 if res["winner"] == ("p1" if chal_is_p1 else "p2") else 0.0
    ci = elo_ci(wins, games)
    print(f"\n=== [独立法] v2 ON vs OFF（hard・{games}局・別シード席交互） ===")
    print(f"v2 勝率 = {wins/games:.3f}  |  Elo = {elo_delta(wins/games):+.0f}  "
          f"(Bernoulli CI [{ci['elo_lo']:+.0f}, {ci['elo_hi']:+.0f}])")
    return wins / games


def main(argv=None):
    ap = argparse.ArgumentParser(description="評価 v2 ON vs OFF 強さA/B（分散低減・対照ペア）")
    ap.add_argument("--pairs", type=int, default=15, help="対照ペア数（総局数=2×pairs）")
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=cpu_arena.DEFAULT_MAX_STEPS)
    ap.add_argument("--compare", action="store_true", help="同局数で独立法との分散を比較")
    args = ap.parse_args(argv)
    run_paired(args.pairs, args.seed0, args.max_steps)
    if args.compare:
        run_independent(2 * args.pairs, args.seed0, args.max_steps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
