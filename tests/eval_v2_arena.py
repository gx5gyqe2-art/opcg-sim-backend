"""評価 v2（L1コア・v0.4）の強さ A/B 計測ハーネス（dev専用・段階導入§9）。

challenger=評価v2 ON / baseline=評価v2 OFF（＝現行の手書きJ値評価）を**同一難易度 hard**で対戦させ、
席を交互に入替えて先手有利を相殺し、勝率・Elo・Wilson 区間を出す。評価以外（探索・プラン・情報方針）は
両者同一なので、差分は**評価関数そのものの強さ**になる。

注意: v2 は first cut（係数未チューニング）。本ハーネスは「現状の v2 がどれだけか」を測る土台であって、
良し悪しの結論は係数チューニング（SPSA）後の再計測まで保留する。

実行例:
    OPCG_LOG_SILENT=1 python tests/eval_v2_arena.py --games 30 --difficulty hard
"""
import argparse
import random
import sys

import conftest  # noqa: F401
import cpu_arena
from cpu_arena import _load_db, play_game, win_rate, elo_delta, elo_ci


def run(games: int, difficulty: str, seed0: int, max_steps: int):
    db = _load_db()
    wins = 0.0
    rows = []
    for i in range(games):
        seed = seed0 + i
        chal_is_p1 = (i % 2 == 0)               # 席交互で先手有利を相殺
        # challenger = 評価v2 ON、baseline = 評価v2 OFF。両者とも同一 difficulty。
        p1_v2, p2_v2 = (True, False) if chal_is_p1 else (False, True)
        res = play_game(seed, db, difficulty, difficulty, max_steps=max_steps,
                        p1_eval_v2=p1_v2, p2_eval_v2=p2_v2)
        chal_seat = "p1" if chal_is_p1 else "p2"
        won = (res["winner"] == chal_seat)
        wins += 1.0 if won else 0.0
        rows.append((seed, chal_seat, res["winner"], won, res["turns"]))
        print(f"  game {i+1}/{games} seed={seed} chal_seat={chal_seat} "
              f"winner={res['winner']} v2_won={won} turns={res['turns']}")
    wr = win_rate(wins, games)
    ci = elo_ci(wins, games)
    print(f"\n=== 評価v2 ON vs OFF（{difficulty}・{games}局・席交互） ===")
    print(f"v2 勝率 = {wins:.0f}/{games} = {wr:.3f}  |  Elo = {elo_delta(wr):+.0f}  "
          f"(Elo95% [{ci['elo_lo']:+.0f}, {ci['elo_hi']:+.0f}])")
    print("判定: " + ("v2 が有意に強い" if ci["elo_lo"] > 0 else
                      "v2 が有意に弱い" if ci["elo_hi"] < 0 else
                      "ノイズ帯（有意差なし）＝係数チューニング前として想定内"))
    return wr


def main(argv=None):
    ap = argparse.ArgumentParser(description="評価 v2 ON vs OFF 強さA/B")
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--difficulty", choices=["hard", "expert"], default="hard")
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=cpu_arena.DEFAULT_MAX_STEPS)
    args = ap.parse_args(argv)
    run(args.games, args.difficulty, args.seed0, args.max_steps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
