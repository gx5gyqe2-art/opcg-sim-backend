"""expert（MCTS）側の伸びしろ（L1 の外）を測る A/B ハーネス（dev専用）。

本番デフォルト CPU は **expert = MCTS**（`app.py` 既定 cpu_difficulty="expert"）。expert は既に
**determinize=True（公平 ISMCTS）・horizon=2・worlds=1** で動く。L1 葉評価が互角天井なので、強さの伸びしろは
**探索側**にある仮説を、expert の未使用レバーで測る。3 モード:

  --mode vs-hard : expert（本番相当・固定反復）vs hard（α-β）。**どちらのデフォルトが強いか**を確定。
  --mode worlds  : expert worlds=K vs worlds=1（**同一総反復**＝多世界アンサンブルの純粋な価値＝真ISMCTS近似）。
  --mode horizon : expert horizon=3 vs 2（読むターン数を1つ深く）。

どちらの席も評価は出荷 J値（eval_v2 OFF）。再現性のため MCTS は固定反復（deadline_ms=None）で測る
（本番は壁時計2秒デッドラインだが、A/B は反復を固定して決定論で比較）。対照ペア＋コア並列＋ペア単位CI。

実行例:
    OPCG_LOG_SILENT=1 python tests/expert_arena.py --mode vs-hard --pairs 20 --iters 160
    OPCG_LOG_SILENT=1 python tests/expert_arena.py --mode worlds  --pairs 20 --iters 160 --worlds 4
    OPCG_LOG_SILENT=1 python tests/expert_arena.py --mode horizon --pairs 20 --iters 160
"""
import argparse
import sys
import time

import conftest  # noqa: F401
import cpu_arena
from cpu_arena import elo_ci
from eval_v2_arena import _pair_level_ci


def run(mode, pairs, seed0, max_steps, iters, worlds, horizon, alpha):
    from arena_parallel import paired_play
    base_mcts = {"iters": iters, "horizon": 2, "worlds": 1, "determinize": True}
    if mode == "vs-hard":
        kw = dict(challenger_difficulty="expert", baseline_difficulty="hard",
                  challenger_mcts=base_mcts)
        label = f"expert(iters={iters},h2,w1,det)  vs  hard(α-β)"
    elif mode == "blend":
        # MCTS×学習価値ブレンド: challenger α>0（学習葉混ぜる）vs baseline α=0（純eval）。両側 expert・同設定。
        kw = dict(challenger_difficulty="expert", baseline_difficulty="expert",
                  challenger_mcts=base_mcts, baseline_mcts=base_mcts,
                  challenger_alpha=alpha, baseline_alpha=0.0)
        label = f"expert blend α={alpha}  vs  α=0  (iters={iters}・h2・w1・det)"
    elif mode == "worlds":
        kw = dict(challenger_difficulty="expert", baseline_difficulty="expert",
                  challenger_mcts={**base_mcts, "worlds": worlds},
                  baseline_mcts={**base_mcts, "worlds": 1})
        label = f"expert worlds={worlds}  vs  worlds=1  (同一総反復 iters={iters}・h2・det)"
    elif mode == "horizon":
        kw = dict(challenger_difficulty="expert", baseline_difficulty="expert",
                  challenger_mcts={**base_mcts, "horizon": horizon},
                  baseline_mcts={**base_mcts, "horizon": 2})
        label = f"expert horizon={horizon}  vs  horizon=2  (iters={iters}・w1・det)"
    else:
        raise SystemExit(f"unknown mode: {mode}")

    t0 = time.time()
    res = paired_play(pairs, seed0=seed0, max_steps=max_steps,
                      challenger_eval_v2=False, baseline_eval_v2=False, **kw)
    dt = time.time() - t0
    ci = _pair_level_ci(res["pair_scores"])
    print(f"\n=== expert 伸びしろ A/B [{mode}]（{pairs}ペア={2*pairs}局・対照ペア・葉=J値固定） ===")
    print(f"{label}")
    print(f"challenger 勝率 = {ci['win_rate']:.3f}  |  Elo = {ci['elo']:+.0f}")
    print(f"  ペア単位CI（分散低減・正）  : Elo95% [{ci['elo_lo']:+.0f}, {ci['elo_hi']:+.0f}]")
    naive = elo_ci(res["win_rate"] * res["games"], res["games"])
    print(f"  素朴Bernoulli CI（参考・広い）: Elo95% [{naive['elo_lo']:+.0f}, {naive['elo_hi']:+.0f}]")
    print("判定: " + ("challenger が有意に強い" if ci["elo_lo"] > 0 else
                     "challenger が有意に弱い" if ci["elo_hi"] < 0 else "互角（有意差なし）"))
    print(f"壁時計: {dt:.1f}s / {res['games']}局 / {res['workers']}並列 ({dt/res['games']:.1f}s/局)")
    return ci


def main(argv=None):
    ap = argparse.ArgumentParser(description="expert(MCTS) 側の伸びしろ A/B（対照ペア）")
    ap.add_argument("--mode", choices=["vs-hard", "worlds", "horizon", "blend"], default="vs-hard")
    ap.add_argument("--pairs", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=cpu_arena.DEFAULT_MAX_STEPS)
    ap.add_argument("--iters", type=int, default=160, help="MCTS 総反復（固定・再現性）")
    ap.add_argument("--worlds", type=int, default=4, help="worlds モードの challenger 世界数")
    ap.add_argument("--horizon", type=int, default=3, help="horizon モードの challenger ホライズン")
    ap.add_argument("--alpha", type=float, default=0.3, help="blend モードの challenger 学習価値ブレンド率")
    args = ap.parse_args(argv)
    run(args.mode, args.pairs, args.seed0, args.max_steps, args.iters, args.worlds, args.horizon, args.alpha)
    return 0


if __name__ == "__main__":
    sys.exit(main())
