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


def run(mode, pairs, seed0, max_steps, iters, worlds, horizon, alpha, eval_v2=False):
    from arena_parallel import paired_play
    base_mcts = {"iters": iters, "horizon": 2, "worlds": 1, "determinize": True}
    leaf = "L1(eval_v2)" if eval_v2 else "J値"
    if mode == "vs-hard":
        # 本番 hard = PIMC K=4・予算75/世界（Dockerfile OPCG_PIMC_WORLDS=4 / OPCG_HARD_PER_MOVE_BUDGET=75・+53Elo）。
        kw = dict(challenger_difficulty="expert", baseline_difficulty="hard",
                  challenger_mcts=base_mcts, baseline_pimc=4, baseline_budget=75)
        label = f"expert(iters={iters},h2,w1,det)  vs  hard(α-β・本番PIMC K=4/予算75)"
    elif mode == "equalize":
        # 探索ロジック以外を揃える: 隠れ情報=両側単一世界（hard PIMC=1 / expert worlds=1）・plan=両側あり・
        # 葉=両側 L1（--eval-v2 前提）。残る差分は α-β(h4) vs マクロMCTS(h2,160反復)＝探索ロジックのみ。
        kw = dict(challenger_difficulty="expert", baseline_difficulty="hard",
                  challenger_mcts=base_mcts, baseline_pimc=1,
                  challenger_force_plan=True, baseline_force_plan=True)
        label = f"expert(MCTS h2/{iters}反復・plan) vs hard(α-β h4・PIMC=1・plan)  ＝探索ロジックのみの差"
    elif mode == "equalize-d":
        # equalize に加え **深さも揃える**: hard の horizon を 2 に下げて expert(h2) と一致。
        # 残差＝同一深さ2での α-β(厳密・ビーム) vs マクロMCTS(サンプリング)＝純粋なアルゴリズム形状の差。
        kw = dict(challenger_difficulty="expert", baseline_difficulty="hard",
                  challenger_mcts=base_mcts, baseline_pimc=1,
                  baseline_search=(2, 52),   # hard を horizon=2 へ（max_ply は h2 で非拘束）
                  challenger_force_plan=True, baseline_force_plan=True)
        label = f"expert(MCTS h2/{iters}反復・plan) vs hard(α-β h2・PIMC=1・plan)  ＝同一深さ2・アルゴリズムのみの差"
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

    label += f"  [葉={leaf}・両側]"
    t0 = time.time()
    res = paired_play(pairs, seed0=seed0, max_steps=max_steps,
                      challenger_eval_v2=eval_v2, baseline_eval_v2=eval_v2, **kw)
    dt = time.time() - t0
    failed = res.get("failed_games", 0)
    print(f"\n=== expert 伸びしろ A/B [{mode}]（要求{pairs}ペア / 成立{res['pairs']}ペア={res['games']}局・対照ペア・葉=J値固定） ===")
    print(f"{label}")
    if failed:
        print(f"⚠ 失敗局 {failed} 件（採点から除外）。例: " + " | ".join(res.get("errors", [])[:3]))
    if not res["pair_scores"]:
        print("有効ペアが 0＝全滅。エラーを確認。")
        return None
    ci = _pair_level_ci(res["pair_scores"])
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
    ap.add_argument("--mode", choices=["vs-hard", "equalize", "equalize-d", "worlds", "horizon", "blend"], default="vs-hard")
    ap.add_argument("--pairs", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=cpu_arena.DEFAULT_MAX_STEPS)
    ap.add_argument("--iters", type=int, default=160, help="MCTS 総反復（固定・再現性）")
    ap.add_argument("--worlds", type=int, default=4, help="worlds モードの challenger 世界数")
    ap.add_argument("--horizon", type=int, default=3, help="horizon モードの challenger ホライズン")
    ap.add_argument("--alpha", type=float, default=0.3, help="blend モードの challenger 学習価値ブレンド率")
    ap.add_argument("--eval-v2", action="store_true", help="両側の葉評価を L1(eval_v2) にする")
    args = ap.parse_args(argv)
    run(args.mode, args.pairs, args.seed0, args.max_steps, args.iters, args.worlds, args.horizon, args.alpha,
        eval_v2=args.eval_v2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
