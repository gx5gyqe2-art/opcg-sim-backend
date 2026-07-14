"""GATE A: 三目並べでAZループ(自己対戦→学習→重み更新＋NN誘導MCTS)が既知最適へ収束するか。

docs/.../cpu_rl_pilot_plan_20260629.md GATE A。RLループ機械の**実装正しさ**を保証する単体テスト。
合否（三目並べの最適＝完全プレイに負けない）:
  ① 改善: gen_final が gen0 にヘッドツーヘッドで明確勝ち越し（policy improvement が機能）。
  ② 最適収束: gen_final が完全プレイ(ミニマックス)に **0敗**（＝最適）。
  ③ 対ランダム: gen_final の敗北率 ~0。
通れば「自己対戦生成→バッチ学習→重み更新」のループは健全＝OPCGの陰性を実装バグから切り離せる。
実行: OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/gate_a_tictactoe.py
"""
import argparse

import numpy as np

import os as _os, sys as _sys  # noqa: E402  test bootstrap (sys.path + google stub)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap  # noqa: E402,F401
import tictactoe as T
from az_loop import run_generations, play_match, net_agent, random_agent


def evaluate(game, nets, eval_sims, c_puct, n_random, n_perfect, n_h2h, seed=123):
    rng = np.random.default_rng(seed)
    gen0 = net_agent(game, nets[0], eval_sims, c_puct)
    final = net_agent(game, nets[-1], eval_sims, c_puct)
    perfect = lambda s, r: T.perfect_action(game, s, r)
    rnd = random_agent(game)

    vs_rand = play_match(game, final, rnd, n_random, rng)
    vs_perf = play_match(game, final, perfect, n_perfect, rng)
    h2h = play_match(game, final, gen0, n_h2h, rng)
    g0_rand = play_match(game, gen0, rnd, n_random, rng)
    g0_perf = play_match(game, gen0, perfect, n_perfect, rng)
    return vs_rand, vs_perf, h2h, g0_rand, g0_perf


def run_gate(gens=8, games=200, sims=60, eval_sims=100, c_puct=1.5,
             hidden=64, epochs=20, seed=0, log=print):
    """GATE A 本体。返り値 (ok: bool, summary: dict)。pytest からも呼べる。"""
    game = T.TicTacToe()
    log(f"=== GATE A: 三目並べ AZループ (gens={gens} games/gen={games} sims={sims}) ===")
    nets = run_generations(game, gens, games, sims, c_puct,
                           hidden=hidden, epochs=epochs, seed=seed, log=log)
    vs_rand, vs_perf, h2h, g0_rand, g0_perf = evaluate(
        game, nets, eval_sims, c_puct, n_random=100, n_perfect=40, n_h2h=60)

    log("\n--- 評価（eval_sims=%d・ノイズ無し・先後交互） ---" % eval_sims)
    log(f"gen0   vs ランダム : {g0_rand}")
    log(f"gen0   vs 完全プレイ: {g0_perf}")
    log(f"final  vs ランダム : {vs_rand}")
    log(f"final  vs 完全プレイ: {vs_perf}")
    log(f"final  vs gen0     : {h2h}")

    # 改善は「完全プレイ(オラクル)への敗北数」で測る。引き分けゲームゆえ最適同士は引分＝
    # final vs gen0 のヘッドツーヘッドは無情報（h2h は参考表示）。loop が suboptimal→optimal へ
    # 動かしたか＝対完全プレイの敗北が gen0 から final で減ったか、を改善の証拠とする。
    improve = vs_perf["a_loss"] < g0_perf["a_loss"]
    optimal = vs_perf["a_loss"] == 0
    beats_rand = vs_rand["a_loss"] == 0
    log("\n--- 判定 ---")
    log(f"① 改善(対完全プレイ敗北 減) : {'OK' if improve else 'NG'}  "
        f"(gen0敗{g0_perf['a_loss']} → final敗{vs_perf['a_loss']} / {sum(vs_perf.values())})")
    log(f"② 最適収束(完全に0敗)     : {'OK' if optimal else 'NG'}  (敗{vs_perf['a_loss']}/{sum(vs_perf.values())})")
    log(f"③ 対ランダム0敗           : {'OK' if beats_rand else 'NG'}  (敗{vs_rand['a_loss']}/{sum(vs_rand.values())})")
    ok = improve and optimal and beats_rand
    log(f"\nGATE A: {'PASS ✅ RLループ機械は健全' if ok else 'FAIL ❌ ループに問題'}")
    return ok, {"vs_rand": vs_rand, "vs_perf": vs_perf, "g0_perf": g0_perf, "h2h": h2h}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=int, default=8)
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--sims", type=int, default=60)
    ap.add_argument("--eval-sims", type=int, default=100)
    ap.add_argument("--c-puct", type=float, default=1.5)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    ok, _ = run_gate(args.gens, args.games, args.sims, args.eval_sims,
                     args.c_puct, args.hidden, args.epochs, args.seed)
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
