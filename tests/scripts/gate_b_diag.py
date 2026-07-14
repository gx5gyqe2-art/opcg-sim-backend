"""GATE B 診断: 価値信号の大きさ・MCTS対ランダム・探索の集中度（健全性の切り分け）。"""
import time
import numpy as np

import os as _os, sys as _sys  # noqa: E402  test bootstrap (sys.path + google stub)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap  # noqa: E402,F401
from opcg_sim.src.core import cpu_ai
from opcg_game import OPCGGame
from az_mcts_tree import TreeMCTS
from cpu_selfplay import _load_db
from gate_b_opcg import mcts_agent, random_agent, play_one


def main():
    db = _load_db()
    game = OPCGGame()
    rnd = random_agent(game)

    # ① evaluate() の生スコア分布（value_scale 妥当性）。
    print("=== ① L1 evaluate 生スコア分布（中盤局面） ===")
    scores = []
    for seed in range(4):
        m = game.new_game(db, seed)
        for _ in range(12):
            name = game.current_player(m)
            if name is None:
                break
            scores.append(cpu_ai.evaluate(m, name, see_opp_hand=False))
            mv = rnd(m, name, np.random.default_rng(seed))
            if mv is None:
                break
            try:
                cpu_ai._apply_move_inplace(m, name, mv)
            except Exception:
                break
    s = np.array(scores)
    print(f"  n={len(s)} 範囲[{s.min():.2f},{s.max():.2f}] 中央{np.median(s):.2f} "
          f"|score|中央{np.median(np.abs(s)):.2f}  tanh(score/6)中央={np.median(np.tanh(s/6)):.3f} "
          f"飽和率(|tanh|>0.9)={np.mean(np.abs(np.tanh(s/6))>0.9):.2f}")

    # ② MCTS(sims) vs ランダム＝探索が機械的に効くか（効けば圧勝のはず）。
    print("\n=== ② MCTS vs ランダム（先後交互6戦） ===")
    for sims in (60, 200):
        ag = mcts_agent(game, sims=sims)
        w = d = loss = 0
        t0 = time.perf_counter()
        for i in range(3):
            for a_is_p1 in (True, False):
                m = game.new_game(db, 500 + i)
                rng = np.random.default_rng(i * 3 + a_is_p1)
                first = ag if a_is_p1 else rnd
                second = rnd if a_is_p1 else ag
                res = play_one(game, m, first, second, rng)
                if res is None:
                    d += 1
                elif (res == "p1") == a_is_p1:
                    w += 1
                else:
                    loss += 1
        print(f"  MCTS{sims} vs random: {w}勝 {d}分 {loss}敗 / 6  ({time.perf_counter()-t0:.0f}s)", flush=True)

    # ③ 探索の集中度: sims を増やすとルート訪問分布が尖るか（健全なら尖る）。
    print("\n=== ③ ルート訪問分布の集中度（中盤局面・sims別） ===")
    m = game.new_game(db, 7)
    for _ in range(10):
        name = game.current_player(m)
        if name is None:
            break
        mv = rnd(m, name, np.random.default_rng(7))
        if mv is None:
            break
        try:
            cpu_ai._apply_move_inplace(m, name, mv)
        except Exception:
            break
    name = game.current_player(m)
    for sims in (30, 90, 270):
        mc = TreeMCTS(game, value_fn=game.value, n_sims=sims,
                      determinize_fn=lambda s, r: game.determinize(s, name, r),
                      rng=np.random.default_rng(0))
        _, N = mc.run(m)
        N = np.asarray(N); p = N / N.sum()
        top = np.sort(p)[::-1][:3]
        ent = -(p[p > 0] * np.log(p[p > 0])).sum()
        print(f"  sims={sims:>3}: 合法{len(N)} top3訪問率={[round(x,2) for x in top]} エントロピー={ent:.2f}")


if __name__ == "__main__":
    main()
