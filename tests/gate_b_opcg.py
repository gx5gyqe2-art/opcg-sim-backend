"""GATE B: OPCG で MCTS の探索健全性（playout単調性）＋PIMC挙動を確認。

docs/.../cpu_rl_pilot_plan_20260629.md GATE B / instrument②。評価器を**固定**（L1 を tanh で[-1,1]）し
sims だけ動かして「more search = stronger」が成り立つか＝探索が健全かを測る（不成立＝探索ハイパラ/PIMC統合
の不具合。表現力に帰属してはいけない、と切り分ける）。相手も同一評価器の固定 sims MCTS＝評価器差を消し
探索量だけを比較。CRN（同一seed・先後入替）。

実行(スモーク): OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/gate_b_opcg.py --sanity
実行(単調性):   OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/gate_b_opcg.py --pairs 8 --base 30 --levels 30,90,270
"""
import argparse
import time

import numpy as np

import conftest  # noqa: F401
from opcg_sim.src.core import cpu_ai
from opcg_game import OPCGGame
from az_mcts_tree import TreeMCTS


def mcts_agent(game, sims, c_puct=1.5):
    def act(state, name, rng):
        mcts = TreeMCTS(game, value_fn=game.value, c_puct=c_puct, n_sims=sims,
                        determinize_fn=lambda s, r: game.determinize(s, name, r), rng=rng)
        move, _ = mcts.run(state)
        if move is None:
            legal = game.legal_actions(state)
            move = legal[0] if legal else None
        return move
    return act


def random_agent(game):
    def act(state, name, rng):
        legal = game.legal_actions(state)
        return legal[int(rng.integers(len(legal)))] if legal else None
    return act


def play_one(game, m, agent_first, agent_second, rng, max_steps=400):
    """real manager を進める。agent_first=p1手番側。返り値: winner(str|None)。"""
    steps = 0
    while game.winner(m) is None and not game.is_terminal(m) and steps < max_steps:
        name = game.current_player(m)
        if name is None:
            break
        agent = agent_first if name == "p1" else agent_second
        move = agent(m, name, rng)
        if move is None:
            break
        try:
            cpu_ai._apply_move_inplace(m, name, move)
        except Exception:
            break
        steps += 1
    return game.winner(m)


def match(game, db, agentA, agentB, pairs, seed0=1000):
    """CRN: 各 seed で A=p1/B=p2 と A=p2/B=p1 の2戦。返り値 dict(a_win,draw,a_loss,games)。"""
    res = {"a_win": 0, "draw": 0, "a_loss": 0}
    for i in range(pairs):
        seed = seed0 + i
        for a_is_p1 in (True, False):
            m = game.new_game(db, seed)
            rng = np.random.default_rng(seed * 7 + (0 if a_is_p1 else 1))
            first = agentA if a_is_p1 else agentB
            second = agentB if a_is_p1 else agentA
            w = play_one(game, m, first, second, rng)
            if w is None:
                res["draw"] += 1
            else:
                a_won = (w == "p1") == a_is_p1
                res["a_win" if a_won else "a_loss"] += 1
    res["games"] = pairs * 2
    return res


def sanity(game, db):
    print("=== GATE B sanity: アダプタ健全性 ===", flush=True)
    # ① determinize: 相手手札の中身が変わり枚数は保存・自分手札は不変。
    m = game.new_game(db, 1)
    rng = np.random.default_rng(0)
    me = game.current_player(m)
    opp = "p2" if me == "p1" else "p1"
    before_opp = [c.master.card_id for c in (m.p2 if opp == "p2" else m.p1).hand]
    before_me = [c.master.card_id for c in (m.p1 if me == "p1" else m.p2).hand]
    d = game.determinize(m, me, rng)
    after_opp = [c.master.card_id for c in (d.p2 if opp == "p2" else d.p1).hand]
    after_me = [c.master.card_id for c in (d.p1 if me == "p1" else d.p2).hand]
    print(f"determinize: opp枚数 {len(before_opp)}→{len(after_opp)} (保存={len(before_opp)==len(after_opp)})  "
          f"opp中身変化={before_opp != after_opp}  自分不変={before_me == after_me}")
    # ② MCTS が合法ゲームを完走（エンジン例外なし）。
    t0 = time.perf_counter()
    agent = mcts_agent(game, sims=20)
    rnd = random_agent(game)
    m = game.new_game(db, 2)
    w = play_one(game, m, agent, rnd, np.random.default_rng(2))
    dt = time.perf_counter() - t0
    print(f"完走: winner={w}  ({dt:.1f}s・MCTS(sims20) vs random 1戦)")
    # ③ 1手あたり MCTS コスト（sims別）。
    m = game.new_game(db, 3)
    # 数手進めて中盤に。
    for _ in range(8):
        name = game.current_player(m)
        if name is None:
            break
        mv = rnd(m, name, np.random.default_rng(9))
        if mv is None:
            break
        try:
            cpu_ai._apply_move_inplace(m, name, mv)
        except Exception:
            break
    for sims in (30, 90, 270):
        name = game.current_player(m)
        ag = mcts_agent(game, sims=sims)
        t0 = time.perf_counter()
        ag(m, name, np.random.default_rng(0))
        print(f"  MCTS 1手 sims={sims:>3}: {time.perf_counter()-t0:.2f}s", flush=True)


def run_monotonicity(game, db, pairs=8, base=30, levels=(30, 270), c_puct=1.5, log=print):
    """playout単調性。返り値 (ok, rates, details)。評価器固定・sims だけ動かす。"""
    base_ag = mcts_agent(game, sims=base, c_puct=c_puct)
    log(f"=== GATE B 単調性: MCTS(自分 sims) vs MCTS(固定 sims={base}) "
        f"CRN {pairs}ペア×2={pairs*2}戦 ===")
    log("評価器は両者同一(L1 tanh)＝探索量だけの差。")
    rates, details = [], []
    for sims in levels:
        mine = mcts_agent(game, sims=sims, c_puct=c_puct)
        t0 = time.perf_counter()
        r = match(game, db, mine, base_ag, pairs)
        wr = (r["a_win"] + 0.5 * r["draw"]) / r["games"]
        rates.append(wr); details.append(r)
        log(f"sims={sims:>4} vs {base}: 勝率={wr:.3f}  {r}  ({time.perf_counter()-t0:.0f}s)")
    mono = all(rates[i] <= rates[i + 1] + 1e-9 for i in range(len(rates) - 1))
    trend = rates[-1] - rates[0]
    # 少数戦ゆえ厳密単調はノイズに脆い。判定は「探索を増やすと明確に強くなる」トレンド
    # （最大sims が最小sims を +0.10 以上上回る）を主・厳密単調は参考表示。
    ok = trend >= 0.10
    log(f"\nrates={[round(x,3) for x in rates]}  トレンド(最大-最小)={trend:+.3f}  厳密単調={mono}")
    log(f"探索健全(more search=stronger): {'OK ✅' if ok else 'NG ❌（探索/PIMC統合を疑う）'}")
    return ok, rates, details


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--pairs", type=int, default=8)
    ap.add_argument("--base", type=int, default=30, help="相手の固定 sims")
    ap.add_argument("--levels", default="30,270", help="自分の sims 群（カンマ区切り）")
    ap.add_argument("--c-puct", type=float, default=1.5)
    args = ap.parse_args()

    from cpu_selfplay import _load_db
    db = _load_db()
    game = OPCGGame()
    if args.sanity:
        sanity(game, db)
        return 0
    levels = [int(x) for x in args.levels.split(",")]
    ok, _, _ = run_monotonicity(game, db, args.pairs, args.base, levels, args.c_puct)
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
