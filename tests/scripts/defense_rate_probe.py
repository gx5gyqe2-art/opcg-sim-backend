"""R1 調査（v5計画 §3-R1）: 防御応答の「守り採択率」を計測する読み取り専用計器。

仮説: v4 の防御応答の温度延長（SELECT_COUNTER/BLOCKER をゲーム全体で温度サンプリング）が
自己対戦データにカウンター/ブロックを過剰注入し、value を「守り＝善」へ振らせた（守りすぎ=C1）。

計測: 既定 net（gen4）で自己対戦し、防御応答（SELECT_COUNTER/SELECT_BLOCKER）局面ごとに
  - argmax:   net が最も訪問した手が「守る(非PASS)」か（＝net の素の意向）
  - temp:     温度1サンプリングで実際に「守る」手が選ばれるか（＝データに入る挙動＝v4 の温度延長）
  - L1:       同局面で L1-hard（時計を手書きで持つ良質プレイ目安）が守るか
を集計し、守り採択率を3系統で比較する。temp >> argmax >> L1 なら仮説を支持（温度延長が過剰注入）。

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/defense_rate_probe.py --games 30 --sims 160
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse

import numpy as np

import os as _os, sys as _sys  # noqa: E402
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap  # noqa: E402,F401
from opcg_sim.src.core import cpu_ai
from opcg_sim.src.core.cpu_learned import LearnedEngine, _value_fn, _priors_fn
from opcg_sim.src.learned.mcts import TreeMCTS
from opcg_sim.src.learned.config import C_PUCT
from opcg_game import OPCGGame
from deckgen import all_leader_ids
from cpu_selfplay import _load_db
import p3_loop as P

_DEFENSE = ("SELECT_COUNTER", "SELECT_BLOCKER")


def _is_pass(game, legal, idx):
    mv = legal[idx]
    d = cpu_ai._describe_move.__wrapped__ if hasattr(cpu_ai._describe_move, "__wrapped__") else cpu_ai._describe_move
    try:
        return (d(_CUR_M[0], mv) or {}).get("action_type") == "PASS"
    except Exception:
        return mv.get("action_type") == "PASS"


_CUR_M = [None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--sims", type=int, default=160)
    ap.add_argument("--max-steps", type=int, default=400)
    args = ap.parse_args()

    db = _load_db()
    eng = LearnedEngine()   # 既定 = gen4(v4)
    P._DB = db
    vocab = eng.vocab
    game = OPCGGame()
    leaders = all_leader_ids(db)
    vf = _value_fn(eng.vnet, vocab, eng.enc_version)
    pf = _priors_fn(eng.pnet, vocab, eng.enc_version)
    rng = np.random.default_rng(20260712)

    # 集計: 防御応答局面での「守る(非PASS)」採択数 / 総数
    stat = {"argmax": [0, 0], "temp": [0, 0], "l1": [0, 0]}
    n_def = 0

    for g in range(args.games):
        m = game.new_game(db=db, seed=int(rng.integers(1 << 30)), leaders=leaders)
        steps = 0
        while game.winner(m) is None and not game.is_terminal(m) and steps < args.max_steps:
            name = game.current_player(m)
            if name is None:
                break
            pend = m.get_pending_request() or {}
            is_def = pend.get("action") in _DEFENSE
            world_seed = int(rng.integers(2 ** 63 - 1))
            mcts = TreeMCTS(game, value_fn=vf, priors_fn=pf, c_puct=C_PUCT, n_sims=args.sims,
                            determinize_fn=lambda s, r, _sd=world_seed:
                                game.determinize(s, name, np.random.default_rng(_sd)),
                            rng=rng)
            move, N, legal = mcts.run(m)
            if move is None or N is None or N.sum() == 0:
                break

            if is_def and len(legal) > 1:
                n_def += 1
                _CUR_M[0] = m
                def _pass_idx(i):
                    mv = legal[i]
                    try:
                        return (cpu_ai._describe_move(m, mv) or {}).get("action_type") == "PASS"
                    except Exception:
                        return mv.get("action_type") == "PASS"
                # argmax: 最多訪問が守り(非PASS)か
                a_arg = int(np.argmax(N))
                stat["argmax"][0] += 0 if _pass_idx(a_arg) else 1
                stat["argmax"][1] += 1
                # temp: 訪問分布サンプリングで守りが出るか（複数サンプルの期待値＝訪問%そのもの）
                p = N / N.sum()
                defend_mass = sum(p[i] for i in range(len(legal)) if not _pass_idx(i))
                stat["temp"][0] += defend_mass
                stat["temp"][1] += 1
                # L1: 同局面で decide_guarded が守るか
                try:
                    import random as _r
                    clone = m.clone()
                    cp = clone.p1 if clone.p1.name == name else clone.p2
                    l1mv = cpu_ai.decide_guarded(clone, cp, "hard", _r.Random(0), {}, pimc_worlds=1)
                    l1_pass = (cpu_ai._describe_move(clone, l1mv) or {}).get("action_type") == "PASS" if l1mv else True
                    stat["l1"][0] += 0 if l1_pass else 1
                    stat["l1"][1] += 1
                except Exception:
                    pass

            # 進行は argmax（データ挙動そのものを測るのが目的なので決定は素直に）
            a = int(np.argmax(N))
            try:
                cpu_ai._apply_move_inplace(m, name, legal[a])
            except Exception:
                break
            steps += 1
        if (g + 1) % 5 == 0:
            print(f"  {g+1}/{args.games} 局・防御局面 {n_def}", flush=True)

    print(f"\n=== 防御応答の守り採択率（gen4・{args.games}局・防御局面 {n_def}）===", flush=True)
    for k, label in (("argmax", "net argmax（素の意向）"),
                     ("temp", "温度1期待（データに入る挙動＝温度延長）"),
                     ("l1", "L1-hard（良質プレイ目安）")):
        s, c = stat[k]
        if c:
            print(f"  {label:<34}: 守り率 {s/c:.3f}  (n={c})", flush=True)
    print("\n判定の見方: temp/argmax が L1 を大きく上回れば『守りすぎ』を net/データが持つ。"
          "\n           temp が argmax を大きく上回れば『温度延長がデータへ守りを過剰注入』（R1支持）。", flush=True)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
