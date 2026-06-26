"""hard(α-β) と expert(MCTS) の **1手レイテンシ**を本番構成で比較する（dev専用・レスポンス確認）。

強さは `expert_arena.py`、ここは**体感（レスポンス）**を測る。両エンジンを本番相当で構成し、複数 seed・複数
ターン深さで「**同一局面**を両エンジンに解かせて1手の所要時間」を採取（クローンで相互非干渉）→ median/p95/max。

- hard: α-β・horizon4・**PIMC K=4・予算75**・fair・自デッキplan（本番 Dockerfile 相当）。
- expert: MCTS・160反復・horizon2・worlds1・determinize・plan無し（本番 _plan_segment 相当）。
- 葉評価は `--eval-v2` で両者 L1(eval_v2)・既定は J値。

実行例:
    OPCG_LOG_SILENT=1 python tests/engine_latency.py --seeds 8 --samples-per-game 3 --eval-v2
"""
import argparse
import random
import statistics
import sys
import time

import conftest  # noqa: F401
from cpu_selfplay import build_deck, _load_db
from opcg_sim.src.core.gamestate import GameManager, Player
from opcg_sim.src.core import action_api, cpu_ai, cpu_mcts, cpu_self_plan, cpu_value_model

KEY_PID = action_api.CONST.get('PENDING_REQUEST_PROPERTIES', {}).get('PLAYER_ID', 'player_id')


def _apply(gm, actor, mv):
    if mv["kind"] == "battle":
        action_api.apply_battle_action(gm, actor, mv["action_type"], mv.get("card_uuid"))
    else:
        action_api.apply_game_action(gm, actor, mv["action_type"], mv.get("payload", {}))


def _hard_decide(gm, actor, plan):
    """本番 hard 相当: PIMC K=4・予算75・fair・plan。"""
    cpu_ai.set_budget_override(75)
    try:
        return cpu_ai.decide_guarded(gm, actor, "hard", random.Random(0), {}, plan=plan,
                                     info_policy="fair", pimc_worlds=4)
    finally:
        cpu_ai.set_budget_override(None)


def _expert_decide(gm, actor):
    """本番 expert 相当: MCTS 160反復・h2・w1・determinize・plan無し。"""
    return cpu_mcts.decide_mcts_macro(gm, actor, "hard", random.Random(0), cache={},
                                      iterations=160, horizon=2, worlds=1,
                                      determinize=True, plan=None, deadline_ms=None)


def _time(fn, n=1):
    t = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t) / n * 1000.0


def run(seeds, samples_per_game, eval_v2):
    db = _load_db()
    if eval_v2:
        cpu_ai.set_eval_v2_override(True)
    hard_ms, expert_ms = [], []
    for s in range(seeds):
        random.seed(s)
        l1, c1 = build_deck(db, "p1")
        l2, c2 = build_deck(db, "p2")
        gm = GameManager(Player("p1", c1, l1), Player("p2", c2, l2))
        gm.start_game()
        plan = None
        try:
            plan = cpu_self_plan.build_plan([ci.master for ci in c1], leader=l1.master if l1 else None)
        except Exception:
            pass
        taken = 0
        steps = 0
        while gm.winner is None and steps < 200 and taken < samples_per_game:
            pend = gm.get_pending_request()
            if not pend:
                break
            pid = pend[KEY_PID]
            actor = gm.p1 if gm.p1.name == pid else gm.p2
            # p1 の MAIN_ACTION 局面でだけ両エンジンを採時（同一局面・クローンで非干渉）。
            is_main = (pend.get("action_type") == "MAIN_ACTION") or (len(gm.get_legal_actions(actor)) >= 4)
            if pid == "p1" and is_main and steps >= 6:
                # 同一局面を各エンジンに（クローンで非干渉に）解かせて1手の所要を採時。
                ch = gm.clone(); ah = ch.p1 if ch.p1.name == "p1" else ch.p2
                hard_ms.append(_time(lambda: _hard_decide(ch, ah, plan)))
                ce = gm.clone(); ae = ce.p1 if ce.p1.name == "p1" else ce.p2
                expert_ms.append(_time(lambda: _expert_decide(ce, ae)))
                taken += 1
            # 実ゲームを hard で前進（代表的な局面遷移）。
            mv = _hard_decide(gm, actor, plan if pid == "p1" else None)
            if mv is None:
                break
            gm.action_events = []
            _apply(gm, actor, mv)
            steps += 1
    if eval_v2:
        cpu_ai.set_eval_v2_override(None)

    def stat(xs):
        xs = sorted(xs)
        if not xs:
            return "(no samples)"
        p95 = xs[min(len(xs) - 1, int(0.95 * len(xs)))]
        return f"n={len(xs)} median={statistics.median(xs):.0f}ms p95={p95:.0f}ms max={xs[-1]:.0f}ms mean={statistics.mean(xs):.0f}ms"

    leaf = "L1(eval_v2)" if eval_v2 else "J値"
    print(f"\n=== 1手レイテンシ（本番構成・葉={leaf}・同一局面ペア採時） ===")
    print(f"hard  (α-β h4・PIMC K=4・予算75): {stat(hard_ms)}")
    print(f"expert(MCTS 160反復・h2・w1・det): {stat(expert_ms)}")
    if hard_ms and expert_ms:
        print(f"中央値比: expert/hard = {statistics.median(expert_ms)/statistics.median(hard_ms):.2f}x")


def main(argv=None):
    ap = argparse.ArgumentParser(description="hard vs expert 1手レイテンシ（本番構成）")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--samples-per-game", type=int, default=3)
    ap.add_argument("--eval-v2", action="store_true", help="両者 L1(eval_v2) 葉で採時")
    args = ap.parse_args(argv)
    run(args.seeds, args.samples_per_game, args.eval_v2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
