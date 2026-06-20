"""C-5（settle 楽観是正）の純 Elo 検証: B-on(hard) vs B-off(hard) ヘッドツーヘッド。

両者とも hard＋デプロイ同等プラン。**唯一の差は `cpu_ai.W_SETTLE_PRESSURE`**（挑戦者=2500 / 基準=0）。
プロセス内グローバルなので、各意思決定の直前に**手番側の重みへ切り替えてから** decide する
（手番は逐次なので衝突しない＝各探索は自分の重みで評価する）。席交互で先手有利を相殺し、
1 局ごとに勝率→Elo を流す（途中経過でも判断できるように）。
"""
import random
import sys
import traceback
from typing import Any, Dict

import conftest  # noqa: F401

from opcg_sim.src.core.gamestate import GameManager, Player
from opcg_sim.src.core import action_api, cpu_ai
from opcg_sim.src.core.invariants import check_invariants, check_turn_boundary
from cpu_selfplay import _load_db, build_deck, DEFAULT_MAX_STEPS, InvariantError
from cpu_arena import _plan_for, elo_delta

CHAL_W = 2500.0   # B-on
BASE_W = 0.0      # B-off


def _make_decider(seat_weight: float, plan):
    mem: Dict[str, Any] = {}

    def _decide(manager, actor):
        cpu_ai.W_SETTLE_PRESSURE = seat_weight   # 手番側の設定へ切替えてから探索
        return cpu_ai.decide_guarded(manager, actor, "hard", random, mem, plan=plan)
    return _decide


def play(seed: int, db, chal_is_p1: bool, max_steps: int = DEFAULT_MAX_STEPS):
    random.seed(seed)
    l1, c1 = build_deck(db, "p1")
    l2, c2 = build_deck(db, "p2")
    manager = GameManager(Player("p1", c1, l1), Player("p2", c2, l2))
    manager.start_game()
    w1, w2 = (CHAL_W, BASE_W) if chal_is_p1 else (BASE_W, CHAL_W)
    deciders = {"p1": _make_decider(w1, _plan_for("hard", l1, c1)),
                "p2": _make_decider(w2, _plan_for("hard", l2, c2))}
    pending_props = action_api.CONST.get('PENDING_REQUEST_PROPERTIES', {})
    KEY_PID = pending_props.get('PLAYER_ID', 'player_id')
    step = 0
    prev_turn = manager.turn_count
    while manager.winner is None and step < max_steps:
        pending = manager.get_pending_request()
        if not pending:
            raise InvariantError([("STUCK", "no pending")], step, [])
        pid = pending[KEY_PID]
        actor = manager.p1 if manager.p1.name == pid else manager.p2
        move = deciders[pid](manager, actor)
        if move is None:
            raise InvariantError([("NO_LEGAL_MOVE", pid)], step, [])
        manager.action_events = []
        if move["kind"] == "battle":
            action_api.apply_battle_action(manager, actor, move["action_type"], move.get("card_uuid"))
        else:
            action_api.apply_game_action(manager, actor, move["action_type"], move.get("payload", {}))
        v = check_invariants(manager)
        if manager.turn_count != prev_turn:
            v += check_turn_boundary(manager); prev_turn = manager.turn_count
        if v:
            raise InvariantError(v, step, [])
        step += 1
    if manager.winner is None:
        raise InvariantError([("MAX_STEPS", str(step))], step, [])
    return manager.winner


def main(argv):
    games = int(argv[0]) if argv else 24
    seed0 = int(argv[1]) if len(argv) > 1 else 1000
    db = _load_db()
    wins = 0.0
    for i in range(games):
        seed = seed0 + i
        chal_is_p1 = (i % 2 == 0)
        try:
            winner = play(seed, db, chal_is_p1)
        except Exception as e:
            print(f"game {i} seed={seed} ERROR {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            continue
        chal_seat = "p1" if chal_is_p1 else "p2"
        won = (winner == chal_seat)
        wins += 1.0 if won else 0.0
        n = i + 1
        wr = wins / n
        print(f"game {i:2d} seed={seed} chal={chal_seat} winner={winner} "
              f"{'WIN ' if won else 'loss'} | running {wins:.0f}/{n} wr={wr:.3f} Elo={elo_delta(wr):+.0f}",
              flush=True)
    wr = wins / games if games else 0.5
    print(f"\nC-5 A/B: B-on(2500) vs B-off(0)  {wins:.0f}/{games}  wr={wr:.3f}  Elo={elo_delta(wr):+.0f}",
          flush=True)
    print("ALLDONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
