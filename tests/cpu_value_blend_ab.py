"""hard(α-β) 学習価値ブレンドの A/B 測定（Phase 2 本体・dev 専用・docs/reports/cpu_value_blend_hard_ab）。

「α版 hard」 vs 「固定基準＝α=0 hard」の **直接対決**（席交互）を回し、勝率・概算 Elo を出す。α は
`cpu_value_model.set_alpha_override` で**席ごと**に切り替える（挑戦者=α / ベースライン=0）。override は
`cpu_ai._hard_blend_alpha()` が読む（hard 葉のブレンド率）。決定論（seed 固定）。

レイテンシ: 各 `decide_guarded` の壁時計を測り、1手平均/最大を出す（1秒目標への影響確認）。

実行例:
    OPCG_LOG_SILENT=1 python tests/cpu_value_blend_ab.py --alpha 0.25 --games 24 --seed 0
"""
import argparse
import json
import math
import random
import sys
import time
import traceback
from typing import Any, Dict, List

import conftest  # noqa: F401

from opcg_sim.src.core.gamestate import GameManager, Player
from opcg_sim.src.core import action_api, cpu_ai, cpu_self_plan, cpu_value_model
from opcg_sim.src.core.invariants import check_invariants, check_turn_boundary

from cpu_selfplay import _load_db, build_deck, DEFAULT_MAX_STEPS, InvariantError


def _plan_for(leader, cards):
    try:
        return cpu_self_plan.build_plan([c.master for c in cards],
                                        leader=leader.master if leader else None)
    except Exception:
        return None


def elo_delta(win_rate: float) -> float:
    p = min(max(win_rate, 1e-4), 1.0 - 1e-4)
    return -400.0 * math.log10(1.0 / p - 1.0)


def play_game_ab(seed: int, db, alpha_p1: float, alpha_p2: float,
                 max_steps: int = DEFAULT_MAX_STEPS, lat: List[float] = None) -> Dict[str, Any]:
    """p1/p2 に別 α を割り当てて 1 局を決定論完走。α は decide 直前に override で切替（席ごと）。"""
    random.seed(seed)
    l1, c1 = build_deck(db, "p1")
    l2, c2 = build_deck(db, "p2")
    manager = GameManager(Player("p1", c1, l1), Player("p2", c2, l2))
    manager.start_game()
    plans = {"p1": _plan_for(l1, c1), "p2": _plan_for(l2, c2)}
    mems: Dict[str, Any] = {"p1": {}, "p2": {}}
    alpha = {"p1": alpha_p1, "p2": alpha_p2}
    pending_props = action_api.CONST.get('PENDING_REQUEST_PROPERTIES', {})
    KEY_PID = pending_props.get('PLAYER_ID', 'player_id')

    step = 0
    prev_turn = manager.turn_count
    while manager.winner is None and step < max_steps:
        pending = manager.get_pending_request()
        if not pending:
            raise InvariantError([("STUCK", "no pending request and no winner")], step, [])
        req_pid = pending[KEY_PID]
        actor = manager.p1 if manager.p1.name == req_pid else manager.p2
        # 席ごとに α を override（hard 葉ブレンド率）。0.0 でも明示設定＝相手手番の影響を受けない。
        cpu_value_model.set_alpha_override(alpha[req_pid])
        t0 = time.perf_counter()
        move = cpu_ai.decide_guarded(manager, actor, "hard", random, mems[req_pid],
                                     plan=plans[req_pid], info_policy="fair")
        if lat is not None:
            lat.append(time.perf_counter() - t0)
        if move is None:
            raise InvariantError([("NO_LEGAL_MOVE", f"no move for {req_pid}")], step, [])
        manager.action_events = []
        try:
            if move["kind"] == "battle":
                action_api.apply_battle_action(manager, actor, move["action_type"], move.get("card_uuid"))
            else:
                action_api.apply_game_action(manager, actor, move["action_type"], move.get("payload", {}))
        except Exception as e:
            raise InvariantError([("ACTION_EXCEPTION", f"{type(e).__name__}: {e}\n{traceback.format_exc()}")],
                                 step, [])
        violations = check_invariants(manager)
        if manager.turn_count != prev_turn:
            violations += check_turn_boundary(manager)
            prev_turn = manager.turn_count
        if violations:
            raise InvariantError(violations, step, [])
        step += 1
    cpu_value_model.set_alpha_override(None)
    if manager.winner is None:
        raise InvariantError([("MAX_STEPS", f"unfinished within {max_steps}")], step, [])
    return {"seed": seed, "winner": manager.winner, "turns": manager.turn_count}


def run_ab(db, alpha: float, games: int, seed0: int, max_steps: int,
           json_path=None) -> Dict[str, Any]:
    """α版 hard（挑戦者）vs α=0 hard（基準）を席交互で games 局。勝率・Elo・レイテンシを返す。"""
    wins = 0.0
    decided = 0
    lat: List[float] = []
    detail: List[Dict[str, Any]] = []
    for i in range(games):
        seed = seed0 + i
        chal_is_p1 = (i % 2 == 0)
        a1, a2 = (alpha, 0.0) if chal_is_p1 else (0.0, alpha)
        try:
            res = play_game_ab(seed, db, a1, a2, max_steps=max_steps, lat=lat)
        except InvariantError as e:
            print(f"ab seed={seed}: FAILED {e.violations}", flush=True)
            continue
        chal_seat = "p1" if chal_is_p1 else "p2"
        won = (res["winner"] == chal_seat)
        wins += 1.0 if won else 0.0
        decided += 1
        detail.append({"seed": seed, "chal_seat": chal_seat, "winner": res["winner"],
                       "won": won, "turns": res["turns"]})
        print(f"  [{decided}] seed={seed} chal={chal_seat} winner={res['winner']} "
              f"{'WIN' if won else 'loss'} turns={res['turns']}", flush=True)
        if json_path:
            _write(json_path, alpha, games, seed0, decided, wins, lat, detail)
    wr = (wins / decided) if decided else 0.5
    rep = {"alpha": alpha, "games": decided, "challenger_wins": wins, "win_rate": wr,
           "elo_delta": elo_delta(wr),
           "lat_mean_ms": (1000.0 * sum(lat) / len(lat)) if lat else 0.0,
           "lat_max_ms": (1000.0 * max(lat)) if lat else 0.0,
           "moves": len(lat), "detail": detail}
    if json_path:
        _write(json_path, alpha, games, seed0, decided, wins, lat, detail)
    return rep


def _write(path, alpha, games, seed0, decided, wins, lat, detail):
    wr = (wins / decided) if decided else 0.5
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"alpha": alpha, "games_requested": games, "seed0": seed0,
                   "games_finished": decided, "challenger_wins": wins, "win_rate": wr,
                   "elo_delta": elo_delta(wr),
                   "lat_mean_ms": (1000.0 * sum(lat) / len(lat)) if lat else 0.0,
                   "lat_max_ms": (1000.0 * max(lat)) if lat else 0.0,
                   "moves": len(lat), "detail": detail}, f, ensure_ascii=False, indent=2)


def main(argv=None):
    ap = argparse.ArgumentParser(description="hard 学習ブレンド A/B（α版 vs α=0）")
    ap.add_argument("--alpha", type=float, required=True, help="挑戦者側のブレンド率（基準は 0）")
    ap.add_argument("--games", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    ap.add_argument("--json", default=None, help="集計 JSON（毎局フラッシュ）")
    args = ap.parse_args(argv)
    db = _load_db()
    rep = run_ab(db, args.alpha, args.games, args.seed, args.max_steps, json_path=args.json)
    print(f"\nAB α={rep['alpha']}: {rep['challenger_wins']:.1f}/{rep['games']} "
          f"win_rate={rep['win_rate']:.3f} Elo={rep['elo_delta']:+.0f} | "
          f"lat mean={rep['lat_mean_ms']:.0f}ms max={rep['lat_max_ms']:.0f}ms (moves={rep['moves']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
