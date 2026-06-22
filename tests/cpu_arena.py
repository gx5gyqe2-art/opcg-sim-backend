"""CPU 検証基盤（フェーズ0）の**絶対強度メトリクス**: 凍結ベースライン Elo ＋ regret ログ
（docs/SPEC.md §2.5.3「2026-06 外部レビュー収束／検証基盤」）。

自己対戦＋インバリアントは自己参照的で、評価関数の改善が「強くなった」のか「相対順位が入れ替わった
だけ」なのかを区別できない。本ハーネスは2つの安価な絶対指標を与える:

  1. 凍結ベースライン Elo（`arena`）: **固定の参照相手**（既定 easy＝正直な 1-ply 貪欲・チューニングで
     変化しない安定相手）に対する挑戦者（normal/hard）の勝率を測り、Elo 差へ変換する。版 N を改善した
     とき、固定相手への勝率が上がる＝絶対的に強くなった、という単調指標。先手有利を相殺するため**席を
     交互に入れ替える**（偶数 seed は p1=挑戦者・奇数 seed は p2=挑戦者）。

  2. regret ログ（`regret`）: 各意思決定で `cpu_ai.decide_with_regret` が返す greedy regret
     ＝ deep_value(深掘り最善手) − deep_value(1-ply 貪欲手) を 1 ゲーム分集計する。大きい regret は
     「浅い読みなら崖に落ちる」局面＝評価/探索が効いている所、恒常的に 0 なら深掘りが効いていない兆候。

いずれも `cpu_selfplay` の決定論ランナーと同じコアパス（action_api）で進行し、本番挙動と乖離しない。
pytest スイートには**機械の健全性のみ**を高速・有界に固定する（`tests/test_cpu_arena.py`）。実ゲームは
低速（normal ≈ 1 手/秒）なので、版間 Elo の本走はこのスクリプトを手動/定期実行する想定:

    OPCG_LOG_SILENT=1 python tests/cpu_arena.py arena --challenger normal --baseline easy --games 20
    OPCG_LOG_SILENT=1 python tests/cpu_arena.py regret --difficulty normal --seed 0
"""
import argparse
import math
import random
import sys
import traceback
from typing import Any, Dict, List, Optional

import conftest  # noqa: F401  (google スタブ注入 & sys.path 設定)

from opcg_sim.src.core.gamestate import GameManager, Player
from opcg_sim.src.core import action_api, cpu_ai, cpu_self_plan
from opcg_sim.src.core.invariants import check_invariants, check_turn_boundary

from cpu_selfplay import _load_db, build_deck, DEFAULT_MAX_STEPS, InvariantError


def _plan_for(difficulty: str, leader, cards):
    """デプロイ（app.py /api/game/create）と同じく normal/hard は自デッキ構成からプランを作る。

    easy はプラン無し（素の 1-ply）。プロファイル（相手テンプレ）は自己対戦では無いので None
    （デプロイのテンプレ未登録フォールバックと同義）。
    """
    if difficulty == "hard":
        try:
            return cpu_self_plan.build_plan([c.master for c in cards],
                                            leader=leader.master if leader else None)
        except Exception:
            return None
    return None


# --- Elo 変換 -----------------------------------------------------------------

def elo_delta(win_rate: float) -> float:
    """勝率 → Elo 差（挑戦者 − ベースライン）。0.5→0・0.76→+200・0.24→-200。

    端（0/1）は ±inf を避けて有限の小/大サンプル境界へクランプする。
    """
    p = min(max(win_rate, 1e-4), 1.0 - 1e-4)
    return -400.0 * math.log10(1.0 / p - 1.0)


def win_rate(wins: float, games: int) -> float:
    """引き分けは 0.5 勝として `wins` に半端で含める想定。games==0 は 0.5（無情報）。"""
    return 0.5 if games <= 0 else wins / games


# --- 非対称（挑戦者 vs ベースライン）対局ランナー -----------------------------

def _make_decider(difficulty: str, plan=None, info_policy: str = cpu_ai.DEFAULT_INFO_POLICY):
    """プレイヤー1人分のターン内メモリ付き意思決定関数を返す（暴走防止ガード付き・デプロイと同じプラン供給）。

    `info_policy`（Phase -1）で情報方針を選ぶ＝凍結 fair-hard vs cheat-hard の A/B を席交互で測れる。
    """
    mem: Dict[str, Any] = {}

    def _decide(manager, actor):
        return cpu_ai.decide_guarded(manager, actor, difficulty, random, mem, plan=plan,
                                     info_policy=info_policy)
    return _decide


def play_game(seed: int, db, p1_difficulty: str, p2_difficulty: str,
              max_steps: int = DEFAULT_MAX_STEPS,
              p1_policy: str = cpu_ai.DEFAULT_INFO_POLICY,
              p2_policy: str = cpu_ai.DEFAULT_INFO_POLICY) -> Dict[str, Any]:
    """p1/p2 に別難易度・別情報方針を割り当てて 1 ゲームを決定論的に完走させ、勝者を返す。

    `cpu_selfplay.run_one_game` は単一 policy 前提なので、非対称対局用に最小実装する
    （同じ action_api コアパス＋各ステップのインバリアント検出）。normal/hard はデプロイと同じく
    自デッキ構成からプランを供給する（easy はプラン無し）。`p1_policy`/`p2_policy` は情報方針
    （fair/cheat・Phase -1）で、フェア化前後の強さ A/B に用いる。
    """
    random.seed(seed)
    l1, c1 = build_deck(db, "p1")
    l2, c2 = build_deck(db, "p2")
    manager = GameManager(Player("p1", c1, l1), Player("p2", c2, l2))
    manager.start_game()
    deciders = {"p1": _make_decider(p1_difficulty, _plan_for(p1_difficulty, l1, c1), p1_policy),
                "p2": _make_decider(p2_difficulty, _plan_for(p2_difficulty, l2, c2), p2_policy)}

    step = 0
    prev_turn = manager.turn_count
    pending_props = action_api.CONST.get('PENDING_REQUEST_PROPERTIES', {})
    KEY_PID = pending_props.get('PLAYER_ID', 'player_id')
    while manager.winner is None and step < max_steps:
        pending = manager.get_pending_request()
        if not pending:
            raise InvariantError([("STUCK", "no pending request and no winner")], step, [])
        req_pid = pending[KEY_PID]
        actor = manager.p1 if manager.p1.name == req_pid else manager.p2
        move = deciders[req_pid](manager, actor)
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
    if manager.winner is None:
        raise InvariantError([("MAX_STEPS", f"unfinished within {max_steps}")], step, [])
    return {"seed": seed, "winner": manager.winner, "steps": step, "turns": manager.turn_count}


def arena(db, challenger: str, baseline: str, games: int, seed0: int = 0,
          max_steps: int = DEFAULT_MAX_STEPS,
          challenger_policy: str = cpu_ai.DEFAULT_INFO_POLICY,
          baseline_policy: str = cpu_ai.DEFAULT_INFO_POLICY) -> Dict[str, Any]:
    """挑戦者 vs 固定ベースラインを `games` 局。**席を交互に入替**して先手有利を相殺し、勝率と Elo を返す。

    偶数 i: p1=挑戦者 / 奇数 i: p2=挑戦者。引き分け（あれば）は 0.5 勝で計上。
    `challenger_policy`/`baseline_policy`（Phase -1・fair/cheat）で情報方針も A/B できる（席入替に追従）。
    """
    wins = 0.0
    decided = 0
    detail: List[Dict[str, Any]] = []
    for i in range(games):
        seed = seed0 + i
        chal_is_p1 = (i % 2 == 0)
        p1d, p2d = (challenger, baseline) if chal_is_p1 else (baseline, challenger)
        p1p, p2p = ((challenger_policy, baseline_policy) if chal_is_p1
                    else (baseline_policy, challenger_policy))
        res = play_game(seed, db, p1d, p2d, max_steps=max_steps, p1_policy=p1p, p2_policy=p2p)
        chal_seat = "p1" if chal_is_p1 else "p2"
        won = (res["winner"] == chal_seat)
        wins += 1.0 if won else 0.0
        decided += 1
        detail.append({"seed": seed, "challenger_seat": chal_seat, "winner": res["winner"],
                       "challenger_won": won, "turns": res["turns"]})
    wr = win_rate(wins, decided)
    return {"challenger": challenger, "baseline": baseline,
            "challenger_policy": challenger_policy, "baseline_policy": baseline_policy,
            "games": decided, "challenger_wins": wins, "win_rate": wr,
            "elo_delta": elo_delta(wr), "detail": detail}


# --- regret ログ --------------------------------------------------------------

def regret_trace(db, seed: int, difficulty: str = "hard",
                 max_steps: int = DEFAULT_MAX_STEPS) -> Dict[str, Any]:
    """1 ゲームを自己対戦し、各 MAIN 意思決定の greedy regret を集計する（mean/max/count/p95）。

    regret は `cpu_ai.decide_with_regret`（deep_value(深掘り最善) − deep_value(1-ply 貪欲)）。
    実際に手を進めるのは返り値の move（＝通常の対局と同じ進行）なので、トレースは本番方策の軌跡上で取る。
    """
    random.seed(seed)
    l1, c1 = build_deck(db, "p1")
    l2, c2 = build_deck(db, "p2")
    manager = GameManager(Player("p1", c1, l1), Player("p2", c2, l2))
    manager.start_game()
    plans = {"p1": _plan_for(difficulty, l1, c1), "p2": _plan_for(difficulty, l2, c2)}
    mems: Dict[str, Any] = {"p1": {}, "p2": {}}
    pending_props = action_api.CONST.get('PENDING_REQUEST_PROPERTIES', {})
    KEY_PID = pending_props.get('PLAYER_ID', 'player_id')
    KEY_ACTION = pending_props.get('ACTION', 'action')

    regrets: List[float] = []
    step = 0
    while manager.winner is None and step < max_steps:
        pending = manager.get_pending_request()
        if not pending:
            break
        req_pid = pending[KEY_PID]
        actor = manager.p1 if manager.p1.name == req_pid else manager.p2
        if pending.get(KEY_ACTION) == "MAIN_ACTION":
            move, regret = cpu_ai.decide_with_regret(manager, actor, difficulty, random,
                                                     plan=plans[req_pid])
            regrets.append(regret)
        else:
            move = cpu_ai.decide_guarded(manager, actor, difficulty, random, mems[req_pid],
                                         plan=plans[req_pid])
        if move is None:
            break
        manager.action_events = []
        if move["kind"] == "battle":
            action_api.apply_battle_action(manager, actor, move["action_type"], move.get("card_uuid"))
        else:
            action_api.apply_game_action(manager, actor, move["action_type"], move.get("payload", {}))
        step += 1

    n = len(regrets)
    s = sorted(regrets)
    return {
        "seed": seed, "difficulty": difficulty, "decisions": n,
        "mean": (sum(regrets) / n) if n else 0.0,
        "max": max(regrets) if n else 0.0,
        "p95": (s[min(n - 1, int(0.95 * n))] if n else 0.0),
        "nonzero": sum(1 for r in regrets if r > 0.0),
    }


# --- value-realization gap（ターン内 楽観崩落・§2.5.3） ----------------------

def realize_trace(db, seed: int, difficulty: str = "hard",
                  max_steps: int = DEFAULT_MAX_STEPS) -> Dict[str, Any]:
    """1 ゲームを自己対戦し、**value-realization gap**（ターン内の楽観崩落）を集計する。

    各 MAIN 意思決定で `decide_with_regret(out=...)` から採用手の深掘りスコア（`chosen_deep`）を取り、
    (player, turn) ごとに時系列で並べる。ターン頭でドン/盤面に過剰コミットし、手番が進んで初めて
    「読み切れなかった代償」が露見すると、採用手の深掘り値は**ターン内で単調に崩落**する（実ケース:
    付与時 +4798 → 攻撃時 -91）。1 ターンの gap = max(そのターンの chosen_deep) − 最終決定の chosen_deep。
    大きい gap が頻発する＝探索が予算地平線の外を楽観視して資源を溶かしている兆候（B が縮める対象）。
    """
    random.seed(seed)
    l1, c1 = build_deck(db, "p1")
    l2, c2 = build_deck(db, "p2")
    manager = GameManager(Player("p1", c1, l1), Player("p2", c2, l2))
    manager.start_game()
    plans = {"p1": _plan_for(difficulty, l1, c1), "p2": _plan_for(difficulty, l2, c2)}
    mems: Dict[str, Any] = {"p1": {}, "p2": {}}
    pending_props = action_api.CONST.get('PENDING_REQUEST_PROPERTIES', {})
    KEY_PID = pending_props.get('PLAYER_ID', 'player_id')
    KEY_ACTION = pending_props.get('ACTION', 'action')

    series: Dict[tuple, List[float]] = {}   # (player, turn) -> [chosen_deep, ...]
    step = 0
    while manager.winner is None and step < max_steps:
        pending = manager.get_pending_request()
        if not pending:
            break
        req_pid = pending[KEY_PID]
        actor = manager.p1 if manager.p1.name == req_pid else manager.p2
        if pending.get(KEY_ACTION) == "MAIN_ACTION":
            info: Dict[str, Any] = {}
            move, _r = cpu_ai.decide_with_regret(manager, actor, difficulty, random,
                                                 plan=plans[req_pid], out=info)
            if "chosen_deep" in info:
                series.setdefault((req_pid, manager.turn_count), []).append(info["chosen_deep"])
        else:
            move = cpu_ai.decide_guarded(manager, actor, difficulty, random, mems[req_pid],
                                         plan=plans[req_pid])
        if move is None:
            break
        manager.action_events = []
        if move["kind"] == "battle":
            action_api.apply_battle_action(manager, actor, move["action_type"], move.get("card_uuid"))
        else:
            action_api.apply_game_action(manager, actor, move["action_type"], move.get("payload", {}))
        step += 1

    gaps = [max(v) - v[-1] for v in series.values() if len(v) >= 2]
    n = len(gaps)
    s = sorted(gaps)
    return {
        "seed": seed, "difficulty": difficulty, "turns_scored": n,
        "mean_gap": (sum(gaps) / n) if n else 0.0,
        "max_gap": max(gaps) if n else 0.0,
        "p95_gap": (s[min(n - 1, int(0.95 * n))] if n else 0.0),
        "big_gaps": sum(1 for g in gaps if g >= 2000.0),   # ライフ ~1/3 相当以上の崩落ターン数
    }


# --- CLI ----------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description="CPU 検証基盤: 凍結ベースライン Elo ＋ regret ログ")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pa = sub.add_parser("arena", help="挑戦者 vs 固定ベースラインの勝率→Elo")
    pa.add_argument("--challenger", choices=["hard"], default="hard")
    pa.add_argument("--baseline", choices=["hard"], default="hard")
    # Phase -1: 情報方針の A/B（既定＝挑戦者 fair vs ベースライン cheat＝フェア化の損失量を測る）。
    pa.add_argument("--challenger-policy", choices=["fair", "cheat"], default="fair")
    pa.add_argument("--baseline-policy", choices=["fair", "cheat"], default="cheat")
    pa.add_argument("--games", type=int, default=10)
    pa.add_argument("--seed", type=int, default=0)
    pa.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)

    pr = sub.add_parser("regret", help="自己対戦 1 局の greedy regret 集計")
    pr.add_argument("--difficulty", choices=["hard"], default="hard")
    pr.add_argument("--seed", type=int, default=0)
    pr.add_argument("--games", type=int, default=1)
    pr.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)

    pz = sub.add_parser("realize", help="自己対戦の value-realization gap（ターン内 楽観崩落）集計")
    pz.add_argument("--difficulty", choices=["hard"], default="hard")
    pz.add_argument("--seed", type=int, default=0)
    pz.add_argument("--games", type=int, default=1)
    pz.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)

    args = ap.parse_args(argv)
    db = _load_db()

    if args.cmd == "arena":
        rep = arena(db, args.challenger, args.baseline, args.games, args.seed, args.max_steps,
                    challenger_policy=args.challenger_policy, baseline_policy=args.baseline_policy)
        for d in rep["detail"]:
            print(f"  seed={d['seed']} challenger={d['challenger_seat']} winner={d['winner']} "
                  f"{'WIN' if d['challenger_won'] else 'loss'} turns={d['turns']}")
        print(f"\narena: {rep['challenger']}[{rep['challenger_policy']}] vs "
              f"{rep['baseline']}[{rep['baseline_policy']}]  "
              f"{rep['challenger_wins']:.1f}/{rep['games']}  win_rate={rep['win_rate']:.3f}  "
              f"Elo={rep['elo_delta']:+.0f}")
        return 0

    if args.cmd == "realize":
        for i in range(args.games):
            rep = realize_trace(db, args.seed + i, args.difficulty, args.max_steps)
            print(f"realize seed={rep['seed']} {rep['difficulty']}: turns={rep['turns_scored']} "
                  f"mean_gap={rep['mean_gap']:.1f} p95_gap={rep['p95_gap']:.1f} "
                  f"max_gap={rep['max_gap']:.1f} big_gaps={rep['big_gaps']}")
        return 0

    # regret
    for i in range(args.games):
        rep = regret_trace(db, args.seed + i, args.difficulty, args.max_steps)
        print(f"regret seed={rep['seed']} {rep['difficulty']}: decisions={rep['decisions']} "
              f"mean={rep['mean']:.1f} p95={rep['p95']:.1f} max={rep['max']:.1f} nonzero={rep['nonzero']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
