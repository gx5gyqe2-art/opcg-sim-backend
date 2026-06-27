"""control plan A/B テスト（dev専用）: control-vs-midrange 戦で control 側のプランのみ変えて効果を測定する。

3条件（control 側）:
  noplan   : plan=None（plan 未使用、旧来挙動）
  ctrl     : build_plan() そのまま（本番と同じ control plan）
  mid      : control デッキに midrange プリセットを上書き（midrange plan を使わせる）

midrange 側は常に build_plan() 由来の本番プランを使う（deployed 挙動再現）。
同一シード・同一マッチアップで 3 条件を回す（paired comparison）。

実行例:
    OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/control_plan_ab.py --games 300 --real-decks --all-leaders --pimc 1
"""
import argparse
import copy
import dataclasses
import math
import multiprocessing as mp
import os
import random
import sys
from typing import Any, Dict, List, Optional

import conftest  # noqa: F401

from opcg_sim.src.core.gamestate import GameManager, Player
from opcg_sim.src.core import action_api, cpu_ai, cpu_self_plan
from opcg_sim.src.core.invariants import check_invariants
from collect_value_data import _build_decks
from cpu_selfplay import _load_db, DEFAULT_MAX_STEPS

_DB = None
_CFG: Dict[str, Any] = {}


def _init(cfg):
    global _DB, _CFG
    _DB = _load_db()
    _CFG = cfg


def _archetype(cards, leader):
    try:
        return getattr(cpu_self_plan.build_plan([c.master for c in cards], leader=leader), "archetype", "midrange")
    except Exception:
        return "midrange"


def _make_mid_plan(ctrl_plan):
    """ctrl_plan の PlanProfile に midrange プリセットを上書きして返す。"""
    mid_preset = cpu_self_plan._PRESETS["midrange"]
    return dataclasses.replace(ctrl_plan, archetype="midrange", **mid_preset)


def _play_one(manager_factory, ctrl_player_name, mid_player_name, ctrl_plan, pimc):
    """manager_factory() → GameManager を受け取り、ctrl_plan で 1 戦回して勝者を返す。"""
    m, c_name, mid_name = manager_factory()
    pid_key = action_api.CONST.get("PENDING_REQUEST_PROPERTIES", {}).get("PLAYER_ID", "player_id")
    mems = {"p1": {}, "p2": {}}
    plans = {c_name: ctrl_plan, mid_name: None}  # mid_plan は後で上書き（呼び出し元が渡す）
    step = 0
    while m.winner is None and step < _CFG["max_steps"]:
        pending = m.get_pending_request()
        if not pending:
            break
        actor = m.p1 if m.p1.name == pending[pid_key] else m.p2
        chosen = cpu_ai.decide_guarded(m, actor, "hard", random, mems[actor.name],
                                       plan=plans[actor.name], pimc_worlds=pimc)
        if chosen is None:
            break
        m.action_events = []
        try:
            if chosen["kind"] == "battle":
                action_api.apply_battle_action(m, actor, chosen["action_type"], chosen.get("card_uuid"))
            else:
                action_api.apply_game_action(m, actor, chosen["action_type"], chosen.get("payload", {}))
        except Exception:
            return None
        if check_invariants(m):
            return None
        step += 1
    return m.winner


def ab_game(seed: int):
    """1シードで 3 条件（noplan/ctrl/mid_preset）を比較して返す。"""
    random.seed(seed)
    l1, c1, l2, c2 = _build_decks(seed, _DB, _CFG["real_decks"], all_leaders=_CFG["all_leaders"])
    if not l1 or not l2:
        return None
    arch = {"p1": _archetype(c1, l1), "p2": _archetype(c2, l2)}
    if set(arch.values()) != {"control", "midrange"}:
        return None

    ctrl_side = "p1" if arch["p1"] == "control" else "p2"
    mid_side = "p2" if ctrl_side == "p1" else "p1"

    # build plans
    ctrl_cards = c1 if ctrl_side == "p1" else c2
    ctrl_leader = l1 if ctrl_side == "p1" else l2
    mid_cards = c2 if mid_side == "p2" else c1
    mid_leader = l2 if mid_side == "p2" else l1

    try:
        ctrl_plan = cpu_self_plan.build_plan([c.master for c in ctrl_cards], leader=ctrl_leader)
        mid_plan_real = cpu_self_plan.build_plan([c.master for c in mid_cards], leader=mid_leader)
        ctrl_as_mid = _make_mid_plan(ctrl_plan)
    except Exception:
        return None

    pimc = _CFG["pimc"]

    pid_key = action_api.CONST.get("PENDING_REQUEST_PROPERTIES", {}).get("PLAYER_ID", "player_id")

    def _run(ctrl_plan_arg, mid_plan_arg):
        random.seed(seed)
        c1f, c2f = copy.deepcopy(c1), copy.deepcopy(c2)
        l1f, l2f = copy.deepcopy(l1), copy.deepcopy(l2)
        m = GameManager(Player("p1", c1f, l1f), Player("p2", c2f, l2f))
        m.start_game()
        mems = {"p1": {}, "p2": {}}
        plans = {ctrl_side: ctrl_plan_arg, mid_side: mid_plan_arg}
        step = 0
        while m.winner is None and step < _CFG["max_steps"]:
            pending = m.get_pending_request()
            if not pending:
                break
            actor = m.p1 if m.p1.name == pending[pid_key] else m.p2
            chosen = cpu_ai.decide_guarded(m, actor, "hard", random, mems[actor.name],
                                           plan=plans[actor.name], pimc_worlds=pimc)
            if chosen is None:
                break
            m.action_events = []
            try:
                if chosen["kind"] == "battle":
                    action_api.apply_battle_action(m, actor, chosen["action_type"], chosen.get("card_uuid"))
                else:
                    action_api.apply_game_action(m, actor, chosen["action_type"], chosen.get("payload", {}))
            except Exception:
                return None
            if check_invariants(m):
                return None
            step += 1
        return m.winner

    w_noplan = _run(None, mid_plan_real)
    w_ctrl = _run(ctrl_plan, mid_plan_real)
    w_mid = _run(ctrl_as_mid, mid_plan_real)

    if w_noplan is None or w_ctrl is None or w_mid is None:
        return None

    return {
        "ctrl_side": ctrl_side,
        "noplan_win": w_noplan == ctrl_side,
        "ctrl_win": w_ctrl == ctrl_side,
        "mid_win": w_mid == ctrl_side,
    }


def _one(seed):
    try:
        return ab_game(seed)
    except Exception:
        return None


def _ci95(wins, n):
    """Wilson 95% CI"""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = wins / n
    z = 1.96
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0, center - margin), min(1, center + margin)


def main(argv=None):
    ap = argparse.ArgumentParser(description="control plan A/B テスト")
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--real-decks", action="store_true")
    ap.add_argument("--all-leaders", action="store_true")
    ap.add_argument("--pimc", type=int, default=1)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    args = ap.parse_args(argv)

    cfg = {"max_steps": args.max_steps, "real_decks": args.real_decks,
           "all_leaders": args.all_leaders, "pimc": args.pimc}
    workers = args.workers or max(1, (os.cpu_count() or 2) - 1)
    seeds = [args.seed + g for g in range(args.games)]

    rows = []
    with mp.Pool(workers, initializer=_init, initargs=(cfg,)) as pool:
        for i, r in enumerate(pool.imap_unordered(_one, seeds), 1):
            if r is not None:
                rows.append(r)
            if i % 40 == 0:
                print(f"  {i}/{args.games} … control-vs-midrange={len(rows)}", flush=True)

    n = len(rows)
    print(f"\n=== control plan A/B: control-vs-midrange {n}戦（pimc={args.pimc}） ===")
    if n < 4:
        print("サンプル不足")
        return 1

    for label, key in [("noplan (plan=None)", "noplan_win"), ("ctrl_plan (本番)", "ctrl_win"),
                        ("mid_plan (midrange上書き)", "mid_win")]:
        wins = sum(1 for r in rows if r[key])
        p, lo, hi = _ci95(wins, n)
        print(f"  {label:30s}: {wins}/{n}  {p:.1%}  95%CI [{lo:.1%}, {hi:.1%}]")

    # ペア比較: ctrl_plan vs noplan, mid_plan vs noplan, mid_plan vs ctrl_plan
    print("\n--- ペア比較（同じ対局で条件だけ変えた差） ---")
    for a_key, b_key, label in [
        ("ctrl_win", "noplan_win", "ctrl_plan - noplan"),
        ("mid_win", "noplan_win", "mid_plan  - noplan"),
        ("mid_win", "ctrl_win",   "mid_plan  - ctrl_plan"),
    ]:
        diff = sum(1 for r in rows if r[a_key] and not r[b_key]) - \
               sum(1 for r in rows if r[b_key] and not r[a_key])
        a_wins = sum(1 for r in rows if r[a_key])
        b_wins = sum(1 for r in rows if r[b_key])
        print(f"  {label}: {a_wins/n:.1%} vs {b_wins/n:.1%}  (純差={diff:+d}局)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
