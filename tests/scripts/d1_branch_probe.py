"""cand_D1 ゲート退行の枝別診断（P4-b・2026-08-24・使い捨てではなく退行診断の再利用計器）。

coach_gate で 1.0→0.0 に落ちた防御2点（m1@14 / m2@58＝どちらも素通しが正）について、
決定点の合法手それぞれを resolved_branch_values（=serve と同じ戦闘箱の物差し）で
gen15 既定と cand_D1 の両方で採点し、順位がどう崩れたかを枝単位で表示する。
"""
import argparse
import os
import sys

import numpy as np

import _bootstrap  # noqa: F401
import counterfactual_referee as CR
import mark_gate as MG
import replay_reeval as RE
import coach_gate as CG
from opcg_sim.src.core import cpu_ai
from opcg_sim.src.core.cpu_learned import LearnedEngine
from opcg_sim.src.learned.mcts import resolved_branch_values
from cpu_selfplay import _load_db


def _describe(mgr, mv):
    try:
        d = cpu_ai._describe_move(mgr, mv) or {}
    except Exception:
        d = {}
    return f"{d.get('action_type', mv.get('action_type'))}:{d.get('card')}"


def probe_point(db, tag, i, engines):
    CR.ARGS = argparse.Namespace(true_board=True)
    replays = {**MG.REPLAYS, **CG.REPLAYS_V2, **CG.REPLAYS_V48, **CG.REPLAYS_HUMAN}
    if tag not in CR.GAMES:
        raw = RE.load_replay_json(replays[tag]); rec = raw.get("replay", raw)
        CR.GAMES[tag] = (rec, {f.get("action_index"): f for f in raw.get("frames") or []},
                         rec["actions"])
    built = CR._restore_board(db, tag, i)
    if isinstance(built, str):
        rec, fbi, actions = CR.GAMES[tag]
        built = MG._restore(db, rec, fbi, actions, i)
    m0, who = built
    name = who if isinstance(who, str) else who.name
    print(f"\n=== {tag}@{i} 手番={name} ===")
    for label, eng in engines.items():
        legal = eng.game.legal_actions(m0)
        vals = resolved_branch_values(eng.game, m0, name, legal,
                                      eng._battle_value_fn())
        rows = sorted(zip(legal, vals), key=lambda t: -(t[1] if t[1] is not None else -9e9))
        print(f"  [{label}]")
        for mv, v in rows:
            vs = "None" if v is None else f"{v:+.4f}"
            print(f"    {vs}  {_describe(m0, mv)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", default="m1:14,m2:58")
    ap.add_argument("--cand", default="/home/user/cand_D1/value.npz,/home/user/cand_D1/policy.npz")
    args = ap.parse_args()
    db = _load_db()
    CR.GAMES = {}
    parts = args.cand.split(",")
    engines = {
        "gen15(既定)": LearnedEngine(),
        "cand": LearnedEngine(value_path=parts[0],
                              policy_path=parts[1] if len(parts) > 1 else None),
    }
    for p in args.points.split(","):
        tag, i = p.split(":")
        probe_point(db, tag, int(i), engines)


if __name__ == "__main__":
    main()
