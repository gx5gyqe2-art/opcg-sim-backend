"""control なぜ下手か診断（dev専用）: control-vs-midrange 戦で **control側 と midrange側 の行動を直接比較**し、
hard が control を「どう」下手に回すかの mechanism を炙り出す。

両者とも hard なので、行動差＝アーキ補正（plan）が生む方策差。control が「受けるべき場面で midrange 並みに攻める」
「カウンター/手札を抱える/吐く」「序盤に捲られる」等のどこで崩れるかを、フェーズ別（序盤1-4/中盤5-8/終盤9+）の
指標で見る:
  - 攻撃宣言数 / ターン（攻め圧）
  - 与ライフ / ターン（クロック）
  - 被ライフ / ターン（守備の漏れ）
  - 平均手札・平均手札カウンター枚数（資源温存）
  - ライフ推移（control視点リード）→ どのフェーズで負けが確定するか
control側を「勝った control / 負けた control」でも分けて、負け筋の行動を見る。

実行例:
    OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/control_diag.py --games 200 --real-decks --all-leaders --pimc 1
"""
import argparse
import multiprocessing as mp
import os
import random
import sys
from typing import Any, Dict, List, Optional

import conftest  # noqa: F401

from opcg_sim.src.core.gamestate import GameManager, Player
from opcg_sim.src.core import action_api, cpu_self_plan
from opcg_sim.src.core.invariants import check_invariants
from collect_value_data import _build_decks, _make_decider
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


def _phase(turn: int) -> str:
    return "early" if turn <= 4 else ("mid" if turn <= 8 else "late")


def _counters(player) -> int:
    return sum(1 for c in player.hand if (getattr(c, "current_counter", 0) or 0) > 0)


def diag_game(seed: int):
    random.seed(seed)
    l1, c1, l2, c2 = _build_decks(seed, _DB, _CFG["real_decks"], all_leaders=_CFG["all_leaders"])
    if not l1 or not l2:
        return None
    arch = {"p1": _archetype(c1, l1), "p2": _archetype(c2, l2)}
    # control-vs-midrange のみ対象（それ以外は捨てる）。
    if set(arch.values()) != {"control", "midrange"}:
        return None
    ctrl = "p1" if arch["p1"] == "control" else "p2"
    mid = "p2" if ctrl == "p1" else "p1"

    m = GameManager(Player("p1", c1, l1), Player("p2", c2, l2))
    m.start_game()
    decide = _make_decider(_CFG["difficulty"], 40, 2, _CFG["pimc"])
    pid_key = action_api.CONST.get("PENDING_REQUEST_PROPERTIES", {}).get("PLAYER_ID", "player_id")

    # side → phase → 指標
    def _blank():
        return {ph: {"atk": 0, "dealt": 0, "turns": 0, "hand_sum": 0, "cnt_sum": 0, "snaps": 0} for ph in ("early", "mid", "late")}
    stats = {ctrl: _blank(), mid: _blank()}
    lead_traj = []  # (turn, ctrl_life - mid_life)

    last_turn = -1
    life_at_turn_start = {"p1": len(m.p1.life), "p2": len(m.p2.life)}
    cur_tp = m.turn_player.name if m.turn_player else "p1"
    step = 0
    while m.winner is None and step < _CFG["max_steps"]:
        tc = getattr(m, "turn_count", 0)
        tp = m.turn_player.name if m.turn_player else cur_tp
        if tc != last_turn:
            lead_traj.append((tc, len(m.p1.life if ctrl == "p1" else m.p2.life) -
                              len(m.p2.life if ctrl == "p1" else m.p1.life)))
            # ターン境界: 直前ターンの与ライフを確定（前 turn_player の相手のライフ減少）
            last_turn = tc
            # スナップショット（手札/カウンター）を現 turn_player に計上
            ph = _phase(tc)
            actor_p = m.p1 if tp == "p1" else m.p2
            if tp in stats:
                stats[tp][ph]["turns"] += 1
                stats[tp][ph]["hand_sum"] += len(actor_p.hand)
                stats[tp][ph]["cnt_sum"] += _counters(actor_p)
                stats[tp][ph]["snaps"] += 1
            life_at_turn_start = {"p1": len(m.p1.life), "p2": len(m.p2.life)}
            cur_tp = tp

        pending = m.get_pending_request()
        if not pending:
            break
        actor = m.p1 if m.p1.name == pending[pid_key] else m.p2
        chosen = decide(m, actor)
        if chosen is None:
            break
        # 攻撃宣言を計上（actor=現 turn_player の攻め）。
        if chosen.get("action_type") == "ATTACK" and actor.name in stats:
            stats[actor.name][_phase(getattr(m, "turn_count", 0))]["atk"] += 1
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
        # 与ライフ: turn_player の相手のライフが減ったぶんを turn_player に計上。
        tp_now = m.turn_player.name if m.turn_player else cur_tp
        if tp_now in stats:
            opp = "p2" if tp_now == "p1" else "p1"
            opp_life_now = len(m.p1.life if opp == "p1" else m.p2.life)
            drop = life_at_turn_start[opp] - opp_life_now
            if drop > 0:
                stats[tp_now][_phase(getattr(m, "turn_count", 0))]["dealt"] += drop
                life_at_turn_start[opp] = opp_life_now
        step += 1
    if m.winner is None:
        return None
    return {"ctrl_win": m.winner == ctrl, "stats": stats, "ctrl": ctrl, "mid": mid, "lead": lead_traj}


def _one(seed):
    try:
        return diag_game(seed)
    except Exception:
        return None


def _agg_side(rows, side_key):
    """side_key in {'ctrl','mid'}: 全戦のそのside統計を phase 別に合算→平均指標。"""
    out = {ph: {"atk": 0.0, "dealt": 0.0, "turns": 0, "hand": 0.0, "cnt": 0.0, "snaps": 0} for ph in ("early", "mid", "late")}
    for r in rows:
        s = r["stats"][r[side_key]]
        for ph in ("early", "mid", "late"):
            out[ph]["atk"] += s[ph]["atk"]
            out[ph]["dealt"] += s[ph]["dealt"]
            out[ph]["turns"] += s[ph]["turns"]
            out[ph]["hand"] += s[ph]["hand_sum"]
            out[ph]["cnt"] += s[ph]["cnt_sum"]
            out[ph]["snaps"] += s[ph]["snaps"]
    return out


def _print_side(name, agg):
    print(f"\n[{name}] フェーズ別（1ターンあたり平均）")
    print(f"{'phase':6} {'攻撃/T':>7} {'与ライフ/T':>10} {'手札':>6} {'手Cnt':>6} {'Tn':>4}")
    for ph in ("early", "mid", "late"):
        a = agg[ph]
        t = max(1, a["turns"]); s = max(1, a["snaps"])
        print(f"{ph:6} {a['atk']/t:7.2f} {a['dealt']/t:10.2f} {a['hand']/s:6.2f} {a['cnt']/s:6.2f} {a['turns']:4d}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="control なぜ下手か診断")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--difficulty", choices=["hard"], default="hard")
    ap.add_argument("--real-decks", action="store_true")
    ap.add_argument("--all-leaders", action="store_true")
    ap.add_argument("--pimc", type=int, default=1)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    args = ap.parse_args(argv)

    cfg = {"difficulty": args.difficulty, "max_steps": args.max_steps, "real_decks": args.real_decks,
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
    print(f"\n=== control なぜ下手か診断: control-vs-midrange {n}戦（pimc={args.pimc}） ===")
    if n < 4:
        print("サンプル不足")
        return 1
    cw = sum(1 for r in rows if r["ctrl_win"])
    print(f"control 勝率 = {cw}/{n} ({cw/n:.0%})")

    # control側 vs midrange側 の行動比較（全戦）
    _print_side("control側", _agg_side(rows, "ctrl"))
    _print_side("midrange側", _agg_side(rows, "mid"))

    # control の 勝ち戦 vs 負け戦 で control側行動を比較（負け筋の行動）
    wins = [r for r in rows if r["ctrl_win"]]
    losses = [r for r in rows if not r["ctrl_win"]]
    if wins and losses:
        _print_side(f"control側・勝った{len(wins)}戦", _agg_side(wins, "ctrl"))
        _print_side(f"control側・負けた{len(losses)}戦", _agg_side(losses, "ctrl"))

    # ライフリード推移（control視点）: 勝ち戦/負け戦の平均（どのフェーズで崩れるか）
    def _avg_lead(rs):
        acc = {}
        for r in rs:
            for (t, d) in r["lead"]:
                acc.setdefault(t, []).append(d)
        return {t: sum(v) / len(v) for t, v in acc.items()}
    al_w, al_l = _avg_lead(wins), _avg_lead(losses)
    print("\n--- control視点ライフリード推移（勝ち戦 / 負け戦の平均） ---")
    print(f"{'turn':>4} {'勝った戦':>8} {'負けた戦':>8}")
    for t in sorted(set(list(al_w) + list(al_l)))[:16]:
        w = f"{al_w[t]:+.2f}" if t in al_w else "  -"
        l = f"{al_l[t]:+.2f}" if t in al_l else "  -"
        print(f"{t:4d} {w:>8} {l:>8}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
