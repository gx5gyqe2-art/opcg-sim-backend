"""game-shape センサス（診断・dev専用）: hard同士のミラー自己対戦で各局の**構造**を記録・集計し、
「どのフェーズ・どの決着モードで勝敗が決まるか」を把握する（負け筋の差分分析の第一歩＝ボトルネック特定）。

eval/探索の高度化が総合Eloを動かさないと確定したので、そもそも何が勝敗を決めているかを直接見る。各局で:
  - 長さ（決着ターン数）
  - 決着モード: ライフ切れ（lifeout） vs デッキ切れ（deckout）
  - ライフ推移（ターン境界ごとの両者ライフ）→ 勝者のリードが固定した「決着ターン」
  - 逆転の有無（敗者が一度でもライフでリードしたか）
  - 勝者/敗者リーダー
を取り、分布を出す。早期決着が多い＝開幕/テンポ律速、長期グラインド＝資源/リーサル律速、deckout多発＝
山札管理律速、のように**次に見るべき場所**を示す。

実行例:
    OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/game_census.py --games 80 --real-decks --all-leaders --pimc 1
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


def _archetype(cards, leader):
    """デッキ構成からアーキタイプ（aggro/midrange/control）を分類（本番 plan と同じ build_plan 由来）。"""
    try:
        p = cpu_self_plan.build_plan([c.master for c in cards], leader=leader)
        return getattr(p, "archetype", "midrange")
    except Exception:
        return "midrange"

_DB = None
_CFG: Dict[str, Any] = {}


def _init(cfg):
    global _DB, _CFG
    _DB = _load_db()
    _CFG = cfg


def census_game(seed: int):
    random.seed(seed)
    l1, c1, l2, c2 = _build_decks(seed, _DB, _CFG["real_decks"], all_leaders=_CFG["all_leaders"])
    if not l1 or not l2:
        return None
    arch = {"p1": _archetype(c1, l1), "p2": _archetype(c2, l2)}
    m = GameManager(Player("p1", c1, l1), Player("p2", c2, l2))
    m.start_game()
    decide = _make_decider(_CFG["difficulty"], 40, 2, _CFG["pimc"])
    pid_key = action_api.CONST.get("PENDING_REQUEST_PROPERTIES", {}).get("PLAYER_ID", "player_id")

    traj = []  # (turn, p1_life, p2_life)
    last_turn = -1
    step = 0
    while m.winner is None and step < _CFG["max_steps"]:
        tc = getattr(m, "turn_count", 0)
        if tc != last_turn:
            traj.append((tc, len(m.p1.life), len(m.p2.life)))
            last_turn = tc
        pending = m.get_pending_request()
        if not pending:
            break
        actor = m.p1 if m.p1.name == pending[pid_key] else m.p2
        chosen = decide(m, actor)
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
    if m.winner is None:
        return None

    win = m.winner
    winner = m.p1 if m.p1.name == win else m.p2
    loser = m.p2 if m.p1.name == win else m.p1
    # 決着モード: 敗者ライフ0=lifeout / それ以外でデッキ空=deckout / それ以外=other
    if len(loser.life) == 0:
        mode = "lifeout"
    elif len(loser.deck) == 0:
        mode = "deckout"
    else:
        mode = "other"

    # 勝者視点のライフ差推移 d(turn)=winner_life - loser_life
    wi = 0 if win == "p1" else 1
    diffs = [(t, (a if wi == 0 else b) - (b if wi == 0 else a)) for (t, a, b) in traj]
    # 決着ターン: これ以降ずっと winner_life >= loser_life（リード固定）になる最初のターン
    decided = traj[-1][0] if traj else 0
    for k in range(len(diffs)):
        if all(d >= 0 for (_t, d) in diffs[k:]):
            decided = diffs[k][0]
            break
    # 逆転: 敗者が一度でも厳密リード（diff<0）を持ったか
    comeback = any(d < 0 for (_t, d) in diffs)

    return {
        "turns": traj[-1][0] if traj else 0,
        "mode": mode,
        "decided": decided,
        "comeback": comeback,
        "win_leader": getattr(winner.leader.master, "card_id", "?") if winner.leader else "?",
        "lose_leader": getattr(loser.leader.master, "card_id", "?") if loser.leader else "?",
        "win_arch": arch[win],
        "lose_arch": arch["p2" if win == "p1" else "p1"],
    }


def _one(seed):
    try:
        return census_game(seed)
    except Exception:
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description="game-shape センサス（hardミラー）")
    ap.add_argument("--games", type=int, default=80)
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
            if i % 20 == 0:
                print(f"  {i}/{args.games} … valid={len(rows)}", flush=True)

    n = len(rows)
    print(f"\n=== game-shape センサス: {n}局（hardミラー・pimc={args.pimc}） ===")
    if n == 0:
        print("有効局なし")
        return 1

    turns = sorted(r["turns"] for r in rows)
    decided = sorted(r["decided"] for r in rows)
    pct = lambda xs, p: xs[min(len(xs) - 1, int(p * len(xs)))]
    print(f"決着ターン数: 中央={pct(turns,0.5)} p25={pct(turns,0.25)} p75={pct(turns,0.75)} "
          f"min={turns[0]} max={turns[-1]} 平均={sum(turns)/n:.1f}")
    print(f"決着ターン（リード固定）: 中央={pct(decided,0.5)} p25={pct(decided,0.25)} p75={pct(decided,0.75)}")
    fracs = sorted((r["decided"] / max(1, r["turns"])) for r in rows)
    print(f"決着/全長 比: 中央={pct(fracs,0.5):.2f}（小さい=早期に勝敗固定・大きい=最後まで競る）")

    modes = {}
    for r in rows:
        modes[r["mode"]] = modes.get(r["mode"], 0) + 1
    print("\n決着モード: " + " / ".join(f"{k}={v}({v/n:.0%})" for k, v in sorted(modes.items(), key=lambda x: -x[1])))
    cb = sum(1 for r in rows if r["comeback"])
    print(f"逆転あり（敗者が一度リード）: {cb}/{n} ({cb/n:.0%})")

    # アーキタイプ別 勝率（ミラーなので登場＝win側 or lose側のどちらか・出場数で正規化）。
    arch_wl: Dict[str, List[int]] = {}
    for r in rows:
        arch_wl.setdefault(r["win_arch"], [0, 0])[0] += 1
        arch_wl.setdefault(r["lose_arch"], [0, 0])[1] += 1
    print("\n--- アーキタイプ別 勝/敗（偏り＝hardのそのプレイスタイルの巧拙＋アーキ強度） ---")
    print(f"{'archetype':10} {'win':>4} {'lose':>4} {'勝率':>6} {'出場':>5}")
    for a in ("aggro", "midrange", "control"):
        if a in arch_wl:
            w, l = arch_wl[a]
            print(f"{a:10} {w:4d} {l:4d} {w/max(1,w+l):6.0%} {w+l:5d}")
    # マッチアップ行列（行アーキ vs 列アーキ・行の勝率）。
    mu: Dict[tuple, List[int]] = {}
    for r in rows:
        mu.setdefault((r["win_arch"], r["lose_arch"]), [0, 0])[0] += 1   # 行=win_arch が勝ち
        mu.setdefault((r["lose_arch"], r["win_arch"]), [0, 0])[1] += 1   # 行=lose_arch が負け
    archs = ("aggro", "midrange", "control")
    print("\n--- マッチアップ勝率（行が列に対して。n小は参考） ---")
    print(f"{'':10} " + " ".join(f"{c:>10}" for c in archs))
    for a in archs:
        cells = []
        for b in archs:
            w, l = mu.get((a, b), [0, 0])
            cells.append(f"{(w/(w+l)) if (w+l) else 0:>8.0%}({w+l})" if (w + l) else f"{'-':>10}")
        print(f"{a:10} " + " ".join(f"{c:>10}" for c in cells))

    wl: Dict[str, List[int]] = {}
    for r in rows:
        wl.setdefault(r["win_leader"], [0, 0])[0] += 1
        wl.setdefault(r["lose_leader"], [0, 0])[1] += 1
    rank = sorted(wl.items(), key=lambda kv: -(kv[1][0] + kv[1][1]))
    print("\n--- リーダー別 勝/敗（登場数上位14） ---")
    print(f"{'leader':12} {'win':>4} {'lose':>4} {'勝率':>6} {'n':>4}")
    for lid, (w, l) in rank[:14]:
        tot = w + l
        print(f"{lid:12} {w:4d} {l:4d} {w/max(1,tot):6.0%} {tot:4d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
