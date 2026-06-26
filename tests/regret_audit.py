"""多観点 regret 診断（dev専用）: 自己対戦の各意思決定で「勝率 regret」を測り、文脈でバケツ分けして
CPU がどの種類の判断で価値を最も失っているかを炙り出す。リーサル決め打ちを避ける（確証バイアス回避）。

統一指標: 各決定で合法手すべてを**価値モデル（判定者）**で評価し
  regret = winprob(最善候補) − winprob(CPUが実際に選んだ手)   （>=0・actor視点）
eval 律速のため深掘り regret は無意味＝勝率モデルを独立判定者に使う。判定者は自己対戦由来（天井0.68）で
相関はあるが、1局面での候補手の相対順位付けには使え、**系統的な偏り**は出せる（絶対の最適性保証は無い）。

文脈バケツ:
  - 守備応答         : 相手ターンの自分の応答（ブロック/カウンター/パス）
  - 攻め・リーサル帯 : 自ターン本体・相手ライフ<=2（詰めの場面）
  - 攻め・通常       : 自ターン本体・相手ライフ>2
  - 目標選択         : 単一対象選択ノード
各バケツで 件数 / 平均regret / blunder率(regret>0.10) / 総regret を出し、攻めバケツでは
「最善手が攻撃だった割合」（＝攻撃見送り信号）も集計する。

実行例:
    OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/regret_audit.py --games 60 --real-decks --all-leaders \
        --pimc 4 --budget 75 --judge /path/to/judge.json
"""
import argparse
import multiprocessing as mp
import os
import random
import sys
from typing import Any, Dict, List, Optional

import conftest  # noqa: F401

from opcg_sim.src.core.gamestate import GameManager, Player
from opcg_sim.src.core import action_api, cpu_ai, cpu_features, cpu_value_model
from opcg_sim.src.core.invariants import check_invariants
from collect_value_data import _build_decks, _make_decider
from cpu_selfplay import _load_db, DEFAULT_MAX_STEPS

BLUNDER = 0.10  # regret がこの閾値超え＝崖（blunder）として率を出す

_DB = None
_CFG: Dict[str, Any] = {}
_JUDGE = None


def _init_worker(cfg: Dict[str, Any]):
    global _DB, _CFG, _JUDGE
    _DB = _load_db()
    _CFG = cfg
    cpu_ai.set_budget_override(cfg.get("budget"))
    _JUDGE = cpu_value_model.load_model_file(cfg["judge"]) if cfg.get("judge") else None


def _wp_after(manager, actor_name: str, move) -> float:
    """move を適用した子局面の actor 視点 winprob（非破壊・判定者モデル）。失敗時 0.5。"""
    def fn(board):
        f = cpu_features.extract_features(board, actor_name, see_opp_hand=False)
        p = cpu_value_model.predict_winprob(f, model=_JUDGE)
        return 0.5 if p is None else float(p)
    try:
        r = cpu_ai._recurse_child(manager, actor_name, move, fn)
    except Exception:
        return 0.5
    return 0.5 if r is None else r


def _candidate_moves(manager, actor):
    """decide と同じ候補手集合（単一対象選択ノード or 合法手＋本番プルーニング）。"""
    sel = cpu_ai._selection_moves(manager, actor.name)
    if sel is not None:
        return sel, True
    moves = manager.get_legal_actions(actor)
    moves = cpu_ai._prune_don_moves(manager, actor.name, moves)
    moves = cpu_ai._prune_futile_attacks(manager, actor.name, moves)
    return moves, False


def _bucket(actor, opp, is_defending: bool, is_select: bool, has_attack: bool) -> str:
    if is_defending:
        return "守備応答"
    if is_select:
        return "目標選択"
    if has_attack and len(opp.life) <= 2:
        return "攻め・リーサル帯"
    return "攻め・通常"


def audit_game(seed: int):
    random.seed(seed)
    l1, c1, l2, c2 = _build_decks(seed, _DB, _CFG["real_decks"], all_leaders=_CFG["all_leaders"])
    if not l1 or not l2:
        return None
    m = GameManager(Player("p1", c1, l1), Player("p2", c2, l2))
    m.start_game()
    decide = _make_decider(_CFG["difficulty"], 40, 2, _CFG["pimc"])
    pid_key = action_api.CONST.get("PENDING_REQUEST_PROPERTIES", {}).get("PLAYER_ID", "player_id")

    # バケツ→[件数, 総regret, blunder件数, 最善が攻撃の件数]
    stats: Dict[str, List[float]] = {}
    step = 0
    while m.winner is None and step < _CFG["max_steps"]:
        pending = m.get_pending_request()
        if not pending:
            break
        actor = m.p1 if m.p1.name == pending[pid_key] else m.p2
        opp = m.p2 if actor is m.p1 else m.p1
        is_defending = (m.turn_player is not None and m.turn_player.name != actor.name)

        moves, is_select = _candidate_moves(m, actor)
        if moves and len(moves) > 1:
            has_attack = any(x.get("action_type") == "ATTACK" for x in moves)
            wps = []
            for x in moves:
                wps.append((_wp_after(m, actor.name, x), x))
            best_wp, best_move = max(wps, key=lambda t: t[0])
            chosen = decide(m, actor)
            csig = cpu_ai._move_sig(chosen) if chosen is not None else None
            chosen_wp = None
            for wp, x in wps:
                if csig is not None and cpu_ai._move_sig(x) == csig:
                    chosen_wp = wp
                    break
            if chosen_wp is None:
                chosen_wp = _wp_after(m, actor.name, chosen) if chosen is not None else best_wp
            regret = max(0.0, best_wp - chosen_wp)
            b = _bucket(actor, opp, is_defending, is_select, has_attack)
            s = stats.setdefault(b, [0.0, 0.0, 0.0, 0.0])
            s[0] += 1
            s[1] += regret
            if regret > BLUNDER:
                s[2] += 1
            if best_move.get("action_type") == "ATTACK":
                s[3] += 1
        else:
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
    return stats


def _audit_one(seed: int):
    try:
        return audit_game(seed)
    except Exception:
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description="多観点 regret 診断")
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--difficulty", choices=["hard"], default="hard")
    ap.add_argument("--real-decks", action="store_true")
    ap.add_argument("--all-leaders", action="store_true")
    ap.add_argument("--pimc", type=int, default=4)
    ap.add_argument("--budget", type=int, default=75)
    ap.add_argument("--judge", default="", help="判定者の価値モデル JSON（未指定なら同梱モデル）")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    args = ap.parse_args(argv)

    cfg = {"difficulty": args.difficulty, "max_steps": args.max_steps, "real_decks": args.real_decks,
           "all_leaders": args.all_leaders, "pimc": args.pimc, "budget": args.budget,
           "judge": args.judge or None}
    workers = args.workers or max(1, (os.cpu_count() or 2) - 1)
    seeds = [args.seed + g for g in range(args.games)]

    agg: Dict[str, List[float]] = {}
    n_games = 0
    with mp.Pool(workers, initializer=_init_worker, initargs=(cfg,)) as pool:
        for i, st in enumerate(pool.imap_unordered(_audit_one, seeds), 1):
            if st is None:
                continue
            n_games += 1
            for b, s in st.items():
                a = agg.setdefault(b, [0.0, 0.0, 0.0, 0.0])
                for k in range(4):
                    a[k] += s[k]
            if i % 20 == 0:
                print(f"  {i}/{args.games} … valid={n_games}", flush=True)

    print(f"\n=== 多観点 regret 診断: {n_games} 局（pimc={args.pimc}・判定者={args.judge or '同梱'}） ===")
    order = ["攻め・リーサル帯", "攻め・通常", "守備応答", "目標選択"]
    tot_regret = sum(a[1] for a in agg.values())
    print(f"{'バケツ':16s} {'決定数':>6} {'平均regret':>10} {'blunder率':>9} {'総regret':>9} {'寄与%':>6} {'最善=攻撃%':>9}")
    for b in order + [k for k in agg if k not in order]:
        if b not in agg:
            continue
        n, sr, bl, atk = agg[b]
        if n == 0:
            continue
        share = sr / tot_regret * 100 if tot_regret > 0 else 0
        print(f"{b:16s} {int(n):6d} {sr/n:10.4f} {bl/n:8.1%} {sr:9.2f} {share:5.1f}% {atk/n:8.1%}")
    print(f"\n総regret={tot_regret:.1f}（高いバケツ＝次に作るべき改善対象）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
