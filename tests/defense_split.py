"""守備regretの eval/探索 分解（診断・dev専用）: 守備応答で「hardの戦闘解決後eval」と「勝率判定者」を
両方とり、守備regret を **eval成分**（evalの誤順位）と**探索成分**（eval最善すら選べてない）に分ける。

各守備決定（相手ターンの自分の応答）で候補手 c ごとに静止点まで畳んで:
  eval[c]  = evaluate_base（hard の葉評価・自分視点）
  wp[c]    = 価値モデル winprob（独立判定者・outcome 由来）
eval_best=argmax eval, judge_best=argmax wp, chosen=hard の実選択 として
  total_regret  = wp[judge_best] − wp[chosen]            （= 監査の守備regret）
  eval成分      = wp[judge_best] − wp[eval_best]  (>=0)   ← evalが守備を誤評価する分（eval問題の純量）
  探索成分      = wp[eval_best]  − wp[chosen]            ← hardがeval最善を選べてない分（ビーム/予算）
eval成分 >> 探索成分 なら **eval問題**確定。高regret実例もダンプしてミスの性質を見る。

実行例:
    OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/defense_split.py --games 40 --real-decks --all-leaders \
        --pimc 4 --budget 75 --judge /path/judge.json --dump 12
"""
import argparse
import heapq
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

_SETTLE_LIMIT = 60
_DB = None
_CFG: Dict[str, Any] = {}
_JUDGE = None


def _init_worker(cfg):
    global _DB, _CFG, _JUDGE
    _DB = _load_db()
    _CFG = cfg
    cpu_ai.set_budget_override(cfg.get("budget"))
    _JUDGE = cpu_value_model.load_model_file(cfg["judge"]) if cfg.get("judge") else None


def _settle(board, root_name):
    bo = action_api.CONST.get('c_to_s_interface', {}).get('BATTLE_ACTIONS', {}).get('TYPES', {})
    ACT_PASS = bo.get('PASS', 'PASS')
    for _ in range(_SETTLE_LIMIT):
        if board.winner is not None:
            break
        pa = board.pending_actor_action()
        if not pa:
            break
        pid, action = pa
        if pid != root_name and action == "MAIN_ACTION":
            break
        actor = board.p1 if board.p1.name == pid else board.p2
        board.action_events = []
        try:
            if action == "MAIN_ACTION":
                action_api.apply_game_action(board, actor, "TURN_END", {})
            elif action in ("SELECT_BLOCKER", "SELECT_COUNTER"):
                action_api.apply_battle_action(board, actor, ACT_PASS, None)
            else:
                pending = board.get_pending_request()
                payload = board.default_interaction_payload(pending)
                action_api.apply_game_action(board, actor, action_api.ACT_RESOLVE_SELECTION, payload)
        except Exception:
            break


def _scores_after(manager, actor_name, move):
    """move を適用し静止点まで畳んだ後の (eval_base, winprob)（自分視点）。失敗時 None。"""
    def fn(board):
        _settle(board, actor_name)
        if board.winner is not None:
            won = (board.winner == actor_name)
            return (cpu_ai.W_WIN if won else -cpu_ai.W_WIN, 1.0 if won else 0.0)
        ev = cpu_ai.evaluate_base(board, actor_name, see_opp_hand=False)
        f = cpu_features.extract_features(board, actor_name, see_opp_hand=False)
        p = cpu_value_model.predict_winprob(f, model=_JUDGE)
        return (float(ev), 0.5 if p is None else float(p))
    try:
        return cpu_ai._recurse_child(manager, actor_name, move, fn)
    except Exception:
        return None


def _candidate_moves(manager, actor):
    sel = cpu_ai._selection_moves(manager, actor.name)
    if sel is not None:
        return sel
    return manager.get_legal_actions(actor)


def _context(manager, defender):
    """高regret実例用の軽量盤面記述。"""
    ab = getattr(manager, "active_battle", None)
    atk_pw = 0
    if ab and ab.get("attacker") is not None:
        try:
            atk_pw = ab["attacker"].get_power(True)
        except Exception:
            atk_pw = 0
    counters = sum(1 for c in defender.hand if (getattr(c, "current_counter", 0) or 0) > 0)
    blockers = sum(1 for c in defender.field if (not c.is_rest) and c.has_keyword("ブロッカー"))
    return {"life": len(defender.life), "atk_pw": atk_pw, "counters": counters, "blockers": blockers}


def audit_game(seed):
    random.seed(seed)
    l1, c1, l2, c2 = _build_decks(seed, _DB, _CFG["real_decks"], all_leaders=_CFG["all_leaders"])
    if not l1 or not l2:
        return None
    m = GameManager(Player("p1", c1, l1), Player("p2", c2, l2))
    m.start_game()
    decide = _make_decider(_CFG["difficulty"], 40, 2, _CFG["pimc"])
    pid_key = action_api.CONST.get("PENDING_REQUEST_PROPERTIES", {}).get("PLAYER_ID", "player_id")

    n = 0
    sum_total = sum_eval = sum_search = 0.0
    bl_total = bl_eval = 0           # blunder件数（>0.10）と、うち eval成分が主因の件数
    examples = []                    # (total, ctx, chosen_at, eval_at, judge_at, wps)
    step = 0
    while m.winner is None and step < _CFG["max_steps"]:
        pending = m.get_pending_request()
        if not pending:
            break
        actor = m.p1 if m.p1.name == pending[pid_key] else m.p2
        is_defending = (m.turn_player is not None and m.turn_player.name != actor.name)
        chosen = None
        if is_defending:
            moves = _candidate_moves(m, actor)
            if moves and len(moves) > 1:
                scored = []
                for x in moves:
                    sc = _scores_after(m, actor.name, x)
                    if sc is not None:
                        scored.append((sc[0], sc[1], x))
                if len(scored) >= 2:
                    ctx = _context(m, actor)
                    chosen = decide(m, actor)
                    csig = cpu_ai._move_sig(chosen) if chosen is not None else None
                    eval_best = max(scored, key=lambda t: t[0])
                    judge_best = max(scored, key=lambda t: t[1])
                    ch = next((s for s in scored if csig is not None and cpu_ai._move_sig(s[2]) == csig), None)
                    if ch is not None:
                        total = max(0.0, judge_best[1] - ch[1])
                        eval_c = judge_best[1] - eval_best[1]
                        search_c = eval_best[1] - ch[1]
                        n += 1
                        sum_total += total
                        sum_eval += eval_c
                        sum_search += search_c
                        if total > 0.10:
                            bl_total += 1
                            if eval_c >= search_c:
                                bl_eval += 1
                            if len(examples) < 200:
                                examples.append((total, ctx,
                                                 ch[2].get("action_type"),
                                                 eval_best[2].get("action_type"),
                                                 judge_best[2].get("action_type"),
                                                 round(ch[1], 2), round(eval_best[1], 2), round(judge_best[1], 2)))
        if chosen is None:
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
    return (n, sum_total, sum_eval, sum_search, bl_total, bl_eval, examples)


def _one(seed):
    try:
        return audit_game(seed)
    except Exception:
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description="守備regretの eval/探索 分解")
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--difficulty", choices=["hard"], default="hard")
    ap.add_argument("--real-decks", action="store_true")
    ap.add_argument("--all-leaders", action="store_true")
    ap.add_argument("--pimc", type=int, default=4)
    ap.add_argument("--budget", type=int, default=75)
    ap.add_argument("--judge", default="")
    ap.add_argument("--dump", type=int, default=12, help="高regret守備の実例ダンプ件数")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    args = ap.parse_args(argv)

    cfg = {"difficulty": args.difficulty, "max_steps": args.max_steps, "real_decks": args.real_decks,
           "all_leaders": args.all_leaders, "pimc": args.pimc, "budget": args.budget,
           "judge": args.judge or None}
    workers = args.workers or max(1, (os.cpu_count() or 2) - 1)
    seeds = [args.seed + g for g in range(args.games)]

    N = 0
    s_total = s_eval = s_search = 0.0
    BL = BLE = 0
    all_ex = []
    n_games = 0
    with mp.Pool(workers, initializer=_init_worker, initargs=(cfg,)) as pool:
        for i, r in enumerate(pool.imap_unordered(_one, seeds), 1):
            if r is None:
                continue
            n_games += 1
            n, st, se, ss, bt, be, ex = r
            N += n; s_total += st; s_eval += se; s_search += ss; BL += bt; BLE += be
            all_ex.extend(ex)
            if i % 20 == 0:
                print(f"  {i}/{args.games} … valid={n_games} defense_decisions={N}", flush=True)

    print(f"\n=== 守備regret分解: {n_games}局・守備決定 {N} 件（pimc={args.pimc}） ===")
    if N == 0:
        print("守備決定なし")
        return 0
    print(f"平均 total regret : {s_total/N:.4f}")
    print(f"  ├ eval成分      : {s_eval/N:.4f}  ({s_eval/max(1e-9,s_total)*100:.0f}%)  ← evalの守備誤評価")
    print(f"  └ 探索成分      : {s_search/N:.4f}  ({s_search/max(1e-9,s_total)*100:.0f}%)  ← eval最善の取りこぼし")
    print(f"blunder(>0.10) {BL}件・うち eval成分が主因 {BLE} ({BLE/max(1,BL):.0%})")
    verdict = "eval問題が主因" if s_eval > s_search * 1.5 else \
              "探索問題が主因" if s_search > s_eval * 1.5 else "eval・探索が拮抗"
    print(f"判定: {verdict}")

    print(f"\n--- 高regret守備の実例（上位{args.dump}） chosen/eval最善/judge最善（括弧=各勝率） ---")
    print(f"{'regret':>7} {'life':>4} {'atkPw':>6} {'cnt':>3} {'blk':>3}  chosen→  eval最善 / judge最善")
    for total, ctx, ch_at, ev_at, jd_at, chw, evw, jdw in heapq.nlargest(args.dump, all_ex, key=lambda t: t[0]):
        print(f"{total:7.3f} {ctx['life']:4d} {ctx['atk_pw']:6d} {ctx['counters']:3d} {ctx['blockers']:3d}  "
              f"{ch_at}({chw}) → eval:{ev_at}({evw}) / judge:{jd_at}({jdw})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
