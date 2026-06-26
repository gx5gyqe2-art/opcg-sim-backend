"""リーサル監査（診断・dev専用）: 自己対戦を再生し「盤面だけで確定リーサルがあったターン」を検出し、
それを勝ちに変換できたか／取りこぼして最終的に負けたかを集計する。

狙い: hard は eval 律速（深掘りで強くならない）と確定済み。残る弱点が「終盤の詰めの甘さ＝リーサル精度」か
どうかを、**ルールベースの厳密判定**（eval/探索に依存しない正解）で切り分ける。①厳密リーサルソルバーを
作る価値があるかの意思決定材料。

リーサル判定（保守的・隠れ情報無視）:
  - 攻撃側 = 自分の手番開始時のアクティブ非新規キャラ＋リーダー。
  - 各攻撃が相手リーダーへ「届く」= 自パワー(自手番) >= 相手リーダーパワー(守備)。ダブルアタックは2点。
  - 相手のアクティブブロッカー数だけ攻撃を相殺。
  - (届く打点 − 相手ブロッカー) >= 相手ライフ なら「盤面リーサル」。
  ※ ドン付与・除去・自カウンターは数えない（与打点を過小評価＝下限）。一方で相手の手札カウンターも
    無視する（相手防御を過小評価＝過大評価）。よって「機会」は近似で、真に重要な指標は
    **『盤面リーサルがあったのに負けた』**＝高コストな取りこぼし（カウンターでは説明しづらい）。

実行例:
    OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/lethal_audit.py --games 100 --real-decks --all-leaders --pimc 4
"""
import argparse
import random
import sys
from typing import Any, Dict, List, Optional

import conftest  # noqa: F401

from opcg_sim.src.core.gamestate import GameManager, Player
from opcg_sim.src.core import action_api
from opcg_sim.src.core.invariants import check_invariants
from collect_value_data import _build_decks, _make_decider
from cpu_selfplay import _load_db, DEFAULT_MAX_STEPS

_BLOCKER = "ブロッカー"
_DOUBLE = "ダブルアタック"


def board_lethal(me, opp):
    """me の手番開始盤面で opp を確定リーサルできるか（保守的・隠れ情報無視）。

    返り: (is_lethal, reach_pts, opp_blockers, opp_life)
    """
    opp_life = len(opp.life)
    if opp_life <= 0:
        return False, 0, 0, 0
    leader_def = opp.leader.get_power(False) if opp.leader is not None else 0
    attackers = []
    if me.leader is not None:
        attackers.append(me.leader)
    attackers += [c for c in me.field if (not c.is_rest) and (not c.is_newly_played)]
    reach = 0
    for c in attackers:
        try:
            pw = c.get_power(True)
        except Exception:
            continue
        if pw >= leader_def:
            reach += 2 if c.has_keyword(_DOUBLE) else 1
    blockers = sum(1 for c in opp.field if (not c.is_rest) and c.has_keyword(_BLOCKER))
    effective = reach - blockers
    return effective >= opp_life, effective, blockers, opp_life


def audit_game(seed, db, difficulty, max_steps, real_decks, all_leaders, pimc):
    """1局を再生。各ターン開始盤面で手番側の盤面リーサルを記録し、(records, winner, final_turn) を返す。

    records: [{"turn":int,"actor":name,"lethal":bool}] （各ターン1件・そのターンの手番側視点）
    """
    random.seed(seed)
    l1, c1, l2, c2 = _build_decks(seed, db, real_decks, all_leaders=all_leaders)
    if not l1 or not l2:
        return None
    m = GameManager(Player("p1", c1, l1), Player("p2", c2, l2))
    m.start_game()
    decide = _make_decider(difficulty, 40, 2, pimc)
    pid_key = action_api.CONST.get("PENDING_REQUEST_PROPERTIES", {}).get("PLAYER_ID", "player_id")

    records: List[Dict[str, Any]] = []
    seen_turns = set()
    prev_turn = None
    step = 0
    while m.winner is None and step < max_steps:
        pending = m.get_pending_request()
        if not pending:
            break
        actor = m.p1 if m.p1.name == pending[pid_key] else m.p2
        # ターンが切り替わった最初の意思決定点＝手番開始盤面（untap 済み）でリーサル判定。
        if m.turn_count != prev_turn and m.turn_count not in seen_turns:
            prev_turn = m.turn_count
            seen_turns.add(m.turn_count)
            opp = m.p2 if actor is m.p1 else m.p1
            is_leth, _reach, _blk, _life = board_lethal(actor, opp)
            records.append({"turn": m.turn_count, "actor": actor.name, "lethal": is_leth})
        move = decide(m, actor)
        if move is None:
            break
        m.action_events = []
        try:
            if move["kind"] == "battle":
                action_api.apply_battle_action(m, actor, move["action_type"], move.get("card_uuid"))
            else:
                action_api.apply_game_action(m, actor, move["action_type"], move.get("payload", {}))
        except Exception:
            return None
        if check_invariants(m):
            return None
        step += 1
    if m.winner is None:
        return None
    return {"records": records, "winner": m.winner, "final_turn": m.turn_count}


def main(argv=None):
    ap = argparse.ArgumentParser(description="リーサル監査（診断）")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--difficulty", choices=["hard"], default="hard")
    ap.add_argument("--real-decks", action="store_true")
    ap.add_argument("--all-leaders", action="store_true")
    ap.add_argument("--pimc", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    args = ap.parse_args(argv)

    db = _load_db()
    n_games = 0
    leth_turns = 0          # 盤面リーサルが立ったターン数
    converted = 0           # うち、そのターンに手番側が勝った
    miss_lost = 0           # 盤面リーサルが立った手番側が、最終的にゲームを落とした
    miss_won_later = 0      # 立ったが当ターン未勝利・ただし最終的には勝った（遅延）
    games_with_leth = 0
    games_leth_then_lost = 0
    for g in range(args.games):
        r = audit_game(args.seed + g, db, args.difficulty, args.max_steps,
                       args.real_decks, args.all_leaders, args.pimc)
        if r is None:
            continue
        n_games += 1
        winner, final_turn = r["winner"], r["final_turn"]
        game_had_leth = False
        game_leth_loser = False
        for rec in r["records"]:
            if not rec["lethal"]:
                continue
            leth_turns += 1
            game_had_leth = True
            actor = rec["actor"]
            won_this_turn = (winner == actor and rec["turn"] == final_turn)
            if won_this_turn:
                converted += 1
            elif winner == actor:
                miss_won_later += 1
            else:
                miss_lost += 1
                game_leth_loser = True
        if game_had_leth:
            games_with_leth += 1
        if game_leth_loser:
            games_leth_then_lost += 1
        if (g + 1) % 20 == 0:
            print(f"  {g+1}/{args.games} … valid={n_games} leth_turns={leth_turns}", flush=True)

    print(f"\n=== リーサル監査: {n_games} 局（pimc={args.pimc}） ===")
    if leth_turns == 0:
        print("盤面リーサルの立ったターンが0（要サンプル増 or 判定見直し）")
        return 0
    print(f"盤面リーサルが立ったターン: {leth_turns}")
    print(f"  そのターンに勝利（変換）   : {converted} ({converted/leth_turns:.1%})")
    print(f"  当ターン未勝利→のち勝利    : {miss_won_later} ({miss_won_later/leth_turns:.1%})")
    print(f"  当ターン未勝利→最終的に敗北: {miss_lost} ({miss_lost/leth_turns:.1%})  ← 高コストな取りこぼし")
    print(f"局単位: 盤面リーサルが立った局 {games_with_leth}/{n_games}・"
          f"うち最終敗北 {games_leth_then_lost} ({games_leth_then_lost/max(1,games_with_leth):.1%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
