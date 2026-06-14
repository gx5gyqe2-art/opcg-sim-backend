"""決定論的 CPU 対 CPU 自己対戦ランナー（効果検証ハーネス）。

ルールエンジン上で CPU 対 CPU を **決定論的・再現可能・自動異常検出付き** で実行し、
カード効果のサイレント失敗・中断リーク・盤面不整合を進行から自動炙り出しする
（docs/TEST_SPEC.md §3.1）。

特徴:
  - 単一の seed で完全再現（全乱数は global random に集約）。
  - 各ステップ後にインバリアント検出（invariants.check_invariants）。違反で即停止＋リプロ出力。
  - 機械可読トレース（JSONL）: 1 行 = 1 ステップ。grep/diff で異常箇所に直行できる。
  - 進行は action_api（HTTP と同一コアパス）で行い、本番挙動と乖離しない。

実行例:
    OPCG_LOG_SILENT=1 python tests/cpu_selfplay.py --seed 0
    OPCG_LOG_SILENT=1 python tests/cpu_selfplay.py --games 20 --seed 0
    OPCG_LOG_SILENT=1 python tests/cpu_selfplay.py --seed 3 --out /tmp/trace.jsonl --verbose
    OPCG_LOG_SILENT=1 python tests/cpu_selfplay.py --p1-leader OP01-001 --p2-leader OP02-001
"""
import argparse
import json
import os
import random
import sys
import traceback
from typing import Any, Dict, List, Optional

import conftest  # noqa: F401  (google スタブ注入 & sys.path 設定)

from opcg_sim.src.core.gamestate import GameManager, Player
from opcg_sim.src.core import action_api, cpu_ai
from opcg_sim.src.core.invariants import check_invariants, check_turn_boundary
from opcg_sim.src.models.models import CardInstance
from opcg_sim.src.utils.loader import CardLoader

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "opcg_sim", "data")

# 1 ゲーム当たりの安全上限（無限ループ検出）。
DEFAULT_MAX_STEPS = 4000


class InvariantError(Exception):
    """インバリアント違反で対局を即停止する例外。"""

    def __init__(self, violations, step, trace_tail):
        self.violations = violations
        self.step = step
        self.trace_tail = trace_tail
        super().__init__(f"invariant violation(s) at step {step}: {violations}")


def _load_db() -> CardLoader:
    db = CardLoader(os.path.join(DATA, "opcg_cards.json"))
    db.load()
    for cid in list(db.raw_db.keys()):
        db.get_card(cid)
    return db


def build_deck(db: CardLoader, owner_id: str, leader_id: Optional[str] = None) -> tuple:
    """カード DB からリーダー + キャラ 50 枚のデッキを構築する。

    leader_id 指定時はそのリーダーを使い、同色のキャラを優先して詰める。
    未指定なら最初に見つかったリーダーを使う。
    """
    leader = None
    if leader_id:
        m = db.get_card(leader_id)
        if m and m.type.name == "LEADER":
            leader = CardInstance(m, owner_id)
    cards: List[CardInstance] = []
    leader_colors = set(getattr(leader.master, "colors", []) or []) if leader else set()
    # 1st pass: 同色キャラ優先
    for cid in db.raw_db.keys():
        c = db.get_card(cid)
        if c is None:
            continue
        if leader is None and c.type.name == "LEADER":
            leader = CardInstance(c, owner_id)
            leader_colors = set(getattr(c, "colors", []) or [])
            continue
        if c.type.name == "CHARACTER" and len(cards) < 50:
            if not leader_colors or (set(getattr(c, "colors", []) or []) & leader_colors):
                cards.append(CardInstance(c, owner_id))
        if leader and len(cards) >= 50:
            break
    # 2nd pass: 不足分は色を問わずキャラで埋める
    if len(cards) < 50:
        for cid in db.raw_db.keys():
            c = db.get_card(cid)
            if c is None or c.type.name != "CHARACTER":
                continue
            cards.append(CardInstance(c, owner_id))
            if len(cards) >= 50:
                break
    return leader, cards


def _zone_snapshot(manager: GameManager) -> Dict[str, int]:
    """ゾーン枚数の軽量スナップショット（効果前後差分 snapshot_diff 用）。"""
    snap: Dict[str, int] = {}
    for p in (manager.p1, manager.p2):
        snap[f"{p.name}_hand"] = len(p.hand)
        snap[f"{p.name}_field"] = len(p.field)
        snap[f"{p.name}_life"] = len(p.life)
        snap[f"{p.name}_trash"] = len(p.trash)
        snap[f"{p.name}_deck"] = len(p.deck)
        snap[f"{p.name}_don"] = len(p.don_active) + len(p.don_rested) + len(p.don_attached_cards)
    return snap


def _snapshot_diff(before: Dict[str, int], after: Dict[str, int]) -> Dict[str, List[int]]:
    """前後スナップショットの差分（変化のあったゾーンのみ {zone: [before, after]}）。"""
    return {k: [before[k], after[k]] for k in before if before[k] != after.get(k)}


def choose_move(manager: GameManager, moves: List[Dict[str, Any]]) -> Dict[str, Any]:
    """PR1 の既定方策: ランダム合法手（global random で決定論的）。

    無限引き延ばし防止のため、攻撃可能なら一定確率で攻撃を優先し、
    そうでなければ TURN_END にやや寄せる軽いバイアスを掛ける。
    PR2 で評価関数ベースの強い方策に差し替える。
    """
    if not moves:
        return None
    attacks = [m for m in moves if m.get("action_type") == "ATTACK"]
    if attacks and random.random() < 0.6:
        return random.choice(attacks)
    end = [m for m in moves if m.get("action_type") == "TURN_END"]
    # メインアクションが TURN_END しか無い、または一定確率でターンを畳む。
    if end and (len(moves) == 1 or random.random() < 0.25):
        return end[0]
    return random.choice(moves)


def _make_policy(policy: str, difficulty: str):
    """手選択関数 (manager, actor, moves) -> move を返す。

    'random' : choose_move（軽いバイアス付きランダム・高速）。
    'ai'     : cpu_ai.decide（評価関数ベースの強い方策・低速＝効果トレース用）。
    """
    if policy == "ai":
        # プレイヤーごとにターン内メモリを保持し、暴走防止ガード付きで意思決定する。
        mem = {"p1": {}, "p2": {}}

        def _ai(manager, actor, moves):
            return cpu_ai.decide_guarded(manager, actor, difficulty, random, mem.setdefault(actor.name, {}))
        return _ai
    return lambda manager, actor, moves: choose_move(manager, moves)


def run_one_game(
    seed: int,
    db: CardLoader,
    p1_leader: Optional[str] = None,
    p2_leader: Optional[str] = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    trace_out=None,
    verbose: bool = False,
    policy: str = "random",
    difficulty: str = "normal",
    oracle: bool = False,
) -> Dict[str, Any]:
    """1 ゲームを決定論的に完走させ、結果サマリを返す。

    インバリアント違反・例外・スタックは InvariantError / 例外で停止し、
    呼び出し側が seed とともに記録できるようにする。
    """
    random.seed(seed)
    l1, c1 = build_deck(db, "p1", p1_leader)
    l2, c2 = build_deck(db, "p2", p2_leader)
    if not l1 or not l2:
        raise RuntimeError("リーダーを含むデッキを構築できませんでした。")
    manager = GameManager(Player("p1", c1, l1), Player("p2", c2, l2))
    manager.start_game()
    pick = _make_policy(policy, difficulty)

    trace_tail: List[Dict[str, Any]] = []
    step = 0
    prev_turn = manager.turn_count

    def emit(line: Dict[str, Any]):
        trace_tail.append(line)
        if len(trace_tail) > 12:
            trace_tail.pop(0)
        if trace_out is not None:
            trace_out.write(json.dumps(line, ensure_ascii=False) + "\n")
        if verbose:
            print(f"[{line['step']:04d}] t{line['turn']} {line['phase']:>13} "
                  f"{line['player']:>2} {line['action']:<22} "
                  f"{'; '.join(e.get('message', '') for e in line.get('events', []))}")

    while manager.winner is None and step < max_steps:
        pending = manager.get_pending_request()
        if not pending:
            # 勝者未確定なのに合法手が無い = スタック（効果バグの疑い）。
            raise InvariantError([("STUCK", "no pending request and no winner")], step, trace_tail)

        pending_props = action_api.CONST.get('PENDING_REQUEST_PROPERTIES', {})
        req_pid = pending[pending_props.get('PLAYER_ID', 'player_id')]
        actor = manager.p1 if manager.p1.name == req_pid else manager.p2

        moves = manager.get_legal_actions(actor)
        if not moves:
            raise InvariantError([("NO_LEGAL_MOVE", f"no legal moves for {req_pid}")], step, trace_tail)

        move = pick(manager, actor, moves)
        before_snap = _zone_snapshot(manager) if oracle else None
        manager.action_events = []
        try:
            if move["kind"] == "battle":
                action_api.apply_battle_action(manager, actor, move["action_type"], move.get("card_uuid"))
            else:
                action_api.apply_game_action(manager, actor, move["action_type"], move.get("payload", {}))
        except Exception as e:
            raise InvariantError(
                [("ACTION_EXCEPTION", f"{type(e).__name__}: {e}\n{traceback.format_exc()}")],
                step, trace_tail,
            )

        events = list(manager.action_events)
        line = {
            "step": step,
            "turn": manager.turn_count,
            "phase": manager.phase.name,
            "player": req_pid,
            "action": move["action_type"],
            "payload": move.get("payload") or {"card_uuid": move.get("card_uuid")},
            "events": events,
        }
        if oracle:
            line["snapshot_diff"] = _snapshot_diff(before_snap, _zone_snapshot(manager))
        emit(line)

        # --- インバリアント検出（各ステップ後） ---
        violations = check_invariants(manager)
        if manager.turn_count != prev_turn:
            violations += check_turn_boundary(manager)
            prev_turn = manager.turn_count
        if violations:
            raise InvariantError(violations, step, trace_tail)

        step += 1

    if manager.winner is None:
        raise InvariantError([("MAX_STEPS", f"game did not finish within {max_steps} steps")], step, trace_tail)

    return {
        "seed": seed,
        "winner": manager.winner,
        "steps": step,
        "turns": manager.turn_count,
        "p1_leader": l1.master.card_id,
        "p2_leader": l2.master.card_id,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="決定論的 CPU 対 CPU 自己対戦ランナー")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--games", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    ap.add_argument("--p1-leader", default=None)
    ap.add_argument("--p2-leader", default=None)
    ap.add_argument("--out", default=None, help="トレース JSONL の出力先")
    ap.add_argument("--policy", choices=["random", "ai"], default="random",
                    help="手選択方策。random=高速ランダム / ai=評価関数ベース（効果トレース用・低速）")
    ap.add_argument("--difficulty", choices=["easy", "normal", "hard"], default="normal",
                    help="--policy ai のときの CPU 難易度")
    ap.add_argument("--oracle", action="store_true",
                    help="効果検証: 各ステップに snapshot_diff（盤面前後差分）を記録する")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    db = _load_db()
    trace_out = open(args.out, "w", encoding="utf-8") if args.out else None
    results, failures = [], []
    try:
        for i in range(args.games):
            seed = args.seed + i
            try:
                res = run_one_game(seed, db, args.p1_leader, args.p2_leader,
                                   max_steps=args.max_steps, trace_out=trace_out, verbose=args.verbose,
                                   policy=args.policy, difficulty=args.difficulty, oracle=args.oracle)
                results.append(res)
                print(f"game seed={seed}: winner={res['winner']} steps={res['steps']} "
                      f"turns={res['turns']} ({res['p1_leader']} vs {res['p2_leader']})")
            except InvariantError as e:
                failures.append((seed, e))
                print(f"game seed={seed}: FAILED step={e.step} violations={e.violations}")
                print("  --- trace tail (repro: --seed {} --verbose) ---".format(seed))
                for line in e.trace_tail:
                    print(f"    [{line['step']:04d}] t{line['turn']} {line['phase']} "
                          f"{line['player']} {line['action']} "
                          f"{'; '.join(ev.get('message','') for ev in line.get('events', []))}")
    finally:
        if trace_out:
            trace_out.close()

    print(f"\nsummary: {len(results)}/{args.games} finished, {len(failures)} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
