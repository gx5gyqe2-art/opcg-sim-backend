"""決定論的 CPU 対 CPU 自己対戦ランナー（効果検証ハーネス）。

ルールエンジン上で CPU 対 CPU を **決定論的・再現可能・自動異常検出付き** で実行し、
カード効果のサイレント失敗・中断リーク・盤面不整合を進行から自動炙り出しする
（docs/TEST_SPEC.md §3.1）。

特徴:
  - 単一の seed で完全再現（全乱数は global random に集約）。
  - 各ステップ後にインバリアント検出（invariants.check_invariants）。違反で即停止＋リプロ出力。
  - 機械可読トレース（JSONL）: 1 行 = 1 ステップ。grep/diff で異常箇所に直行できる。
  - 進行は action_api（HTTP と同一コアパス）で行い、本番挙動と乖離しない。

対局ループ本体は `game_driver.run_game`（設計⑥・全 CPU 検証ハーネス共通）へ集約した。本モジュールは
その **観測 observer**（JSONL emit・oracle 差分）と CLI を担う。共有部品（`load_db`/`build_deck`/
`InvariantError`/`DEFAULT_MAX_STEPS`/`choose_move`/スナップショット）は `game_driver` へ移設し、
本モジュールは後方互換のため `_load_db` などを **再エクスポート**する。

実行例:
    OPCG_LOG_SILENT=1 python tests/cpu_selfplay.py --seed 0
    OPCG_LOG_SILENT=1 python tests/cpu_selfplay.py --games 20 --seed 0
    OPCG_LOG_SILENT=1 python tests/cpu_selfplay.py --seed 3 --out /tmp/trace.jsonl --verbose
    OPCG_LOG_SILENT=1 python tests/cpu_selfplay.py --p1-leader OP01-001 --p2-leader OP02-001
"""
import argparse
import json
import sys
from typing import Any, Dict, List, Optional

import os as _os, sys as _sys  # noqa: E402  test bootstrap (sys.path + google stub)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap  # noqa: E402,F401

# 対局ドライバ＋共有部品（再エクスポート＝既存の `from cpu_selfplay import ...` を壊さない）。
from game_driver import (  # noqa: F401
    load_db as _load_db,
    build_deck,
    InvariantError,
    DEFAULT_MAX_STEPS,
    choose_move,
    make_seat,
    leader_deck_builder,
    run_game,
    _zone_snapshot,
    _snapshot_diff,
)
from opcg_sim.src.core.invariants import check_invariants, check_turn_boundary  # noqa: F401


def _make_policy(policy: str, difficulty: str):
    """手選択関数（seat）を返す（後方互換ラッパ。内部は game_driver.make_seat）。

    'random' : choose_move（軽いバイアス付きランダム・高速）。
    'ai'     : cpu_ai.decide（評価関数ベースの強い方策・低速＝効果トレース用）。
    """
    if policy == "ai":
        mem = {"p1": {}, "p2": {}}
        return {pid: make_seat(difficulty, kind="ai", mem=mem.setdefault(pid, {}))
                for pid in ("p1", "p2")}
    return {pid: make_seat(kind="random") for pid in ("p1", "p2")}


class _SelfplayObserver:
    """1 ステップ = 1 JSONL 行を emit する観測子（効果検証用）。oracle 時は盤面前後差分を付す。"""

    def __init__(self, emit, oracle: bool):
        self.emit = emit
        self.oracle = oracle
        self._before: Optional[Dict[str, int]] = None

    def on_decision(self, ctx, move):
        # 旧 run_one_game は「decide 後・apply 前」に before スナップショットを採る。
        self._before = _zone_snapshot(ctx.manager) if self.oracle else None

    def on_step(self, ctx, move, events):
        m = ctx.manager
        line = {
            "step": ctx.step,
            "turn": m.turn_count,        # apply 後の値（旧実装と一致）
            "phase": m.phase.name,
            "player": ctx.actor.name,
            "action": move["action_type"],
            "payload": move.get("payload") or {"card_uuid": move.get("card_uuid")},
            "events": events,
        }
        if self.oracle:
            line["snapshot_diff"] = _snapshot_diff(self._before, _zone_snapshot(m))
        self.emit(line)


def run_one_game(
    seed: int,
    db,
    p1_leader: Optional[str] = None,
    p2_leader: Optional[str] = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    trace_out=None,
    verbose: bool = False,
    policy: str = "random",
    difficulty: str = "hard",
    oracle: bool = False,
) -> Dict[str, Any]:
    """1 ゲームを決定論的に完走させ、結果サマリを返す。

    インバリアント違反・例外・スタックは InvariantError / 例外で停止し、
    呼び出し側が seed とともに記録できるようにする。
    """
    trace_tail: List[Dict[str, Any]] = []

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

    seats = _make_policy(policy, difficulty)
    obs = _SelfplayObserver(emit, oracle)
    result = run_game(seed, db, seats=seats,
                      deck_builder=leader_deck_builder(p1_leader, p2_leader),
                      observers=[obs], max_steps=max_steps,
                      legal_moves="check", invariants="raise", trace_tail=trace_tail)
    return {
        "seed": seed,
        "winner": result.winner,
        "steps": result.steps,
        "turns": result.turns,
        "p1_leader": result.p1_leader,
        "p2_leader": result.p2_leader,
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
    ap.add_argument("--difficulty", choices=["hard"], default="hard",
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
