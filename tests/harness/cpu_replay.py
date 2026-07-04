"""決定論的 CPU 対局リプレイ＋思考トレース・ハーネス（CPU 挙動改善用・Phase 1）。

`cpu_selfplay.py` の決定論ランナー（全乱数は global random に集約・action_api で本番同一コアパス）を
土台に、**CPU の意思決定トレース**（選んだ手・上位候補スコア・regret・J値成分内訳・読み筋）を
1 局ぶん 1 ファイルへローカル出力する。GCS には一切行かない＝手元で grep/diff して挙動を読める。

なぜ「リプレイ種」か（docs の設計方針）:
  エンジンは単一 seed で完全決定論（shuffle も CPU 思考も global random）。だから重いログを運ぶ代わりに
  **seed＋リーダー＋難易度（＝リプレイ種）** だけ残せば、ここで対局を丸ごと再生して思考トレースを
  好きな詳細度でローカル再生成できる。種は数 KB でチャットに貼れる。Phase 2 で実アプリ対局の
  人間操作列を種へ載せれば、同じ仕組みで実対局を再生・回帰テスト化できる。

トレースの手記述は uuid（実行ごとに変わる）でなく card_id 基準＝同一 seed で安定再現する。

実行例:
    # 新規対局を決定論再生し、盤面ステップ＋思考トレースを JSONL へ
    OPCG_LOG_SILENT=1 python tests/cpu_replay.py --seed 7 --difficulty hard --out /tmp/replay.jsonl
    # リプレイ種を書き出す（後で --descriptor で完全再現できる）
    OPCG_LOG_SILENT=1 python tests/cpu_replay.py --seed 7 --difficulty hard --record /tmp/seed.json
    # 種から再生（seed/leader/difficulty は種の値を使う）
    OPCG_LOG_SILENT=1 python tests/cpu_replay.py --descriptor /tmp/seed.json --out /tmp/replay.jsonl
    # 思考トレースだけ見る（盤面ステップを省く）
    OPCG_LOG_SILENT=1 python tests/cpu_replay.py --seed 7 --difficulty hard --decisions-only --out -
"""
import argparse
import json
import sys
from typing import Any, Dict, List, Optional

import os as _os, sys as _sys  # noqa: E402  test bootstrap (sys.path + google stub)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap  # noqa: E402,F401

from game_driver import (
    load_db as _load_db,
    build_deck,  # noqa: F401  (再エクスポート互換)
    DEFAULT_MAX_STEPS,
    InvariantError,
    make_seat,
    leader_deck_builder,
    run_game,
)

SCHEMA = "opcg-replay/v1"


def _open_out(path: Optional[str]):
    """`--out -` は stdout、None は無出力、それ以外はファイル。"""
    if path is None:
        return None, False
    if path == "-":
        return sys.stdout, False
    return open(path, "w", encoding="utf-8"), True


class _ReplayObserver:
    """CPU の思考トレース（decision）と盤面ステップ（step）を JSONL へ書き出す観測子。

    decision 行は decide 直後（apply 前）に seat が書いた `ctx.trace` を載せる。step 行は apply 後の
    盤面（turn/phase/events）で構成する。手記述は card_id 基準＝同一 seed で安定再現する。
    """

    def __init__(self, emit, decisions, emit_steps: bool, emit_decisions: bool, verbose: bool):
        self.emit = emit
        self.decisions = decisions
        self.emit_steps = emit_steps
        self.emit_decisions = emit_decisions
        self.verbose = verbose

    def on_decision(self, ctx, move):
        tr = ctx.trace
        decision_rec = {
            "type": "decision", "step": ctx.step, "turn": ctx.turn,
            "phase": ctx.phase, "player": ctx.actor.name, **tr,
        }
        self.decisions.append({"step": ctx.step, "turn": ctx.turn, "player": ctx.actor.name,
                               "chosen": tr.get("chosen")})
        if self.emit_decisions:
            self.emit(decision_rec)
        if self.verbose:
            ch = tr.get("chosen") or {}
            print(f"[{ctx.step:04d}] t{ctx.turn} {ctx.actor.name} DECIDE {ch.get('action_type','?'):<14} "
                  f"{ch.get('card','')} regret={tr.get('regret',0)} folded={tr.get('folded',False)}")

    def on_step(self, ctx, move, events):
        if not self.emit_steps:
            return
        m = ctx.manager
        self.emit({
            "type": "step", "step": ctx.step, "turn": m.turn_count,
            "phase": m.phase.name, "player": ctx.actor.name,
            "action": move["action_type"],
            "payload": move.get("payload") or {"card_uuid": move.get("card_uuid")},
            "events": events,
        })


def run_replay(
    seed: int,
    db,
    p1_leader: Optional[str] = None,
    p2_leader: Optional[str] = None,
    p1_difficulty: str = "hard",
    p2_difficulty: str = "hard",
    max_steps: int = DEFAULT_MAX_STEPS,
    trace_out=None,
    emit_steps: bool = True,
    emit_decisions: bool = True,
    verbose: bool = False,
    stop_after_decisions: Optional[int] = None,
) -> Dict[str, Any]:
    """1 局を決定論再生し、盤面ステップ＋ CPU 思考トレースを JSONL に出力してサマリを返す。

    両者 AI（暴走防止ガード付き）。seed で完全再現。返り値は結果サマリ。
    `stop_after_decisions` を指定すると、その数の意思決定で打ち切る（決着前で `winner=None` のまま
    返す＝テストの有界化用。再現性比較には決着不要なため）。
    """
    decisions: List[Dict[str, Any]] = []   # card_id 基準の決定列（再現比較・回帰用）
    trace_tail: List[Dict[str, Any]] = []

    def emit(line: Dict[str, Any]):
        trace_tail.append(line)
        if len(trace_tail) > 12:
            trace_tail.pop(0)
        if trace_out is not None:
            trace_out.write(json.dumps(line, ensure_ascii=False) + "\n")

    # 席別難易度の AI（trace 採取 ON）。全乱数は global random（決定論契約）。
    mem: Dict[str, Dict[str, Any]] = {"p1": {}, "p2": {}}
    seats = {
        "p1": make_seat(p1_difficulty, kind="ai", mem=mem["p1"], want_trace=True),
        "p2": make_seat(p2_difficulty, kind="ai", mem=mem["p2"], want_trace=True),
    }
    obs = _ReplayObserver(emit, decisions, emit_steps, emit_decisions, verbose)
    result = run_game(seed, db, seats=seats,
                      deck_builder=leader_deck_builder(p1_leader, p2_leader),
                      observers=[obs], max_steps=max_steps,
                      legal_moves="check", invariants="raise",
                      stop_after_decisions=stop_after_decisions, trace_tail=trace_tail)

    return {
        "seed": seed, "winner": result.winner, "steps": result.steps, "turns": result.turns,
        "p1_leader": result.p1_leader, "p2_leader": result.p2_leader,
        "p1_difficulty": p1_difficulty, "p2_difficulty": p2_difficulty,
        "decisions": decisions,
    }


def make_descriptor(res: Dict[str, Any], seed: int) -> Dict[str, Any]:
    """結果サマリから「リプレイ種」を作る（seed＋リーダー＋難易度＝完全再現に十分）。"""
    return {
        "schema": SCHEMA, "seed": seed,
        "p1_leader": res["p1_leader"], "p2_leader": res["p2_leader"],
        "p1_difficulty": res["p1_difficulty"], "p2_difficulty": res["p2_difficulty"],
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="決定論的 CPU 対局リプレイ＋思考トレース")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--p1-leader", default=None)
    ap.add_argument("--p2-leader", default=None)
    ap.add_argument("--difficulty", choices=["hard"], default="hard",
                    help="両者の既定難易度（--p1-difficulty/--p2-difficulty で個別上書き）")
    ap.add_argument("--p1-difficulty", choices=["hard"], default=None)
    ap.add_argument("--p2-difficulty", choices=["hard"], default=None)
    ap.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    ap.add_argument("--out", default=None, help="トレース JSONL の出力先（'-' で stdout）")
    ap.add_argument("--record", default=None, help="リプレイ種（descriptor JSON）の出力先")
    ap.add_argument("--descriptor", default=None,
                    help="リプレイ種から再生（seed/leader/difficulty は種の値を使う）")
    ap.add_argument("--decisions-only", action="store_true", help="盤面ステップを省き思考トレースのみ")
    ap.add_argument("--steps-only", action="store_true", help="思考トレースを省き盤面ステップのみ")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    seed = args.seed
    p1_leader, p2_leader = args.p1_leader, args.p2_leader
    p1_diff = args.p1_difficulty or args.difficulty
    p2_diff = args.p2_difficulty or args.difficulty
    if args.descriptor:
        with open(args.descriptor, "r", encoding="utf-8") as f:
            d = json.load(f)
        seed = d["seed"]
        p1_leader, p2_leader = d.get("p1_leader"), d.get("p2_leader")
        p1_diff, p2_diff = d.get("p1_difficulty", "hard"), d.get("p2_difficulty", "hard")

    db = _load_db()
    trace_out, should_close = _open_out(args.out)
    try:
        res = run_replay(
            seed, db, p1_leader, p2_leader, p1_diff, p2_diff,
            max_steps=args.max_steps, trace_out=trace_out,
            emit_steps=not args.decisions_only, emit_decisions=not args.steps_only,
            verbose=args.verbose,
        )
    except InvariantError as e:
        print(f"replay seed={seed}: FAILED step={e.step} violations={e.violations}")
        for line in e.trace_tail:
            print(f"    {json.dumps(line, ensure_ascii=False)[:200]}")
        return 1
    finally:
        if should_close and trace_out:
            trace_out.close()

    print(f"replay seed={seed}: winner={res['winner']} steps={res['steps']} turns={res['turns']} "
          f"({res['p1_leader']}[{p1_diff}] vs {res['p2_leader']}[{p2_diff}]) "
          f"decisions={len(res['decisions'])}")

    if args.record:
        with open(args.record, "w", encoding="utf-8") as f:
            json.dump(make_descriptor(res, seed), f, ensure_ascii=False, indent=2)
        print(f"  wrote replay seed -> {args.record}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
