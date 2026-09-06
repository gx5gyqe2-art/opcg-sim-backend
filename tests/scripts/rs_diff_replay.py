"""Rust エンジンの差分照合ハーネス（`docs/rust_engine_plan.md` §4・全段で使う受け入れ道具）。

**問い**: Rust 版は Python 版（＝正本／オラクル）と**同じ入力に同じ出力**を返すか。

やること:
  1. Python エンジンで `--games` 局を打ち（`--policy random|l1`）、
     seed・初期盤面（デッキ/ライフ/手札の並びを含む）・行動列・**各行動後の盤面 dict** を記録する。
  2. 同じ記録を Rust 側 `opcg_engine.replay(json)` へ渡し、返ってきた各行動後の盤面を照合する。
  3. `RS_DIFF {...}` の1行を stdout に出す（games / actions / mismatch / unimplemented / first）。

P0（本段階）では Rust 側に盤面もルールも無いため、`replay` は契約検査だけを行って
`NotImplementedError` を返す。本ハーネスはそれを **unimplemented** として集計する
（「一致」とも「不一致」とも数えない＝黙って緑にしない）。P1 以降で Rust が実装されるに従って
unimplemented が減り、mismatch が 0 のまま maintained されることが各 WP の受け入れ条件になる。

実行例:
    OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/rs_diff_replay.py --games 5 --policy random
    OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/rs_diff_replay.py \\
      --games 20 --policy l1 --seed-base 500000 --dump /tmp/rec.json

照合の規約（計画 §4）:
  - 盤面の比較は **キー順に依存しない正規化 JSON**（`json.dumps(..., sort_keys=True)`）で行う。
  - 記録した局の再生では乱数を使わない。初期のデッキ/ライフ/手札の**並び**を記録して渡す。
  - 盤面 dict は API と同じ形（`presenters.build_game_result_hybrid` の `game_state` 相当＝
    `turn_info` / `players`（`Player.to_dict`）/ `active_battle`）＋ `pending_request`。
    Python 側のコードは一切変更せず、この読み取りだけで組み立てる（P0 は追加のみ）。

既知の穴（P1 で決める・記録形式 version を上げる）:
  - **対局中のシャッフル**（マリガン・デッキシャッフル効果）は再生側で再現できない。
    `--hidden` を付けると各行動後の隠しゾーン（デッキ/ライフの並び）も記録するので、
    「記録した並びを与える」方式でこれを塞げる。既定 off（記録が数百 KB/局 増えるため）。
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import traceback

import os as _os, sys as _sys  # noqa: E402  test bootstrap (sys.path + google スタブ)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: E402,F401

from harness.game_driver import DEFAULT_MAX_STEPS, InvariantError, load_db, make_seat, run_game  # noqa: E402

try:                        # Rust 拡張は未導入でも動く（その場合は全局 unimplemented）。
    import opcg_engine
except ImportError:         # pragma: no cover - 実行環境依存
    opcg_engine = None

# 記録ペイロードの形式バージョン。Rust 側 `state::RECORD_VERSION` と一致させること
# （形を非互換に変えたら両方 +1 する）。
RECORD_VERSION = 1


# --- 盤面スナップショット -----------------------------------------------------

def board_dict(manager) -> dict:
    """API と同じ形の盤面 dict（`presenters.build_game_result_hybrid` の `game_state` 相当）。

    Python 側を変更しない方針（P0 は追加のみ）なので、ここで同じ形を組み立てる。
    `GameManager` に `to_dict` が生えたらこの関数はそれに置き換える。
    """
    active_battle = None
    if manager.active_battle:
        active_battle = {
            "attacker_uuid": manager.active_battle["attacker"].uuid,
            "target_uuid": manager.active_battle["target"].uuid,
            "counter_buff": manager.active_battle.get("counter_buff", 0),
        }
    return {
        "turn_info": {
            "turn_count": manager.turn_count,
            "current_phase": manager.phase.name,
            "active_player_id": "p1" if manager.turn_player is manager.p1 else "p2",
            "winner": manager.winner,
        },
        "players": {
            "p1": manager.p1.to_dict(is_my_turn=(manager.turn_player is manager.p1)),
            "p2": manager.p2.to_dict(is_my_turn=(manager.turn_player is manager.p2)),
        },
        "active_battle": active_battle,
        "pending_request": manager.get_pending_request(),
    }


def _zone_order(cards) -> list:
    """隠しゾーン（デッキ/ライフ等）の並びを card_id + uuid で記録する。"""
    return [{"card_id": c.master.card_id, "uuid": c.uuid, "is_face_up": c.is_face_up} for c in cards]


def hidden_dict(manager) -> dict:
    """再生に必要な非公開情報（各ゾーンの並び）。to_dict では見えない部分。"""
    out = {}
    for p in (manager.p1, manager.p2):
        out[p.name] = {
            "leader": ({"card_id": p.leader.master.card_id, "uuid": p.leader.uuid}
                       if p.leader else None),
            "deck": _zone_order(p.deck),
            "hand": _zone_order(p.hand),
            "life": _zone_order(p.life),
            "field": _zone_order(p.field),
            "trash": _zone_order(p.trash),
            "don": {"deck": len(p.don_deck), "active": len(p.don_active),
                    "rested": len(p.don_rested), "attached": len(p.don_attached_cards)},
        }
    return out


# --- 記録 observer ------------------------------------------------------------

class Recorder:
    """1 局分の（初期盤面・行動列・各行動後の盤面）を集める observer。

    `game_driver.run_game` の呼び出し規約: on_start（start_game 直後）→
    on_decision（apply 前・move 確定）→ on_step（apply 後）→ … → on_end。
    観測専用で manager は一切変更しない（決定論契約）。
    """

    def __init__(self, seed: int, policy: str, record_hidden: bool = False):
        self.seed = seed
        self.policy = policy
        self.record_hidden = record_hidden
        self.setup = None
        self.steps = []
        self._pending_move = None
        self.result = None

    def on_start(self, ctx):
        m = ctx.manager
        self.setup = {
            "first_player": "p1" if m.turn_player is m.p1 else "p2",
            "hidden": hidden_dict(m),
            "state": board_dict(m),
        }

    def on_decision(self, ctx, move):
        self._pending_move = {
            "index": len(self.steps),
            "actor": ctx.actor.name,
            "phase": ctx.phase,
            "turn": ctx.turn,
            "move": move,
        }

    def on_step(self, ctx, move, events):
        step = self._pending_move or {"index": len(self.steps), "actor": ctx.actor.name, "move": move}
        step["state"] = board_dict(ctx.manager)
        if self.record_hidden:
            step["hidden"] = hidden_dict(ctx.manager)
        self.steps.append(step)
        self._pending_move = None

    def on_end(self, ctx, result):
        self.result = {"winner": result.winner, "steps": result.steps, "turns": result.turns,
                       "p1_leader": result.p1_leader, "p2_leader": result.p2_leader}

    def payload(self) -> dict:
        """Rust `replay()` へ渡す JSON dict（形は `docs/rust_engine_plan.md` §4）。"""
        return {
            "version": RECORD_VERSION,
            "seed": self.seed,
            "policy": self.policy,
            "setup": self.setup,
            "steps": self.steps,
            "result": self.result,
        }


# --- 照合 ---------------------------------------------------------------------

def _norm(value) -> str:
    """キー順に依存しない正規化 JSON（比較の正本）。"""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def first_diff(expected, actual, path="") -> str:
    """最初に食い違う場所を `players.p1.zones.field[0].power` のようなパスで返す（無ければ ""）。"""
    if type(expected) is not type(actual) and not (
            isinstance(expected, (int, float)) and isinstance(actual, (int, float))):
        return path or "<root>"
    if isinstance(expected, dict):
        for key in sorted(set(expected) | set(actual)):
            if key not in expected or key not in actual:
                return f"{path}.{key}" if path else str(key)
            sub = first_diff(expected[key], actual[key], f"{path}.{key}" if path else str(key))
            if sub:
                return sub
        return ""
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return f"{path}[len {len(expected)}!={len(actual)}]"
        for i, (e, a) in enumerate(zip(expected, actual)):
            sub = first_diff(e, a, f"{path}[{i}]")
            if sub:
                return sub
        return ""
    return "" if expected == actual else (path or "<root>")


def compare(record: dict, replayed: dict) -> dict:
    """Rust の replay 出力（`{"version":..,"states":[...]}`）を記録と突き合わせる。"""
    states = replayed.get("states")
    if not isinstance(states, list):
        return {"status": "bad_output", "detail": "replay result has no 'states' list"}
    expected = [s["state"] for s in record["steps"]]
    if len(states) != len(expected):
        return {"status": "mismatch",
                "action": min(len(states), len(expected)),
                "path": f"states[len {len(expected)}!={len(states)}]"}
    for i, (exp, got) in enumerate(zip(expected, states)):
        if _norm(exp) != _norm(got):
            return {"status": "mismatch", "action": i, "path": first_diff(exp, got)}
    return {"status": "match"}


# --- 1 局 ---------------------------------------------------------------------

def run_one(seed: int, db, policy: str, max_steps: int, record_hidden: bool):
    """1 局を Python で打って記録し、Rust で再生して照合する。戻り値: (record, verdict)。"""
    kind = "random" if policy == "random" else "ai"
    seats = {"p1": make_seat(kind=kind), "p2": make_seat(kind=kind)}
    rec = Recorder(seed, policy, record_hidden)
    try:
        run_game(seed, db, seats=seats, observers=(rec,), max_steps=max_steps)
    except InvariantError as e:
        return rec, {"status": "python_error", "detail": f"InvariantError: {e.violations[:1]}"}
    except Exception as e:  # noqa: BLE001 - ハーネスは落とさず集計に載せる
        return rec, {"status": "python_error", "detail": f"{type(e).__name__}: {e}"}

    record = rec.payload()
    if opcg_engine is None:
        return rec, {"status": "unimplemented", "detail": "opcg_engine is not installed"}
    try:
        out = opcg_engine.replay(json.dumps(record, ensure_ascii=False, default=str))
    except NotImplementedError as e:
        return rec, {"status": "unimplemented", "detail": str(e)}
    except ValueError as e:      # 契約違反（記録側のバグ）＝黙って通さない
        return rec, {"status": "bad_payload", "detail": str(e)}
    try:
        replayed = json.loads(out)
    except (TypeError, ValueError) as e:
        return rec, {"status": "bad_output", "detail": f"replay returned non-JSON: {e}"}
    return rec, compare(record, replayed)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Python エンジンの局を Rust で再生して照合する（P0 骨組み）")
    ap.add_argument("--games", type=int, default=5, help="局数（既定 5）")
    ap.add_argument("--seed-base", type=int, default=500000, help="seed の基点（既定 500000）")
    ap.add_argument("--policy", choices=["random", "l1"], default="random",
                    help="両席の方策: random=ランダム合法手 / l1=古典CPU（既定 random）")
    ap.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS, help="1局の上限ステップ")
    ap.add_argument("--hidden", action="store_true",
                    help="各行動後の隠しゾーン（デッキ/ライフの並び）も記録する")
    ap.add_argument("--dump", default=None, help="1局目の記録をこのパスへ書き出す（Rust 側の開発用）")
    ap.add_argument("--verbose", action="store_true", help="局ごとの判定を出す")
    args = ap.parse_args(argv)

    db = load_db()
    totals = {"games": 0, "actions": 0, "match": 0, "mismatch": 0, "unimplemented": 0,
              "python_error": 0, "bad_payload": 0, "bad_output": 0}
    first = None

    for i in range(args.games):
        seed = args.seed_base + i
        try:
            rec, verdict = run_one(seed, db, args.policy, args.max_steps, args.hidden)
        except Exception as e:  # noqa: BLE001 - ハーネス自身の事故も集計に載せる
            rec, verdict = None, {"status": "python_error",
                                  "detail": f"{type(e).__name__}: {e}\n{traceback.format_exc()}"}
        totals["games"] += 1
        if rec is not None:
            totals["actions"] += len(rec.steps)
        status = verdict["status"]
        totals[status] = totals.get(status, 0) + 1
        if status in ("mismatch", "bad_payload", "bad_output", "python_error") and first is None:
            first = dict(verdict, seed=seed, game=i)
        if args.verbose:
            print(f"[game {i}] seed={seed} steps={len(rec.steps) if rec else 0} -> {status}"
                  f"{'  ' + verdict.get('detail', '')[:120] if verdict.get('detail') else ''}")
        if args.dump and i == 0 and rec is not None:
            with open(args.dump, "w", encoding="utf-8") as f:
                json.dump(rec.payload(), f, ensure_ascii=False, default=str)
            print(f"[dump] {args.dump} ({os.path.getsize(args.dump)/1e6:.2f} MB)")

    summary = {
        "games": totals["games"],
        "actions": totals["actions"],
        "match": totals["match"],
        "mismatch": totals["mismatch"],
        "unimplemented": totals["unimplemented"],
        "python_error": totals["python_error"],
        "bad_payload": totals["bad_payload"],
        "bad_output": totals["bad_output"],
        "policy": args.policy,
        "seed_base": args.seed_base,
        "engine": (opcg_engine.version() if opcg_engine is not None else None),
        "record_version": RECORD_VERSION,
        "first": first,
    }
    print("RS_DIFF " + json.dumps(summary, ensure_ascii=False))
    # 終了コード: 不一致・契約違反があれば 1（CI は無いが、呼び出し側が機械判定できるように）。
    return 1 if (totals["mismatch"] or totals["bad_payload"] or totals["bad_output"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
