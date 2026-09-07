"""原始操作のオラクル照合（`docs/rust_engine_plan.md` §3 P1 の受け入れ (b)・WP `rs-p1-journal`）。

**問い**: Rust の原始操作（`rust/opcg_engine/src/ops.rs`）は Python の原始操作
（`move_card` / `draw_card` / `pay_cost` / `_return_one_don` / ドン!!付与 / `reset_turn_status` /
ライフ→手札 / デッキ→ライフ / レスト切替 / `record_turn_event`）と**同じ盤面**を作るか。

やること:
  1. `rs_diff_replay.Recorder` で局を打ち、各行の `hidden`（記録 v2 = 完全な内部状態）を得る。
  2. 各行の `hidden` から Python 側の `GameManager` を復元し（`tests/harness/rs_record.py`）、
     **復元直後の盤面 dict が記録の `state` と一致する**ことを先に検査する
     （＝Python 側だけで閉じる自己検査。Rust が無くても回る）。
  3. その盤面へ**ランダムに合法な原始操作の台本**（§9.5 の ops_json）を K 件流し、各操作後の
     盤面 dict を集める。
  4. 同じ `hidden` と台本を Rust の `opcg_engine.apply_ops` へ渡し、各操作後の盤面を照合する
     （Rust 側は各 op の前に transaction+rollback を 1 回挟み、journal が bit 一致で戻ることも見る）。
  5. `RS_OPS {...}` の 1 行を stdout に出す。

`opcg_engine` に `apply_ops` が無い／`GameState::from_record` が未実装（WP `rs-p1-model` 待ち）の
間は **unimplemented** として集計する（黙って緑にしない）。自己検査（2.）はその間も走る。

実行例:
    OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/rs_ops_oracle.py \\
      --games 100 --ops-per-state 20 --seed-base 700000
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import traceback  # noqa: E402

import os as _os, sys as _sys  # noqa: E402  test bootstrap (sys.path + google スタブ)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: E402,F401

from harness.game_driver import DEFAULT_MAX_STEPS, InvariantError, load_db, make_seat, run_game  # noqa: E402
from harness.rs_record import apply_python_op, manager_from_hidden  # noqa: E402
from rs_diff_replay import Recorder, board_dict, canon, first_diff  # noqa: E402

try:                        # Rust 拡張は未導入でも動く（その場合は全行 unimplemented）。
    import opcg_engine
except ImportError:         # pragma: no cover - 実行環境依存
    opcg_engine = None

DEFAULT_EFFECTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))),
    "opcg_sim", "data", "opcg_effects.json",
)

_TURN_EVENTS = ("DON_RETURNED", "CHAR_LEFT_BY_OWN_EFFECT", "NAVY_DISCARD", "TRIGGER_CHAR_PLAYED")
_DEST_ZONES = ("FIELD", "HAND", "TRASH", "LIFE", "DECK", "TEMP")
_CARD_ZONES = ("hand", "field", "life", "trash", "deck", "temp_zone")


def _norm(value) -> str:
    return json.dumps(canon(value), sort_keys=True, ensure_ascii=False, default=str)


def _public(state: dict) -> dict:
    """照合対象の盤面（`pending_request` は P2 の責務なので外す）。"""
    out = dict(state)
    out.pop("pending_request", None)
    return out


# --- ランダムな合法台本 --------------------------------------------------------

def _all_cards(manager) -> list:
    cards = []
    for p in (manager.p1, manager.p2):
        if p.leader:
            cards.append(p.leader)
        if p.stage:
            cards.append(p.stage)
        for zone in _CARD_ZONES:
            cards.extend(getattr(p, zone))
    return cards


def _units(manager) -> list:
    """ドン!!を付与できる先（場のキャラ・リーダー・ステージ）。"""
    units = []
    for p in (manager.p1, manager.p2):
        if p.leader:
            units.append(p.leader)
        if p.stage:
            units.append(p.stage)
        units.extend(p.field)
    return units


def _pick_op(manager, rng: random.Random):
    """今の盤面で**適用できる**原始操作を 1 件作る（作れなければ None）。

    「適用できる」＝Python 側が例外を投げない・P1 の責務外へ踏み込まない、の 2 点:
      - `pay_cost` は枚数不足で ValueError を投げるので必ず足りる形にする。
      - `draw` はデッキを空にしない（空にすると `check_victory` が走る＝敗北判定は P2 の責務）。
    """
    kind = rng.choice([
        "move_card", "move_card", "draw", "pay_cost", "return_don", "attach_don",
        "reset_turn_status", "life_to_hand", "deck_to_life", "set_rest", "record_turn_event",
    ])
    p1, p2 = manager.p1, manager.p2
    if kind == "move_card":
        cards = _all_cards(manager)
        if not cards:
            return None
        return {"op": "move_card", "card": rng.choice(cards).uuid,
                "to": rng.choice(_DEST_ZONES), "player": rng.choice(["p1", "p2"]),
                "pos": rng.choice(["TOP", "BOTTOM"])}
    if kind == "draw":
        cand = [p for p in (p1, p2) if len(p.deck) >= 2]
        if not cand:
            return None
        p = rng.choice(cand)
        return {"op": "draw", "player": p.name, "n": rng.randint(1, min(2, len(p.deck) - 1))}
    if kind == "pay_cost":
        cand = [p for p in (p1, p2) if p.don_active or p.don_attached_cards]
        if not cand:
            return None
        p = rng.choice(cand)
        if p.don_active and rng.random() < 0.5:
            return {"op": "pay_cost", "player": p.name, "cost": rng.randint(1, len(p.don_active))}
        pool = list(p.don_active) + list(p.don_attached_cards)
        picked = rng.sample(pool, rng.randint(1, len(pool)))
        return {"op": "pay_cost", "player": p.name, "cost": rng.randint(0, len(picked)),
                "dons": [d.uuid for d in picked]}
    if kind == "return_don":
        cand = [p for p in (p1, p2) if p.don_active or p.don_rested or p.don_attached_cards]
        if not cand:
            return None
        p = rng.choice(cand)
        pool = list(p.don_active) + list(p.don_rested) + list(p.don_attached_cards)
        return {"op": "return_don", "player": p.name, "don": rng.choice(pool).uuid}
    if kind == "attach_don":
        units = _units(manager)
        cand = [p for p in (p1, p2) if p.don_active or p.don_rested]
        if not units or not cand:
            return None
        p = rng.choice(cand)
        from_rested = bool(p.don_rested) and rng.random() < 0.3
        return {"op": "attach_don", "player": p.name, "card": rng.choice(units).uuid,
                "from_rested": from_rested}
    if kind == "reset_turn_status":
        cards = _all_cards(manager)
        if not cards:
            return None
        return {"op": "reset_turn_status", "card": rng.choice(cards).uuid,
                "keep_don": rng.random() < 0.5, "clear_usage": rng.random() < 0.5}
    if kind == "life_to_hand":
        cand = [p for p in (p1, p2) if p.life]
        if not cand:
            return None
        return {"op": "life_to_hand", "player": rng.choice(cand).name,
                "from": rng.choice(["TOP", "BOTTOM"])}
    if kind == "deck_to_life":
        cand = [p for p in (p1, p2) if p.deck]
        if not cand:
            return None
        return {"op": "deck_to_life", "player": rng.choice(cand).name}
    if kind == "set_rest":
        cards = _all_cards(manager)
        if not cards:
            return None
        return {"op": "set_rest", "card": rng.choice(cards).uuid, "value": rng.random() < 0.5}
    return {"op": "record_turn_event", "name": rng.choice(_TURN_EVENTS), "n": rng.randint(1, 2)}


# --- 1 行（1 盤面）の検査 ------------------------------------------------------

def check_row(db, row: dict, rng: random.Random, ops_per_state: int, effects: str, totals: dict):
    """1 行の hidden について: 復元の自己検査 → 台本の作成/適用 → Rust との照合。

    戻り値は最初の異常 dict（無ければ None）。`totals` を更新する。
    """
    manager = manager_from_hidden(db, row["hidden"])
    totals["rows"] += 1

    # (a) 復元の自己検査（Python 側だけで閉じる）。
    expected = _public(row["state"])
    restored = _public(board_dict(manager))
    if _norm(expected) != _norm(restored):
        totals["restore_mismatch"] += 1
        return {"status": "restore_mismatch",
                "path": first_diff(canon(expected), canon(restored))}

    # (b) ランダムな合法台本を作りながら Python 側で適用する。
    script, states = [], []
    for _ in range(ops_per_state):
        op = _pick_op(manager, rng)
        if op is None:
            continue
        apply_python_op(manager, op)
        script.append(op)
        states.append(_public(board_dict(manager)))
    totals["ops"] += len(script)
    if not script:
        return None

    # (c) Rust と照合する。1 度 NotImplementedError が出たら以後は呼ばない
    #     （`apply_ops` は毎回 7.9MB の効果 JSON を読み直すので、1 万行ぶん叩くと桁違いに遅い）。
    if opcg_engine is None or not hasattr(opcg_engine, "apply_ops") or totals["unimplemented"]:
        totals["unimplemented"] += 1
        return None
    try:
        out = opcg_engine.apply_ops(
            json.dumps(row["hidden"], ensure_ascii=False, default=str),
            json.dumps(script, ensure_ascii=False, default=str),
            effects,
        )
    except NotImplementedError as e:
        totals["unimplemented"] += 1
        return {"status": "unimplemented", "detail": str(e)} if totals["unimplemented"] == 1 else None
    except ValueError as e:
        totals["bad_payload"] += 1
        return {"status": "bad_payload", "detail": str(e)}
    try:
        got = json.loads(out).get("states")
    except (TypeError, ValueError) as e:
        totals["bad_output"] += 1
        return {"status": "bad_output", "detail": f"apply_ops returned non-JSON: {e}"}
    if not isinstance(got, list) or len(got) != len(states):
        totals["mismatch"] += 1
        return {"status": "mismatch", "path": f"states[len {len(states)}!={len(got) if isinstance(got, list) else None}]"}
    for i, (exp, act) in enumerate(zip(states, got)):
        act = _public(act)
        if _norm(exp) != _norm(act):
            totals["mismatch"] += 1
            return {"status": "mismatch", "op": i, "op_json": script[i],
                    "path": first_diff(canon(exp), canon(act))}
    totals["match"] += 1
    return None


def run_one(seed: int, db, args, totals: dict):
    """1 局: 打って記録 → 各行を検査。戻り値は最初の異常（無ければ None）。"""
    seats = {"p1": make_seat(kind="random"), "p2": make_seat(kind="random")}
    rec = Recorder(seed, "random", record_hidden=True)
    try:
        run_game(seed, db, seats=seats, observers=(rec,), max_steps=args.max_steps)
    except InvariantError as e:
        totals["python_error"] += 1
        return {"status": "python_error", "detail": f"InvariantError: {e.violations[:1]}"}
    except Exception as e:  # noqa: BLE001 - ハーネスは落とさず集計に載せる
        totals["python_error"] += 1
        return {"status": "python_error", "detail": f"{type(e).__name__}: {e}"}

    rng = random.Random(args.op_seed + seed)
    first = None
    for i, row in enumerate([rec.setup] + rec.steps):
        try:
            bad = check_row(db, row, rng, args.ops_per_state, args.effects, totals)
        except Exception as e:  # noqa: BLE001
            totals["python_error"] += 1
            bad = {"status": "python_error",
                   "detail": f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}"}
        if bad is not None:
            first = _keep_first(first, dict(bad, seed=seed, row=i))
    return first


def _keep_first(current, candidate):
    """報告する「最初の異常」を選ぶ。`unimplemented`（Rust 未実装＝待ち）より本物の異常を優先する。"""
    if current is None:
        return candidate
    if current["status"] == "unimplemented" and candidate["status"] != "unimplemented":
        return candidate
    return current


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Python の原始操作と Rust の ops.rs を照合する（P1）")
    ap.add_argument("--games", type=int, default=5, help="局数（既定 5）")
    ap.add_argument("--ops-per-state", type=int, default=20, help="1 盤面あたりの操作数（既定 20）")
    ap.add_argument("--seed-base", type=int, default=700000, help="seed の基点（既定 700000）")
    ap.add_argument("--op-seed", type=int, default=1234, help="台本の乱数 seed（局 seed に足す）")
    ap.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS, help="1 局の上限ステップ")
    ap.add_argument("--effects", default=DEFAULT_EFFECTS,
                    help="効果 JSON（Rust のマスター表。既定 opcg_sim/data/opcg_effects.json）")
    ap.add_argument("--verbose", action="store_true", help="局ごとの判定を出す")
    args = ap.parse_args(argv)

    effects = args.effects if args.effects and os.path.exists(args.effects) else None
    args.effects = effects
    db = load_db()
    totals = {"games": 0, "rows": 0, "ops": 0, "match": 0, "mismatch": 0,
              "restore_mismatch": 0, "unimplemented": 0, "python_error": 0,
              "bad_payload": 0, "bad_output": 0}
    first = None

    for i in range(args.games):
        seed = args.seed_base + i
        bad = run_one(seed, db, args, totals)
        totals["games"] += 1
        if bad is not None:
            first = _keep_first(first, bad)
        if args.verbose:
            print(f"[game {i}] seed={seed} rows={totals['rows']} ops={totals['ops']}"
                  f" mismatch={totals['mismatch']} restore_mismatch={totals['restore_mismatch']}")

    summary = dict(
        totals,
        ops_per_state=args.ops_per_state,
        seed_base=args.seed_base,
        effects=bool(effects),
        engine=(opcg_engine.version() if opcg_engine is not None else None),
        first=first,
    )
    print("RS_OPS " + json.dumps(summary, ensure_ascii=False, default=str))
    return 1 if (totals["mismatch"] or totals["restore_mismatch"] or totals["bad_payload"]
                 or totals["bad_output"] or totals["python_error"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
