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
      --mode state --games 500 --policy random          # P1-model の受け入れ
    OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/rs_diff_replay.py \\
      --games 20 --policy l1 --seed-base 500000 --dump /tmp/rec.json

`--effects`（既定 `opcg_sim/data/opcg_effects.json`）は Rust 側が読むカード定義（効果構造 JSON）。
生成物・git 管理外なので、無ければ `export_effects_json` を呼んで作り、`opcg_engine.load_masters`
へ渡す（拡張に `load_masters` が無い旧版では何もしない）。

照合の規約（計画 §4）:
  - 盤面の比較は **キー順に依存しない正規化 JSON**（`json.dumps(..., sort_keys=True)`）で行う。
  - 記録した局の再生では乱数を使わない。初期のデッキ/ライフ/手札の**並び**を記録して渡す。
  - 盤面 dict は API と同じ形（`presenters.build_game_result_hybrid` の `game_state` 相当＝
    `turn_info` / `players`（`Player.to_dict`）/ `active_battle`）＋ `pending_request`。
    Python 側のコードは一切変更せず、この読み取りだけで組み立てる（P0 は追加のみ）。

記録形式 v2（P1・`docs/rust_engine_plan.md` §9.1）:
  - `--hidden` で各行動後の**完全な内部状態**（全ゾーンの並び・カード実体の実行時フィールド・
    ドン!!の所在・進行状態）も記録する。対局中のシャッフル（マリガン・シャッフル効果）は
    「記録した並びを与える」方式で塞ぐ（乱数列は Rust へ流さない）。
  - `--mode state`（P1-model の受け入れ）: 各行の `hidden` から Rust が `GameState` を組み立て、
    `state_roundtrip(hidden_json)` の盤面 dict が同じ行の `state` と一致するか（`pending_request` は
    P2 の責務なので除外）。`--mode replay`（既定）: 行動列の再生（P2 以降）。
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import subprocess
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
RECORD_VERSION = 2

# 効果構造 JSON（`opcg_sim/tools/export_effects_json.py` の生成物・git 管理外・約 8MB）。
# Rust 側は起動時にこれを 1 度だけ読んで `CardMaster` 表を作る（`opcg_engine.load_masters`）。
_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DEFAULT_EFFECTS_PATH = _os.path.join(_REPO_ROOT, "opcg_sim", "data", "opcg_effects.json")


def ensure_effects_json(path: str) -> str:
    """効果構造 JSON を用意する（無ければ exporter を呼んで生成する）。"""
    if os.path.exists(path):
        return path
    print(f"[effects] {path} が無いので生成する: python -m opcg_sim.tools.export_effects_json")
    subprocess.run([_sys.executable, "-m", "opcg_sim.tools.export_effects_json", "--out", path],
                   cwd=_REPO_ROOT, check=True, stdout=subprocess.DEVNULL)
    return path


def load_masters(path: str) -> None:
    """Rust 側へカード定義を読み込ませる（`load_masters` が無い版の拡張では何もしない）。"""
    if opcg_engine is None or not hasattr(opcg_engine, "load_masters"):
        return
    n = opcg_engine.load_masters(ensure_effects_json(path))
    print(f"[effects] load_masters({path}) -> {n} cards")


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


# 記録形式 v2（P1 の契約・`docs/rust_engine_plan.md` §9.1）: `hidden` は「盤面を完全に再構成できる」
# 内部状態＝カード実体の実行時フィールド全部・ドン!!の所在と付与先・マネージャの進行状態を持つ。
# Rust 側はこれだけから `GameState` を組み立て、`to_dict` 相当が同じ行の `state` と一致することで
# P1-model を受け入れる（`--mode state`）。順序（並び）は全ゾーンで記録順＝Python の list 順。

_CARD_RUNTIME_FIELDS = (
    "is_rest", "is_newly_played", "attached_don", "is_face_up", "power_buff", "cost_buff",
    "passive_power", "passive_power_override", "passive_counter", "base_power_override",
    "base_cost_override", "negated", "ability_disabled", "timed_power", "timed_cost",
)
_CARD_SET_FIELDS = ("current_keywords", "flags", "timed_flags", "timed_keywords")


def card_record(c) -> dict:
    """カード実体 1 枚の完全な記録（マスター参照＝card_id＋実行時フィールド全部）。"""
    d = {"card_id": c.master.card_id, "uuid": c.uuid, "owner_id": c.owner_id}
    for f in _CARD_RUNTIME_FIELDS:
        d[f] = getattr(c, f)
    for f in _CARD_SET_FIELDS:
        d[f] = sorted(getattr(c, f))
    d["ability_used_this_turn"] = {str(k): v for k, v in sorted(c.ability_used_this_turn.items())}
    return d


def _zone_order(cards) -> list:
    return [card_record(c) for c in cards]


def don_record(d) -> dict:
    return {"uuid": d.uuid, "owner_id": d.owner_id, "is_rest": d.is_rest,
            "attached_to": d.attached_to, "is_frozen": d.is_frozen}


def hidden_dict(manager) -> dict:
    """再生に必要な非公開情報＝盤面の完全な内部状態（形式 v2）。to_dict では見えない部分を含む。"""
    out = {"players": {}}
    for p in (manager.p1, manager.p2):
        out["players"][p.name] = {
            "name": p.name,
            "leader": card_record(p.leader) if p.leader else None,
            "stage": card_record(p.stage) if p.stage else None,
            "deck": _zone_order(p.deck),
            "hand": _zone_order(p.hand),
            "life": _zone_order(p.life),
            "field": _zone_order(p.field),
            "trash": _zone_order(p.trash),
            "temp_zone": _zone_order(p.temp_zone),
            "don": {"deck": [don_record(d) for d in p.don_deck],
                    "active": [don_record(d) for d in p.don_active],
                    "rested": [don_record(d) for d in p.don_rested],
                    "attached": [don_record(d) for d in p.don_attached_cards]},
            "negate_onplay_until": p.negate_onplay_until,
            "restrictions": {k: dict(v) for k, v in p.restrictions.items()},
        }
    ab = manager.active_battle
    out["manager"] = {
        "turn_count": manager.turn_count,
        "phase": manager.phase.name,
        "turn_player": manager.turn_player.name,
        "winner": manager.winner,
        "active_battle": ({"attacker": ab["attacker"].uuid, "target": ab["target"].uuid,
                           "counter_buff": ab.get("counter_buff", 0)} if ab else None),
        "turn_events": dict(manager._turn_events),
        "mulligan_done": sorted(manager.mulligan_done),
        "setup_phase_pending": manager.setup_phase_pending,
        "turn_start_pending": manager.turn_start_pending,
        # 対話スタック・遅延継続・誘発待ち行列は P2/P3 の契約で足す（P1 では件数だけ持ち、
        # 0 でない行は state モードの照合対象から外す＝`interaction_depth`）。
        "interaction_depth": len(manager._interaction_stack),
        "pending_triggers": len(manager._pending_triggers),
        "pending_end_of_turn": len(manager.pending_end_of_turn),
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

# 順序を持たない list 欄（Python 側が set から作る＝プロセス毎の hash 乱択で並びが変わる）。
# 照合前にソートして正規化する。Rust 側はソート済みで出せばよい。
_UNORDERED_LIST_KEYS = frozenset({"keywords"})


def canon(value, key=None):
    """順序を持たない list 欄をソートした複製を返す（比較の前処理）。"""
    if isinstance(value, dict):
        return {k: canon(v, k) for k, v in value.items()}
    if isinstance(value, list):
        items = [canon(v) for v in value]
        if key in _UNORDERED_LIST_KEYS:
            items = sorted(items, key=lambda x: json.dumps(x, sort_keys=True, ensure_ascii=False, default=str))
        return items
    return value


def _norm(value) -> str:
    """キー順に依存しない正規化 JSON（比較の正本）。順序を持たない list 欄はソートする。"""
    return json.dumps(canon(value), sort_keys=True, ensure_ascii=False, default=str)


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
            return {"status": "mismatch", "action": i, "path": first_diff(canon(exp), canon(got))}
    return {"status": "match"}


def compare_states(record: dict) -> dict:
    """`--mode state`: 各行の hidden → Rust の `state_roundtrip` → 盤面 dict が記録の `state` と一致するか。"""
    rows = [record["setup"]] + record["steps"]
    for i, row in enumerate(rows):
        hidden = row.get("hidden")
        if hidden is None:
            return {"status": "bad_payload", "detail": "--mode state requires --hidden"}
        try:
            out = opcg_engine.state_roundtrip(json.dumps(hidden, ensure_ascii=False, default=str))
        except NotImplementedError as e:
            return {"status": "unimplemented", "detail": str(e)}
        except ValueError as e:
            return {"status": "bad_payload", "detail": f"row {i}: {e}"}
        try:
            got = json.loads(out)
        except (TypeError, ValueError) as e:
            return {"status": "bad_output", "detail": f"state_roundtrip returned non-JSON: {e}"}
        exp = dict(row["state"])
        exp.pop("pending_request", None)
        got.pop("pending_request", None)
        if _norm(exp) != _norm(got):
            return {"status": "mismatch", "action": i - 1, "path": first_diff(canon(exp), canon(got))}
    return {"status": "match"}


# --- 1 局 ---------------------------------------------------------------------

def run_one(seed: int, db, policy: str, max_steps: int, record_hidden: bool, mode: str = "replay"):
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
    if mode == "state":
        return rec, compare_states(record)
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
                    help="各行動後の完全な内部状態（形式 v2 の hidden）も記録する（--mode state は必須）")
    ap.add_argument("--mode", choices=["replay", "state"], default="replay",
                    help="replay=行動列の再生を照合（既定）／state=hidden→GameState→盤面 dict の往復を照合（P1-model）")
    ap.add_argument("--effects", default=DEFAULT_EFFECTS_PATH,
                    help="効果構造 JSON（Rust の CardMaster 表・既定 opcg_sim/data/opcg_effects.json）。"
                         "無ければ export_effects_json で生成する")
    ap.add_argument("--dump", default=None, help="1局目の記録をこのパスへ書き出す（Rust 側の開発用）")
    ap.add_argument("--verbose", action="store_true", help="局ごとの判定を出す")
    args = ap.parse_args(argv)

    load_masters(args.effects)
    db = load_db()
    totals = {"games": 0, "actions": 0, "match": 0, "mismatch": 0, "unimplemented": 0,
              "python_error": 0, "bad_payload": 0, "bad_output": 0}
    first = None

    for i in range(args.games):
        seed = args.seed_base + i
        try:
            if args.mode == "state" and not args.hidden:
                args.hidden = True
            rec, verdict = run_one(seed, db, args.policy, args.max_steps, args.hidden, args.mode)
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
        "mode": args.mode,
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
