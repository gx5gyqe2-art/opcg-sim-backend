"""問合せオラクル（`docs/rust_engine_plan.md` §11.1・WP `rs-p3-core` の受け入れ道具）。

**問い**: Rust の「対象・条件・値」（`effects/{matcher,cond,value}.rs`）は Python 版
（`matcher.get_target_cards`／`EffectResolver._check_condition`／`_calculate_value`）と
**同じ盤面・同じ発生源で同じ答え**を返すか。

やること:
  1. `rs_diff_replay.Recorder` で局を打ち（random／l1）、各行の `hidden`（記録 v3＝完全な内部状態）を
     集めて **等間隔に `--boards` 局面**を抜く。
  2. カード DB から **全ての `TargetQuery`／`Condition`／`ValueSource`** を、効果木の中の位置
     （`path`＝`effect.actions[1].target` のようなフィールド名の連結）付きで列挙する。
  3. 各局面で、発生源カードを **p1 の場（リーダー／ステージ含む）・手札の各カード**へ置き換えて
     Python 側で評価する（`effect_context` は空）。局面と問合せの組で発生源を巡回させるので、
     `--boards` 枚の局面を通して各問合せが多数の発生源で評価される。
  4. 同じ `hidden` と問合せ列を Rust の `opcg_engine.eval_queries` に渡して照合する。
  5. `RS_QUERY {...}` の 1 行を stdout に出す。

**例外は黙って除外しない**: どちらかが例外を投げた組合せは "error" として扱い、
**両側が error なら一致・片側だけ error なら不一致**と数える（Python が落ちる組合せで Rust が
黙って答えを返す、の逆も含めて検出する）。

`opcg_engine` が無い／`eval_queries` を持たない古い拡張では全行 unimplemented として集計する
（黙って緑にしない）。

実行例:
    OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/rs_query_oracle.py --boards 200
    OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/rs_query_oracle.py \\
      --boards 20 --games 2 --policy random --verbose     # 開発中の速い一次チェック
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import os as _os, sys as _sys  # noqa: E402  test bootstrap (sys.path + google スタブ)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: E402,F401

from harness.game_driver import DEFAULT_MAX_STEPS, InvariantError, load_db, make_seat, run_game  # noqa: E402
from harness.rs_record import manager_from_hidden  # noqa: E402
from rs_diff_replay import Recorder, board_dict, canon, first_diff  # noqa: E402

from opcg_sim.src.core.effects.matcher import get_target_cards  # noqa: E402
from opcg_sim.src.core.effects.resolver import EffectResolver  # noqa: E402
from opcg_sim.src.models.effect_types import (  # noqa: E402
    Branch, Choice, Condition, GameAction, Sequence, TargetQuery, ValueSource,
)

try:                        # Rust 拡張は未導入でも動く（その場合は全行 unimplemented）。
    import opcg_engine
except ImportError:         # pragma: no cover - 実行環境依存
    opcg_engine = None

_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DEFAULT_EFFECTS = _os.path.join(_REPO_ROOT, "opcg_sim", "data", "opcg_effects.json")
CARD_DB_PATH = _os.path.join(_REPO_ROOT, "opcg_sim", "data", "opcg_cards.json")


# --- 効果 JSON（Rust が読むカード定義）------------------------------------------

def ensure_effects_json(path: str, db) -> str:
    """効果構造 JSON を用意する（無い／カード DB と食い違うなら作り直す）。

    Rust と Python が**同じカード DB**を見ていることが照合の前提。exporter は
    `source.db_hash` を書くので、それが今の DB と違えば黙って再生成する
    （古い生成物で「一致」を出すのが一番たちが悪い）。
    """
    want = db.db_hash()
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                head = json.load(f)
            if head.get("source", {}).get("db_hash") == want:
                return path
            print(f"[effects] {path} はカード DB と食い違う（db_hash 不一致）ので再生成する")
        except (OSError, ValueError):
            print(f"[effects] {path} が読めないので再生成する")
    else:
        print(f"[effects] {path} が無いので生成する")
    subprocess.run([_sys.executable, "-m", "opcg_sim.tools.export_effects_json", "--out", path],
                   cwd=_REPO_ROOT, check=True, stdout=subprocess.DEVNULL)
    return path


# --- 問合せカタログ（カード DB の全ノードを path 付きで列挙）-----------------------

class Query:
    """1 件の問合せ（カード DB 上の位置と、Python 側で評価するためのノード実体）。"""

    __slots__ = ("kind", "card_id", "ability_index", "path", "node")

    def __init__(self, kind, card_id, ability_index, path, node):
        self.kind = kind
        self.card_id = card_id
        self.ability_index = ability_index
        self.path = path
        self.node = node


def collect_queries(db, card_ids=None) -> list:
    """カード DB の全カード×全能力から `TargetQuery`／`Condition`／`ValueSource` を列挙する。

    `path` は Rust 側 `effects::eval` の path 文法と 1:1（`Ability` からのフィールド名を
    `.` で繋ぎ、list は `name[i]`）。
    """
    out = []

    def emit(kind, card_id, ability_index, path, node):
        out.append(Query(kind, card_id, ability_index, path, node))

    def walk_value(vs, card_id, ai, path):
        emit("value", card_id, ai, path, vs)
        if getattr(vs, "count_query", None) is not None:
            emit("target", card_id, ai, f"{path}.count_query", vs.count_query)

    def walk_cond(c, card_id, ai, path):
        emit("condition", card_id, ai, path, c)
        if c.target is not None:
            emit("target", card_id, ai, f"{path}.target", c.target)
        for i, arg in enumerate(c.args or ()):
            walk_cond(arg, card_id, ai, f"{path}.args[{i}]")

    def walk_node(n, card_id, ai, path):
        if isinstance(n, Sequence):
            for i, sub in enumerate(n.actions or ()):
                walk_node(sub, card_id, ai, f"{path}.actions[{i}]")
        elif isinstance(n, Branch):
            if n.condition is not None:
                walk_cond(n.condition, card_id, ai, f"{path}.condition")
            if n.if_true is not None:
                walk_node(n.if_true, card_id, ai, f"{path}.if_true")
            if n.if_false is not None:
                walk_node(n.if_false, card_id, ai, f"{path}.if_false")
        elif isinstance(n, Choice):
            for i, opt in enumerate(n.options or ()):
                walk_node(opt, card_id, ai, f"{path}.options[{i}]")
        elif isinstance(n, GameAction):
            if n.target is not None:
                emit("target", card_id, ai, f"{path}.target", n.target)
            if n.value is not None:
                walk_value(n.value, card_id, ai, f"{path}.value")
            if n.sub_effect is not None:
                walk_node(n.sub_effect, card_id, ai, f"{path}.sub_effect")

    for card_id in sorted(db.raw_db.keys()):
        if card_ids is not None and card_id not in card_ids:
            continue
        master = db.get_card(card_id)
        if master is None:
            continue
        for ai, ab in enumerate(master.abilities or ()):
            if ab.condition is not None:
                walk_cond(ab.condition, master.card_id, ai, "condition")
            if ab.cost is not None:
                walk_node(ab.cost, master.card_id, ai, "cost")
            if ab.effect is not None:
                walk_node(ab.effect, master.card_id, ai, "effect")
    return out


# --- 盤面の収集 ----------------------------------------------------------------

def collect_boards(db, games: int, policy: str, seed_base: int, max_steps: int) -> list:
    """`games` 局を打ち、全行（setup＋各行動後）の `hidden`／`state` を集める。"""
    rows = []
    kind = "random" if policy == "random" else "ai"
    for i in range(games):
        seed = seed_base + i
        seats = {"p1": make_seat(kind=kind), "p2": make_seat(kind=kind)}
        rec = Recorder(seed, policy, record_hidden=True)
        try:
            run_game(seed, db, seats=seats, observers=(rec,), max_steps=max_steps)
        except (InvariantError, Exception) as e:  # noqa: BLE001 - 局が落ちてもそこまでの行は使える
            print(f"[boards] seed={seed} policy={policy} で打ち切り: {type(e).__name__}: {e}")
        if rec.setup is not None:
            rows.append({"seed": seed, "policy": policy, "hidden": rec.setup["hidden"],
                         "state": rec.setup["state"]})
        for step in rec.steps:
            if "hidden" in step:
                rows.append({"seed": seed, "policy": policy, "hidden": step["hidden"],
                             "state": step["state"]})
    return rows


def evenly(rows: list, n: int) -> list:
    """等間隔に n 件抜く（n >= len(rows) なら全部）。"""
    if n >= len(rows) or n <= 0:
        return list(rows)
    step = len(rows) / n
    return [rows[min(len(rows) - 1, int(i * step))] for i in range(n)]


# --- Python 側の評価 -----------------------------------------------------------

def sources_of(manager) -> list:
    """発生源の候補＝p1 の場（リーダー・ステージ含む）と手札の各カード。"""
    p = manager.p1
    out = []
    if p.leader:
        out.append(p.leader)
    out.extend(c for c in p.field if c is not None)
    if p.stage:
        out.append(p.stage)
    out.extend(c for c in p.hand if c is not None)
    return out


def python_eval(manager, resolver, q: Query, source):
    """Python 側で 1 件評価する。戻り値 `(True, 値)` か `(False, 例外の説明)`。"""
    try:
        if q.kind == "target":
            cards = get_target_cards(manager, q.node, source)
            return True, [c.uuid for c in cards]
        if q.kind == "condition":
            return True, bool(resolver._check_condition(manager.p1, q.node, source, None))
        if q.kind == "value":
            return True, int(resolver._calculate_value(manager.p1, q.node, []))
        raise ValueError(f"unknown query kind {q.kind!r}")
    except Exception as e:  # noqa: BLE001 - 例外は「error」という答えとして照合する
        return False, f"{type(e).__name__}: {e}"


def synthetic_context(manager, source):
    """文脈依存の条件・値を実際に踏むための合成 `effect_context`。

    空の context だけで照合すると、`PREV_ACTION`／`REVEALED_CARD_TRAIT`／
    `DECLARED_COST_MATCH`／`COUNT_QUERY`／`REFERENCE_POWER(selected)` が**既定値の枝しか
    通らない**（＝Rust 側の実装が間違っていても気付けない）。盤面から決まる値を詰めた
    context を 2 本目の条件として流す。

    戻り値 `(python_context_patch, rust_ctx_json)` — 同じ内容の Python 側 dict と
    `eval_queries` の `ctx` 欄。
    """
    revealed = next((c for c in manager.p2.deck), None) or next((c for c in manager.p1.deck), None)
    selected = next((c for c in manager.p2.field if c is not None), None) or source
    py = {
        "last_action_success": False,
        "_last_had_targets": False,
        "_last_action_count": 3,
        "_source_card_uuid": source.uuid,
        "saved_targets": {"selected_card": [selected], "selected": [selected]},
    }
    rs = {
        "last_action_success": False,
        "last_had_targets": False,
        "last_action_count": 3,
        "source_card_uuid": source.uuid,
        "saved_targets": {"selected_card": [selected.uuid], "selected": [selected.uuid]},
    }
    if revealed is not None:
        py["last_revealed_card"] = revealed
        py["declared_cost"] = revealed.master.cost
        rs["last_revealed_card"] = revealed.uuid
        rs["declared_cost"] = revealed.master.cost
    return py, rs


# --- 1 局面の照合 --------------------------------------------------------------

def check_board(db, row: dict, queries: list, board_index: int, totals: dict, firsts: list,
                effects_path: str, with_context: bool = False, null_source: bool = False) -> None:
    hidden = row["hidden"]
    manager = manager_from_hidden(db, hidden)

    # 自己検査: 復元した盤面が記録の `state` と一致するか（Python 側だけで閉じる）。
    got = dict(board_dict(manager))
    got.pop("pending_request", None)
    exp = dict(row["state"])
    exp.pop("pending_request", None)
    if json.dumps(canon(exp), sort_keys=True, ensure_ascii=False, default=str) != \
            json.dumps(canon(got), sort_keys=True, ensure_ascii=False, default=str):
        totals["restore_mismatch"] += 1
        if not firsts:
            firsts.append({"kind": "restore", "board": board_index, "seed": row["seed"],
                           "path": first_diff(canon(exp), canon(got))})
        return

    sources = sources_of(manager)
    if not sources:
        totals["boards_without_source"] += 1
        return

    expected = []
    payload = []
    for qi, q in enumerate(queries):
        source = sources[(board_index + qi) % len(sources)]
        # context は問合せごとに作り直す（空 context の resolver は状態を持たないが、
        # 合成 context を入れた resolver は問合せごとに発生源が変わるため）。
        resolver = EffectResolver(manager)
        rs_ctx = None
        if with_context:
            py_ctx, rs_ctx = synthetic_context(manager, source)
            resolver.context.update(py_ctx)
        # 発生源を落とす条件（`--null-source`）: Python は対象クエリで必ず例外になり、条件は
        # 型により「例外」と「False」に分かれる。**両側が同じように失敗する**ことまで見る。
        py_source = None if null_source else source
        expected.append(python_eval(manager, resolver, q, py_source))
        row = {"kind": q.kind, "card_id": q.card_id, "ability_index": q.ability_index,
               "path": q.path, "actor": "p1",
               "source": None if null_source else source.uuid, "host": None}
        if rs_ctx is not None:
            row["ctx"] = rs_ctx
        payload.append(row)

    if opcg_engine is None or not hasattr(opcg_engine, "eval_queries"):
        totals["unimplemented"] += len(queries)
        return

    try:
        out = opcg_engine.eval_queries(
            json.dumps(hidden, ensure_ascii=False, default=str),
            json.dumps(payload, ensure_ascii=False, default=str),
            effects_path,
        )
    except NotImplementedError as e:
        totals["unimplemented"] += len(queries)
        if not firsts:
            firsts.append({"kind": "unimplemented", "board": board_index, "detail": str(e)})
        return
    except ValueError as e:      # 契約違反（ハーネス側／Rust 側のバグ）＝黙って通さない
        totals["bad_payload"] += len(queries)
        if not firsts:
            firsts.append({"kind": "bad_payload", "board": board_index, "detail": str(e)})
        return
    results = json.loads(out).get("results")
    if not isinstance(results, list) or len(results) != len(queries):
        totals["bad_output"] += len(queries)
        if not firsts:
            firsts.append({"kind": "bad_output", "board": board_index,
                           "detail": f"results={type(results).__name__} len="
                                     f"{len(results) if isinstance(results, list) else '?'}"})
        return

    for q, (ok, value), got_row in zip(queries, expected, results):
        rust_ok = got_row.get("status") == "ok"
        if not ok and not rust_ok:
            totals["error_match"] += 1          # 両側で例外＝一致
            continue
        if ok and rust_ok and value == got_row.get("value"):
            totals["match"] += 1
            # 照合が空振りでないことの目安（対象が非空／条件が真／値が非零だった件数）。
            # これが 0 のまま「一致」だと、何も見ていないのに緑になっている恐れがある。
            if q.kind == "target" and value:
                totals["nonempty_targets"] += 1
            elif q.kind == "condition" and value:
                totals["true_conditions"] += 1
            elif q.kind == "value" and value:
                totals["nonzero_values"] += 1
            continue
        totals["mismatch"] += 1
        totals["mismatch_by_kind"][q.kind] = totals["mismatch_by_kind"].get(q.kind, 0) + 1
        if not firsts:
            firsts.append({
                "kind": "mismatch", "board": board_index, "seed": row["seed"],
                "query": {"kind": q.kind, "card_id": q.card_id,
                          "ability_index": q.ability_index, "path": q.path},
                "python": ("error: " + value) if not ok else value,
                "rust": got_row.get("value") if rust_ok else ("error: " + str(got_row.get("error"))),
            })


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="カード DB の全 TargetQuery/Condition/ValueSource を Python と Rust で照合する")
    ap.add_argument("--boards", type=int, default=200, help="照合する局面数（既定 200）")
    ap.add_argument("--games", type=int, default=10, help="局面を採る局数（方策ごと・既定 10）")
    ap.add_argument("--policy", choices=["random", "l1", "both"], default="both",
                    help="局面を採る方策（既定 both＝random と l1 の両方から等分に採る）")
    ap.add_argument("--seed-base", type=int, default=900000, help="seed の基点（既定 900000）")
    ap.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS, help="1局の上限ステップ")
    ap.add_argument("--cards", default=None,
                    help="カード ID をカンマ区切りで指定（既定は全カード。開発中の絞り込み用）")
    ap.add_argument("--effects", default=DEFAULT_EFFECTS,
                    help="効果構造 JSON（Rust のカード定義表・カード DB と食い違えば再生成する）")
    ap.add_argument("--ctx", choices=["empty", "synthetic", "both"], default="both",
                    help="effect_context の条件（既定 both＝空 context に加えて、盤面から作った"
                         "合成 context でも照合する。PREV_ACTION／REVEALED_CARD_TRAIT／"
                         "DECLARED_COST_MATCH／COUNT_QUERY／REFERENCE_POWER の既定値以外の枝を踏む）")
    ap.add_argument("--no-null-source", action="store_true",
                    help="発生源を落とした条件での照合を省く（既定は行う＝Python が例外を投げる"
                         "組合せで Rust も同じように失敗することを確かめる）")
    ap.add_argument("--verbose", action="store_true", help="局面ごとの進捗を出す")
    args = ap.parse_args(argv)

    db = load_db()
    effects_path = ensure_effects_json(args.effects, db)
    if opcg_engine is not None and hasattr(opcg_engine, "load_masters"):
        n = opcg_engine.load_masters(effects_path)
        print(f"[effects] load_masters({effects_path}) -> {n} cards")

    card_ids = set(args.cards.split(",")) if args.cards else None
    t0 = time.time()
    queries = collect_queries(db, card_ids)
    kinds = {}
    for q in queries:
        kinds[q.kind] = kinds.get(q.kind, 0) + 1
    print(f"[catalog] {len(queries)} queries {kinds} ({time.time() - t0:.1f}s)")

    policies = ["random", "l1"] if args.policy == "both" else [args.policy]
    rows = []
    for i, policy in enumerate(policies):
        t0 = time.time()
        got = collect_boards(db, args.games, policy, args.seed_base + 1000 * i, args.max_steps)
        print(f"[boards] policy={policy} games={args.games} rows={len(got)} "
              f"({time.time() - t0:.1f}s)")
        rows.append(got)
    per_policy = max(1, args.boards // len(rows)) if rows else 0
    boards = []
    for got in rows:
        boards.extend(evenly(got, per_policy))
    boards = boards[:args.boards] if args.boards > 0 else boards

    # 照合の条件（パス）: (合成 context か, 発生源を落とすか)。
    ctx_modes = {"empty": [False], "synthetic": [True], "both": [False, True]}[args.ctx]
    passes = [(c, False) for c in ctx_modes]
    if not args.no_null_source:
        passes.append((False, True))
    totals = {"match": 0, "mismatch": 0, "error_match": 0, "unimplemented": 0,
              "bad_payload": 0, "bad_output": 0, "restore_mismatch": 0,
              "boards_without_source": 0, "nonempty_targets": 0, "true_conditions": 0,
              "nonzero_values": 0, "mismatch_by_kind": {}}
    firsts = []
    t0 = time.time()
    for i, row in enumerate(boards):
        try:
            for with_context, null_source in passes:
                check_board(db, row, queries, i, totals, firsts, effects_path,
                            with_context, null_source)
        except Exception as e:  # noqa: BLE001 - ハーネス自身の事故も集計に載せる
            totals["bad_payload"] += 1
            if not firsts:
                firsts.append({"kind": "harness_error", "board": i,
                               "detail": f"{type(e).__name__}: {e}\n{traceback.format_exc()}"})
        if args.verbose and (i + 1) % 10 == 0:
            print(f"[boards] {i + 1}/{len(boards)} match={totals['match']} "
                  f"mismatch={totals['mismatch']} ({time.time() - t0:.1f}s)", flush=True)

    summary = {
        "boards": len(boards),
        "queries": totals["match"] + totals["mismatch"] + totals["error_match"],
        "match": totals["match"],
        "mismatch": totals["mismatch"],
        "mismatch_by_kind": totals["mismatch_by_kind"],
        "error_match": totals["error_match"],
        "unimplemented": totals["unimplemented"],
        "bad_payload": totals["bad_payload"],
        "bad_output": totals["bad_output"],
        "restore_mismatch": totals["restore_mismatch"],
        # 空振り検査（一致の内訳）: 対象が非空／条件が真／値が非零だった件数。
        "nonempty_targets": totals["nonempty_targets"],
        "true_conditions": totals["true_conditions"],
        "nonzero_values": totals["nonzero_values"],
        "catalog": len(queries),
        "catalog_kinds": kinds,
        "games": args.games,
        "policy": args.policy,
        "ctx": args.ctx,
        "null_source_pass": not args.no_null_source,
        "seed_base": args.seed_base,
        "engine": (opcg_engine.version() if opcg_engine is not None else None),
        "seconds": round(time.time() - t0, 1),
        "first": firsts[0] if firsts else None,
    }
    print("RS_QUERY " + json.dumps(summary, ensure_ascii=False))
    bad = (totals["mismatch"] or totals["bad_payload"] or totals["bad_output"]
           or totals["restore_mismatch"] or totals["unimplemented"])
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
