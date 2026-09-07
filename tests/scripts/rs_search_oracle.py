"""探索オラクル（`docs/rust_engine_plan.md` §12.2・WP `rs-p4-legal` の受け入れ道具）。

**問い**: Rust の「探索用の候補・世界サンプル・手の適用」（`search/{adapter,macro,prune,
determinize,apply}.rs`）は Python 版（`learned/adapter.py::OPCGGame.legal_actions`／
`cpu_ai._determinize_opponent`／`cpu_ai._apply_move_inplace`）と**同じ答え**を返すか。

3 本を `--what` で選ぶ（既定は 3 本とも）:

  legal        `OPCGGame.legal_actions` を**順序込み**で照合する。局面そのもの（pass A）に加え、
               先頭の合法手を数手打った先（pass B＝`--prefix`）でも照合する。記録 v5 の
               `hidden` は中断スタックを持たない（`GameState::from_record` も同じ）ため、
               復元しただけの盤面では対話（`merged_search_actions` の代替手併合・防御箱）を
               1 度も踏まない。実際に手を打って中断へ入り、そこでの候補を照合する。
  determinize  Python の `rng.shuffle(pool)` の**出目（並び）**を記録して Rust へ渡し、
               できた盤面 dict を照合する（両席ぶん）。
  apply        各局面の全合法手（上限 `--max-moves`）を両側で適用し、盤面 dict
               （`pending_request` 込み）を照合する。**例外も答え**として扱い、
               両側 error なら一致・片側だけ error なら不一致と数える。

出力は `RS_SEARCH {...}` の 1 行（`--what` ごとに 1 行）。

`opcg_engine` が無い／`search_legal` を持たない古い拡張では全件 unimplemented として集計する
（黙って緑にしない）。

実行例:
    OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/rs_search_oracle.py \\
      --what legal,determinize,apply --boards 200
    OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/rs_search_oracle.py \\
      --what legal --boards 20 --games 2 --policy random --verbose   # 開発中の一次チェック
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import os as _os, sys as _sys  # noqa: E402  test bootstrap (sys.path + google スタブ)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: E402,F401

from harness.game_driver import DEFAULT_MAX_STEPS, load_db  # noqa: E402
from harness.rs_record import manager_from_hidden  # noqa: E402
from rs_diff_replay import board_dict, canon, first_diff  # noqa: E402
from rs_query_oracle import collect_boards, ensure_effects_json, evenly  # noqa: E402

from opcg_sim.src.core import cpu_ai  # noqa: E402
from opcg_sim.src.learned.adapter import OPCGGame  # noqa: E402

try:                        # Rust 拡張は未導入でも動く（その場合は全件 unimplemented）。
    import opcg_engine
except ImportError:         # pragma: no cover - 実行環境依存
    opcg_engine = None

_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DEFAULT_EFFECTS = _os.path.join(_REPO_ROOT, "opcg_sim", "data", "opcg_effects.json")

WHATS = ("legal", "determinize", "apply")


# --- Python 側の盤面（記録 v5 の hidden から「動く」manager を組む）------------------

def live_manager(db, hidden: dict):
    """`manager_from_hidden` の盤面を**要求（pending request）が読める形**にする。

    `harness/rs_record.manager_from_hidden` は P1 の照合用に
      (a) `get_pending_request` をインスタンス属性で潰して常に None にし、
      (b) `active_battle` を `{attacker, target, counter_buff}` だけで組む
    （記録 v2 の `active_battle` には所在の持ち主が無かったため）。P4 の探索オラクルは
    要求そのものを使うので、記録 v3 以降が持つ `attacker_owner`／`target_owner` から
    戦闘の持ち主を補い、(a) の潰しを外す（クラス側の実装が再び見えるようになる）。

    `opcg_sim/` は変更しない（ハーネス側の復元だけを厚くする）。
    """
    manager = manager_from_hidden(db, hidden)
    del manager.get_pending_request          # インスタンス属性を外す＝クラスの実装に戻る
    battle = (hidden.get("manager") or {}).get("active_battle")
    if battle and manager.active_battle is not None:
        by_name = {"p1": manager.p1, "p2": manager.p2}
        manager.active_battle["attacker_owner"] = by_name[battle["attacker_owner"]]
        manager.active_battle["target_owner"] = by_name[battle["target_owner"]]
    return manager


def game_for(args) -> OPCGGame:
    """探索アダプタ（探索設定は Rust 側 `SearchOptions` の既定と同じ serve 既定）。"""
    return OPCGGame(prune_futile=args.prune_futile, macro_moves=args.macro_moves,
                    defense_box=args.defense_box, don_margin=args.don_margin)


def rust_opts(args, prefix=None) -> str:
    opts = {}
    for key, value in (("prune_futile", args.prune_futile), ("macro_moves", args.macro_moves),
                       ("defense_box", args.defense_box), ("don_margin", args.don_margin)):
        if value is not None:
            opts[key] = value
    if prefix:
        opts["prefix"] = prefix
    return json.dumps(opts, ensure_ascii=False, default=str)


def norm(value) -> str:
    """キー順に依存しない正規化 JSON（`rs_diff_replay._norm` と同じ規約）。"""
    return json.dumps(canon(value), sort_keys=True, ensure_ascii=False, default=str)


def strip_request_id(board: dict) -> dict:
    """`pending_request.request_id`（フロント専用の sha1）は照合から外す。"""
    b = dict(board)
    pr = b.get("pending_request")
    if isinstance(pr, dict):
        pr = dict(pr)
        pr.pop("request_id", None)
        b["pending_request"] = pr
    return b


class ShuffleCounter:
    """適用中に `random.shuffle` が呼ばれたかを数える（観測だけ・並びは本物に任せる）。

    エンジンが山札を混ぜる手（マリガン・サーチ効果の混ぜ直し）は、Rust 側が**自前の並び**で
    混ぜる（計画 §6「乱数列は流さない」）ので盤面が食い違って当然＝照合から外す。
    `rs_diff_replay` は記録の `hidden` から並びを取り直して同じことを避けているが、
    本オラクルは 1 手だけを適用する（取り直す先の記録が無い）ので、件数を数えて報告する。
    """

    def __enter__(self):
        self._real = random.shuffle
        self.n = 0

        def wrapped(seq, *args, **kwargs):
            self._real(seq, *args, **kwargs)
            self.n += 1

        random.shuffle = wrapped
        return self

    def __exit__(self, *exc):
        random.shuffle = self._real
        return False


class Totals(dict):
    """集計（欠けたキーは 0 から数える）。"""

    def bump(self, key, n=1):
        self[key] = self.get(key, 0) + n


def note_first(firsts: list, row: dict) -> None:
    if not firsts:
        firsts.append(row)


# --- 1. 候補（legal）-------------------------------------------------------------

def check_legal(db, row: dict, board_index: int, args, totals: Totals, firsts: list,
                effects_path: str) -> None:
    hidden = row["hidden"]
    hidden_json = json.dumps(hidden, ensure_ascii=False, default=str)
    game = game_for(args)

    # pass A＝復元した局面そのもの、pass B＝先頭の合法手を prefix として打った先
    # （中断＝対話の代替手併合・防御箱を踏むための経路）。
    prefixes = [[]]
    manager = live_manager(db, hidden)
    if args.prefix > 0:
        seen_moves = game.legal_actions(manager)
        for mv in seen_moves[:args.prefix_width]:
            prefixes.append([mv])

    for prefix in prefixes:
        manager = live_manager(db, hidden)
        try:
            with ShuffleCounter() as shuffles:
                for mv in prefix:
                    actor = game.current_player(manager)
                    if actor is None:
                        raise ValueError("prefix を打つ主体が居ない")
                    cpu_ai._apply_move_inplace(manager, actor, mv, stop_at_select=True)
            expected = game.legal_actions(manager)
        except Exception as e:  # noqa: BLE001 - prefix が例外になる局面はこの pass を飛ばす
            totals.bump("prefix_skipped")
            del e
            continue
        if shuffles.n:
            totals.bump("shuffle_skipped")   # 山札を混ぜた先は Rust と並びが違って当然
            continue

        if opcg_engine is None or not hasattr(opcg_engine, "search_legal"):
            totals.bump("unimplemented")
            continue
        try:
            got = json.loads(opcg_engine.search_legal(hidden_json, rust_opts(args, prefix)))
        except NotImplementedError as e:
            totals.bump("unimplemented")
            note_first(firsts, {"kind": "unimplemented", "board": board_index, "detail": str(e)})
            continue
        except ValueError as e:      # 契約違反（ハーネス／Rust のバグ）＝黙って通さない
            totals.bump("bad_payload")
            note_first(firsts, {"kind": "bad_payload", "board": board_index, "detail": str(e)})
            continue

        totals.bump("moves", len(expected))
        totals.bump("checks")
        if norm(expected) == norm(got):
            totals.bump("match")
            # 空振り検査: 箱・対話の代替手・防御窓を実際に踏んだ件数。
            kinds = {m.get("action_type") for m in expected}
            if "DON_BOX" in kinds:
                totals.bump("boards_with_box")
            if "RESOLVE_EFFECT_SELECTION" in kinds:
                totals.bump("boards_with_selection")
            if {"SELECT_COUNTER", "PASS"} & kinds:
                totals.bump("boards_with_defense")
            if len(expected) > 1:
                totals.bump("boards_with_choice")
            continue
        totals.bump("mismatch")
        note_first(firsts, {
            "kind": "mismatch", "board": board_index, "seed": row["seed"],
            "prefix": prefix, "path": first_diff(canon(expected), canon(got)),
            "python_n": len(expected), "rust_n": len(got) if isinstance(got, list) else None,
            "python": expected[:6], "rust": got[:6] if isinstance(got, list) else got,
        })


# --- 2. 世界サンプル（determinize）-------------------------------------------------

class RecordingRng:
    """`rng.shuffle` の**出目（並び）**を記録するラッパ（ハーネス内・`opcg_sim/` は触らない）。"""

    def __init__(self, rng):
        self.rng = rng
        self.orders = []

    def shuffle(self, seq):
        self.rng.shuffle(seq)
        self.orders.append([getattr(c, "uuid", None) for c in seq])


def check_determinize(db, row: dict, board_index: int, args, totals: Totals, firsts: list) -> None:
    hidden = row["hidden"]
    hidden_json = json.dumps(hidden, ensure_ascii=False, default=str)
    for seat in ("p1", "p2"):
        manager = live_manager(db, hidden)
        rng = RecordingRng(random.Random(args.seed_base + board_index))
        clone = cpu_ai._determinize_opponent(manager, seat, rng)
        expected = strip_request_id(board_dict(clone))
        # `pool` が空なら Python は shuffle を呼ばない（Rust も並びを要らない）。
        order = rng.orders[0] if rng.orders else []

        totals.bump("checks")
        if opcg_engine is None or not hasattr(opcg_engine, "search_determinize"):
            totals.bump("unimplemented")
            continue
        try:
            got = json.loads(opcg_engine.search_determinize(
                hidden_json, seat, json.dumps(order, ensure_ascii=False)))
        except NotImplementedError as e:
            totals.bump("unimplemented")
            note_first(firsts, {"kind": "unimplemented", "board": board_index, "detail": str(e)})
            continue
        except ValueError as e:
            totals.bump("bad_payload")
            note_first(firsts, {"kind": "bad_payload", "board": board_index, "seat": seat,
                                "detail": str(e)})
            continue
        got = strip_request_id(got)
        if norm(expected) == norm(got):
            totals.bump("match")
            if order:
                totals.bump("nonempty_pools")
            continue
        totals.bump("mismatch")
        note_first(firsts, {"kind": "mismatch", "board": board_index, "seed": row["seed"],
                            "seat": seat, "pool": len(order),
                            "path": first_diff(canon(expected), canon(got))})


# --- 3. 適用（apply）---------------------------------------------------------------

def python_apply(manager, actor: str, move: dict):
    """`_apply_move_inplace` をクローン上で回す。

    戻り値は `(状態, 値, 混ぜた回数)`。状態は `True`＝盤面／`False`＝例外の説明。
    """
    clone = manager.clone()
    clone.action_events = []
    with ShuffleCounter() as shuffles:
        try:
            cpu_ai._apply_move_inplace(clone, actor, move, stop_at_select=True)
        except Exception as e:  # noqa: BLE001 - 例外は「答え」として照合する
            return False, f"{type(e).__name__}: {e}", shuffles.n
    return True, strip_request_id(board_dict(clone)), shuffles.n


def check_apply(db, row: dict, board_index: int, args, totals: Totals, firsts: list) -> None:
    hidden = row["hidden"]
    hidden_json = json.dumps(hidden, ensure_ascii=False, default=str)
    manager = live_manager(db, hidden)
    game = game_for(args)
    actor = game.current_player(manager)
    if actor is None:
        totals.bump("boards_without_actor")
        return
    moves = game.legal_actions(manager)[:args.max_moves]
    for move in moves:
        ok, expected, shuffled = python_apply(manager, actor, move)
        if shuffled:
            totals.bump("shuffle_skipped")   # 山札を混ぜた手は Rust と並びが違って当然
            continue
        totals.bump("checks")
        totals.bump("moves")
        if opcg_engine is None or not hasattr(opcg_engine, "search_apply"):
            totals.bump("unimplemented")
            continue
        move_json = json.dumps(move, ensure_ascii=False, default=str)
        rust_ok, got = True, None
        try:
            got = json.loads(opcg_engine.search_apply(hidden_json, actor, move_json, True))
        except NotImplementedError as e:
            totals.bump("unimplemented")
            note_first(firsts, {"kind": "unimplemented", "board": board_index,
                                "move": move, "detail": str(e)})
            continue
        except ValueError as e:      # Python の例外に対応する Rust 側の失敗
            rust_ok, got = False, str(e)

        if not ok and not rust_ok:
            totals.bump("error_match")      # 両側で失敗＝一致
            continue
        if ok and rust_ok and norm(expected) == norm(strip_request_id(got)):
            totals.bump("match")
            if (move.get("action_type") or "") == "DON_BOX":
                totals.bump("box_moves")
            if isinstance(got, dict) and got.get("pending_request"):
                totals.bump("with_pending")
            continue
        totals.bump("mismatch")
        note_first(firsts, {
            "kind": "mismatch", "board": board_index, "seed": row["seed"], "move": move,
            "python": ("error: " + expected) if not ok else "board",
            "rust": ("error: " + str(got)) if not rust_ok else "board",
            "path": (first_diff(canon(expected), canon(strip_request_id(got)))
                     if ok and rust_ok else None),
        })


CHECKERS = {
    "legal": lambda db, row, i, args, totals, firsts, effects: check_legal(
        db, row, i, args, totals, firsts, effects),
    "determinize": lambda db, row, i, args, totals, firsts, effects: check_determinize(
        db, row, i, args, totals, firsts),
    "apply": lambda db, row, i, args, totals, firsts, effects: check_apply(
        db, row, i, args, totals, firsts),
}


def run_one(what: str, db, boards: list, args, effects_path: str) -> int:
    totals = Totals()
    firsts: list = []
    t0 = time.time()
    for i, row in enumerate(boards):
        try:
            CHECKERS[what](db, row, i, args, totals, firsts, effects_path)
        except Exception as e:  # noqa: BLE001 - ハーネス自身の事故も集計に載せる
            totals.bump("harness_error")
            note_first(firsts, {"kind": "harness_error", "board": i,
                                "detail": f"{type(e).__name__}: {e}\n{traceback.format_exc()}"})
        if args.verbose and (i + 1) % 10 == 0:
            print(f"[{what}] {i + 1}/{len(boards)} match={totals.get('match', 0)} "
                  f"mismatch={totals.get('mismatch', 0)} ({time.time() - t0:.1f}s)", flush=True)

    summary = {
        "what": what,
        "boards": len(boards),
        "moves": totals.get("moves", 0),
        "checks": totals.get("checks", 0),
        "match": totals.get("match", 0),
        "mismatch": totals.get("mismatch", 0),
        "error_match": totals.get("error_match", 0),
        "unimplemented": totals.get("unimplemented", 0),
        "bad_payload": totals.get("bad_payload", 0),
        "harness_error": totals.get("harness_error", 0),
    }
    # 空振り検査（一致の内訳＝照合が本当に対象を踏んだか）。
    for key in ("boards_with_box", "boards_with_selection", "boards_with_defense",
                "boards_with_choice", "prefix_skipped", "shuffle_skipped", "nonempty_pools",
                "box_moves", "with_pending", "boards_without_actor"):
        if key in totals:
            summary[key] = totals[key]
    summary.update({
        "games": args.games, "policy": args.policy, "seed_base": args.seed_base,
        "engine": (opcg_engine.version() if opcg_engine is not None else None),
        "seconds": round(time.time() - t0, 1),
        "first": firsts[0] if firsts else None,
    })
    print("RS_SEARCH " + json.dumps(summary, ensure_ascii=False, default=str))
    bad = (totals.get("mismatch", 0) or totals.get("bad_payload", 0)
           or totals.get("unimplemented", 0) or totals.get("harness_error", 0))
    return 1 if bad else 0


def _tri_state(value):
    """`--prune-futile` 等の 3 値（既定＝config／on／off）。"""
    if value is None:
        return None
    return value not in ("0", "false", "off", "no")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="探索用の候補・世界サンプル・手の適用を Python と Rust で照合する")
    ap.add_argument("--what", default=",".join(WHATS),
                    help="照合する項目（legal,determinize,apply のカンマ区切り・既定は全部）")
    ap.add_argument("--boards", type=int, default=200, help="照合する局面数（既定 200）")
    ap.add_argument("--games", type=int, default=10, help="局面を採る局数（方策ごと・既定 10）")
    ap.add_argument("--policy", choices=["random", "l1", "both"], default="both",
                    help="局面を採る方策（既定 both＝random と l1 の両方から等分に採る）")
    ap.add_argument("--seed-base", type=int, default=940000, help="seed の基点（既定 940000）")
    ap.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS, help="1局の上限ステップ")
    ap.add_argument("--max-moves", type=int, default=20,
                    help="apply で 1 局面あたり適用する合法手の上限（既定 20）")
    ap.add_argument("--prefix", type=int, default=1,
                    help="legal で「手を打った先」も照合するか（0=しない・既定 1）")
    ap.add_argument("--prefix-width", type=int, default=3,
                    help="legal の prefix に使う先頭の合法手の本数（既定 3）")
    ap.add_argument("--prune-futile", dest="prune_futile", default=None, type=_tri_state,
                    help="枝刈り（既定＝config の SERVE_PRUNE_FUTILE）")
    ap.add_argument("--macro-moves", dest="macro_moves", default=None, type=_tri_state,
                    help="配分箱／アタック箱（既定＝config の SERVE_MACRO_MOVES）")
    ap.add_argument("--defense-box", dest="defense_box", default=None, type=_tri_state,
                    help="防御箱（既定＝config の SERVE_DEFENSE_BOX）")
    ap.add_argument("--don-margin", dest="don_margin", default=None, type=_tri_state,
                    help="マージン付与（既定＝cpu_ai.DON_MARGIN_ATTACH）")
    ap.add_argument("--effects", default=DEFAULT_EFFECTS,
                    help="効果構造 JSON（Rust のカード定義表・カード DB と食い違えば再生成する）")
    ap.add_argument("--verbose", action="store_true", help="局面ごとの進捗を出す")
    args = ap.parse_args(argv)

    whats = [w.strip() for w in args.what.split(",") if w.strip()]
    unknown = [w for w in whats if w not in CHECKERS]
    if unknown:
        ap.error(f"unknown --what: {unknown} (choose from {list(WHATS)})")

    db = load_db()
    effects_path = ensure_effects_json(args.effects, db)
    if opcg_engine is not None and hasattr(opcg_engine, "load_masters"):
        n = opcg_engine.load_masters(effects_path)
        print(f"[effects] load_masters({effects_path}) -> {n} cards")

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

    rc = 0
    for what in whats:
        rc |= run_one(what, db, boards, args, effects_path)
    return rc


if __name__ == "__main__":
    sys.exit(main())
