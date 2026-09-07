"""探索オラクル（`docs/rust_engine_plan.md` §12.2・WP `rs-p4-legal`／`rs-p4-mcts` の受け入れ道具）。

**問い**: Rust の探索（`search/{adapter,macro,prune,determinize,apply,quiesce,mcts,decide}.rs`）は
Python 版（`learned/adapter.py::OPCGGame`／`cpu_ai._determinize_opponent`／
`cpu_ai._apply_move_inplace`／`learned/mcts.py::TreeMCTS`／`core/cpu_learned.py::LearnedEngine.decide`）と
**同じ答え**を返すか。

4 本を `--what` で選ぶ（既定は 3 本＝legal,determinize,apply。decide は明示指定）:

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
  decide       **1 手の決定そのもの**を照合する（WP `rs-p4-mcts`）。Python の `LearnedEngine`
               （a1・serve 既定）で `--games` 局を打ち、各決定点で
                 (a) 決定前の `hidden`、(b) その decide が引いた乱数の出目（世界サンプルの並び・
                 Dirichlet・温度サンプルの一様乱数）、(c) 箱コミットの残り手順、
               を採って Rust の `decide` を**同じ出目**で走らせ、**手**と**根の訪問数 N** を
               照合する。窓（window）・コミット（commit）の決定点は手の一致のみ
               （どちらも訪問を配らないため N が無い）。

               同点で argmax が割れた決定点（float の丸めで PUCT の順位が入れ替わりうる）は
               「訪問分布の L1 ≤ 2/sims」で通し、`ties` として件数を報告する。
               探索の途中で山札を混ぜた（`random.shuffle` を消費した）決定点は、Rust 側が
               自前の擬似乱数で混ぜる（計画 §12.4-4）ため盤面が食い違って当然＝
               `shuffle_skipped` として照合から外す（黙って一致にしない）。

出力は `RS_SEARCH {...}` の 1 行（`--what` ごとに 1 行）。

`opcg_engine` が無い／必要な関数を持たない古い拡張では全件 unimplemented として集計する
（黙って緑にしない）。

実行例:
    OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/rs_search_oracle.py \\
      --what legal,determinize,apply --boards 200
    OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/rs_search_oracle.py \\
      --what decide --games 100 --jobs 8
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

import numpy as np  # noqa: E402

import os as _os, sys as _sys  # noqa: E402  test bootstrap (sys.path + google スタブ)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: E402,F401

from harness.game_driver import DEFAULT_MAX_STEPS, load_db  # noqa: E402
from harness.rs_record import manager_from_hidden  # noqa: E402
from rs_diff_replay import board_dict, canon, first_diff, hidden_dict  # noqa: E402
from rs_query_oracle import collect_boards, ensure_effects_json, evenly  # noqa: E402

from opcg_sim.src.core import cpu_ai  # noqa: E402
from opcg_sim.src.learned.adapter import OPCGGame  # noqa: E402

try:                        # Rust 拡張は未導入でも動く（その場合は全件 unimplemented）。
    import opcg_engine
except ImportError:         # pragma: no cover - 実行環境依存
    opcg_engine = None

_REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DEFAULT_EFFECTS = _os.path.join(_REPO_ROOT, "opcg_sim", "data", "opcg_effects.json")
DEFAULT_NET = _os.path.join(_REPO_ROOT, "opcg_sim", "data", "learned", "nrel_a1.npz")

WHATS = ("legal", "determinize", "apply", "decide")


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


MAX_REPORTED = {"n": 1}


def note_first(firsts: list, row: dict) -> None:
    """不一致の記録（既定は先頭 1 件・`--max-report` で増やせる）。"""
    if len(firsts) < MAX_REPORTED["n"]:
        firsts.append(row)


_LOUD = {"on": False}


def note_loud(row: dict) -> None:
    """`--verbose` のとき不一致を**その場で**出す（1 件目だけでは原因が追えないため）。"""
    if _LOUD["on"]:
        print("  ! " + json.dumps(row, ensure_ascii=False, default=str)[:1400], flush=True)


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


# --- 4. 決定（decide）--------------------------------------------------------------
#
# `opcg_sim/` は変えない。ここで足すのは 3 つのハーネス部品だけ:
#   `RecordingGenerator` … `np.random.Generator` を包んで shuffle／dirichlet／choice の出目を記録する
#                    （`RecordingRng` は determinize 用の別物＝`random.Random` を包む）
#   `RecordingEngine`… `LearnedEngine._world_rng` の返り値を上のラッパへ差し替える（席別 seam）
#   `_TracingMCTS`   … `cpu_learned` が参照する `TreeMCTS` を包んで `last_stats` を取り出す
# どれも観測専用で、Python 側の決定そのものは 1 bit も変えない。

class RecordingGenerator:
    """`np.random.Generator` の薄い記録ラッパ（出目を `sink` へ書く）。

    `choice(n, p=...)` だけは**自前で組む**（numpy の実装と同じ「累積分布を cdf[-1] で割り
    `searchsorted(u, side="right")`」＝2,000 回の照合で完全一致）。こうすると Rust へ渡せる
    「引いた一様乱数」がそのまま採れる（numpy の内部からは取り出せない）。
    """

    def __init__(self, rng, sink):
        self._rng = rng
        self._sink = sink

    def shuffle(self, seq):
        self._rng.shuffle(seq)
        self._sink["shuffles"].append([getattr(c, "uuid", None) for c in seq])

    def dirichlet(self, alpha, size=None):
        v = self._rng.dirichlet(alpha, size)
        self._sink["dirichlets"].append([float(x) for x in np.asarray(v).reshape(-1)])
        return v

    def choice(self, a, size=None, replace=True, p=None):
        if p is None or size is not None:
            return self._rng.choice(a, size=size, replace=replace, p=p)
        u = float(self._rng.random())
        self._sink["uniforms"].append(u)
        cdf = np.asarray(p, np.float64).cumsum()
        cdf /= cdf[-1]
        return int(min(int(cdf.searchsorted(u, side="right")), len(cdf) - 1))

    def __getattr__(self, name):     # integers / random / …（記録しない口は素通し）
        return getattr(self._rng, name)


def _engine_class():
    """`LearnedEngine` に `_world_rng` の記録だけを足した席別 seam（ハーネス内）。"""
    from opcg_sim.src.core import cpu_learned

    class RecordingEngine(cpu_learned.LearnedEngine):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.rec = None      # 現在の決定点の出目シンク（None＝記録しない）

        def _world_rng(self, manager, name, rng):
            g = super()._world_rng(manager, name, rng)
            return RecordingGenerator(g, self.rec) if self.rec is not None else g

    return RecordingEngine


_MCTS_STATS = {}


def _install_mcts_probe():
    """`cpu_learned` が使う `TreeMCTS` を包み、`run` 後の `last_stats` を横取りする。

    `decide` は木のインスタンスを返さないので、根の訪問数 N を採るにはここしかない
    （`record["groups"]` は**マージ後**の集計＝素の N ではない）。観測専用。
    """
    from opcg_sim.src.core import cpu_learned

    base = cpu_learned.TreeMCTS
    if getattr(base, "_rs_probe", False):
        return

    class _TracingMCTS(base):
        _rs_probe = True

        def run(self, real_state):
            out = super().run(real_state)
            _MCTS_STATS["last"] = getattr(self, "last_stats", None)
            return out

    cpu_learned.TreeMCTS = _TracingMCTS


def _step_to_json(step):
    """Python の箱コミット手順 → Rust の `Step` JSON。"""
    if isinstance(step, tuple) and len(step) == 3 and step[0] == "__box__":
        return {"kind": "box", "sig": [_jsonable(x) for x in step[1]], "left": int(step[2])}
    return {"kind": "sig", "sig": [_jsonable(x) for x in step]}


def _jsonable(x):
    return list(x) if isinstance(x, tuple) else x


def _commit_steps(engine, manager, name):
    """その決定点で有効なコミット（Python の `_commits` から読む・無ければ空）。"""
    hit = engine._commits.get(engine._commit_key(manager, name))
    if hit is None or hit[0]() is not manager:
        return []
    return [_step_to_json(s) for s in hit[1]]


def _decide_opts(args, engine, manager, name):
    opts = {"sims": args.sims, "commit": _commit_steps(engine, manager, name),
            "resact_pending": bool(engine._resact_pending)}
    for key, value in (("prune_futile", args.prune_futile), ("macro_moves", args.macro_moves),
                       ("defense_box", args.defense_box), ("don_margin", args.don_margin)):
        if value is not None:
            opts[key] = value
    return opts


def _restorable(hidden: dict) -> bool:
    """記録 v5 の `hidden` だけで盤面を完全に復元できるか。

    `hidden` は中断スタック・誘発待ち行列・ターン終了待ちを**件数でしか**持たない
    （`hidden_dict` の注記）ので、0 でない決定点は `prefix`（そこへ至る手順の再適用）が要る。
    """
    m = hidden.get("manager") or {}
    return not (m.get("interaction_depth") or m.get("pending_triggers")
                or m.get("pending_end_of_turn"))


class GlobalShuffleCounter:
    """`random.shuffle` の呼び出し回数を局のあいだ数え続ける（観測だけ）。

    `prefix` を再適用する経路（Rust）は自前の並びで混ぜるので、**基準点から今までの間に
    山札を混ぜていたら照合できない**。`ShuffleCounter` と違い局全体で開きっぱなしにする。
    """

    def __enter__(self):
        self._real = random.shuffle
        self.n = 0

        def wrapped(seq, *a, **kw):
            self._real(seq, *a, **kw)
            self.n += 1

        random.shuffle = wrapped
        return self

    def __exit__(self, *exc):
        random.shuffle = self._real
        return False


class DecideRunner:
    """1 局を打ちながら、各決定点で Python と Rust の decide を照合する。"""

    def __init__(self, db, args, totals: "Totals", firsts: list):
        self.db, self.args, self.totals, self.firsts = db, args, totals, firsts
        _install_mcts_probe()
        self.engine = _engine_class()(value_path=args.net)
        self.shuffles = None          # 局のあいだ開きっぱなしの `GlobalShuffleCounter`
        self.reset_game()

    def reset_game(self):
        """局の頭で基準点（復元できる決定点）と手順を空にする。"""
        self.base_hidden = None       # 完全に復元できる直近の `hidden`
        self.prefix = []              # そこから今までに打った手（`run_game` と同じ適用）
        self.base_shuffles = 0

    def seat(self, ctx):
        manager, player = ctx.manager, ctx.actor
        name = player.name
        totals = self.totals
        args = self.args
        hidden = hidden_dict(manager)
        shuffled_now = self.shuffles.n if self.shuffles is not None else 0
        if _restorable(hidden):
            # ここから復元できる＝基準点を更新して手順を空にする。
            self.base_hidden, self.prefix, self.base_shuffles = hidden, [], shuffled_now
        commit_in = _commit_steps(self.engine, manager, name)
        opts = _decide_opts(args, self.engine, manager, name)
        prefix = list(self.prefix)
        if prefix:
            opts["prefix"] = prefix
        sink = {"shuffles": [], "dirichlets": [], "uniforms": []}
        self.engine.rec = sink
        record = {}
        _MCTS_STATS.pop("last", None)
        try:
            with ShuffleCounter() as shuffles:
                move = self.engine.decide(manager, player, sims=args.sims, record=record)
        finally:
            self.engine.rec = None
        stats = _MCTS_STATS.pop("last", None)
        kind = record.get("kind")
        totals.bump("decisions")
        totals.bump(f"kind_{kind}")
        if commit_in:
            totals.bump("with_commit_in")
        if prefix:
            totals.bump("with_prefix")

        skip = None
        if self.base_hidden is None:
            skip = "unrestorable_skipped"     # 局頭から一度も復元できる点が無い（起こらない想定）
        elif shuffled_now != self.base_shuffles:
            # 基準点から今までのあいだに山札を混ぜた＝prefix の再適用で並びが食い違う。
            skip = "shuffle_skipped"
        elif shuffles.n:
            # 探索の中で山札を混ぜた＝Rust は自前の擬似乱数で混ぜる（§12.4-4）ので
            # 盤面が食い違って当然。どちらも「黙って一致」にしない。
            skip = "shuffle_skipped"
        elif len(prefix) > args.max_prefix:
            skip = "long_prefix_skipped"
        if skip is not None:
            totals.bump(skip)
        else:
            self._compare(self.base_hidden, name, opts, sink, move, kind, stats, record)
        self.prefix.append({"actor": name, "move": move})
        return move

    def _dump(self, hidden, name, opts, sink, move, kind, stats, perm) -> None:
        """不一致の**再現一式**を書き出す（`--dump-dir`）。

        `hidden`＋`prefix`＋出目＋Python の答えがあれば、あとから 1 決定点だけを
        両側で打ち直せる（100 局を回し直さずに原因を詰められる）。
        """
        out = self.args.dump_dir
        if not out:
            return
        _os.makedirs(out, exist_ok=True)
        import numpy as _np
        path = _os.path.join(out, f"mm_{_os.getpid()}_{len(_os.listdir(out))}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "hidden": hidden, "name": name, "opts": opts, "rng": sink,
                "python_kind": kind, "python_move": move,
                "python_N": ([float(x) for x in _np.asarray(stats["N"]).reshape(-1)]
                             if stats else None),
                "python_Q": ([float(x) for x in _np.asarray(stats["Q"]).reshape(-1)]
                             if stats else None),
                "python_legal": (list(stats["legal"]) if stats else None),
                "sorted_q_max_diff": perm,
                "sims": self.args.sims,
            }, f, ensure_ascii=False, default=str)

    def _compare(self, hidden, name, opts, sink, move, kind, stats, record):
        totals, firsts = self.totals, self.firsts
        if opcg_engine is None or not hasattr(opcg_engine, "decide"):
            totals.bump("unimplemented")
            return
        totals.bump("checks")
        try:
            got = json.loads(opcg_engine.decide(
                json.dumps(hidden, ensure_ascii=False, default=str), name,
                json.dumps(opts, ensure_ascii=False, default=str),
                json.dumps(sink, ensure_ascii=False)))
        except NotImplementedError as e:
            totals.bump("unimplemented")
            note_first(firsts, {"kind": "unimplemented", "detail": str(e)})
            return
        except ValueError as e:
            totals.bump("bad_payload")
            row = {"kind": "bad_payload", "decide_kind": kind, "detail": str(e),
                   "commit_in": opts.get("commit"), "rng": sink}
            note_first(firsts, row)
            note_loud(row)
            return

        if got.get("kind") != kind:
            totals.bump("mismatch")
            totals.bump("kind_mismatch")
            row = {"kind": "kind", "python": kind, "rust": got.get("kind"),
                   "commit_in": opts.get("commit"), "python_move": move,
                   "rust_move": got.get("move")}
            note_first(firsts, row)
            note_loud(row)
            return
        same_move = norm(move) == norm(got.get("move"))
        rs_pre_legal = (got.get("stats") or {}).get("legal")

        # 窓・コミットの決定点は手の一致だけを見る（訪問を配らないので N が無い）。
        if kind != "main":
            if same_move:
                totals.bump("match")
                return
            # 窓の根畳みは枝の**出口 value** の argmax。float32 の行列積は加算順が numpy と
            # 違う（`rs_net_oracle` の許容 1e-5）ので、同値の枝（同名カードの別実体など）で
            # 順位が割れうる。Rust 側の Q で「その差が許容以下＝同点」と確かめられたときだけ
            # 通し、`ties` として件数を報告する（§12.4-5 の同点規約）。
            tie = False
            rs_q = [float(x) for x in ((got.get("stats") or {}).get("Q") or [])]
            if kind == "window" and rs_q and rs_pre_legal:
                want = norm(move)
                idx = next((i for i, m in enumerate(rs_pre_legal) if norm(m) == want), None)
                best = max(range(len(rs_q)), key=lambda i: rs_q[i])
                tie = (idx is not None and idx < len(rs_q)
                       and abs(rs_q[idx] - rs_q[best]) <= self.args.tie_tol)
            if tie:
                totals.bump("match")
                totals.bump("ties")
                totals.bump("tie_window")
                return
            totals.bump("mismatch")
            totals.bump("move_mismatch")
            row = {"kind": "move", "decide_kind": kind, "python": move,
                   "rust": got.get("move"), "commit_in": opts.get("commit"),
                   "rust_Q": rs_q[:12],
                   "rust_legal": [m.get("action_type") for m in (rs_pre_legal or [])][:12]}
            note_first(firsts, row)
            note_loud(row)
            return

        rs = got.get("stats") or {}
        py_legal = list(stats["legal"]) if stats and stats.get("legal") is not None else []
        py_n = [float(x) for x in np.asarray(stats["N"]).reshape(-1)] if stats else []
        rs_legal = list(rs.get("legal") or [])
        rs_n = [float(x) for x in (rs.get("N") or [])]
        if norm(py_legal) != norm(rs_legal):
            totals.bump("mismatch")
            totals.bump("legal_mismatch")
            row = {"kind": "legal", "python_n": len(py_legal), "rust_n": len(rs_legal),
                   "path": first_diff(canon(py_legal), canon(rs_legal))}
            note_first(firsts, row)
            note_loud(row)
            return
        l1 = (sum(abs(a - b) for a, b in zip(py_n, rs_n))
              if len(py_n) == len(rs_n) else None)
        if same_move and l1 == 0.0:
            totals.bump("match")
            if len(py_legal) > 1:
                totals.bump("multi_choice")
            return
        # 同点で argmax／PUCT の順位が割れた: 訪問分布の L1 ≤ 2/sims なら通す（件数を報告）。
        if l1 is not None and l1 <= 2.0:
            totals.bump("match")
            totals.bump("ties")
            if not same_move:
                totals.bump("tie_moves")
            return
        totals.bump("mismatch")
        totals.bump("n_mismatch" if same_move else "move_mismatch")
        # **枝の入れ替わりか、値そのものの差か**を分ける（原因の性質が違う）:
        # Q を昇順に並べた「集合」が一致するなら、同じ枝の値が別の添字に付いただけ
        # ＝float の同点で PUCT の順位が割れた形。集合まで違うなら値が本当に違う。
        perm = None
        if len(py_q) == len(rs_q) and py_q:
            perm = max(abs(a - b) for a, b in zip(sorted(py_q), sorted(rs_q)))
            totals.bump("mismatch_permutation" if perm <= self.args.tie_tol
                        else "mismatch_values")
        self._dump(hidden, name, opts, sink, move, kind, stats, perm)
        py_q = [float(x) for x in np.asarray(stats["Q"]).reshape(-1)] if stats else []
        rs_q = [float(x) for x in (rs.get("Q") or [])]
        dq = (max((abs(a - b) for a, b in zip(py_q, rs_q)), default=0.0)
              if len(py_q) == len(rs_q) else None)
        totals.bump("n_mismatch_same_move", int(bool(same_move)))
        if l1 is not None:
            totals["l1_max"] = max(totals.get("l1_max", 0.0), l1)
        row = {
            "kind": "decide", "same_move": same_move, "l1": l1, "sims": self.args.sims,
            "max_abs_dq": dq,
            "python_move": move, "rust_move": got.get("move"),
            "python_N": py_n[:12], "rust_N": rs_n[:12],
            "python_Q": [round(x, 9) for x in py_q[:12]],
            "rust_Q": [round(x, 9) for x in rs_q[:12]],
            "legal": [m.get("action_type") for m in py_legal][:12],
            "sig": record.get("sig"),
            "sorted_q_max_diff": (max(abs(a - b) for a, b in zip(sorted(py_q), sorted(rs_q)))
                                  if len(py_q) == len(rs_q) and py_q else None),
        }
        note_first(firsts, row)
        note_loud(row)


def _play_decide_games(db, args, seeds, totals: "Totals", firsts: list) -> None:
    from harness.game_driver import run_game
    _LOUD["on"] = bool(args.verbose)
    runner = DecideRunner(db, args, totals, firsts)
    seat = {"p1": runner.seat, "p2": runner.seat}
    for seed in seeds:
        t0 = time.time()
        runner.reset_game()
        try:
            with GlobalShuffleCounter() as counter:
                runner.shuffles = counter
                run_game(seed, db, seats=seat, max_steps=args.max_steps,
                         stop_after_decisions=(args.max_decisions or None))
        except Exception as e:  # noqa: BLE001 - 局が落ちてもそこまでの決定点は使える
            totals.bump("game_aborted")
            note_first(firsts, {"kind": "game_aborted", "seed": seed,
                                "detail": f"{type(e).__name__}: {e}"})
        finally:
            runner.shuffles = None
        totals.bump("games_played")
        if args.verbose:
            print(f"[decide] seed={seed} decisions={totals.get('decisions', 0)} "
                  f"match={totals.get('match', 0)} mismatch={totals.get('mismatch', 0)} "
                  f"({time.time() - t0:.1f}s)", flush=True)


def _decide_worker(payload):
    """並列実行の 1 ワーカー（seed の部分集合を打って集計を返す）。"""
    args = argparse.Namespace(**payload["args"])
    db = load_db()
    if opcg_engine is not None and hasattr(opcg_engine, "load_masters"):
        opcg_engine.load_masters(payload["effects"])
        opcg_engine.load_net(args.net)
    MAX_REPORTED["n"] = max(1, int(getattr(args, "max_report", 1)))
    totals, firsts = Totals(), []
    _play_decide_games(db, args, payload["seeds"], totals, firsts)
    return {"totals": dict(totals), "firsts": firsts}


def run_decide(db, args, effects_path: str) -> int:
    """`--what decide` の本体（`run_one` と同じ形の 1 行を出す）。"""
    t0 = time.time()
    MAX_REPORTED["n"] = max(1, int(getattr(args, "max_report", 1)))
    seeds = [args.seed_base + i for i in range(args.games)]
    totals, firsts = Totals(), []
    if args.jobs > 1 and len(seeds) > 1:
        import concurrent.futures as cf
        shards = [seeds[i::args.jobs] for i in range(args.jobs)]
        payloads = [{"args": vars(args), "seeds": s, "effects": effects_path}
                    for s in shards if s]
        with cf.ProcessPoolExecutor(max_workers=len(payloads)) as pool:
            for out in pool.map(_decide_worker, payloads):
                for k, v in out["totals"].items():
                    totals.bump(k, v)
                for row in out["firsts"]:
                    note_first(firsts, row)
    else:
        if opcg_engine is not None and hasattr(opcg_engine, "load_net"):
            opcg_engine.load_net(args.net)
        _play_decide_games(db, args, seeds, totals, firsts)

    summary = {
        "what": "decide",
        "games": args.games,
        "games_played": totals.get("games_played", 0),
        "decisions": totals.get("decisions", 0),
        "checks": totals.get("checks", 0),
        "match": totals.get("match", 0),
        "mismatch": totals.get("mismatch", 0),
        "ties": totals.get("ties", 0),
        "tie_moves": totals.get("tie_moves", 0),
        "tie_window": totals.get("tie_window", 0),
        "move_mismatch": totals.get("move_mismatch", 0),
        "n_mismatch": totals.get("n_mismatch", 0),
        "n_mismatch_same_move": totals.get("n_mismatch_same_move", 0),
        "mismatch_permutation": totals.get("mismatch_permutation", 0),
        "mismatch_values": totals.get("mismatch_values", 0),
        "l1_max": totals.get("l1_max", 0.0),
        "legal_mismatch": totals.get("legal_mismatch", 0),
        "kind_mismatch": totals.get("kind_mismatch", 0),
        "shuffle_skipped": totals.get("shuffle_skipped", 0),
        "long_prefix_skipped": totals.get("long_prefix_skipped", 0),
        "unrestorable_skipped": totals.get("unrestorable_skipped", 0),
        "with_prefix": totals.get("with_prefix", 0),
        "unimplemented": totals.get("unimplemented", 0),
        "bad_payload": totals.get("bad_payload", 0),
        "harness_error": totals.get("harness_error", 0),
        "game_aborted": totals.get("game_aborted", 0),
        # 空振り検査: どの読み出し経路をどれだけ踏んだか（main=木／window=窓の根畳み／
        # commit=箱コミットの機械実行）と、候補が 2 つ以上あった決定点の数。
        "kind_main": totals.get("kind_main", 0),
        "kind_window": totals.get("kind_window", 0),
        "kind_commit": totals.get("kind_commit", 0),
        "with_commit_in": totals.get("with_commit_in", 0),
        "multi_choice": totals.get("multi_choice", 0),
        "sims": args.sims,
        "net": _os.path.basename(args.net),
        "seed_base": args.seed_base,
        "jobs": args.jobs,
        "engine": (opcg_engine.version() if opcg_engine is not None else None),
        "seconds": round(time.time() - t0, 1),
        "first": firsts[0] if firsts else None,
    }
    if len(firsts) > 1:
        summary["mismatches"] = firsts
    print("RS_SEARCH " + json.dumps(summary, ensure_ascii=False, default=str))
    bad = (totals.get("mismatch", 0) or totals.get("bad_payload", 0)
           or totals.get("unimplemented", 0) or totals.get("harness_error", 0)
           or not totals.get("checks", 0))
    return 1 if bad else 0


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
    ap.add_argument("--net", default=DEFAULT_NET,
                    help="decide が使う NRel の npz（既定 nrel_a1.npz＝出荷既定）")
    ap.add_argument("--sims", type=int, default=None,
                    help="decide の探索回数（既定＝config.SERVE_SIMS）")
    ap.add_argument("--max-decisions", type=int, default=0,
                    help="decide で 1 局あたりの決定点の上限（0=無制限）")
    ap.add_argument("--jobs", type=int, default=1,
                    help="decide を何プロセスに分けて打つか（既定 1）")
    ap.add_argument("--dump-dir", default=None,
                    help="decide の不一致の再現一式（hidden／prefix／出目／Python の答え）を書き出す先")
    ap.add_argument("--max-report", type=int, default=1,
                    help="decide で残す不一致の件数（既定 1・原因の性質を見るときは増やす）")
    ap.add_argument("--tie-tol", type=float, default=1e-5,
                    help="decide の窓で「同点」とみなす出口 value の差（既定 1e-5＝forward の許容）")
    ap.add_argument("--max-prefix", type=int, default=24,
                    help="decide で中断へ入り直す手順の上限（超えたら照合から外す・既定 24）")
    ap.add_argument("--verbose", action="store_true", help="局面ごとの進捗を出す")
    args = ap.parse_args(argv)
    if args.sims is None:
        from opcg_sim.src.learned.config import SERVE_SIMS
        args.sims = SERVE_SIMS

    whats = [w.strip() for w in args.what.split(",") if w.strip()]
    unknown = [w for w in whats if w not in CHECKERS and w != "decide"]
    if unknown:
        ap.error(f"unknown --what: {unknown} (choose from {list(WHATS)})")

    db = load_db()
    effects_path = ensure_effects_json(args.effects, db)
    if opcg_engine is not None and hasattr(opcg_engine, "load_masters"):
        n = opcg_engine.load_masters(effects_path)
        print(f"[effects] load_masters({effects_path}) -> {n} cards")

    if whats == ["decide"]:
        return run_decide(db, args, effects_path)

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
        if what == "decide":
            rc |= run_decide(db, args, effects_path)
        else:
            rc |= run_one(what, db, boards, args, effects_path)
    return rc


if __name__ == "__main__":
    sys.exit(main())
