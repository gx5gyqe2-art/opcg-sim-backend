"""CPU 検証ハーネスの共通対局ドライバ（設計⑥ `docs/refactoring_harness_driver.md`）。

`cpu_selfplay` / `cpu_replay` / `cpu_arena` に散在していた「同型の対局ループ」を 1 箇所へ集約する。
各ハーネスは **席（seat）＝意思決定** と **observer＝観測** を差し込むだけになり、新しい計器の追加コストが
observer 1 個へ下がる。エンジンのループ接点（pending request 種別・action_api 適用規約）が変わったときの
追従点も 1 箇所に集まる。

決定論契約（最重要・`docs/...` §2.5）:
  同一 seed → 完全同一対局が本スイートの生命線（凍結ベースライン Elo・リプレイ種・挙動ベースラインが依存）。
  ドライバは乱数を従来どおり **global `random`** に集約し、**乱数消費点の順序を旧経路と一致**させる:
    `random.seed(seed)` → `deck_builder` → `start_game()` → 各 seat の decide。
  `get_legal_actions` の事前呼び出し有無（`legal_moves`）も旧経路に合わせて保存する
  （「乱数を消費しないはず」という仮定には依拠しない）。observer は観測専用で manager を変更しない。

共有部品（旧 `cpu_selfplay` からの移設。既存 import は `cpu_selfplay` の再エクスポートで互換維持）:
  `load_db` / `build_deck` / `InvariantError` / `DEFAULT_MAX_STEPS` / `choose_move` /
  `_zone_snapshot` / `_snapshot_diff`。
"""
import json
import os
import random
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple

import os as _os, sys as _sys  # noqa: E402  test bootstrap (sys.path + google stub)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap  # noqa: E402,F401

from opcg_sim.src.core.gamestate import GameManager, Player
from opcg_sim.src.core import action_api, cpu_ai
from opcg_sim.src.core.invariants import check_invariants, check_turn_boundary
from opcg_sim.src.models.models import CardInstance
from opcg_sim.src.utils.loader import CardLoader

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    "opcg_sim", "data")

# 1 ゲーム当たりの安全上限（無限ループ検出）。
DEFAULT_MAX_STEPS = 4000


class InvariantError(Exception):
    """インバリアント違反で対局を即停止する例外。"""

    def __init__(self, violations, step, trace_tail):
        self.violations = violations
        self.step = step
        self.trace_tail = trace_tail
        super().__init__(f"invariant violation(s) at step {step}: {violations}")

    def __reduce__(self):
        # multiprocessing でワーカー→親へ転送できるよう picklable に（既定は args=(message,) で復元不能）。
        return (InvariantError, (self.violations, self.step, self.trace_tail))


# --- カード DB / デッキ構築 --------------------------------------------------

def load_db() -> CardLoader:
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


def leader_deck_builder(p1_leader: Optional[str] = None, p2_leader: Optional[str] = None):
    """`run_game(deck_builder=…)` 用: 固定（または既定）リーダーで両者のデッキを組む builder を返す。

    `random.seed(seed)` の直後・`start_game()` の前に呼ばれる（旧 run_one_game/run_replay と同位置）。
    build_deck は raw_db を決定論走査するだけで global random を消費しない。
    """
    def _build(db, seed):
        l1, c1 = build_deck(db, "p1", p1_leader)
        l2, c2 = build_deck(db, "p2", p2_leader)
        return l1, c1, l2, c2
    return _build


# --- 盤面スナップショット（効果検証の oracle 用） ----------------------------

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


# --- 席（seat）＝意思決定関数 ------------------------------------------------

def choose_move(manager: GameManager, moves: List[Dict[str, Any]]) -> Dict[str, Any]:
    """既定方策: ランダム合法手（global random で決定論的）。

    無限引き延ばし防止のため、攻撃可能なら一定確率で攻撃を優先し、
    そうでなければ TURN_END にやや寄せる軽いバイアスを掛ける。
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


def make_seat(difficulty: str = "hard", *, kind: str = "ai", mem: Optional[Dict] = None,
              want_trace: bool = False,
              info_policy: str = cpu_ai.DEFAULT_INFO_POLICY, policy_rng=None,
              pimc_worlds: int = 1, budget=None, search=None, coeffs=None,
              sims: int = 160, engine=None):
    """1 席ぶんの意思決定関数 `seat(ctx) -> move` を返す。

    kind:
      'random'  — `choose_move`（軽バイアス付きランダム・高速。cpu_selfplay の既定）。
      'ai'      — `cpu_ai.decide_guarded`（評価関数ベース。cpu_selfplay --policy ai / cpu_replay 用）。
                  `want_trace` 指定時のみ trace を採り `ctx.trace` へ書く（進行不変・観測専用）。
      'arena'   — `decide_guarded` に席別の情報方針/CRN rng/PIMC/予算/深さ/L1係数を掛ける（cpu_arena 用）。
                  呼び出しごとに一時オーバーライドを適用→finally で既定へ戻す（単一スレッド前提）。
      'learned' — Gen2 学習型・NN誘導MCTS（本番既定 CPU）。numpy 必須なので遅延 import。rng は global random
                  由来（PR-D2）＝run_game の seed で決定論再生できる。`want_trace` 時は MCTS root 統計
                  （訪問%・Q値・L1第二意見）を `ctx.trace` へ書く。`engine`（`cpu_learned.LearnedEngine`）を渡すと
                  **そのネット**で決める＝net-vs-net（新Gen vs 凍結Gen2・A3）用。未指定は出荷 Gen2 既定エンジン。
    """
    mem = mem if mem is not None else {}

    if kind == "random":
        def _random(ctx):
            return choose_move(ctx.manager, ctx.moves)
        return _random

    if kind == "learned":
        from opcg_sim.src.core import cpu_learned

        def _learned(ctx):
            tr = ctx.trace if want_trace else None
            if engine is not None:
                return engine.decide(ctx.manager, ctx.actor, sims=sims, trace=tr)
            return cpu_learned.decide_learned(ctx.manager, ctx.actor, sims=sims, trace=tr)
        return _learned

    if kind == "ai":
        def _ai(ctx):
            tr = ctx.trace if want_trace else None
            return cpu_ai.decide_guarded(ctx.manager, ctx.actor, difficulty, random, mem, trace=tr)
        return _ai

    if kind == "arena":
        prng = policy_rng if policy_rng is not None else random
        s_horizon, s_max_ply = (search if search is not None else (None, None))

        def _arena(ctx):
            cpu_ai.set_budget_override(budget)
            cpu_ai.set_search_override(s_horizon, s_max_ply)
            _apply_v2_coeffs(coeffs)
            try:
                return cpu_ai.decide_guarded(ctx.manager, ctx.actor, difficulty, prng, mem,
                                             info_policy=info_policy, pimc_worlds=pimc_worlds)
            finally:
                cpu_ai.set_budget_override(None)
                cpu_ai.set_search_override(None, None)
                _apply_v2_coeffs(None)
        return _arena

    raise ValueError(f"unknown seat kind: {kind!r}")


# L1（cpu_eval_v2）係数の席別上書き用: 出荷時の既定値を一度だけ退避し、各 decide 前に
# 「既定へ戻す→その席の coeffs を適用」する（席別 A/B＝SPSA で候補θ vs 凍結基準を測る）。
_V2_DEFAULTS: Optional[Dict[str, float]] = None


def _v2_defaults() -> Dict[str, float]:
    global _V2_DEFAULTS
    if _V2_DEFAULTS is None:
        from opcg_sim.src.core import cpu_eval_v2
        _V2_DEFAULTS = {k: getattr(cpu_eval_v2, k) for k in dir(cpu_eval_v2)
                        if k.startswith("V2_") and isinstance(getattr(cpu_eval_v2, k), (int, float))}
    return _V2_DEFAULTS


def _apply_v2_coeffs(coeffs: Optional[Dict[str, float]]) -> None:
    """L1 係数を「既定へリセット→coeffs を上書き」。coeffs=None なら既定のまま。arena seat 専用。"""
    from opcg_sim.src.core import cpu_eval_v2
    base = _v2_defaults()
    for k, v in base.items():
        setattr(cpu_eval_v2, k, v)
    if coeffs:
        for k, v in coeffs.items():
            setattr(cpu_eval_v2, k, v)


# --- observer コンテキスト ---------------------------------------------------

class StepCtx:
    """observer / seat へ渡す読み取り窓（1 意思決定点のスナップショット）。

    observer は manager を変更しない（決定論契約）。`trace` は当該 decision 用のスクラッチ dict で、
    seat が診断情報を書き、observer が読む（trace 有効時のみ・進行に影響しない）。
    """
    __slots__ = ("manager", "step", "turn", "phase", "actor", "pending", "moves", "trace")

    def __init__(self, manager: GameManager):
        self.manager = manager
        self.step = 0
        self.turn = manager.turn_count
        self.phase = manager.phase.name
        self.actor = None
        self.pending = None
        self.moves = None
        self.trace: Dict[str, Any] = {}

    def _update(self, step, actor, pending, moves):
        self.step = step
        self.turn = self.manager.turn_count
        self.phase = self.manager.phase.name
        self.actor = actor
        self.pending = pending
        self.moves = moves
        self.trace = {}


def _notify(observers, hook: str, *args):
    for ob in observers:
        fn = getattr(ob, hook, None)
        if fn is not None:
            fn(*args)


class GameResult:
    """1 局の結果サマリ。"""
    __slots__ = ("seed", "winner", "steps", "turns", "p1_leader", "p2_leader", "decisions")

    def __init__(self, seed, winner, steps, turns, p1_leader, p2_leader, decisions):
        self.seed = seed
        self.winner = winner
        self.steps = steps
        self.turns = turns
        self.p1_leader = p1_leader
        self.p2_leader = p2_leader
        self.decisions = decisions


# --- 統一対局ループ ----------------------------------------------------------

def run_game(seed: int, db, *, seats: Dict[str, Callable],
             deck_builder: Optional[Callable] = None,
             observers: Tuple = (),
             max_steps: int = DEFAULT_MAX_STEPS,
             legal_moves: str = "check",
             invariants: str = "raise",
             stop_after_decisions: Optional[int] = None,
             trace_tail: Optional[List] = None,
             first_player: Optional[str] = None) -> Optional[GameResult]:
    """1 局を決定論的に進める統一ループ。observer で観測し、seat で意思決定する。

    seats: {"p1": seat_fn, "p2": seat_fn}（`seat_fn(ctx) -> move`）。
    deck_builder: `(db, seed) -> (l1, c1, l2, c2)`。既定は `leader_deck_builder()`（既定リーダー）。
    legal_moves: "check"=各手番で `get_legal_actions` を事前取得し NO_LEGAL_MOVE を検査（cpu_selfplay/cpu_replay。
                 `ctx.moves` に格納）。"skip"=事前取得せず seat 内で解決（cpu_arena。乱数消費順の保存）。
    invariants: "raise"=違反/例外/スタックで `InvariantError`（selfplay/replay/play_game）。
                "skip"=検査せず・スタック/None で break・apply 例外は素通し（regret/realize トレース）。
    stop_after_decisions: 指定数の意思決定で打ち切り（決着前なら winner=None のまま返す＝有界化）。
    trace_tail: InvariantError に載せる直近ログの共有リスト（observer が emit で充填・省略時は内部生成）。
    """
    if deck_builder is None:
        deck_builder = leader_deck_builder()
    if trace_tail is None:
        trace_tail = []

    random.seed(seed)
    built = deck_builder(db, seed)
    if built is None:
        return None
    l1, c1, l2, c2 = built
    if not l1 or not l2:
        if invariants == "skip":
            return None
        raise RuntimeError("リーダーを含むデッキを構築できませんでした。")

    manager = GameManager(Player("p1", c1, l1), Player("p2", c2, l2))
    # first_player の再現（実対局リプレイ）: "random"＝コイントス（global random を 1 消費＝API と同順）／
    # "p1"/"p2"＝明示（消費なし）／None＝既定（`start_game()` 相当・既存ハーネスは全てこれ＝挙動不変）。
    # デッキ復元は乱数を消費しないので、seed 直後のこの位置が API の coin toss と同じ乱数位置。
    if first_player == "random":
        fp = random.choice([manager.p1, manager.p2])
        manager.start_game(fp)
    elif first_player in ("p1", "p2"):
        manager.start_game(manager.p1 if first_player == "p1" else manager.p2)
    else:
        manager.start_game()

    pending_props = action_api.CONST.get("PENDING_REQUEST_PROPERTIES", {})
    KEY_PID = pending_props.get("PLAYER_ID", "player_id")

    ctx = StepCtx(manager)
    _notify(observers, "on_start", ctx)

    n_decisions = 0
    step = 0
    prev_turn = manager.turn_count

    while manager.winner is None and step < max_steps:
        pending = manager.get_pending_request()
        if not pending:
            if invariants == "skip":
                break
            raise InvariantError([("STUCK", "no pending request and no winner")], step, trace_tail)
        req_pid = pending[KEY_PID]
        actor = manager.p1 if manager.p1.name == req_pid else manager.p2

        moves = None
        if legal_moves == "check":
            moves = manager.get_legal_actions(actor)
            if not moves:
                if invariants == "skip":
                    break
                raise InvariantError([("NO_LEGAL_MOVE", f"no legal moves for {req_pid}")], step, trace_tail)

        ctx._update(step, actor, pending, moves)
        _notify(observers, "on_decision_point", ctx)

        move = seats[req_pid](ctx)
        if move is None:
            if invariants == "skip":
                break
            raise InvariantError([("NO_LEGAL_MOVE", f"no move for {req_pid}")], step, trace_tail)

        n_decisions += 1
        _notify(observers, "on_decision", ctx, move)

        manager.action_events = []
        try:
            if move["kind"] == "battle":
                action_api.apply_battle_action(manager, actor, move["action_type"], move.get("card_uuid"))
            else:
                action_api.apply_game_action(manager, actor, move["action_type"], move.get("payload", {}))
        except Exception as e:
            if invariants == "skip":
                raise  # regret/realize は apply 例外を素通しする（旧挙動）
            raise InvariantError(
                [("ACTION_EXCEPTION", f"{type(e).__name__}: {e}\n{traceback.format_exc()}")],
                step, trace_tail,
            )

        events = list(manager.action_events)
        _notify(observers, "on_step", ctx, move, events)

        if invariants == "raise":
            violations = check_invariants(manager)
            if manager.turn_count != prev_turn:
                violations += check_turn_boundary(manager)
                prev_turn = manager.turn_count
            if violations:
                raise InvariantError(violations, step, trace_tail)

        step += 1
        if stop_after_decisions is not None and n_decisions >= stop_after_decisions:
            break

    if manager.winner is None and stop_after_decisions is None and invariants == "raise":
        raise InvariantError([("MAX_STEPS", f"game did not finish within {max_steps} steps")], step, trace_tail)

    result = GameResult(seed, manager.winner, step, manager.turn_count,
                        l1.master.card_id, l2.master.card_id, [])
    _notify(observers, "on_end", ctx, result)
    return result
