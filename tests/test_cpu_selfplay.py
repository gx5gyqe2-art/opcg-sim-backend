"""CPU 対 CPU 自己対戦ハーネスのスモーク/不変条件テスト（PR1 基盤）。

docs/CPU_BATTLE_PLAN.md §5 の検証項目:
  - CPU 対 CPU を seed 固定で完走（クラッシュ/無限ループ無し・必ず決着）。
  - clone() が本番状態を破壊しない。
  - get_legal_actions の生成手が適用可能（_validate_action を通る）。
  - インバリアント検出が既知の破れを検出する。
"""
import copy
import random

import conftest  # noqa: F401  (google スタブ注入 & sys.path 設定)
import pytest

from opcg_sim.src.core.gamestate import GameManager, Player, FIELD_LIMIT
from opcg_sim.src.core import action_api
from opcg_sim.src.core.invariants import check_invariants
from cpu_selfplay import build_deck, run_one_game, _load_db, InvariantError


@pytest.fixture(scope="module")
def db():
    return _load_db()


@pytest.mark.parametrize("seed", list(range(8)))
def test_selfplay_finishes_cleanly(db, seed):
    """各 seed の CPU 対 CPU が決着し、途中でインバリアント違反を起こさない。"""
    res = run_one_game(seed, db, max_steps=4000)
    assert res["winner"] in ("p1", "p2")
    assert res["steps"] > 0


def test_selfplay_is_deterministic(db):
    """同一 seed は同一の結果（勝者/手数/ターン数）を再現する。"""
    a = run_one_game(3, db, max_steps=4000)
    b = run_one_game(3, db, max_steps=4000)
    assert a == b


def test_clone_does_not_mutate_original(db):
    """clone() で得たコピー上の進行は本体に影響しない。"""
    random.seed(0)
    l1, c1 = build_deck(db, "p1")
    l2, c2 = build_deck(db, "p2")
    gm = GameManager(Player("p1", c1, l1), Player("p2", c2, l2))
    gm.start_game()

    before = (gm.turn_count, gm.phase.name, len(gm.p1.hand), len(gm.p2.hand),
              gm.winner, (gm.get_pending_request() or {}).get("player_id"))

    clone = gm.clone()
    assert clone is not gm
    # クローン上でマリガンを進める
    pending = clone.get_pending_request()
    actor = clone.p1 if clone.p1.name == pending["player_id"] else clone.p2
    clone.action_events = []
    action_api.apply_game_action(clone, actor, "KEEP_HAND", {})

    after = (gm.turn_count, gm.phase.name, len(gm.p1.hand), len(gm.p2.hand),
             gm.winner, (gm.get_pending_request() or {}).get("player_id"))
    assert before == after, "clone 上の操作が本体を変化させた"


def test_legal_moves_are_applicable(db):
    """get_legal_actions の各手が apply_*（_validate_action 含む）で例外なく適用できる。

    クローン上で各候補手を1手だけ適用し、ValueError 等が出ないことを確認する。
    """
    random.seed(1)
    l1, c1 = build_deck(db, "p1")
    l2, c2 = build_deck(db, "p2")
    gm = GameManager(Player("p1", c1, l1), Player("p2", c2, l2))
    gm.start_game()

    # マリガンを抜けてメインフェイズへ。
    for _ in range(2):
        pending = gm.get_pending_request()
        actor = gm.p1 if gm.p1.name == pending["player_id"] else gm.p2
        gm.action_events = []
        action_api.apply_game_action(gm, actor, "KEEP_HAND", {})

    pending = gm.get_pending_request()
    actor = gm.p1 if gm.p1.name == pending["player_id"] else gm.p2
    moves = gm.get_legal_actions(actor)
    assert moves, "メインフェイズで合法手が空"

    for move in moves:
        trial = gm.clone()
        t_actor = trial.p1 if trial.p1.name == actor.name else trial.p2
        trial.action_events = []
        # 例外を投げないこと（TURN_END/PLAY/ATTACK/ATTACH_DON/ACTIVATE_MAIN）。
        if move["kind"] == "battle":
            action_api.apply_battle_action(trial, t_actor, move["action_type"], move.get("card_uuid"))
        else:
            action_api.apply_game_action(trial, t_actor, move["action_type"], move.get("payload", {}))


def test_invariant_detects_injected_violation(db):
    """インバリアント検出が、人為的に壊した盤面（場 6 体・対話なし）を検出する。"""
    random.seed(0)
    l1, c1 = build_deck(db, "p1")
    l2, c2 = build_deck(db, "p2")
    gm = GameManager(Player("p1", c1, l1), Player("p2", c2, l2))
    gm.start_game()

    assert check_invariants(gm) == [], "初期状態でインバリアント違反"

    # 場に上限超過のキャラを直接積む（対話なし）→ FIELD_LIMIT 違反になるはず。
    from opcg_sim.src.models.models import CardInstance
    char_master = next(c for c in (db.get_card(cid) for cid in db.raw_db) if c and c.type.name == "CHARACTER")
    gm.p1.field = [CardInstance(char_master, "p1") for _ in range(FIELD_LIMIT + 1)]
    gm.active_interaction = None
    codes = {v[0] for v in check_invariants(gm)}
    assert "FIELD_LIMIT" in codes


def test_invariant_detects_don_violation(db):
    """ドン!! 総数が 10 から崩れたら DON_CONSERVATION を検出する。"""
    random.seed(0)
    l1, c1 = build_deck(db, "p1")
    l2, c2 = build_deck(db, "p2")
    gm = GameManager(Player("p1", c1, l1), Player("p2", c2, l2))
    gm.start_game()
    gm.p1.don_deck.pop()  # 1 枚消す → 合計 9
    codes = {v[0] for v in check_invariants(gm)}
    assert "DON_CONSERVATION" in codes
