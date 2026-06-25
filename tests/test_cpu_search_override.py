"""探索深さ／ply 上限の per-decide オーバーライド（`set_search_override`）の不変条件テスト。

L1 外の伸びしろ（探索深さの A/B）を測るための席別オーバーライド。確認するのは「機械が正しく動くこと」:
  - getter（`_effective_horizon`/`_effective_max_ply`）が override を反映し、None で既定へ戻る。
  - **不活性時（未設定）は従来と完全同値**＝本番/テストに無影響（set→reset しても決定不変）。
  - override 下でも decide が合法手を返し例外を出さない（深い探索パスが壊れていない）。
強さの主張はしない（それはアリーナ A/B＝`tests/depth_arena.py` の仕事）。
"""
import random

import conftest  # noqa: F401
import pytest

from opcg_sim.src.core import cpu_ai
from opcg_sim.src.core import action_api
from opcg_sim.src.core.gamestate import GameManager, Player
from cpu_selfplay import build_deck, _load_db


@pytest.fixture(scope="module")
def db():
    return _load_db()


def _apply(gm, actor, move):
    if move["kind"] == "battle":
        action_api.apply_battle_action(gm, actor, move["action_type"], move.get("card_uuid"))
    else:
        action_api.apply_game_action(gm, actor, move["action_type"], move.get("payload", {}))


def _midgame(db, seed=0, steps=24):
    """数手進めた途中局面を決定論的に作る（葉でなくルートに選択肢がある局面が欲しい）。"""
    random.seed(seed)
    l1, c1 = build_deck(db, "p1")
    l2, c2 = build_deck(db, "p2")
    gm = GameManager(Player("p1", c1, l1), Player("p2", c2, l2))
    gm.start_game()
    KEY_PID = action_api.CONST.get('PENDING_REQUEST_PROPERTIES', {}).get('PLAYER_ID', 'player_id')
    mem = {}
    n = 0
    while gm.winner is None and n < steps:
        pend = gm.get_pending_request()
        if not pend:
            break
        pid = pend[KEY_PID]
        actor = gm.p1 if gm.p1.name == pid else gm.p2
        mv = cpu_ai.decide_guarded(gm, actor, "hard", random, mem)
        if mv is None:
            break
        gm.action_events = []
        _apply(gm, actor, mv)
        n += 1
    return gm


def test_override_getters_reflect_and_reset():
    cpu_ai.set_search_override(None, None)
    assert cpu_ai._effective_horizon() == cpu_ai.HARD_HORIZON
    assert cpu_ai._effective_max_ply() == cpu_ai.HARD_MAX_PLY
    cpu_ai.set_search_override(5, 65)
    assert cpu_ai._effective_horizon() == 5
    assert cpu_ai._effective_max_ply() == 65
    cpu_ai.set_search_override(None, None)   # 必ず既定へ戻す（他テスト汚染防止）
    assert cpu_ai._effective_horizon() == cpu_ai.HARD_HORIZON
    assert cpu_ai._effective_max_ply() == cpu_ai.HARD_MAX_PLY


def test_override_clamps_to_minimum():
    cpu_ai.set_search_override(0, 0)
    assert cpu_ai._effective_horizon() >= 1
    assert cpu_ai._effective_max_ply() >= 1
    cpu_ai.set_search_override(None, None)


def test_inactive_override_is_identical(db):
    """未設定（reset 後）の decide は、override 機構に一切触れない decide と同一手＝本番無影響。"""
    gm = _midgame(db, seed=3)
    pend = gm.get_pending_request()
    if not pend:
        pytest.skip("no pending decision at midgame")
    pid = pend.get("player_id", "p1")
    actor = gm.p1 if pid == "p1" else gm.p2

    mv_plain = cpu_ai.decide(gm, actor, "hard", random.Random(7))
    cpu_ai.set_search_override(5, 65)
    cpu_ai.set_search_override(None, None)   # set→reset：機構を通っても既定なら同一であるべき
    mv_reset = cpu_ai.decide(gm, actor, "hard", random.Random(7))
    assert cpu_ai._move_sig(mv_plain) == cpu_ai._move_sig(mv_reset)


def test_deep_override_returns_legal_move(db):
    """深い override 下でも decide が合法手を返し例外を出さない（深い探索パスの健全性）。"""
    gm = _midgame(db, seed=5)
    pend = gm.get_pending_request()
    if not pend:
        pytest.skip("no pending decision at midgame")
    pid = pend.get("player_id", "p1")
    actor = gm.p1 if pid == "p1" else gm.p2
    legal = gm.get_legal_actions(actor)
    legal_sigs = {cpu_ai._move_sig(m) for m in legal}

    cpu_ai.set_budget_override(450)
    cpu_ai.set_search_override(5, 65)
    try:
        mv = cpu_ai.decide(gm, actor, "hard", random.Random(11))
    finally:
        cpu_ai.set_search_override(None, None)
        cpu_ai.set_budget_override(None)
    assert mv is not None
    # 選択ノード展開で合法手の sig 集合に含まれる（または単一手 forced）。
    assert cpu_ai._move_sig(mv) in legal_sigs or len(legal) == 1
