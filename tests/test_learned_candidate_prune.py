"""learned MCTS 候補生成の無駄手枝刈り（adapter.OPCGGame.legal_actions・v5 §4-1補）の単体検証。

L1/α-β は `_prune_don_moves`/`_prune_futile_attacks` で無駄攻撃・無意味なドン付与をルート候補から
除外するが、learned MCTS の候補生成（`merged_search_actions`）は従来これを掛けておらず、net が
無駄手に visit を割いて選ぶ実害があった（v4 実測マーク @19/@102/@38）。adapter が同じ枝刈りを
適用することを、決定的な合成盤面で確認する。SERVE_PRUNE_FUTILE=False で従来（枝刈り無し）に戻る。
"""
import pytest

import conftest  # noqa: F401
from cpu_selfplay import build_deck, _load_db
from opcg_sim.src.core.gamestate import GameManager, Player
from opcg_sim.src.core import cpu_ai
from opcg_sim.src.learned.adapter import OPCGGame
from opcg_sim.src.learned import config as CFG

pytestmark = pytest.mark.cpu_infra   # 基盤健全性（探索候補生成の機構）


def _game(seed=1):
    import random
    random.seed(seed)
    db = _load_db()
    l1, c1 = build_deck(db, "p1"); l2, c2 = build_deck(db, "p2")
    m = GameManager(Player("p1", c1, l1), Player("p2", c2, l2)); m.start_game()
    return m


def _keys(m, moves):
    return {(cpu_ai._describe_move(m, x) or {}).get("action_type") for x in moves}


def test_adapter_applies_futile_prune_and_gate():
    """adapter.legal_actions の候補は、同一盤面での _prune_* 適用後と一致する（ON）。
    SERVE_PRUNE_FUTILE=False では merged_search_actions 素の集合に戻る（ゲート）。"""
    game = OPCGGame()
    # 探索を数手進めて攻撃/付与が候補に出る局面を作る（決定論・seed 固定）。
    m = _game(3)
    for _ in range(12):
        name = m.pending_actor_action()[0] if m.pending_actor_action() else None
        if name is None or m.winner is not None:
            break
        actor = m.p1 if m.p1.name == name else m.p2
        legal = m.get_legal_actions(actor)
        if not legal:
            break
        cpu_ai._apply_move_inplace(m, name, legal[0])

    name = m.pending_actor_action()[0] if m.pending_actor_action() else None
    if name is None:
        pytest.skip("盤面が終局（合成局面の都合）")

    base = m.get_legal_actions(m.p1 if m.p1.name == name else m.p2)
    merged = cpu_ai.merged_search_actions(m, name, base)
    expect = cpu_ai._prune_futile_attacks(m, name, cpu_ai._prune_don_moves(m, name, list(merged)))

    old = CFG.SERVE_PRUNE_FUTILE
    try:
        CFG.SERVE_PRUNE_FUTILE = True
        on = game.legal_actions(m)
        assert len(on) == len(expect), "枝刈りON が _prune_* 適用後と件数不一致"
        CFG.SERVE_PRUNE_FUTILE = False
        off = game.legal_actions(m)
        assert len(off) == len(merged), "枝刈りOFF が merged 素集合に戻らない（ゲート破れ）"
        # TURN_END は常に残る（枝刈りで手詰まりにしない）。
        assert "TURN_END" in _keys(m, on) or "TURN_END" not in _keys(m, merged)
    finally:
        CFG.SERVE_PRUNE_FUTILE = old


def test_prune_never_empties_candidates():
    """枝刈りは候補を空にしない（最低 TURN_END 等が残る）＝decide が None にならない前提。"""
    game = OPCGGame()
    m = _game(5)
    name = m.pending_actor_action()[0]
    assert len(game.legal_actions(m)) >= 1
