"""学習型CPU本番配線の検証（CI内）: 合法手を返す・routing・フォールバック・**符号化ドリフト検知**。

配線は tests/ の学習コードを opcg_sim/src/learned へ**忠実コピー**した単一ソース。コピーが訓練時の
符号化と一致し続けることを保証する（ドリフトすると net にゴミ入力＝サイレント劣化）。
"""
import numpy as np

import conftest  # noqa: F401
import rl_encoder as TEST_E          # 訓練時の符号化（tests側）
import opcg_action as TEST_A
from cpu_selfplay import build_deck, _load_db
from opcg_sim.src.core.gamestate import GameManager, Player
from opcg_sim.src.core import cpu_learned
from opcg_sim.src.learned import encoder as PROD_E
from opcg_sim.src.learned.action import legal_action_matrix as prod_lam


def _game(seed=1):
    import random
    random.seed(seed)
    db = _load_db()
    l1, c1 = build_deck(db, "p1"); l2, c2 = build_deck(db, "p2")
    m = GameManager(Player("p1", c1, l1), Player("p2", c2, l2)); m.start_game()
    return m


def _actor(m):
    name = m.pending_actor_action()[0]
    return name, (m.p1 if m.p1.name == name else m.p2)


def test_available_and_decides_legal_move():
    assert cpu_learned.available(), "Gen2 重みが同梱されていない"
    m = _game(2); name, actor = _actor(m)
    legal = m.get_legal_actions(actor)
    mv = cpu_learned.decide_learned(m, actor, sims=30)
    assert mv in legal, "学習型CPUが合法手を返さない"


def test_decide_client_routes_learned():
    from opcg_sim.api import decide_client
    m = _game(3); name, actor = _actor(m)
    legal = m.get_legal_actions(actor)
    mv = decide_client.decide(m, actor, "learned", mem={})
    assert mv in legal, "decide_client 経由の learned が合法手を返さない"


def test_learned_only_no_l1_fallback():
    """learned-only: decide/plan_segment とも常に学習型が手を返す（L1へ落ちない）。"""
    from opcg_sim.api import decide_client
    m = _game(4); name, actor = _actor(m)
    legal = m.get_legal_actions(actor)
    mv = decide_client.decide(m, actor, "learned", mem={})
    assert mv in legal, "learned が合法手を返さない"
    seg = decide_client.plan_segment(m, actor, "learned", mem={})
    assert isinstance(seg, list) and (not seg or seg[0] in legal), "plan_segment(learned) が不正"


def test_encoder_no_drift():
    """製品コピーの符号化が訓練時(tests)と厳密一致＝netへ同じ入力を与える。"""
    m = _game(5); name, _ = _actor(m)
    vocab_t = TEST_E.build_vocab(_load_db())
    vocab_p = PROD_E.build_vocab(_load_db())
    et = TEST_E.encode(m, name, vocab_t)
    ep = PROD_E.encode(m, name, vocab_p)
    for k in ("scalars", "field", "card_idx"):
        assert np.array_equal(et[k], ep[k]), f"encoder ドリフト: {k}"


def test_action_features_no_drift():
    m = _game(6); name, actor = _actor(m)
    legal = m.get_legal_actions(actor)
    at = TEST_A.legal_action_matrix(m, legal, name)
    ap = prod_lam(m, legal, name)
    assert np.array_equal(at, ap), "action 符号化ドリフト"
