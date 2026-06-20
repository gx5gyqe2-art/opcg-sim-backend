"""② 自明手の即答（OPCG_TRIVIAL_FASTPATH）の健全性。

既定 OFF＝従来挙動。ON 時は 1-ply で best が圧倒する自明局面で深掘りを省いて即答するが、
返す手は必ず合法・対局は正常に完走する（ショートカットが壊れないこと）。
"""
import random
import conftest  # noqa: F401
import pytest

from opcg_sim.src.core import cpu_ai, action_api
from opcg_sim.src.core.gamestate import GameManager, Player
from cpu_selfplay import build_deck, _load_db


@pytest.fixture(scope="module")
def db():
    return _load_db()


def test_trivial_fastpath_off_by_default():
    """既定では無効（OPCG_TRIVIAL_FASTPATH 未設定）＝従来挙動と完全同値。"""
    assert cpu_ai._TRIVIAL_FASTPATH is False


def test_trivial_fastpath_returns_legal_and_fires(db):
    """フラグ ON＋小マージンで、即答が発火（collect['trivial']）しつつ合法手で完走する。"""
    random.seed(0)
    l1, c1 = build_deck(db, "p1")
    l2, c2 = build_deck(db, "p2")
    m = GameManager(Player("p1", c1, l1), Player("p2", c2, l2))
    m.start_game()
    mem = {"p1": {}, "p2": {}}
    orig_flag, orig_margin = cpu_ai._TRIVIAL_FASTPATH, cpu_ai._TRIVIAL_MARGIN
    cpu_ai._TRIVIAL_FASTPATH = True
    cpu_ai._TRIVIAL_MARGIN = 1.0  # ほぼ常に発火させて機構の健全性を見る
    fired = {"n": 0}
    try:
        steps = 0
        while m.winner is None and steps < 250:
            pa = m.pending_actor_action()
            if not pa:
                break
            actor = cpu_ai._player_by_name(m, pa[0])
            legal = m.get_legal_actions(actor)
            tr = {}
            mv = cpu_ai.decide(m, actor, "hard", random.Random(steps), trace=tr)
            assert mv is not None
            assert cpu_ai._move_sig(mv) in {cpu_ai._move_sig(x) for x in legal}
            steps += 1
            m.action_events = []
            if mv.get("kind") == "battle":
                action_api.apply_battle_action(m, actor, mv["action_type"], mv.get("card_uuid"))
            else:
                action_api.apply_game_action(m, actor, mv["action_type"], mv.get("payload", {}))
    finally:
        cpu_ai._TRIVIAL_FASTPATH, cpu_ai._TRIVIAL_MARGIN = orig_flag, orig_margin
    assert m.winner is not None
