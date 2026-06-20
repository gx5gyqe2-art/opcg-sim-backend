"""方式B（PyPy 探索ワーカー）の不変ゲート。

「探索を別プロセス/別ランタイムへオフロードしても**手選択・盤面が変わらない**」ことを機械照合する。
- pickle round-trip: GameManager を pickle→unpickle した盤面が、同じ decide で**同一手**を返す。
- profile/plan/mem の往復: 採点補助オブジェクトと turn-memory が pickle を跨いで一致する。
- ブリッジ等価: decide_client.decide（worker 無効時のインプロセス経路）が cpu_ai.decide_guarded と完全同値。

実 PyPy ワーカー起動を要しない（pickle 等価＝オフロードの健全性の核心を CPython 内で固定）。
実 IPC のスモークは別途（pypy3 が要るため CI ではスキップ）。
"""
import os
import pickle
import random

import pytest

os.environ.setdefault("OPCG_LOG_SILENT", "1")

from opcg_sim.src.core.gamestate import GameManager, Player
from opcg_sim.src.core import action_api, cpu_ai

try:
    from cpu_selfplay import _load_db, build_deck
except ImportError:  # tests/ を sys.path に載せて再試行
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from cpu_selfplay import _load_db, build_deck

KEY_PID = action_api.CONST.get('PENDING_REQUEST_PROPERTIES', {}).get('PLAYER_ID', 'player_id')


def _midgame(seed=7, steps=12, difficulty="hard"):
    """決定論セルフプレイで mid-game の GameManager を作る（CPU の手番で止める）。"""
    random.seed(seed)
    db = _load_db()
    l1, c1 = build_deck(db, "p1")
    l2, c2 = build_deck(db, "p2")
    mgr = GameManager(Player("p1", c1, l1), Player("p2", c2, l2))
    mgr.start_game()
    mem = {}
    for _ in range(steps):
        if mgr.winner:
            break
        pend = mgr.get_pending_request()
        if not pend:
            break
        pid = pend[KEY_PID]
        actor = mgr.p1 if mgr.p1.name == pid else mgr.p2
        mv = cpu_ai.decide_guarded(mgr, actor, difficulty, mem=mem.setdefault(pid, {}))
        if mv is None:
            break
        mgr.action_events = []
        if mv["kind"] == "battle":
            action_api.apply_battle_action(mgr, actor, mv["action_type"], mv.get("card_uuid"))
        else:
            action_api.apply_game_action(mgr, actor, mv["action_type"], mv.get("payload", {}))
    return mgr


def _decide_key(move):
    if move is None:
        return None
    return (move.get("kind"), move.get("action_type"), move.get("card_uuid"),
            repr(move.get("payload")))


def test_gamemanager_pickle_roundtrips():
    """GameManager は __getstate__ 無しで pickle round-trip する（方式B の IPC 前提）。"""
    mgr = _midgame()
    blob = pickle.dumps(mgr, protocol=pickle.HIGHEST_PROTOCOL)
    mgr2 = pickle.loads(blob)
    assert mgr2 is not None
    # 盤面の主要不変量が一致（ターン・勝者・両者の手札/ライフ/場の枚数）。
    assert mgr2.turn_count == mgr.turn_count
    assert mgr2.winner == mgr.winner
    for a, b in ((mgr.p1, mgr2.p1), (mgr.p2, mgr2.p2)):
        assert len(b.hand) == len(a.hand)
        assert len(b.life) == len(a.life)
        assert len(b.field) == len(a.field)


def test_decide_same_move_after_pickle_roundtrip():
    """pickle 前後の盤面で、同一 RNG 状態の decide が同一手を返す（オフロードの核心）。"""
    mgr = _midgame()
    pid = mgr.get_pending_request()[KEY_PID]

    state = random.getstate()
    rng_a = random.Random(); rng_a.setstate(state)
    rng_b = random.Random(); rng_b.setstate(state)

    a = mgr.p1 if pid == "p1" else mgr.p2
    move_orig = cpu_ai.decide_guarded(mgr, a, "hard", rng=rng_a, mem={})

    mgr2 = pickle.loads(pickle.dumps(mgr, protocol=pickle.HIGHEST_PROTOCOL))
    b = mgr2.p1 if pid == "p1" else mgr2.p2
    move_round = cpu_ai.decide_guarded(mgr2, b, "hard", rng=rng_b, mem={})

    assert _decide_key(move_orig) == _decide_key(move_round)


def test_profile_plan_mem_roundtrip():
    """profile/plan/mem（採点補助・turn-memory）が pickle を跨いで等価に往復する。"""
    import cpu_arena
    random.seed(7)
    db = _load_db()
    l1, c1 = build_deck(db, "p1")
    plan = cpu_arena._plan_for("hard", l1, c1)  # PlanProfile（None もあり得る）
    mem = {"acted": 3, "repeat": {"X": 2}}
    payload = (plan, mem)
    plan2, mem2 = pickle.loads(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
    assert mem2 == mem
    # plan は dataclass 等。pickle 往復で型・主要属性が保たれる（None なら None）。
    assert (plan is None) == (plan2 is None)
    if plan is not None:
        assert type(plan2) is type(plan)


def test_bridge_inprocess_equals_decide_guarded():
    """decide_client.decide（worker 無効＝インプロセス経路）は cpu_ai.decide_guarded と完全同値。"""
    os.environ["OPCG_PYPY_WORKER"] = "0"
    import importlib
    from opcg_sim.api import decide_client
    importlib.reload(decide_client)  # USE_WORKER をモジュールロード時に確定するため

    mgr = _midgame()
    pid = mgr.get_pending_request()[KEY_PID]
    actor = mgr.p1 if pid == "p1" else mgr.p2

    state = random.getstate()
    random.setstate(state)
    move_ref = cpu_ai.decide_guarded(mgr, actor, "hard", mem={})
    random.setstate(state)
    move_bridge = decide_client.decide(mgr, actor, "hard", mem={})

    assert _decide_key(move_ref) == _decide_key(move_bridge)
