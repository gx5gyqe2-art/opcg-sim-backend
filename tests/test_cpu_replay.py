"""CPU 思考トレース＋決定論リプレイ（Phase 1）のテスト。

検証項目:
  - トレースは観測専用: `decide` に trace を渡しても渡さなくても**同じ手**を返す（挙動不変）。
  - トレース構築は RNG 中立: trace 有無でグローバル random の状態が変わらない（進行を分岐させない）。
  - 決定論再現: 同一 seed の `run_replay` は同一の決定列（card_id 基準）を再現する。
  - トレースの内容: 多候補の意思決定で chosen/candidates/regret/j_components/read_ahead が揃う。

速度方針: フル対局は重い（normal は ~1 手/秒＋読み筋クローン）ため、
  - 単発の意思決定検証は 1 局面のみ（normal）。
  - 局面到達と再現性比較は easy（1-ply・高速）＋少手数で有界化する。
"""
import random
from collections import Counter

import conftest  # noqa: F401  (google スタブ注入 & sys.path 設定)
import pytest

from opcg_sim.src.core.gamestate import GameManager, Player
from opcg_sim.src.core import action_api, cpu_ai
from cpu_selfplay import build_deck, _load_db
from cpu_replay import run_replay


@pytest.fixture(scope="module")
def db():
    return _load_db()


def _new_game(db, seed=0):
    random.seed(seed)
    l1, c1 = build_deck(db, "p1")
    l2, c2 = build_deck(db, "p2")
    gm = GameManager(Player("p1", c1, l1), Player("p2", c2, l2))
    gm.start_game()
    return gm


def _advance_to_multi_choice(gm, max_steps=200):
    """高速方策（easy）で進め、MAIN_ACTION かつ合法手が複数ある意思決定点で止めて返す。"""
    mem = {"p1": {}, "p2": {}}
    for _ in range(max_steps):
        pending = gm.get_pending_request()
        if not pending:
            return None
        pid = pending["player_id"]
        actor = gm.p1 if gm.p1.name == pid else gm.p2
        moves = gm.get_legal_actions(actor)
        if pending.get("action") == "MAIN_ACTION" and len(moves) > 1:
            return actor
        move = cpu_ai.decide_guarded(gm, actor, "easy", random, mem.setdefault(pid, {}))
        gm.action_events = []
        if move["kind"] == "battle":
            action_api.apply_battle_action(gm, actor, move["action_type"], move.get("card_uuid"))
        else:
            action_api.apply_game_action(gm, actor, move["action_type"], move.get("payload", {}))
    return None


def test_trace_does_not_change_decision(db):
    """trace を渡しても返る手は trace 無しと同一（観測専用＝挙動不変）。"""
    gm = _new_game(db, seed=2)
    actor = _advance_to_multi_choice(gm)
    assert actor is not None, "複数候補の MAIN_ACTION 局面に到達できなかった"

    move_plain = cpu_ai.decide(gm, actor, "normal", random.Random(0))
    tr = {}
    move_traced = cpu_ai.decide(gm, actor, "normal", random.Random(0), trace=tr)
    assert cpu_ai._move_sig(move_plain) == cpu_ai._move_sig(move_traced)


def test_trace_is_rng_neutral(db):
    """trace は探索ぶん以上にグローバル random を消費しない（trace 有無で進行が分岐しない）。

    探索（クローン上の効果解決）はグローバル random を消費し得るが、それは trace 有無で同一。
    trace 構築（追加クローン）は getstate/setstate で中立化されるので、両者の最終状態は一致する。
    """
    gm = _new_game(db, seed=4)
    actor = _advance_to_multi_choice(gm)
    assert actor is not None

    random.seed(123)
    cpu_ai.decide(gm, actor, "normal", random.Random(0))            # trace 無し
    state_plain = random.getstate()
    random.seed(123)
    cpu_ai.decide(gm, actor, "normal", random.Random(0), trace={})  # trace 有り
    assert random.getstate() == state_plain, "trace 構築がグローバル random を余分に消費した"


def test_trace_has_expected_fields(db):
    """多候補の意思決定で、要求した 4 項目（候補スコア/regret/J値成分/読み筋）が揃う。"""
    gm = _new_game(db, seed=2)
    actor = _advance_to_multi_choice(gm)
    assert actor is not None

    tr = {}
    cpu_ai.decide(gm, actor, "normal", random.Random(0), trace=tr)
    assert tr.get("chosen") and "action_type" in tr["chosen"]
    assert isinstance(tr.get("candidates"), list) and 1 <= len(tr["candidates"]) <= cpu_ai.TRACE_TOPN
    assert "regret" in tr and tr["regret"] >= 0.0
    assert "j_components" in tr and "total" in tr["j_components"]
    assert "me" in tr["j_components"] and "opp" in tr["j_components"]
    # 候補は chosen を含む。
    sigs = {(c["move"] or {}).get("action_type") for c in tr["candidates"]}
    assert tr["chosen"]["action_type"] in sigs

    # 読み筋 PV: 有界（max_steps 以内）で、同一の繰り返し手が REPEAT_CAP を超えて並ばない。
    ra = tr.get("read_ahead") or []
    assert isinstance(ra, list) and len(ra) <= 12
    rep = Counter()
    for entry in ra:
        mv = entry.get("move") or {}
        if mv.get("action_type") in ("ACTIVATE_MAIN", "ATTACH_DON"):
            rep[(mv.get("action_type"), mv.get("card"), tuple(mv.get("targets") or []))] += 1
    assert all(v <= cpu_ai.REPEAT_CAP for v in rep.values()), f"PV に繰り返し膨張: {rep}"


def test_replay_is_deterministic(db):
    """同一 seed の run_replay は同一の決定列（card_id 基準）を再現する（有界・部分再生・easy）。"""
    a = run_replay(5, db, p1_difficulty="easy", p2_difficulty="easy", stop_after_decisions=20)
    b = run_replay(5, db, p1_difficulty="easy", p2_difficulty="easy", stop_after_decisions=20)
    assert a["decisions"] == b["decisions"]
    assert len(a["decisions"]) == 20
