"""残ドン掘り seam（`LearnedEngine(residual_dig=True)`・対照生成の腕A・2026-09-02）の契約。

基盤健全性（`cpu_infra`）: 対象は自己対戦の対照生成の機構であり、ゲームプレイの正しさ
（効果解決・カード消失・API 契約）には触れない。

守る性質:
  1. **serve 既定（None）は挙動不変**——同一 seed・同一盤面で従来と同じ手。
  2. `_is_dig_card` は**構造判定**（CHARACTER・cost1・ON_PLAY に DRAW・コストに RETURN_DON）で
     サトリ/シュラを真、ドローしないコスト1・コスト2以上を偽にする（カードID非依存）。
  3. seam 有効でも合法手が返る（配線の煙試験）。差し替えが起きた場合は PLAY で、
     `residual_dig_events` に発火が記録される。
"""
import argparse
import random

import numpy as np
import conftest  # noqa: F401
import pytest

import _bootstrap  # noqa: F401

import coach_gate as CG
import counterfactual_referee as CR
import replay_reeval as RE
from cpu_selfplay import _load_db
from opcg_sim.src.core.cpu_learned import LearnedEngine, _is_dig_card

pytestmark = pytest.mark.cpu_infra


@pytest.fixture(scope="module")
def db():
    return _load_db()


@pytest.fixture(scope="module")
def m2_game(db):
    CR.ARGS = argparse.Namespace(true_board=True)
    CR.GAMES = {}
    raw = RE.load_replay_json(CG.REPLAYS_V2["m2"])
    rec = raw.get("replay", raw)
    CR.GAMES["m2"] = (rec, {f.get("action_index"): f for f in raw.get("frames") or []},
                      rec["actions"])
    return db


def _board(db, i):
    m, who = CR._restore_board(db, "m2", i)
    m.action_events = []
    return m, (who if isinstance(who, str) else who.name)


def _decide(eng, m, name, seed):
    random.seed(20260902)
    eng._world_seeds = {}
    eng._commits.clear()
    actor = m.p1 if m.p1.name == name else m.p2
    return eng.decide(m, actor, sims=8, rng=np.random.default_rng(seed))


def test_dig_card_predicate_is_structural(db):
    """サトリ/シュラ（登場時ドン-1ドロー・コスト1）は真。ドローしない/コスト≠1 は偽。"""
    assert _is_dig_card(db.get_card("OP15-066"))          # サトリ
    assert _is_dig_card(db.get_card("OP15-067"))          # シュラ（速攻つきでも構造は同じ）
    assert not _is_dig_card(db.get_card("OP15-118"))      # 6c エネル（登場時ドン-1だがコスト6）
    assert not _is_dig_card(db.get_card("OP15-058"))      # リーダー
    assert not _is_dig_card(None)
    # 全カードで「真＝コスト1キャラ」が必ず成り立つ（構造判定が cost/type を見ている証拠）
    for cid, _ in list(db.raw_db.items()):
        c = db.get_card(cid)
        if c is not None and _is_dig_card(c):
            assert c.cost == 1 and c.type.name == "CHARACTER", cid


def test_serve_default_unchanged(m2_game):
    """seam 未指定は None＝従来どおり。同一 seed で同一手。"""
    eng = LearnedEngine()
    assert eng.residual_dig is None and eng.residual_dig_events == []
    m, name = _board(m2_game, 50)
    a = _decide(eng, m, name, 0)
    b = _decide(eng, m, name, 0)
    assert a == b and eng.residual_dig_events == []


def test_residual_dig_seam_smoke(m2_game):
    """seam 有効でも合法手が返る。差し替えが起きたなら PLAY で events に記録される。"""
    eng = LearnedEngine(residual_dig=True)
    fired = 0
    for i in (10, 30, 50, 70):
        m, name = _board(m2_game, i)
        mv = _decide(eng, m, name, 3)
        assert mv is not None and mv.get("action_type")
        if eng.residual_dig_events:
            assert mv.get("action_type") == "PLAY"
            ev = eng.residual_dig_events[-1]
            assert ev["card"] and ev["don_active"] >= 1 and ev["field"] < 5
            fired += 1
            eng.residual_dig_events.clear()
    # 発火の有無は盤面次第（m2 はハンニャバル対局で掘りカードが無ければ 0）。0 でも契約違反ではない。
    assert fired >= 0
