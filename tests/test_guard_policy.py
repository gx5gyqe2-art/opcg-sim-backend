"""受け方針箱（マクロ手化 P6-c・2026-08-25）の契約。

`guard.shape_moves`（方針は SELECT_COUNTER/SELECT_BLOCKER にだけ作用・効果/対話は常に残す・
空にはしない）と `guard.select_guard_policy`（台本比較・CRN・平坦なら local）、エンジン統合
（相手ターンの防御窓で方針が候補整形として効く・既定 OFF は挙動不変）を固定する。
"""
import argparse
import types

import numpy as np
import conftest  # noqa: F401
import pytest

import _bootstrap  # noqa: F401

import coach_gate as CG
import counterfactual_referee as CR
import replay_reeval as RE
from cpu_selfplay import _load_db
from opcg_sim.src.core.cpu_learned import LearnedEngine
from opcg_sim.src.learned import guard as GD

pytestmark = pytest.mark.cpu_infra


def _unit(power):
    return types.SimpleNamespace(get_power=lambda attacking, _p=power: _p)


def _c(uuid, counter):
    return types.SimpleNamespace(uuid=uuid, current_counter=counter)


def _mgr(atk, tgt, hand):
    bat = {"attacker": _unit(atk), "target": _unit(tgt), "counter_buff": 0}
    p1 = types.SimpleNamespace(name="p1", hand=list(hand))
    return types.SimpleNamespace(p1=p1, p2=types.SimpleNamespace(name="p2"),
                                 active_battle=bat)


def _sc(uuid):
    return {"kind": "battle", "action_type": "SELECT_COUNTER", "card_uuid": uuid}


_PASS = {"kind": "battle", "action_type": "PASS"}
_BLK = {"kind": "battle", "action_type": "SELECT_BLOCKER", "card_uuid": "b"}
_EVT = {"kind": "battle", "action_type": "ACTIVATE_MAIN", "card_uuid": "e"}


def test_shape_pass_drops_hand_and_board_costs_only():
    m = _mgr(6000, 5000, [_c("a", 2000)])
    out = GD.shape_moves("pass", m, "p1", [_sc("a"), _BLK, _EVT, _PASS])
    assert out == [_EVT, _PASS]            # カウンター/ブロッカーだけ落ちる（効果は残る）
    assert GD.shape_moves("local", m, "p1", [_sc("a"), _PASS]) == [_sc("a"), _PASS]


def test_shape_minimal_keeps_smallest_counter():
    m = _mgr(6000, 5000, [_c("a", 2000), _c("b", 1000)])       # need=2000
    out = GD.shape_moves("minimal", m, "p1", [_sc("a"), _sc("b"), _PASS])
    assert out == [_sc("b"), _PASS]        # 最小の印字1枚＋素通しだけ残る
    m0 = _mgr(4000, 5000, [_c("a", 2000)])                      # need=0＝止まっている
    assert GD.shape_moves("minimal", m0, "p1", [_sc("a"), _PASS]) == [_PASS]


def test_shape_hold_drops_pass_only_when_savable():
    m = _mgr(6000, 5000, [_c("a", 2000), _c("b", 1000)])       # 総量3000 ≥ need=2000
    out = GD.shape_moves("hold", m, "p1", [_sc("a"), _sc("b"), _PASS])
    assert out == [_sc("a"), _sc("b")]     # 守れるなら素通しを落とす
    m2 = _mgr(9000, 5000, [_c("a", 1000)])                      # 総量不足＝落とさない
    moves = [_sc("a"), _PASS]
    assert GD.shape_moves("hold", m2, "p1", moves) == moves


def test_shape_mixed_window_untouched():
    m = _mgr(6000, 5000, [_c("a", 0)])                          # 印字0＝算術で閉じない
    moves = [_sc("a"), _PASS]
    assert GD.shape_moves("minimal", m, "p1", moves) == moves
    assert GD.shape_moves("hold", m, "p1", moves) == moves


@pytest.fixture(scope="module")
def m2_58():
    CR.ARGS = argparse.Namespace(true_board=True)
    CR.GAMES = {}
    raw = RE.load_replay_json(CG.REPLAYS_V2["m2"])
    rec = raw.get("replay", raw)
    CR.GAMES["m2"] = (rec, {f.get("action_index"): f for f in raw.get("frames") or []},
                      rec["actions"])
    db = _load_db()
    m, who = CR._restore_board(db, "m2", 58)
    return m, (who if isinstance(who, str) else who.name)


def test_select_guard_policy_on_real_board(m2_58):
    m, name = m2_58
    from opcg_sim.src.core import cpu_learned as CL
    eng = LearnedEngine()
    vf = CL._value_fn(eng.vnet, eng.vocab, eng.enc_version)
    pf = CL._priors_fn(eng.pnet, eng.vocab, eng.enc_version)
    world = eng.game.determinize(m, name, np.random.default_rng(11))
    pol, diag = GD.select_guard_policy(eng.game, world, name, vf, pf,
                                       battle_value_fn=eng._battle_value_fn())
    assert pol in GD.POLICIES
    assert set(diag.get("scores", {})) == set(GD.POLICIES)


def test_engine_decide_with_guard_policy_returns_legal(m2_58):
    m, name = m2_58
    actor = m.p1 if m.p1.name == name else m.p2
    on = LearnedEngine(guard_policy=True)
    off = LearnedEngine()
    mv = on.decide(m, actor, sims=8, rng=np.random.default_rng(1))
    legal = off.game.legal_actions(m)
    assert any(mv == x for x in legal)     # 方針経由でも出力は従来の合法手
    from opcg_sim.src.learned import config as C
    assert C.SERVE_GUARD_POLICY is False   # 既定 OFF
