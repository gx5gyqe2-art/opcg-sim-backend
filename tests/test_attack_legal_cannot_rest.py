"""CANNOT_REST と攻撃列挙の整合（bb0 発見の欠陥B・2026-08-11）。

**欠陥**: 「レストにできない」継続効果（PREVENT_REST → timed_flags "CANNOT_REST"）を
持つカードの攻撃が**合法手列挙に載る**のに、`declare_attack`（engine/battle.py）は
ValueError で拒否する——列挙と検証の不整合。実害は探索の例外手封印（dead_child）の
汚染と apply 失敗（bb0 実測: seed880014 ほか2局で APPLY_NONE）。

**修正**: 列挙（gamestate の攻撃者収集）が検証側と同じ `timed_flags` の CANNOT_REST を
見る（リーダー・キャラの両方）。

**固定する性質**: 列挙された全 ATTACK は declare_attack を通る（=適用可能）。
"""
import conftest  # noqa: F401

import pytest

from engine_helpers import make_game, make_master
from opcg_sim.src.models.models import CardInstance, CardType
from opcg_sim.src.models.enums import Phase


def _setup(gm, p1, p2):
    p1.leader = CardInstance(make_master(card_id="T-L01", type=CardType.LEADER,
                                         power=5000, life=5), p1.name)
    p2.leader = CardInstance(make_master(card_id="T-L02", type=CardType.LEADER,
                                         power=5000, life=5), p2.name)
    gm.turn_count = 5                      # 双方の初回ターン制限を外す
    gm.turn_player = p1
    gm.phase = Phase.MAIN                  # MAIN_ACTION pending の導出条件
    c = CardInstance(make_master(card_id="T-C01", cost=3, power=5000), p1.name)
    c.is_newly_played = False
    p1.field.append(c)
    return c


def _attack_cards(gm, p):
    moves = gm.get_legal_actions(p)
    out = []
    for mv in moves:
        if mv.get("action_type") == "ATTACK":
            out.append(mv["payload"]["uuid"])
    return out


def test_cannot_rest_character_not_enumerated():
    gm, p1, p2 = make_game()
    c = _setup(gm, p1, p2)
    assert c.uuid in _attack_cards(gm, p1), "対照: 通常時は攻撃が列挙されるはず"
    gm.continuous.apply(c, "FLAG", "THIS_TURN", flag="CANNOT_REST")
    assert "CANNOT_REST" in c.timed_flags
    assert c.uuid not in _attack_cards(gm, p1), "CANNOT_REST の攻撃が列挙に残っている（欠陥B）"


def test_cannot_rest_leader_not_enumerated():
    gm, p1, p2 = make_game()
    _setup(gm, p1, p2)
    assert p1.leader.uuid in _attack_cards(gm, p1)
    gm.continuous.apply(p1.leader, "FLAG", "THIS_TURN", flag="CANNOT_REST")
    assert p1.leader.uuid not in _attack_cards(gm, p1)


def test_enumerated_attacks_pass_declare_attack():
    """整合の直接検査: 列挙された全 ATTACK は declare_attack が受理する。"""
    gm, p1, p2 = make_game()
    c = _setup(gm, p1, p2)
    gm.continuous.apply(c, "FLAG", "THIS_TURN", flag="CANNOT_REST")
    moves = gm.get_legal_actions(p1)
    by_uuid = {p1.leader.uuid: p1.leader, c.uuid: c,
               p2.leader.uuid: p2.leader}
    for mv in moves:
        if mv.get("action_type") != "ATTACK":
            continue
        atk = by_uuid[mv["payload"]["uuid"]]
        tgt = by_uuid[mv["payload"]["target_ids"][0]]
        g2 = gm.clone()
        a2 = next(x for x in (g2.p1.leader, g2.p2.leader, *g2.p1.field, *g2.p2.field)
                  if x.uuid == atk.uuid)
        t2 = next(x for x in (g2.p1.leader, g2.p2.leader, *g2.p1.field, *g2.p2.field)
                  if x.uuid == tgt.uuid)
        try:
            g2.declare_attack(a2, t2)
        except ValueError as e:
            pytest.fail(f"列挙された攻撃を declare_attack が拒否: {e}")
