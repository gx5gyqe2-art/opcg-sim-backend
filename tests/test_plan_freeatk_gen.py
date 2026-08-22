"""V8 フリー攻撃族生成器の判定規約（系統3・2026-08-22）。

`tests/scripts/plan_freeatk_gen.py` の適用条件と、本族のために追加した
`plan._attack_candidates`（ATTACK 意図の対象指定・3要素形）を固定する:
  - free_kill_target: 自リーダーがアクティブ、相手に「レスト済み・パワー≤リーダー」の
    キャラがいて、相手にアクティブなブロッカーがいない時だけ (リーダー, 最大の獲物)
  - _attack_candidates: 攻撃者 uuid で絞り、対象 uuid 指定時は target_ids[0] も一致
"""
import types

import pytest

import _bootstrap  # noqa: F401

import plan_freeatk_gen as PF
from opcg_sim.src.learned import plan as PL

pytestmark = pytest.mark.cpu_infra


def _card(uuid, power=5000, rest=False, blocker=False):
    return types.SimpleNamespace(
        uuid=uuid, is_rest=rest,
        get_power=lambda attacking, _p=power: _p,
        has_keyword=lambda k, _b=blocker: (k == "ブロッカー" and _b))


def _mgr(lead, my_field=(), opp_field=()):
    p1 = types.SimpleNamespace(name="p1", leader=lead, field=list(my_field))
    p2 = types.SimpleNamespace(name="p2", leader=_card("OL"), field=list(opp_field))
    return types.SimpleNamespace(p1=p1, p2=p2)


def test_free_kill_picks_biggest_rested_at_or_below_leader_power():
    m = _mgr(_card("L", 6000), opp_field=[
        _card("small", 3000, rest=True), _card("big", 6000, rest=True),
        _card("over", 7000, rest=True), _card("active", 5000, rest=False)])
    lead, tgt = PF.free_kill_target(m, "p1")
    assert lead.uuid == "L" and tgt.uuid == "big"   # 同値は取れる・格上と非レストは対象外


def test_free_kill_none_when_leader_rested_or_no_prey():
    assert PF.free_kill_target(_mgr(_card("L", 6000, rest=True),
                                    opp_field=[_card("x", 3000, rest=True)]), "p1") is None
    assert PF.free_kill_target(_mgr(_card("L", 6000),
                                    opp_field=[_card("x", 7000, rest=True)]), "p1") is None
    assert PF.free_kill_target(_mgr(None), "p1") is None


def test_free_kill_none_when_active_blocker_present():
    m = _mgr(_card("L", 6000), opp_field=[
        _card("prey", 3000, rest=True), _card("blk", 4000, rest=False, blocker=True)])
    assert PF.free_kill_target(m, "p1") is None
    # ブロッカーがレスト済みなら阻止できない＝対象になる/対は立つ
    m2 = _mgr(_card("L", 6000), opp_field=[
        _card("prey", 3000, rest=True), _card("blk", 4000, rest=True, blocker=True)])
    lead, tgt = PF.free_kill_target(m2, "p1")
    assert tgt.uuid == "blk"                        # レストブロッカー4000 > prey3000


def _atk(uuid, tgt):
    return {"action_type": "ATTACK", "payload": {"uuid": uuid, "target_ids": [tgt]}}


def test_attack_candidates_filters_by_attacker_and_target():
    legal = [_atk("L", "a"), _atk("L", "b"), _atk("c1", "a"),
             {"action_type": "TURN_END", "payload": {}}]
    assert [m["payload"]["target_ids"][0]
            for m in PL._attack_candidates(legal, "L")] == ["a", "b"]
    assert [m["payload"]["target_ids"][0]
            for m in PL._attack_candidates(legal, "L", "b")] == ["b"]
    assert PL._attack_candidates(legal, "L", "zzz") == []
