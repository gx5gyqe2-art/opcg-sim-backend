"""D族＝防御支配ペア生成器の判定規約（P4-b・2026-08-24）。

`tests/scripts/plandef_gen.py` の純関数部を固定する:
  - battle_need: 攻撃が atk≥def で通る規約（need = atk−def+1000・止まっていれば 0）
  - d_family: D1（総量不足→素通しが任意の支払いを支配）／D2（最小で守れて余りがある→
    最小札組が +1枚 を支配）／必要0や札なしでは対を立てない
"""
import types

import pytest

import _bootstrap  # noqa: F401

import plandef_gen as PD

pytestmark = pytest.mark.cpu_infra


def _c(uuid, counter):
    return types.SimpleNamespace(uuid=uuid, current_counter=counter)


def _unit(power):
    return types.SimpleNamespace(get_power=lambda attacking, _p=power: _p)


def _mgr(atk, tgt, hand, buff=0):
    bat = {"attacker": _unit(atk), "target": _unit(tgt), "counter_buff": buff}
    p1 = types.SimpleNamespace(name="p1", hand=list(hand))
    return types.SimpleNamespace(p1=p1, p2=types.SimpleNamespace(name="p2"),
                                 active_battle=bat)


def test_battle_need_arithmetic():
    assert PD.battle_need(_mgr(6000, 5000, []), "p1") == 2000   # 6000 vs 5000 → 2000で7000
    assert PD.battle_need(_mgr(5000, 5000, []), "p1") == 1000   # 同値は通る → 1000
    assert PD.battle_need(_mgr(4000, 5000, []), "p1") == 0      # 止まっている
    assert PD.battle_need(_mgr(6000, 5000, [], buff=1000), "p1") == 1000  # 既払い分を考慮


def test_d1_when_total_insufficient():
    m = _mgr(9000, 5000, [_c("a", 1000), _c("b", 2000)])        # need=5000 > 総量3000
    fam = PD.d_family(m, "p1")
    assert fam == [("D1", [], ["b"])]                           # 素通し vs 最大1枚


def test_d2_when_min_save_has_leftover():
    m = _mgr(6000, 5000, [_c("a", 2000), _c("b", 1000)])        # need=2000 → 最小={a}・余り=b
    fam = PD.d_family(m, "p1")
    assert fam == [("D2", ["a"], ["a", "b"])]


def test_no_family_when_need_zero_or_exact():
    assert PD.d_family(_mgr(4000, 5000, [_c("a", 2000)]), "p1") == []     # 止まっている
    m = _mgr(6000, 5000, [_c("a", 2000)])                       # 最小={a}・余りなし
    assert PD.d_family(m, "p1") == []
    assert PD.d_family(_mgr(9000, 5000, []), "p1") == []        # 札なし


def test_counter_cards_sorted_and_filtered():
    cs = PD.counter_cards([_c("x", 0), _c("y", 2000), _c("z", 1000)])
    assert cs == [(2000, "y"), (1000, "z")]
