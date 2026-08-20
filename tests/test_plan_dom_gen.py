"""P1/P2 支配ペア生成器の判定規約（系統3・A段・2026-08-20）。

`tests/scripts/plan_dom_gen.py` の純関数部を固定する:
  - 死に付与先の判定（V1=素の死に／V1s=【ドン!!×N】閾値達成済み・V1'）
  - 「相手のターン」常在（チョッパー型）と閾値未達（ドレーク1枚目）は死にではない
  - 支配ペアの構成（V1系: 差はドンの置き場所のみ／V4: 差は付与と攻撃の順序のみ）
"""
import types

import pytest

import _bootstrap  # noqa: F401

import plan_dom_gen as PD

pytestmark = pytest.mark.cpu_infra


def _card(uuid, text="", rest=True, attached=0, new=False, cost=3):
    return types.SimpleNamespace(
        uuid=uuid, is_rest=rest, is_newly_played=new, attached_don=attached,
        has_keyword=lambda k: False,
        master=types.SimpleNamespace(cost=cost, card_id=uuid, effect_text=text))


def _player(field=(), don=3, leader=None):
    return types.SimpleNamespace(name="p1", field=list(field), don_active=[1] * don,
                                 leader=leader, hand=[])


DRAKE = "【ドン!!×1】【自分のターン中】このキャラがレストの場合、リーダー+1000。"
CHOPPER = "【ドン!!×2】相手のターン中、このキャラのパワー+2000。"


def test_dead_plain_rested_vanilla():
    p = _player(field=[_card("v", text="", rest=True)])
    assert [(c.uuid, t) for c, t in PD.dead_targets(p)] == [("v", "V1")]


def test_not_dead_when_active_or_opponent_turn_text():
    p = _player(field=[_card("a", text="", rest=False),          # アクティブ＝対象外
                       _card("c", text=CHOPPER, rest=True, attached=2)])  # 相手ターン常在
    assert PD.dead_targets(p) == []


def test_v1s_threshold_satisfied_is_dead_but_first_attach_is_not():
    drake0 = _card("d0", text=DRAKE, rest=True, attached=0)   # 1枚目＝条件を開く正当な付与
    drake2 = _card("d2", text=DRAKE, rest=True, attached=2)   # 達成済み＝追加は死に（#1/#2型）
    p = _player(field=[drake0, drake2])
    assert [(c.uuid, t) for c, t in PD.dead_targets(p)] == [("d2", "V1s")]


def test_v1s_sorted_before_plain_v1():
    p = _player(field=[_card("v", text="", rest=True),
                       _card("d", text=DRAKE, rest=True, attached=1)])
    tags = [t for _, t in PD.dead_targets(p)]
    assert tags == ["V1s", "V1"]


def test_don_cond_max_n_variants():
    assert PD._don_cond_max_n(_card("x", text="【ドン‼×2】…【ドン!!×1】…")) == 2
    assert PD._don_cond_max_n(_card("x", text="効果なし")) is None


def test_dominance_pairs_v1_only_don_placement_differs():
    lead = _card("L", text="", rest=False, new=False)
    dead = _card("dead", text="", rest=True)
    m = types.SimpleNamespace(p1=_player(field=[dead], don=2, leader=lead),
                              p2=_player(don=0))
    pairs = dict((t, (g, b)) for t, g, b in PD.dominance_pairs(m, "p1"))
    tag = next(t for t in pairs if t.startswith("V1"))
    good, bad = pairs[tag]
    # 差はドンの置き場所のみ: ATTACH の対象以外（攻撃列）は同一
    assert [x for x in good if x[0] == "ATTACK"] == [x for x in bad if x[0] == "ATTACK"]
    assert all(u == "L" for k, u in good if k == "ATTACH")
    assert all(u == "dead" for k, u in bad if k == "ATTACH")


def test_dominance_pairs_v4_only_order_differs():
    lead = _card("L", text="", rest=False)
    ch = _card("ch", text="", rest=False)
    m = types.SimpleNamespace(p1=_player(field=[ch], don=2, leader=lead),
                              p2=_player(don=0))
    pairs = dict((t, (g, b)) for t, g, b in PD.dominance_pairs(m, "p1"))
    tag = next(t for t in pairs if t.startswith("V4"))
    good, bad = pairs[tag]
    assert sorted(good) == sorted(bad)                     # 同じ手の集合
    assert good.index(("ATTACH", "ch")) < good.index(("ATTACK", "ch"))   # 正順=付与→攻撃
    assert bad.index(("ATTACK", "ch")) < bad.index(("ATTACH", "ch"))    # 誤順=攻撃→付与


def test_no_pairs_without_spare_don_or_attackers():
    m = types.SimpleNamespace(p1=_player(field=[_card("v")], don=0,
                                         leader=_card("L", rest=False)),
                              p2=_player(don=0))
    assert PD.dominance_pairs(m, "p1") == []
    m2 = types.SimpleNamespace(p1=_player(field=[], don=3, leader=None), p2=_player(don=0))
    assert PD.dominance_pairs(m2, "p1") == []
