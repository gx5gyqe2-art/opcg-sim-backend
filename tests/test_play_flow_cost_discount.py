"""「このターン中、次に手札から登場させる〜の支払うコストは N 少なくなる」を実対局の流れで固定する。

OP02-025（錦えもん・コスト3以上の特徴《ワノ国》を1少なく）と OP12-061（ロシナンテ・コスト4以上の
「トラファルガー・ロー」を2少なく）。起動 → 対象外のカードを先に出す（消費されない）→ 対象を出す
（軽減が効き、残りの軽減は消える）→ 同じ対象をもう 1 枚出す（満額）を、`RsGame` で 1 手ずつ見る。
ゲームプレイの退行を見逃さないための必須/標準テスト。
"""
import os

import pytest

import _bootstrap  # noqa: F401
from harness.rs_scenario import Scenario

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(os.path.dirname(__file__), "..", "opcg_sim", "data", "opcg_effects.json")),
    reason="効果 JSON（生成物）が無い")

KINEMON = "OP02-025"
ROSINANTE = "OP12-061"
WANO4 = "ST09-002"       # コスト 4・ワノ国
WANO3 = "ST09-009"       # コスト 3・ワノ国
WANO2 = "ST09-006"       # コスト 2・ワノ国（対象外）
LAW4 = "P-088"           # コスト 4・トラファルガー・ロー
LAW1 = "ST03-008"        # コスト 1・トラファルガー・ロー（対象外）
VANILLA = "EB01-005"     # コスト 1・ワノ国でない
OTHER4 = VANILLA         # 別名のカード（対象外）


def _spend(p, card_id):
    before = p.don_active("p1")
    p.play("P1", card_id)
    return before - p.don_active("p1")


def _activate_leader(p):
    p.act("P1", "ACTIVATE_MAIN", {"uuid": p.leader("p1")["uuid"]})
    while p.action() in ("SELECT_RESOURCE", "CONFIRM_OPTIONAL"):     # ドン!!-1 の選択など
        pe = p.pending()
        if pe["action"] == "SELECT_RESOURCE":
            p.resolve("P1", pe["selectable_uuids"][:1])
        else:
            p.answer(accept=True)
    assert p.action() == "MAIN_ACTION"


def test_kinemon_discount_applies_to_the_next_qualifying_card_only():
    s = Scenario(p1_leader=KINEMON, turn=5)
    s.don["p1"] = 10
    for c in (WANO4, WANO4, WANO2, VANILLA):
        s.put("p1", c, "hand")
    p = s.build()
    _activate_leader(p)
    assert _spend(p, VANILLA) == 1, "ワノ国でないカードは満額（かつ軽減を消費しない）"
    assert _spend(p, WANO2) == 2, "コスト 3 未満のワノ国は対象外（軽減を消費しない）"
    assert _spend(p, WANO4) == 3, "最初の対象は 1 少なく（4→3）"
    assert _spend(p, WANO4) == 4, "軽減は 1 度きり。2 枚目は満額"


def test_kinemon_discount_vanishes_for_every_card_that_was_in_hand():
    """起動時に手札にあった対象全部へ付く軽減の 1 枚目が使われたら、残りの対象からも消える。"""
    s = Scenario(p1_leader=KINEMON, turn=5)
    s.don["p1"] = 10
    for c in (WANO4, WANO3):
        s.put("p1", c, "hand")
    p = s.build()
    _activate_leader(p)
    assert _spend(p, WANO3) == 2
    assert _spend(p, WANO4) == 4


def test_kinemon_cannot_activate_with_two_characters():
    s = Scenario(p1_leader=KINEMON, turn=5)
    s.put("p1", VANILLA, "field")
    s.put("p1", VANILLA, "field")
    p = s.build()
    kinds = {(m["action_type"], m["payload"].get("uuid")) for m in p.legal("P1")}
    assert ("ACTIVATE_MAIN", p.leader("p1")["uuid"]) not in kinds


def test_rosinante_discount_is_two_for_the_next_qualifying_law():
    s = Scenario(p1_leader=ROSINANTE, turn=5)
    s.don["p1"] = 10
    for c in (LAW4, LAW4, LAW1, OTHER4):
        s.put("p1", c, "hand")
    p = s.build()
    _activate_leader(p)
    assert p.don_active("p1") == 9 or p.don_active("p1") == 10 - 1, "ドン!!-1 のコスト"
    assert _spend(p, OTHER4) == 1, "別名は対象外"
    assert _spend(p, LAW1) == 1, "コスト 4 未満のローは対象外"
    assert _spend(p, LAW4) == 2, "最初の対象は 2 少なく（4→2）"
    assert _spend(p, LAW4) == 4, "軽減は 1 度きり"
