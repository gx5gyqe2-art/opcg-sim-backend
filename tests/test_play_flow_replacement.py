"""KO の置換・保護が**実対局の流れ**（API と同じ `RsGame` を 1 手ずつ進める）で本文どおり動くこと。

2026-10-01 のカード効果点検は単体のプローブで直した。ここでは実カード ID で盤面を組み、
攻撃・登場時効果・ドン!!の付け替えを通して「確認が出る→断れば KO・受ければ代わりの処理」
「自分の効果による KO にも保護/置換が働く」「【ターン1回】が盤面の動きで勝手に消費されない」を固定する。
ゲームプレイの退行（誤った効果解決）を見逃さないための必須/標準テスト。
"""
import os

import pytest

import _bootstrap  # noqa: F401
from harness.rs_scenario import Scenario

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(os.path.dirname(__file__), "..", "opcg_sim", "data", "opcg_effects.json")),
    reason="効果 JSON（生成物）が無い")

FUKABOSHI = "OP11-110"      # KOされる場合、代わりに「魚人島」かリーダーの「しらほし」1枚をレストにできる
UOJIMA = "OP11-117"         # ステージ「魚人島」
NAMI_EGG = "ST29-008"       # エッグヘッドのキャラが相手の効果でKOされる場合、代わりにライフの上を表向きにできる
ACE = "ST09-010"            # 【ターン1回】KOされる場合、代わりにライフの上か下1枚をトラッシュに置ける
CRACKER = "ST20-002"        # 【ターン1回】効果でKOされる場合、代わりにライフの上1枚をトラッシュに置ける
LUFFY_118 = "OP10-118"      # ターンに1回、相手の効果でKOされない
MISS_ALL_SUNDAY = "OP12-075"  # 登場時: 相手のコスト3以下のキャラ1枚までをKO
KAIDO = "OP01-094"          # 登場時 ドン!!-6: このキャラ以外のキャラすべてをKO（自分のものも）
KAIDO_LEADER = "OP01-061"
SHIRAHOSHI_LEADER = "OP11-022"
VANILLA = "EB01-005"


def _battle_ko_attempt(defender_id, extra_p2=(), p2_leader="EB01-001"):
    """p1 の 8000 パワーのキャラが、p2 のレストのキャラを攻撃する（ブロッカー・カウンターはパス）。"""
    s = Scenario(p2_leader=p2_leader, turn=5)
    s.put("p1", VANILLA, "field", don=5)
    s.put("p2", defender_id, "field", rest=True)
    for card_id, zone in extra_p2:
        s.put("p2", card_id, zone)
    p = s.build()
    p.attack("P1", p.uuid("p1", "field", VANILLA), p.uuid("p2", "field", defender_id))
    p.pass_battle()
    return p


def test_fukaboshi_battle_ko_asks_and_declining_lets_it_die():
    p = _battle_ko_attempt(FUKABOSHI, [(UOJIMA, "stage")])
    pe = p.pending()
    assert pe["action"] == "CONFIRM_OPTIONAL" and pe["player_id"] == "P2", pe   # 確認は持ち主へ
    p.answer(accept=False)
    assert FUKABOSHI not in p.ids("p2", "field") and FUKABOSHI in p.ids("p2", "trash")
    assert not p.stage("p2")["is_rest"], "断ったので魚人島はレストにならない"


def test_fukaboshi_battle_ko_accepting_rests_the_stage_instead():
    p = _battle_ko_attempt(FUKABOSHI, [(UOJIMA, "stage")])
    assert p.action() == "CONFIRM_OPTIONAL"
    p.answer(accept=True)
    assert FUKABOSHI in p.ids("p2", "field") and FUKABOSHI not in p.ids("p2", "trash")
    assert p.stage("p2")["is_rest"], "代わりに魚人島がレストになる"


def test_fukaboshi_replacement_can_use_the_shirahoshi_leader():
    p = _battle_ko_attempt(FUKABOSHI, p2_leader=SHIRAHOSHI_LEADER)
    assert p.action() == "CONFIRM_OPTIONAL"
    p.answer(accept=True)
    assert FUKABOSHI in p.ids("p2", "field") and p.leader("p2")["is_rest"]


def test_fukaboshi_without_a_restable_substitute_just_dies():
    p = _battle_ko_attempt(FUKABOSHI)    # 魚人島もしらほしも居ない
    assert p.action() == "MAIN_ACTION"
    assert FUKABOSHI in p.ids("p2", "trash")


def _opponent_effect_ko(target_id, p2_leader="EB01-001"):
    s = Scenario(p2_leader=p2_leader, turn=5)
    s.put("p1", MISS_ALL_SUNDAY, "hand")
    s.put("p2", target_id, "field")
    p = s.build()
    p.play("P1", MISS_ALL_SUNDAY)
    assert p.action() == "SEARCH_AND_SELECT"
    p.resolve("P1", [p.uuid("p2", "field", target_id)])
    return p


def test_egghead_nami_protects_against_an_opponent_effect_ko_only_when_accepted():
    p = _opponent_effect_ko(NAMI_EGG)
    pe = p.pending()
    assert pe["action"] == "CONFIRM_OPTIONAL" and pe["player_id"] == "P2", pe
    p.answer(accept=True)
    assert NAMI_EGG in p.ids("p2", "field")
    assert p.face_up_life("p2") == ["EB01-005"], "代わりにライフの上 1 枚が表向きになる"
    p = _opponent_effect_ko(NAMI_EGG)
    p.answer(accept=False)
    assert NAMI_EGG in p.ids("p2", "trash") and not p.face_up_life("p2")


def test_egghead_nami_does_not_answer_the_owners_own_effect():
    """本文は「相手の効果で」。自分の効果（カイドウの全体 KO）による KO には働かない＝確認も出ない。"""
    s = Scenario(p1_leader=KAIDO_LEADER, turn=5)
    s.don["p1"] = 10
    s.put("p1", KAIDO, "hand")
    s.put("p1", NAMI_EGG, "field")
    p = s.build()
    p.play("P1", KAIDO)
    p.answer(accept=True)                                   # コストの使用確認
    pe = p.pending()
    p.resolve("P1", pe["selectable_uuids"][:6])             # ドン!!-6
    assert p.action() == "MAIN_ACTION"
    assert NAMI_EGG in p.ids("p1", "trash")


def test_ace_replacement_in_battle_asks_and_the_once_per_turn_survives_board_changes():
    # ドン!!の付与（盤面の再計算が走る）を挟んでから攻撃しても、置換の【ターン1回】は残っている。
    s = Scenario(turn=5)
    s.put("p1", VANILLA, "field", don=4)
    s.put("p2", ACE, "field", rest=True)
    p = s.build()
    p.act("P1", "ATTACH_DON", {"uuid": p.leader("p1")["uuid"]})
    p.attack("P1", p.uuid("p1", "field", VANILLA), p.uuid("p2", "field", ACE))
    p.pass_battle()
    assert p.action() == "CONFIRM_OPTIONAL"
    life_before = len(p.zone("p2", "life"))
    p.answer(accept=True)
    assert ACE in p.ids("p2", "field")
    assert len(p.zone("p2", "life")) == life_before - 1 and "EB01-005" in p.ids("p2", "trash")


def test_ace_declined_replacement_means_ko():
    s = Scenario(turn=5)
    s.put("p1", VANILLA, "field", don=5)
    s.put("p2", ACE, "field", rest=True)
    p = s.build()
    p.attack("P1", p.uuid("p1", "field", VANILLA), p.uuid("p2", "field", ACE))
    p.pass_battle()
    assert p.action() == "CONFIRM_OPTIONAL"
    p.answer(accept=False)
    assert ACE in p.ids("p2", "trash") and len(p.zone("p2", "life")) == 5


def _kaido_board(p1_field):
    s = Scenario(p1_leader=KAIDO_LEADER, turn=5)
    s.don["p1"] = 10
    s.put("p1", KAIDO, "hand")
    for c in p1_field:
        s.put("p1", c, "field")
    p = s.build()
    p.play("P1", KAIDO)
    p.answer(accept=True)                                   # 「コストを払う」の確認
    p.resolve("P1", p.pending()["selectable_uuids"][:6])    # ドン!!-6
    return p


def test_own_effect_ko_asks_cracker_and_ace_and_accepting_keeps_both():
    """自分のカイドウの全体 KO でも、置換（ST20-002＝効果でKO・ST09-010＝KO）は確認を出す。"""
    p = _kaido_board([CRACKER, ACE])
    seen = 0
    while p.action() == "CONFIRM_OPTIONAL":
        assert p.pending()["player_id"] == "P1"
        p.answer(accept=True)
        seen += 1
        assert seen <= 2
    assert seen == 2, "クラッカーとエースの両方に確認が出る"
    assert sorted(p.ids("p1", "field")) == sorted([KAIDO, CRACKER, ACE])
    assert len(p.zone("p1", "life")) == 3, "ライフ 1 枚ずつがトラッシュへ"


def test_own_effect_ko_declined_replacement_means_ko():
    p = _kaido_board([CRACKER])
    assert p.action() == "CONFIRM_OPTIONAL"
    p.answer(accept=False)
    assert CRACKER in p.ids("p1", "trash") and len(p.zone("p1", "life")) == 5


def test_once_per_turn_ko_immunity_holds_against_the_first_opponent_effect_ko():
    """OP10-118「ターンに1回、相手の効果でKOされない」: 相手の効果 KO（カイドウの全体 KO）を 1 度防ぐ。"""
    s = Scenario(p1_leader=KAIDO_LEADER, turn=5)
    s.don["p1"] = 10
    s.put("p1", KAIDO, "hand")
    s.put("p2", LUFFY_118, "field")
    p = s.build()
    p.play("P1", KAIDO)
    p.answer(accept=True)
    p.resolve("P1", p.pending()["selectable_uuids"][:6])
    assert LUFFY_118 in p.ids("p2", "field")
