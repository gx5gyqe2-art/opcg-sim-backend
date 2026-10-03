"""任意（「〜してもよい／できる」）・確認・選択枚数の扱いを、実対局の流れで固定する。

- OP11-024: 「捨て、ドン!!をレストにしてもよい。そうした場合、登場させる」＝断れば「そうした場合」が走らない。
- OP13-001: 「ドン!!を**任意の枚数**レストにできる。レストにした 1 枚につき +2000」＝0〜N 枚の選択と倍率。
- OP12-075・OP15-059: 「相手は〜してもよい」の確認は**相手に**出る（持ち主ではなく）。
- OP17-041: 「持ち主が好きな順番でデッキの下に置く」の並び替えは**持ち主（相手）**が行う。
ゲームプレイの退行を見逃さないための必須/標準テスト。
"""
import os

import pytest

import _bootstrap  # noqa: F401
from harness.rs_scenario import Scenario

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(os.path.dirname(__file__), "..", "opcg_sim", "data", "opcg_effects.json")),
    reason="効果 JSON（生成物）が無い")

VANILLA = "EB01-005"      # コスト1・パワー3000
KAIDO = "OP01-094"
KAIDO_LEADER = "OP01-061"
ALADDIN = "OP11-024"      # 相手の効果でKOされた時、手札1枚を捨て・ドン!!1枚をレストにしてもよい。そうした場合、魚人族か人魚族(コスト6以下)を登場
MERMAID3 = "OP11-110"     # コスト3・人魚族
LUFFY_LEADER = "OP13-001"
MISS_ALL_SUNDAY = "OP12-075"
AMAZON = "OP15-059"
WANGZHI = "OP17-041"


# --- OP11-024 アラディン -----------------------------------------------------------------------

def _aladdin_ko(accept):
    s = Scenario(p2_leader=KAIDO_LEADER, turn=6)
    s.turn_player = "p2"
    s.don["p2"] = 10
    s.put("p2", KAIDO, "hand")
    s.put("p1", ALADDIN, "field")
    s.put("p1", MERMAID3, "hand")
    s.put("p1", VANILLA, "hand")
    p = s.build()
    p.play("P2", KAIDO)
    p.answer(accept=True)
    p.resolve("P2", p.pending()["selectable_uuids"][:6])
    pe = p.pending()
    assert pe["player_id"] == "P1" and pe["action"] == "CONFIRM_OPTIONAL" and "アラディン" in pe["message"], pe
    p.answer(accept=accept)
    return p


def test_aladdin_declined_pays_nothing_and_plays_nothing():
    p = _aladdin_ko(False)
    assert p.action() == "MAIN_ACTION"
    assert sorted(p.ids("p1", "hand")) == sorted([MERMAID3, VANILLA]), "捨てていない"
    assert p.don_active("p1") == 5, "ドン!!もレストにしていない"
    assert p.ids("p1", "field") == [] and p.ids("p1", "trash") == [ALADDIN]


def test_aladdin_accepted_discards_rests_don_and_plays_the_mermaid():
    p = _aladdin_ko(True)
    pe = p.pending()
    assert pe["action"] == "SEARCH_AND_SELECT"                                  # 捨てる 1 枚
    p.resolve("P1", [p.uuid("p1", "hand", VANILLA)])
    pe = p.pending()
    assert pe["action"] == "SEARCH_AND_SELECT" and p.uuid("p1", "hand", MERMAID3) in pe["selectable_uuids"]
    p.resolve("P1", [p.uuid("p1", "hand", MERMAID3)])
    assert p.ids("p1", "field") == [MERMAID3]
    p.settle(accept=False)                      # 登場したフカボシ自身の【登場時】（任意）は断る
    assert p.action() == "MAIN_ACTION"
    assert VANILLA in p.ids("p1", "trash") and p.don_active("p1") == 4


# --- OP13-001 ルフィ（任意の枚数のドン!!をレスト→1 枚につき +2000）----------------------------------

def _luffy_attacked(active):
    s = Scenario(p1_leader=LUFFY_LEADER, turn=6)
    s.turn_player = "p2"
    s.don["p1"] = active
    s.leader_don["p1"] = 1                       # 【ドン!!×1】
    s.put("p2", VANILLA, "field")
    p = s.build()
    p.attack("P2", p.uuid("p2", "field", VANILLA), p.leader("p1")["uuid"])
    return p


def test_luffy_rests_any_number_of_don_and_gets_2000_each():
    p = _luffy_attacked(4)
    pe = p.pending()
    assert pe["player_id"] == "P1" and pe["action"] == "CONFIRM_OPTIONAL" and "ルフィ" in pe["message"], pe
    p.answer(accept=True)
    pe = p.pending()
    assert pe["action"] == "SEARCH_AND_SELECT" and pe["constraints"]["min"] == 0, pe      # 0 枚から選べる
    assert len(pe["selectable_uuids"]) == 4
    p.resolve("P1", pe["selectable_uuids"][:2])
    assert p.pending()["action"] == "SEARCH_AND_SELECT"                                   # 付与先（1 枚まで）
    p.resolve("P1", [p.leader("p1")["uuid"]])
    assert p.leader("p1")["power"] == 5000 + 2 * 2000, "2 枚レスト → +4000"
    assert p.don_active("p1") == 2


def test_luffy_choosing_zero_don_changes_nothing():
    p = _luffy_attacked(4)
    p.answer(accept=True)
    p.resolve("P1", [])
    p.settle()
    assert p.leader("p1")["power"] == 5000 and p.don_active("p1") == 4


def test_luffy_does_not_ask_with_six_or_more_active_don():
    p = _luffy_attacked(6)
    assert p.action() == "SELECT_COUNTER"


# --- 「相手は〜してもよい」の確認は相手に出る -------------------------------------------------------

def test_miss_all_sunday_asks_the_opponent_whether_to_gain_a_don():
    for accept, expected in ((True, 6), (False, 5)):
        s = Scenario(turn=5)
        s.put("p1", MISS_ALL_SUNDAY, "hand")
        s.put("p2", VANILLA, "field")
        p = s.build()
        p.play("P1", MISS_ALL_SUNDAY)
        p.resolve("P1", [p.uuid("p2", "field", VANILLA)])
        pe = p.pending()
        assert pe["player_id"] == "P2" and pe["action"] == "CONFIRM_OPTIONAL", pe      # 確認は相手（P2）へ
        p.answer(accept=accept)
        assert p.don_active("p2") == expected


def _amazon_attacked():
    s = Scenario(turn=6)
    s.turn_player = "p2"
    s.put("p1", AMAZON, "field")
    s.put("p2", VANILLA, "field")
    p = s.build()
    p.attack("P2", p.uuid("p2", "field", VANILLA), p.leader("p1")["uuid"])
    pe = p.pending()
    assert pe["player_id"] == "P1" and "アマゾン" in pe["message"], pe              # コストを払う確認は持ち主
    p.answer(accept=True)
    return p


def test_amazon_asks_the_attacker_to_return_a_don_and_accepting_stops_the_debuff():
    p = _amazon_attacked()
    pe = p.pending()
    assert pe["player_id"] == "P2" and pe["action"] == "CONFIRM_OPTIONAL", pe          # 確認は攻撃側（相手）へ
    p.answer(accept=True)
    pe = p.pending()
    assert pe["player_id"] == "P2" and pe["action"] == "SELECT_RESOURCE"
    p.resolve("P2", pe["selectable_uuids"][:1])
    p.settle()
    assert p.don_active("p2") == 4 and p.leader("p2")["power"] == 5000
    assert [c["power"] for c in p.zone("p2", "field")] == [3000], "戻したのでパワー-2000 は無い"


def test_amazon_declined_debuffs_the_chosen_opponent_card_by_2000():
    p = _amazon_attacked()
    p.answer(accept=False)                       # 攻撃側はドン!!を戻さない
    pe = p.pending()
    assert pe["player_id"] == "P1" and pe["action"] == "SEARCH_AND_SELECT", pe       # 「そうしなかった場合」は P1 が対象を選ぶ
    p.resolve("P1", [p.uuid("p2", "field", VANILLA)])
    assert [c["power"] for c in p.zone("p2", "field")] == [1000]
    assert p.don_active("p2") == 5


# --- OP17-041 王直（持ち主が好きな順番でデッキの下に置く）------------------------------------------

def test_wangzhi_lets_the_owner_order_the_cost_1_characters_to_the_deck_bottom():
    s = Scenario(turn=5)
    s.put("p1", WANGZHI, "hand")
    s.put("p1", VANILLA, "hand")
    for _ in range(3):
        s.put("p2", VANILLA, "field")
    s.put("p2", "OP01-076", "field")             # コスト 2（対象外）
    p = s.build()
    p.play("P1", WANGZHI)
    pe = p.pending()
    assert pe["player_id"] == "P1" and pe["action"] == "CONFIRM_OPTIONAL", pe        # コスト（手札を捨てる）の確認
    p.answer(accept=True)
    pe = p.pending()
    assert pe["player_id"] == "P2" and pe["action"] == "ARRANGE_DECK", pe            # 並び替えは持ち主（P2）
    assert len(pe["selectable_uuids"]) == 3
    p.resolve("P2", list(reversed(pe["selectable_uuids"])))
    assert p.action() == "MAIN_ACTION"
    assert p.ids("p2", "field") == ["OP01-076"] and VANILLA in p.ids("p1", "trash")


# --- OP02-066 インペルダウンオールスター（カード2枚までを引く＝0〜2 枚を選ぶ）--------------------------

IMPEL_ALL_STAR = "OP02-066"
IMPEL_LEADER = "OP02-049"     # 特徴《インペルダウン》のリーダー


def _all_star(answers):
    s = Scenario(p1_leader=IMPEL_LEADER, turn=5)
    s.don["p1"] = 5
    s.put("p1", IMPEL_ALL_STAR, "hand")
    for _ in range(3):      # 発動したイベント自身は候補に入らない＝選ぶには自分以外が 3 枚要る
        s.put("p1", VANILLA, "hand")
    p = s.build()
    before = p.hand_count("p1")
    p.play("P1", IMPEL_ALL_STAR)
    pe = p.pending()
    assert pe["action"] == "SEARCH_AND_SELECT"             # コスト: 手札 2 枚を捨てる
    assert all(c["card_id"] == VANILLA for c in pe["candidates"]), "イベント自身は候補に出ない"
    p.resolve("P1", [c["uuid"] for c in pe["candidates"] if c["card_id"] == VANILLA][:2])
    for a in answers:
        pe = p.pending()
        assert pe and pe["action"] == "CONFIRM_OPTIONAL" and pe["player_id"] == "P1", pe
        p.answer(accept=a)
    return p, before


def test_all_star_draw_up_to_two_can_stop_at_zero_one_or_two():
    for answers, drawn in (([True, True], 2), ([True, False], 1), ([False], 0)):
        p, before = _all_star(answers)
        assert p.action() == "MAIN_ACTION"
        # 手札: 発動前 − イベント 1 枚 − 捨てる 2 枚 + 引いた枚数
        assert p.hand_count("p1") == before - 3 + drawn, (answers, before, p.hand_count("p1"))
