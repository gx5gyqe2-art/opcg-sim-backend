"""登場させる枚数の「分配」と、登場／付与の対象の受け渡しを実対局の流れで固定する。

- OP06-086（モリア）: トラッシュから「コスト4以下」と「コスト2以下」を 1 枚ずつ選び、**1 枚をアクティブで登場・残りをレスト**で登場。
- OP10-058（レベッカ）: 引いた後、手札のドレスローザ（コスト7以下）を最大 2 枚公開し、**1 枚を登場・残りがコスト4以下ならレスト**で登場。
- OP16-079（ヤマト）: 自分のトラッシュから登場した特徴《ワノ国》のキャラだけが【速攻】を得る。
- OP15-093（リスキー兄弟）: 選んだ「モンキー・D・ルフィ」が【速攻:キャラ】と属性(斬)を得る。
ゲームプレイの退行を見逃さないための必須/標準テスト。
"""
import os

import pytest

import _bootstrap  # noqa: F401
from harness.rs_scenario import Scenario

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(os.path.dirname(__file__), "..", "opcg_sim", "data", "opcg_effects.json")),
    reason="効果 JSON（生成物）が無い")

MORIA = "OP06-086"
REBECCA = "OP10-058"
YAMATO_LEADER = "OP16-079"
RISKY = "OP15-093"
WANO4, WANO1 = "ST09-002", "OP01-036"          # コスト 4／1 の特徴《ワノ国》
DRESS2, DRESS3 = "OP01-076", "OP10-054"       # ドレスローザ コスト 2／3
BIG8 = "OP05-044"
LUFFY3 = "ST09-011"                            # 「モンキー・D・ルフィ」コスト3
VANILLA = "EB01-005"


def _field(p, seat="p1"):
    return {c["card_id"]: c for c in p.zone(seat, "field")}


def _hidden_card(p, uuid):
    h = p.g.hidden()
    for pl in h["players"].values():
        for c in pl["field"]:
            if c["uuid"] == uuid:
                return c
    raise KeyError(uuid)


def test_moria_plays_one_active_and_the_rest_rested():
    s = Scenario(turn=5)
    s.don["p1"] = 10
    s.put("p1", MORIA, "hand")
    s.put("p1", WANO4, "trash")     # コスト 4 以下の枠
    s.put("p1", DRESS2, "trash")    # コスト 2 以下の枠
    p = s.build()
    p.play("P1", MORIA)
    p.resolve("P1", [p.uuid("p1", "trash", WANO4)])        # コスト4以下 1 枚
    p.resolve("P1", [p.uuid("p1", "trash", DRESS2)])       # コスト2以下 1 枚
    pe = p.pending()
    assert pe["action"] == "SEARCH_AND_SELECT" and pe["constraints"] == {"min": 1, "max": 1}, pe
    assert len(pe["selectable_uuids"]) == 2
    p.resolve("P1", [p.uuid("p1", "trash", DRESS2)])       # アクティブで出す 1 枚
    assert p.action() == "MAIN_ACTION"
    f = _field(p)
    assert set(f) == {MORIA, WANO4, DRESS2}
    assert not f[DRESS2]["is_rest"] and f[WANO4]["is_rest"], "選んだ 1 枚がアクティブ・残りはレスト"
    assert not p.ids("p1", "trash")


def test_moria_with_only_one_candidate_plays_it_active():
    s = Scenario(turn=5)
    s.don["p1"] = 10
    s.put("p1", MORIA, "hand")
    s.put("p1", WANO4, "trash")
    p = s.build()
    p.play("P1", MORIA)
    p.resolve("P1", [p.uuid("p1", "trash", WANO4)])
    p.settle()
    f = _field(p)
    assert WANO4 in f and not f[WANO4]["is_rest"]


def _rebecca(hand_extra, big):
    s = Scenario(turn=5)
    s.don["p1"] = 10
    s.put("p1", REBECCA, "hand")
    for c in hand_extra:
        s.put("p1", c, "hand")
    if big:
        s.put("p1", BIG8, "field")
    return s.build()


def test_rebecca_reveals_two_plays_one_active_and_the_cheap_remainder_rested():
    p = _rebecca([DRESS2, DRESS3], big=True)
    p.play("P1", REBECCA)
    assert p.hand_count("p1") == 3, "コスト8以上がいるので 1 枚引く"
    pe = p.pending()
    assert pe["action"] == "SEARCH_AND_SELECT" and pe["constraints"]["max"] == 2, pe       # 手札から最大 2 枚公開
    p.resolve("P1", [p.uuid("p1", "hand", DRESS2), p.uuid("p1", "hand", DRESS3)])
    pe = p.pending()
    assert pe["action"] == "SEARCH_AND_SELECT" and pe["constraints"] == {"min": 1, "max": 1}, pe   # 公開した中から 1 枚登場
    p.resolve("P1", [p.uuid("p1", "hand", DRESS3)])
    assert p.action() == "MAIN_ACTION"
    f = _field(p)
    assert not f[DRESS3]["is_rest"], "選んだ 1 枚はアクティブ"
    assert f[DRESS2]["is_rest"], "残りはコスト4以下なのでレストで登場"


def test_rebecca_does_not_draw_without_a_cost_8_character():
    p = _rebecca([DRESS2], big=False)
    p.play("P1", REBECCA)
    assert p.hand_count("p1") == 1
    p.settle()


def test_yamato_gives_haste_only_to_wano_characters_played_from_the_trash():
    s = Scenario(p1_leader=YAMATO_LEADER, turn=5)
    s.don["p1"] = 10
    s.put("p1", MORIA, "hand")
    s.put("p1", WANO4, "trash")
    s.put("p1", DRESS2, "trash")
    p = s.build()
    p.play("P1", MORIA)
    p.resolve("P1", [p.uuid("p1", "trash", WANO4)])
    p.resolve("P1", [p.uuid("p1", "trash", DRESS2)])
    p.resolve("P1", [p.uuid("p1", "trash", WANO4)])        # ワノ国をアクティブで登場
    f = _field(p)
    assert "速攻" in f[WANO4]["keywords"] and "速攻" not in f[DRESS2]["keywords"] and "速攻" not in f[MORIA]["keywords"]
    attackers = {m["payload"]["uuid"] for m in p.legal("P1") if m["action_type"] == "ATTACK"}
    assert f[WANO4]["uuid"] in attackers, "登場したターンにアタックできる"
    assert f[MORIA]["uuid"] not in attackers and f[DRESS2]["uuid"] not in attackers


def test_risky_brothers_grants_haste_chara_and_slash_to_the_chosen_luffy():
    s = Scenario(turn=5)
    s.put("p1", RISKY, "field")
    s.put("p1", LUFFY3, "field")
    for _ in range(15):
        s.put("p1", VANILLA, "trash")
    p = s.build()
    luffy = _field(p)[LUFFY3]
    assert luffy["attribute"] == "打" and "速攻:キャラ" not in luffy["keywords"]
    p.act("P1", "ACTIVATE_MAIN", {"uuid": p.uuid("p1", "field", RISKY)})
    p.resolve("P1", [luffy["uuid"]])
    assert p.action() == "MAIN_ACTION"
    luffy = _field(p)[LUFFY3]
    assert "速攻:キャラ" in luffy["keywords"]
    assert "ATTR:斬" in _hidden_card(p, luffy["uuid"])["timed_flags"], "属性(斬)は一時のフラグとして載る"
    assert RISKY in p.ids("p1", "trash"), "コストでトラッシュへ"


def test_risky_brothers_needs_fifteen_cards_in_trash():
    s = Scenario(turn=5)
    s.put("p1", RISKY, "field")
    s.put("p1", LUFFY3, "field")
    for _ in range(13):                                    # コストの自身を足しても 14 枚
        s.put("p1", VANILLA, "trash")
    p = s.build()
    p.act("P1", "ACTIVATE_MAIN", {"uuid": p.uuid("p1", "field", RISKY)})
    p.settle()
    luffy = _field(p)[LUFFY3]
    assert "速攻:キャラ" not in luffy["keywords"]
    assert "ATTR:斬" not in _hidden_card(p, luffy["uuid"])["timed_flags"]
