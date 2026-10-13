"""発動中のイベントは手札を離れている＝解決中の手札の候補・枚数・条件に自分自身を含めない。

エンジンは Python 版の移植で、イベントを手札に置いたまま解決し解決後にトラッシュへ送る。
そのため「手札N枚を捨てる」候補・「手札が N 枚以下」の条件・「手札の枚数分」に、発動した
イベント自身が入っていた（`GameState.resolving_event` で除外して直した）。
- OP02-066: 【メイン】「手札2枚を捨てることができる」の候補に自分自身が出ない（【メイン】経路）。
- OP14-059: 「手札が2枚以下の場合」＝自分を除いた手札で数える（条件）。
- P-002: 「手札すべてをデッキに戻す→戻した枚数分引く」に自分自身が入らない。
- EB04-029: 【カウンター】経路でも「手札1枚を捨てる」の候補に自分自身が出ない。
ゲームプレイの退行を見逃さないための必須/標準テスト。
"""
import os

import pytest

import _bootstrap  # noqa: F401
from harness.rs_scenario import Scenario

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(os.path.dirname(__file__), "..", "opcg_sim", "data", "opcg_effects.json")),
    reason="効果 JSON（生成物）が無い")

VANILLA = "EB01-005"
IMPEL_LEADER = "OP02-049"     # インペルダウン
JINBE_LEADER = "OP14-040"
IMPEL_ALLSTAR = "OP02-066"    # 【メイン】手札2枚を捨てることができる：…
JINBE_DRAW = "OP14-059"       # 【メイン】リーダーがジンベエで手札が2枚以下の場合、カード2枚を引く
RESHUFFLE = "P-002"           # 【メイン】手札すべてをデッキに戻しシャッフル、戻した枚数分引く
SANJI_COUNTER = "EB04-029"    # 【カウンター】手札1枚を捨てることができる：…


def test_main_event_is_not_a_discard_candidate_for_itself():
    s = Scenario(p1_leader=IMPEL_LEADER, turn=6)
    s.don["p1"] = 10
    s.put("p1", IMPEL_ALLSTAR, "hand")
    s.put("p1", VANILLA, "hand")
    s.put("p1", VANILLA, "hand")
    p = s.build()
    me = p.uuid("p1", "hand", IMPEL_ALLSTAR)
    p.play("P1", IMPEL_ALLSTAR)
    # 候補は残りの 2 枚だけ＝選択を挟まず自動で捨てられる（自分が入ると 3 枚から選ばせる中断が出る）。
    for _ in range(4):
        pe = p.pending()
        assert pe["action"] != "SEARCH_AND_SELECT", pe
        assert me not in pe["selectable_uuids"], "発動したイベント自身は捨てる候補に出ない"
        if pe["action"] == "MAIN_ACTION":
            break
        p.answer(accept=True)
    assert p.ids("p1", "trash").count(VANILLA) == 2 and IMPEL_ALLSTAR in p.ids("p1", "trash")


def test_hand_count_condition_does_not_count_the_event_itself():
    s = Scenario(p1_leader=JINBE_LEADER, turn=6)
    s.don["p1"] = 10
    s.put("p1", JINBE_DRAW, "hand")
    s.put("p1", VANILLA, "hand")
    s.put("p1", VANILLA, "hand")
    p = s.build()
    p.play("P1", JINBE_DRAW)
    p.settle(accept=True)
    # 残り手札 2 枚（≤2）→ 条件成立で 2 枚引く。自分を数えると 3 枚で不成立になる。
    assert p.hand_count("p1") == 4
    assert p.ids("p1", "trash") == [JINBE_DRAW]


def test_return_all_hand_to_deck_does_not_take_the_event_with_it():
    s = Scenario(turn=6)
    s.put("p1", RESHUFFLE, "hand")
    s.put("p1", VANILLA, "hand")
    s.put("p1", VANILLA, "hand")
    p = s.build()
    p.play("P1", RESHUFFLE)
    p.settle(accept=True)
    assert p.hand_count("p1") == 2, "戻した 2 枚の分だけ引く（自分を含めて 3 枚引かない）"
    assert p.ids("p1", "trash") == [RESHUFFLE]


def test_counter_event_is_not_a_discard_candidate_for_itself():
    s = Scenario(turn=6)
    s.turn_player = "p2"
    s.put("p2", VANILLA, "field")
    s.put("p1", SANJI_COUNTER, "hand")
    s.put("p1", VANILLA, "hand")
    p = s.build()
    me = p.uuid("p1", "hand", SANJI_COUNTER)
    p.attack("P2", p.uuid("p2", "field", VANILLA), p.leader("p1")["uuid"])
    pe = p.pending()
    if pe["action"] == "SELECT_BLOCKER":
        p.battle(pe["player_id"], "PASS")
        pe = p.pending()
    assert pe["action"] == "SELECT_COUNTER", pe
    p.battle("P1", "SELECT_COUNTER", me)
    pe = p.pending()
    assert pe["action"] != "SELECT_COUNTER" or me not in pe["selectable_uuids"], pe
    assert me not in pe.get("selectable_uuids", []), pe
