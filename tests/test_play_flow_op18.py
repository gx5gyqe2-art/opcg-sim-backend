"""OP18 で直した効果（2026-11-22・作業パッケージ p1）を実対局の流れで固定する。

- OP18-099 カク／OP18-100 カリファ: 手札の『CP』を含む特徴のカードを、ライフの上か下（選べる）に表向きで加える。
- OP18-022 モンキー・D・ルフィ（リーダー）: アタック時に自分をアクティブにしてもよい。そうした場合だけ、
  次の自分のリフレッシュフェイズでアクティブにならない。
- OP18-028 チムニー: 「ゴンベ」1枚とこのキャラをレストにして、リーダー「モンキー・D・ルフィ」をアクティブにする
  （「ゴンベ」がいなければ起動できない）。
- OP18-105 スパンダム【トリガー】: 公開したカードの【トリガー】効果を発動する。
- OP18-058 お前に国は救えない【トリガー】: コスト付きの【メイン】を発動する＝払うかを選び、払ったときだけ解決する。
ゲームプレイの退行（未実装句で対局が止まる・コストを払わずに効果が出る）を見逃さないための必須/標準テスト。
"""
import os

import pytest

import _bootstrap  # noqa: F401
from harness.rs_scenario import Scenario

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(os.path.dirname(__file__), "..", "opcg_sim", "data", "opcg_effects.json")),
    reason="効果 JSON（生成物）が無い")

VANILLA = "EB01-005"
KAKU = "OP18-099"
KALIFA = "OP18-100"
KUMADORI = "OP18-101"
LUFFY_LEADER = "OP18-022"
GONBE = "OP18-025"
CHIMNEY = "OP18-028"
SPANDAM = "OP18-105"
NO_KINGDOM = "OP18-058"
CP_LEADER = "OP18-079"     # スパンダム（リーダー・特徴 CP9）


# --- OP18-099／OP18-100: 手札の CP カードをライフの上か下に表向きで ----------------------------------

@pytest.mark.parametrize("index,top", [(0, True), (1, False)])
def test_kaku_adds_a_cp_card_from_hand_face_up_to_top_or_bottom_of_life(index, top):
    s = Scenario(turn=5)
    s.don["p1"] = 10
    s.put("p1", KAKU, "hand")
    s.put("p1", KUMADORI, "hand")
    s.put("p1", VANILLA, "hand")
    p = s.build()
    p.play("P1", KAKU)
    pe = p.pending()
    assert pe["action"] == "CHOICE", pe
    p.act("P1", "RESOLVE_EFFECT_SELECTION", {"index": index})
    pe = p.pending()
    assert pe["action"] == "SEARCH_AND_SELECT", pe
    assert pe["selectable_uuids"] == [p.uuid("p1", "hand", KUMADORI)], "CP を含む特徴のカードだけ"
    p.resolve("P1", [p.uuid("p1", "hand", KUMADORI)])
    assert p.action() == "MAIN_ACTION"
    life = p.zone("p1", "life")
    assert len(life) == 6
    card = life[0] if top else life[-1]
    assert card["card_id"] == KUMADORI and card.get("is_face_up"), life
    assert p.ids("p1", "hand") == [VANILLA]


def test_kalifa_draws_then_adds_a_cp_card_to_life():
    s = Scenario(turn=5)
    s.don["p1"] = 10
    s.put("p1", KALIFA, "hand")
    s.put("p1", KAKU, "hand")
    p = s.build()
    p.play("P1", KALIFA)
    assert p.hand_count("p1") == 2, "先に 1 枚引く"
    p.act("P1", "RESOLVE_EFFECT_SELECTION", {"index": 0})
    p.resolve("P1", [p.uuid("p1", "hand", KAKU)])
    assert p.action() == "MAIN_ACTION"
    assert p.face_up_life("p1") == [KAKU]


# --- OP18-022 ルフィ（リーダー） --------------------------------------------------------------------

def _luffy_attacks(accept):
    s = Scenario(p1_leader=LUFFY_LEADER, turn=5)
    s.leader_don["p1"] = 3
    p = s.build()
    p.attack("P1", p.leader("p1")["uuid"], p.leader("p2")["uuid"])
    pe = p.pending()
    assert pe["action"] == "CONFIRM_OPTIONAL" and pe["player_id"] == "P1", pe
    p.answer(accept=accept)
    p.pass_battle()
    p.settle(accept=False)
    assert p.action() == "MAIN_ACTION"
    return p


def _to_next_own_main(p):
    p.end_turn("P1")
    p.settle(accept=False)
    p.end_turn("P2")
    p.settle(accept=False)
    assert p.pending()["player_id"] == "P1"


def test_luffy_activated_stays_rested_through_the_next_refresh():
    p = _luffy_attacks(True)
    assert not p.leader("p1")["is_rest"], "アクティブにした"
    # もう一度アタックする（【ターン1回】なので 2 回目は聞かれない）。
    p.attack("P1", p.leader("p1")["uuid"], p.leader("p2")["uuid"])
    assert p.pending()["action"] != "CONFIRM_OPTIONAL"
    p.pass_battle()
    p.settle(accept=False)
    assert p.leader("p1")["is_rest"]
    _to_next_own_main(p)
    assert p.leader("p1")["is_rest"], "次の自分のリフレッシュでアクティブにならない"


def test_luffy_declined_refreshes_normally():
    p = _luffy_attacks(False)
    assert p.leader("p1")["is_rest"], "アクティブにしていない"
    _to_next_own_main(p)
    assert not p.leader("p1")["is_rest"], "断ったなら凍結しない"


# --- OP18-028 チムニー ------------------------------------------------------------------------------

def _chimney_board(with_gonbe):
    s = Scenario(p1_leader=LUFFY_LEADER, turn=5)
    s.put("p1", CHIMNEY, "field")
    if with_gonbe:
        s.put("p1", GONBE, "field")
    p = s.build()
    return p


def _activate_kinds(p):
    return [m for m in p.legal("P1") if "ACTIVATE" in str(m).upper()]


def test_chimney_rests_gonbe_and_itself_to_activate_luffy():
    p = _chimney_board(True)
    # リーダーでアタックしてレストにする。
    p.attack("P1", p.leader("p1")["uuid"], p.leader("p2")["uuid"])
    p.settle(accept=False)
    assert p.leader("p1")["is_rest"]
    p.act("P1", "ACTIVATE_MAIN", {"card_id": p.uuid("p1", "field", CHIMNEY)})
    p.settle()
    assert p.action() == "MAIN_ACTION"
    field = {c["card_id"]: c for c in p.zone("p1", "field")}
    assert field[CHIMNEY]["is_rest"] and field[GONBE]["is_rest"], "2 枚ともレストで払う"
    assert not p.leader("p1")["is_rest"], "リーダーをアクティブにした"


def test_chimney_cannot_activate_without_gonbe():
    p = _chimney_board(False)
    chimney = p.uuid("p1", "field", CHIMNEY)
    assert not any(chimney in str(m) for m in p.legal("P1") if "ACTIVATE" in str(m).upper()), p.legal("P1")


# --- 【トリガー】: OP18-105 スパンダム／OP18-058 お前に国は救えない --------------------------------

def _life_trigger(top_card, p1_leader, setup):
    s = Scenario(p1_leader=p1_leader, turn=6)
    s.turn_player = "p2"
    s.put("p1", top_card, "life")
    setup(s)
    s.put("p2", VANILLA, "field", don=2)
    p = s.build()
    assert p.zone("p1", "life")[0]["card_id"] == top_card
    p.attack("P2", p.uuid("p2", "field", VANILLA), p.leader("p1")["uuid"])
    p.pass_battle()
    pe = p.pending()
    assert pe["action"] == "CONFIRM_TRIGGER" and pe["player_id"] == "P1", pe
    p.answer(accept=True)
    return p


def test_spandam_activates_the_revealed_cards_trigger():
    # 公開するカク（CP9）の【トリガー】=「リーダーが CP なら、1 枚引き、コスト8以下のキャラ1枚までをデッキの下」。
    p = _life_trigger(SPANDAM, CP_LEADER, lambda s: s.put("p1", KAKU, "hand"))
    hand0 = p.hand_count("p1")
    pe = p.pending()
    assert pe["action"] == "SEARCH_AND_SELECT" and pe["selectable_uuids"] == [p.uuid("p1", "hand", KAKU)], pe
    p.resolve("P1", [p.uuid("p1", "hand", KAKU)])
    pe = p.pending()
    assert pe["action"] == "SEARCH_AND_SELECT", pe              # カクの【トリガー】のデッキ下
    p.resolve("P1", [p.uuid("p2", "field", VANILLA)])
    p.settle(accept=False)
    assert p.hand_count("p1") == hand0 + 1, "カクの【トリガー】の 1 枚ドロー"
    assert KAKU in p.ids("p1", "hand"), "公開しただけ（手札に残る）"
    assert VANILLA not in p.ids("p2", "field"), "カクの【トリガー】でデッキの下へ"


def test_spandam_revealing_nothing_does_nothing():
    p = _life_trigger(SPANDAM, CP_LEADER, lambda s: s.put("p1", KAKU, "hand"))
    hand0 = p.hand_count("p1")
    p.resolve("P1", [])
    p.settle(accept=False)
    assert p.hand_count("p1") == hand0 and VANILLA in p.ids("p2", "field")


def test_no_kingdom_trigger_pays_the_main_cost_then_resolves():
    p = _life_trigger(NO_KINGDOM, "EB01-001", lambda s: s.put("p1", VANILLA, "field"))
    pe = p.pending()
    assert pe["action"] == "CONFIRM_OPTIONAL" and pe["player_id"] == "P1", pe
    p.answer(accept=True)
    # コスト: 自分のキャラ 1 枚を手札へ（候補が 1 枚なので自動で選ばれる）。
    assert p.ids("p1", "field") == [] and VANILLA in p.ids("p1", "hand")
    pe = p.pending()
    assert pe["action"] == "SEARCH_AND_SELECT" and pe["selectable_uuids"] == [p.uuid("p2", "field", VANILLA)], pe
    p.resolve("P1", [p.uuid("p2", "field", VANILLA)])        # 効果: コスト4以下をデッキの下へ
    p.settle(accept=False)
    assert VANILLA in p.ids("p1", "hand") and p.ids("p1", "field") == []
    assert p.ids("p2", "field") == []


def test_no_kingdom_trigger_declined_resolves_nothing():
    p = _life_trigger(NO_KINGDOM, "EB01-001", lambda s: s.put("p1", VANILLA, "field"))
    p.answer(accept=False)
    p.settle(accept=False)
    assert p.ids("p1", "field") == [VANILLA], "コストを払っていない"
    assert p.ids("p2", "field") == [VANILLA], "払わなければ効果も出ない"


def test_no_kingdom_trigger_without_own_character_resolves_nothing():
    p = _life_trigger(NO_KINGDOM, "EB01-001", lambda s: None)
    p.settle(accept=True)
    assert p.ids("p2", "field") == [VANILLA], "払えないコスト＝効果は出ない"
