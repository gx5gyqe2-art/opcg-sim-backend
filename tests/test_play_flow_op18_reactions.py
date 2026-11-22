"""OP18 の誘発 2 枚を、実対局の流れ（`RsGame`＝API と同じ口）で 1 手ずつ進めて固定する。

- OP18-060 軍子宮（リーダー）「【ターン1回】自分のトラッシュからキャラが登場した時、カード2枚を引き、
  自分の手札1枚を捨てる」: トラッシュからの登場でだけ・ターン1回だけ誘発する（手札からの登場では誘発しない）。
  従来は起動メイン扱いで、登場に反応しなかった。
- OP18-080 ヴァン・オーガー「【自分のターン中】【ターン1回】自分のリーダー「マーシャル・D・ティーチ」の効果で
  自分の手札が捨てられた時、自分の他の「ヴァン・オーガー」がいない場合、カード1枚を引く」: 発生源が
  自分のリーダー「マーシャル・D・ティーチ」のときだけ・他の同名がいないときだけ・ターン1回。
  従来はどの効果で捨てても誘発した。
あわせて、盤面で真偽が変わる常在の条件 2 枚（従来は条件が常に真だった）:
- OP18-004 カルー「このキャラのパワーによって…5000以上の場合【速攻】…」: パワー 4000 では速攻なし・
  ドン!!1 枚付与（5000）で登場したターンにアタックできる。
- OP18-087 スタンセン「自分のキャラがコスト12以上のキャラのみの場合、このキャラのパワー+2000」:
  自身（コスト+12＝15）だけなら +2000・コスト1 のキャラが並ぶと外れる。
ゲームプレイの退行を見逃さないための必須/標準テスト。
"""
import os

import pytest

import _bootstrap  # noqa: F401
from harness.rs_scenario import Scenario

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(os.path.dirname(__file__), "..", "opcg_sim", "data", "opcg_effects.json")),
    reason="効果 JSON（生成物）が無い")

GUNKO = "OP18-060"            # 軍子宮（リーダー）
VAN_AUGUR = "OP18-080"        # ヴァン・オーガー
TEACH = "OP09-081"            # マーシャル・D・ティーチ（リーダー）: 【起動メイン】手札1枚を捨てることができる
SMILE_RAISER = "OP08-090"     # 【登場時】トラッシュからコスト2以下の《SMILE》キャラ1枚までを登場させる
SMILE2 = "OP01-104"           # コスト2・《SMILE》・能力なし
VANILLA = "EB01-005"


def _activate_leader(p):
    p.act("P1", "ACTIVATE_MAIN", {"uuid": p.leader("p1")["uuid"]})
    return p.settle(pick=[p.uuid("p1", "hand", VANILLA)])


# --- OP18-060 --------------------------------------------------------------------------------

def _gunko_board(n_raisers=1):
    s = Scenario(p1_leader=GUNKO, turn=5)
    s.don["p1"] = 10
    for _ in range(n_raisers):
        s.put("p1", SMILE_RAISER, "hand")
        s.put("p1", SMILE2, "trash")
    for _ in range(3):
        s.put("p1", VANILLA, "hand")
    return s.build()


def _play_raiser(p):
    p.play("P1", SMILE_RAISER)
    return p.settle(pick=[p.uuid("p1", "trash", SMILE2)] + [p.uuid("p1", "hand", VANILLA)])


def test_gunko_draws_two_and_discards_one_when_a_character_comes_from_the_trash():
    p = _gunko_board()
    assert p.hand_count("p1") == 4
    _play_raiser(p)
    assert SMILE2 in p.ids("p1", "field"), "トラッシュから登場した"
    # 4 枚 − 登場させた 1 枚 ＋ 2 枚引く − 1 枚捨てる ＝ 4 枚。
    assert p.hand_count("p1") == 4
    assert p.ids("p1", "trash").count(VANILLA) == 1, "1 枚捨てた"


def test_gunko_does_not_fire_for_a_character_played_from_the_hand():
    s = Scenario(p1_leader=GUNKO, turn=5)
    s.don["p1"] = 10
    s.put("p1", SMILE2, "hand")
    s.put("p1", VANILLA, "hand")
    p = s.build()
    p.play("P1", SMILE2)
    log = p.settle()
    assert p.hand_count("p1") == 1, log
    assert p.ids("p1", "trash") == []


def test_gunko_fires_only_once_per_turn():
    p = _gunko_board(n_raisers=2)
    assert p.hand_count("p1") == 5
    _play_raiser(p)
    assert p.hand_count("p1") == 5          # −1 ＋2 −1
    _play_raiser(p)
    assert p.ids("p1", "field").count(SMILE2) == 2
    assert p.hand_count("p1") == 4, "2 回目のトラッシュからの登場では誘発しない（ターン1回）"


# --- OP18-080 --------------------------------------------------------------------------------

def _augur_board(leader, n_augur=1, n_hand=3):
    s = Scenario(p1_leader=leader, turn=5)
    for _ in range(n_augur):
        s.put("p1", VAN_AUGUR, "field")
    for _ in range(n_hand):
        s.put("p1", VANILLA, "hand")
    return s.build()


def test_van_augur_draws_when_the_teach_leader_discards_and_only_once_per_turn():
    p = _augur_board(TEACH)
    _activate_leader(p)
    assert p.hand_count("p1") == 3, "捨てて 1 枚引く"
    _activate_leader(p)
    assert p.hand_count("p1") == 2, "2 回目は引かない（ターン1回）"


def test_van_augur_does_not_draw_with_another_van_augur_on_the_field():
    p = _augur_board(TEACH, n_augur=2)
    _activate_leader(p)
    assert p.hand_count("p1") == 2, "自分の他の「ヴァン・オーガー」がいる"


def test_van_augur_ignores_discards_by_another_leaders_effect():
    # 軍子宮の起動メイン（手札1枚を捨てるコスト）＝リーダーの効果だが「マーシャル・D・ティーチ」ではない。
    p = _augur_board(GUNKO)
    _activate_leader(p)
    assert p.hand_count("p1") == 2, "ティーチ以外のリーダーの効果で捨てても引かない"


def test_van_augur_stays_quiet_on_the_opponents_turn():
    s = Scenario(p1_leader=TEACH, p2_leader=TEACH, turn=6)
    s.turn_player = "p2"
    s.put("p2", VAN_AUGUR, "field")
    s.put("p1", VAN_AUGUR, "field")
    for _ in range(3):
        s.put("p2", VANILLA, "hand")
        s.put("p1", VANILLA, "hand")
    p = s.build()
    p.act("P2", "ACTIVATE_MAIN", {"uuid": p.leader("p2")["uuid"]})
    p.settle(pick=[p.uuid("p2", "hand", VANILLA)])
    assert p.hand_count("p2") == 3, "手番側の p2 は引く"
    assert p.hand_count("p1") == 3, "p1 のヴァン・オーガーは相手の手札・相手のターンでは動かない"


# --- 常在の条件（OP18-004・OP18-087）-----------------------------------------------------------

def test_carue_gains_rush_only_at_5000_power():
    s = Scenario(turn=5)
    s.don["p1"] = 6
    s.put("p1", "OP18-004", "hand")
    p = s.build()
    p.play("P1", "OP18-004")
    p.settle()
    u = p.uuid("p1", "field", "OP18-004")

    def can_attack():
        return any(a["action_type"] == "ATTACK" and a["payload"]["uuid"] == u for a in p.legal("P1"))

    assert not can_attack(), "4000 では【速攻】なし"
    p.act("P1", "ATTACH_DON", {"uuid": u})
    assert can_attack(), "ドン!!付与で 5000 以上＝【速攻】"


def test_stansen_buff_needs_only_cost12_characters():
    s = Scenario(turn=5)
    s.don["p1"] = 6
    s.put("p1", "OP18-087", "hand")
    s.put("p1", VANILLA, "hand")
    p = s.build()
    p.play("P1", "OP18-087")
    p.settle()
    st = p.zone("p1", "field")[0]
    assert (st["cost"], st["power"]) == (15, 6000), "自身だけ（コスト15）＝+2000"
    p.play("P1", VANILLA)
    p.settle()
    st = next(c for c in p.zone("p1", "field") if c["card_id"] == "OP18-087")
    assert st["power"] == 4000, "コスト1 のキャラがいる＝+2000 なし"
