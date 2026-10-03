"""反応型の誘発（「〜した時／された時」）を、実際の戦闘・KO・ドン!!付与・登場を通して固定する。

単体プローブではなく `RsGame`（API と同じ口）で盤面を組み 1 手ずつ進め、誘発の**有無・回数・要因の絞り込み**
（自分の効果／相手の効果／バトル、自分のターン／相手のターン、キャラ／リーダー、種別）を見る:
OP08-056・OP09-080（ステージの「場を離れた時」）／OP07-038（自分の効果で離れた時）／
OP12-081・OP11-088（攻撃先・アタッカー・属性）／OP02-002（ドン!!が付与された時）／
OP10-042（KOされた時か相手の効果で離れた時の二重誘発が無い）。
ゲームプレイの退行を見逃さないための必須/標準テスト。
"""
import os

import pytest

import _bootstrap  # noqa: F401
from harness.rs_scenario import Scenario

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(os.path.dirname(__file__), "..", "opcg_sim", "data", "opcg_effects.json")),
    reason="効果 JSON（生成物）が無い")

MISS_ALL_SUNDAY = "OP12-075"   # 登場時: 相手のコスト3以下のキャラ1枚までをKO（その後 相手がドン!!を得てもよい）
KAIDO = "OP01-094"             # 登場時 ドン!!-6: このキャラ以外のキャラすべてをKO
KAIDO_LEADER = "OP01-061"
VANILLA = "EB01-005"           # コスト1・属性斬・特徴「白ひげ海賊団傘下」（『白ひげ海賊団』を含む）
VANILLA_STRIKE = "ST22-004"    # コスト3・属性打・白ひげ海賊団傘下
STRAWHAT1 = "OP07-099"         # コスト1・麦わらの一味
DRESSROSA2 = "OP01-076"        # コスト2・ドレスローザ（ウソップ OP10-042 で +1）
BIG8 = "OP05-044"              # コスト8・能力なし
MID3 = "OP10-054"              # コスト3・ドレスローザ


def _kaido_ko_all(s):
    """p1（カイドウ・リーダー）が自分のターンにカイドウを出して全体 KO を起動する。"""
    p = s.build()
    p.play("P1", KAIDO)
    p.answer(accept=True)
    p.resolve("P1", p.pending()["selectable_uuids"][:6])
    return p


def _opp_mas(s, target):
    """p2 の手番に p2 がミス・オールサンデーを出し、p1 の `target`（card_id）を KO 対象に選ぶ。"""
    p = s.build()
    p.play("P2", MISS_ALL_SUNDAY)
    return p, p.settle(pick=[p.uuid("p1", "field", target)])


# --- OP08-056 モビー・ディック号（自分のターン中・ターン1回・白ひげ系キャラが効果で離れた時）------------------

def test_moby_dick_draws_once_for_two_whitebeard_characters_leaving_together():
    s = Scenario(p1_leader=KAIDO_LEADER, turn=5)
    s.don["p1"] = 10
    s.put("p1", KAIDO, "hand")
    for _ in range(2):
        s.put("p1", VANILLA, "hand")
    s.put("p1", "OP08-056", "stage")
    s.put("p1", VANILLA, "field")
    s.put("p1", VANILLA_STRIKE, "field")
    p = _kaido_ko_all(s)
    assert p.hand_count("p1") == 3, "カイドウを出して 2 枚→誘発で 1 枚引く"
    log = p.settle()
    kinds = [k for _, k, _ in log]
    assert kinds.count("SEARCH_AND_SELECT") == 1 and "ARRANGE_DECK" in kinds, log   # 1 回だけ誘発→手札 1 枚を上か下へ
    assert p.hand_count("p1") == 2, "引いて 1 枚をデッキへ戻す（ターン1回）"


def test_moby_dick_stays_quiet_on_the_opponents_turn():
    s = Scenario(p1_leader="EB01-001", turn=6)
    s.turn_player = "p2"
    s.put("p1", "OP08-056", "stage")
    s.put("p1", VANILLA, "field")
    s.put("p1", VANILLA, "hand")
    s.put("p2", MISS_ALL_SUNDAY, "hand")
    p, log = _opp_mas(s, VANILLA)
    assert VANILLA not in p.ids("p1", "field")
    assert not any("モビー" in m for _, _, m in log), log
    assert p.hand_count("p1") == 1


# --- OP09-080 サウザンド・サニー号（相手のターン中・麦わらのキャラが相手の効果で離れた時）----------------------

def test_sunny_asks_and_accepting_rests_the_stage_for_a_rested_don():
    s = Scenario(turn=6)
    s.turn_player = "p2"
    s.put("p1", "OP09-080", "stage")
    s.put("p1", STRAWHAT1, "field")
    s.put("p2", MISS_ALL_SUNDAY, "hand")
    p = s.build()
    p.play("P2", MISS_ALL_SUNDAY)
    log = p.settle(pick=[p.uuid("p1", "field", STRAWHAT1)], answers={"ミス・オールサンデー": False, "サニー": True})
    assert any("サニー" in m for _, _, m in log), log
    b = p.board()["players"]["p1"]
    assert p.stage("p1")["is_rest"] and len(b["don_rested"]) == 1 and len(b["don_active"]) == 5


def test_sunny_declined_gives_nothing():
    s = Scenario(turn=6)
    s.turn_player = "p2"
    s.put("p1", "OP09-080", "stage")
    s.put("p1", STRAWHAT1, "field")
    s.put("p2", MISS_ALL_SUNDAY, "hand")
    p = s.build()
    p.play("P2", MISS_ALL_SUNDAY)
    p.settle(pick=[p.uuid("p1", "field", STRAWHAT1)], answers={"ミス・オールサンデー": False, "サニー": False})
    b = p.board()["players"]["p1"]
    assert not p.stage("p1")["is_rest"] and not b["don_rested"]


def test_sunny_ignores_a_non_strawhat_character_and_the_owners_own_effect():
    # 麦わらでないキャラが相手の効果で KO されても誘発しない。
    s = Scenario(turn=6)
    s.turn_player = "p2"
    s.put("p1", "OP09-080", "stage")
    s.put("p1", VANILLA, "field")
    s.put("p2", MISS_ALL_SUNDAY, "hand")
    p, log = _opp_mas(s, VANILLA)
    assert not any("サニー" in m for _, _, m in log), log
    # 自分（p1）の効果で自分の麦わらが場を離れても「相手の効果で」ではない。
    s = Scenario(p1_leader=KAIDO_LEADER, turn=5)
    s.don["p1"] = 10
    s.put("p1", KAIDO, "hand")
    s.put("p1", "OP09-080", "stage")
    s.put("p1", STRAWHAT1, "field")
    p = _kaido_ko_all(s)
    assert not any("サニー" in m for _, _, m in p.settle()), "自分の効果では誘発しない"


# --- OP07-038 ボア・ハンコック（自分のターン中・ターン1回・キャラが自分の効果で場を離れた時）---------------------

def test_hancock_draws_when_my_effect_removes_a_character_but_not_for_a_battle_ko():
    s = Scenario(p1_leader="OP07-038", turn=5)
    s.put("p1", MISS_ALL_SUNDAY, "hand")
    s.put("p2", VANILLA, "field")
    p = s.build()
    p.play("P1", MISS_ALL_SUNDAY)
    log = p.settle(answers={"ミス・オールサンデー": False})
    assert any("ハンコック" in m for _, _, m in log), log
    assert p.hand_count("p1") == 1
    # バトルでの KO は「自分の効果で」ではない。
    s = Scenario(p1_leader="OP07-038", turn=5)
    s.put("p1", VANILLA, "field", don=5)
    s.put("p2", VANILLA, "field", rest=True)
    p = s.build()
    p.attack("P1", p.uuid("p1", "field", VANILLA), p.uuid("p2", "field", VANILLA))
    log = p.settle()
    assert VANILLA in p.ids("p2", "trash") and p.hand_count("p1") == 0
    assert not any("ハンコック" in m for _, _, m in log), log


# --- OP12-081 コアラ ---------------------------------------------------------------------------

def _koala_attack(n_big, at_leader):
    s = Scenario(p1_leader="OP12-081", turn=5)
    for _ in range(n_big):
        s.put("p1", KAIDO, "field")
    s.put("p2", VANILLA, "field", rest=True)
    p = s.build()
    tgt = p.leader("p2")["uuid"] if at_leader else p.uuid("p2", "field", VANILLA)
    p.attack("P1", p.leader("p1")["uuid"], tgt)
    p.settle()
    return p.hand_count("p1")


def test_koala_draws_only_when_attacking_the_leader_with_two_big_characters():
    assert _koala_attack(2, at_leader=True) == 1
    assert _koala_attack(1, at_leader=True) == 0, "コスト8以上が 1 枚では引かない"
    assert _koala_attack(2, at_leader=False) == 0, "相手のキャラへのアタックでは引かない"


def _koala_opponent_plays(card_id):
    s = Scenario(p1_leader="OP12-081", turn=6)
    s.turn_player = "p2"
    s.don["p2"] = 10
    s.put("p2", card_id, "hand")
    p = s.build()
    life, hand = len(p.zone("p2", "life")), p.hand_count("p2")
    p.play("P2", card_id)
    log = p.settle()
    return p, log, life, hand


def test_koala_makes_the_opponent_take_a_life_when_a_cost_8_character_is_played():
    p, log, life, hand = _koala_opponent_plays(BIG8)
    assert [(s, k) for s, k, _ in log] == [("P1", "CONFIRM_TRIGGER")], log
    assert len(p.zone("p2", "life")) == life - 1 and p.hand_count("p2") == hand, "出した 1 枚＋ライフから 1 枚"
    p, log, life, hand = _koala_opponent_plays(MID3)
    assert not log and len(p.zone("p2", "life")) == life, "コスト8未満のキャラ（効果で出したのでもない）では誘発しない"


# --- OP11-088 シュウ（相手のキャラがアタックした時・属性斬ならパワー+5000）-----------------------------------

def _shuu(attacker_id, at_leader=True, attacker_is_leader=False):
    s = Scenario(turn=6)
    s.turn_player = "p2"
    s.put("p1", "OP11-088", "field", rest=not at_leader)
    if not attacker_is_leader:
        s.put("p2", attacker_id, "field")
    p = s.build()
    att = p.leader("p2")["uuid"] if attacker_is_leader else p.uuid("p2", "field", attacker_id)
    tgt = p.leader("p1")["uuid"] if at_leader else p.uuid("p1", "field", "OP11-088")
    p.attack("P2", att, tgt)
    return p


def _shuu_power(p):
    return [c["power"] for c in p.zone("p1", "field") if c["card_id"] == "OP11-088"][0]


def test_shuu_boosts_only_against_a_slash_attribute_character_attacker():
    p = _shuu(VANILLA)                              # 斬
    pe = p.pending()
    assert pe["player_id"] == "P1" and pe["action"] == "CONFIRM_OPTIONAL" and "シュウ" in pe["message"], pe
    assert _shuu_power(p) == 5000
    p.answer(accept=True)
    assert _shuu_power(p) == 10000, "このバトル中 +5000"
    p = _shuu(VANILLA_STRIKE)                       # 打
    assert p.action() == "SELECT_BLOCKER" and _shuu_power(p) == 5000
    p = _shuu(VANILLA, attacker_is_leader=True)     # 「相手のキャラが」＝リーダーのアタックでは誘発しない
    assert p.action() == "SELECT_BLOCKER" and _shuu_power(p) == 5000


def test_shuu_triggers_when_it_is_the_attack_target_too():
    p = _shuu(VANILLA, at_leader=False)
    assert p.action() == "CONFIRM_OPTIONAL"
    p.answer(accept=True)
    assert _shuu_power(p) == 10000


# --- OP02-002 ガープ（自分のターン中・ドン!!が付与された時）-----------------------------------------------

def test_garp_lowers_an_opponent_cost_each_time_don_is_attached():
    s = Scenario(p1_leader="OP02-002", turn=5)
    s.put("p1", VANILLA, "field")
    s.put("p2", VANILLA_STRIKE, "field")
    p = s.build()
    cost = lambda: [c["cost"] for c in p.zone("p2", "field")][0]
    assert cost() == 3
    p.act("P1", "ATTACH_DON", {"uuid": p.leader("p1")["uuid"]})                  # リーダーへ
    assert p.action() == "SEARCH_AND_SELECT"
    p.settle()
    assert cost() == 2
    p.act("P1", "ATTACH_DON", {"uuid": p.uuid("p1", "field", VANILLA)})          # キャラへ
    assert p.action() == "SEARCH_AND_SELECT"
    p.settle()
    assert cost() == 1


# --- OP10-042 ウソップ（相手のターン中・ターン1回・KOされた時か相手の効果で場を離れた時）------------------------

def test_usopp_draws_once_for_an_effect_ko_which_is_both_ko_and_leave():
    s = Scenario(p1_leader="OP10-042", turn=6)
    s.turn_player = "p2"
    s.put("p1", DRESSROSA2, "field")
    for _ in range(2):
        s.put("p1", VANILLA, "hand")
    s.put("p2", MISS_ALL_SUNDAY, "hand")
    p = s.build()
    p.play("P2", MISS_ALL_SUNDAY)
    log = p.settle(pick=[p.uuid("p1", "field", DRESSROSA2)])
    assert [m for _, k, m in log if "ウソップ" in m] and len([1 for _, _, m in log if "ウソップ" in m]) == 1, log
    assert p.hand_count("p1") == 3, "二重に引かない（ターン1回・同じ除去は 1 誘発）"


def test_usopp_triggers_for_a_battle_ko_too():
    s = Scenario(p1_leader="OP10-042", turn=6)
    s.turn_player = "p2"
    s.put("p1", DRESSROSA2, "field", rest=True)
    s.put("p2", VANILLA, "field", don=5)
    for _ in range(2):
        s.put("p1", VANILLA, "hand")
    p = s.build()
    p.attack("P2", p.uuid("p2", "field", VANILLA), p.uuid("p1", "field", DRESSROSA2))
    log = p.settle()
    assert DRESSROSA2 in p.ids("p1", "trash") and p.hand_count("p1") == 3, log
