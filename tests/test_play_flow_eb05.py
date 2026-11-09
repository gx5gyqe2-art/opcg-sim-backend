"""EB05 の置換・誘発が**実対局の流れ**（API と同じ `RsGame` を 1 手ずつ進める）で本文どおり動くこと。

2026-11-09 の EB05 追加の厳格レビューで ng になった置換・誘発（`docs/reports/2026-11-09_card_fix_eb05.md`）:

- EB05-052 グロリオーサ「自分がダメージを受ける場合、代わりにこのキャラをトラッシュに置くことができる」
  ＝ダメージの置換（1 点ごと・任意）。直す前は継続効果の再計算で能力が走り、**場に出ただけで自壊**していた。
- EB05-061 ナミ／OP15-098 ルフィ「相手によって場を離れる場合、代わりに〜」＝相手の効果だけでなく
  相手のアタックによるバトル KO も置換できる（直す前は効果の除去だけ）。
- EB05-010 ニコ・ロビン（リーダー）「自分の属性(知)を持ち、【ブロッカー】を持たないキャラがKOされた時」
  ＝属性とブロッカーの限定（直す前はどのキャラの KO でも誘発）。

ゲームプレイの退行（誤った効果解決）を見逃さないための必須/標準テスト。
"""
import os

import pytest

import _bootstrap  # noqa: F401
from harness.rs_scenario import Scenario

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(os.path.dirname(__file__), "..", "opcg_sim", "data", "opcg_effects.json")),
    reason="効果 JSON（生成物）が無い")

GLORIOSA = "EB05-052"     # 自分がダメージを受ける場合、代わりにこのキャラをトラッシュに置くことができる
NAMI = "EB05-061"         # 【ターン1回】元々のパワー6000以下のキャラが相手によって場を離れる場合、代わりにライフ上1枚を手札に
LUFFY_LEADER = "OP15-098"  # 元々のパワー6000以上の《空島》のキャラが相手によって場を離れる場合、代わりにライフ上1枚を手札に
ROBIN_LEADER = "EB05-010"  # 【ターン1回】属性(知)を持ち【ブロッカー】を持たないキャラがKOされた時、デッキ上1枚までをライフの上に
HAITEI = "OP06-116"       # 【メイン】以下から1つ:・相手のコスト5以下を KO／・相手のライフ1枚なら相手に1ダメージ。その後、自分のライフ上1枚を手札に
ACE_DA = "P-028"          # 【ダブルアタック】6000
VANILLA = "EB01-005"      # 3000・斬
NORA = "OP05-108"         # 7000・《空島》・バニラ
GENZO = "OP03-046"        # 4000・知・バニラ
SHAKA = "OP07-101"        # 6000・知・【ブロッカー】
BIG = "OP02-055"          # 8000・バニラ


def _attack_leader(p2_field=(GLORIOSA,), attacker=VANILLA, p2_life=5):
    s = Scenario(turn=5)
    s.put("p1", attacker, "field", don=5)
    for cid in p2_field:
        s.put("p2", cid, "field")
    s.life_n["p2"] = p2_life
    p = s.build()
    p.attack("P1", p.uuid("p1", "field", attacker), p.leader("p2")["uuid"])
    p.pass_battle()
    return p


def _life(p, seat):
    return len(p.zone(seat, "life"))


def _winner(p):
    return p.board()["turn_info"].get("winner")


# --- EB05-052 グロリオーサ（ダメージの置換） -----------------------------------------------


def test_gloriosa_does_not_trash_itself_when_played():
    s = Scenario(turn=5)
    s.put("p1", GLORIOSA, "hand")
    p = s.build()
    p.play("P1", GLORIOSA)
    p.settle()
    assert GLORIOSA in p.ids("p1", "field") and GLORIOSA not in p.ids("p1", "trash")
    p.end_turn("P1")
    p.settle()
    assert GLORIOSA in p.ids("p1", "field"), "ターンが進んでも自壊しない（継続効果の再計算で走らない）"


def test_gloriosa_battle_damage_asks_and_accepting_trashes_it_instead_of_life():
    p = _attack_leader()
    pe = p.pending()
    assert pe["action"] == "CONFIRM_OPTIONAL" and pe["player_id"] == "P2", pe   # 確認はダメージを受ける側へ
    p.answer(accept=True)
    assert _life(p, "p2") == 5, "ダメージは置き換わった"
    assert GLORIOSA in p.ids("p2", "trash") and GLORIOSA not in p.ids("p2", "field")
    assert p.action() == "MAIN_ACTION"


def test_gloriosa_declining_takes_the_damage_and_keeps_it():
    p = _attack_leader()
    assert p.action() == "CONFIRM_OPTIONAL"
    p.answer(accept=False)
    assert _life(p, "p2") == 4
    assert GLORIOSA in p.ids("p2", "field")


def test_gloriosa_double_attack_replaces_only_one_point():
    """【ダブルアタック】の 2 点は 1 点ずつ: 1 点目を置き換えると、場を離れたので 2 点目はそのまま受ける。"""
    p = _attack_leader(attacker=ACE_DA)
    assert p.action() == "CONFIRM_OPTIONAL"
    p.answer(accept=True)
    assert p.action() != "CONFIRM_OPTIONAL", "1 枚しか無いので 2 点目は聞かない"
    p.settle(accept=False)
    assert _life(p, "p2") == 4 and GLORIOSA in p.ids("p2", "trash")


def test_gloriosa_double_attack_decline_then_accept_on_the_second_point():
    p = _attack_leader(attacker=ACE_DA)
    p.answer(accept=False)                     # 1 点目は受ける
    assert p.action() == "CONFIRM_OPTIONAL", "2 点目もダメージの置換を聞く"
    p.answer(accept=True)
    assert _life(p, "p2") == 4 and GLORIOSA in p.ids("p2", "trash")


def test_gloriosa_at_zero_life_saves_the_game_only_when_accepted():
    p = _attack_leader(p2_life=0)
    assert p.action() == "CONFIRM_OPTIONAL"
    p.answer(accept=True)
    assert _winner(p) is None and GLORIOSA in p.ids("p2", "trash")
    p = _attack_leader(p2_life=0)
    p.answer(accept=False)
    assert _winner(p) == "P1"


def test_gloriosa_replaces_effect_damage_and_the_rest_of_the_effect_continues():
    """効果のダメージ（OP06-116「相手に1ダメージを与える。その後、自分のライフの上から1枚を手札に加える」）。"""
    s = Scenario(turn=5)
    s.put("p1", HAITEI, "hand")
    s.put("p2", GLORIOSA, "field")
    s.life_n["p2"] = 1
    p = s.build()
    p1_life = _life(p, "p1")
    p.play("P1", HAITEI)
    pe = p.pending()
    assert pe["action"] == "CHOICE", pe
    p.act("P1", "RESOLVE_EFFECT_SELECTION", {"index": 1})
    pe = p.pending()
    assert pe["action"] == "CONFIRM_OPTIONAL" and pe["player_id"] == "P2", pe
    p.answer(accept=True)
    assert _life(p, "p2") == 1 and GLORIOSA in p.ids("p2", "trash")
    assert _life(p, "p1") == p1_life - 1, "効果の残り（その後、自分のライフ上1枚を手札へ）も実行される"


# --- EB05-061 ナミ／OP15-098 ルフィ（相手によって場を離れる場合） ----------------------------


def _battle_ko(defender, extra_p2=(), p2_leader="EB01-001"):
    s = Scenario(p2_leader=p2_leader, turn=5)
    s.put("p1", BIG, "field", don=5)
    s.put("p2", defender, "field", rest=True)
    for cid in extra_p2:
        s.put("p2", cid, "field")
    p = s.build()
    p.attack("P1", p.uuid("p1", "field", BIG), p.uuid("p2", "field", defender))
    p.pass_battle()
    return p


def test_nami_replaces_a_battle_ko_by_the_opponents_attack():
    p = _battle_ko(GENZO, [NAMI])
    pe = p.pending()
    assert pe["action"] == "CONFIRM_OPTIONAL" and pe["player_id"] == "P2", pe
    hand = p.hand_count("p2")
    p.answer(accept=True)
    assert GENZO in p.ids("p2", "field"), "代わりにライフの上 1 枚を手札へ"
    assert _life(p, "p2") == 4 and p.hand_count("p2") == hand + 1


def test_nami_declined_lets_the_character_die():
    p = _battle_ko(GENZO, [NAMI])
    p.answer(accept=False)
    assert GENZO in p.ids("p2", "trash") and _life(p, "p2") == 5


def test_luffy_leader_replaces_a_battle_ko_of_a_sky_island_character():
    p = _battle_ko(NORA, p2_leader=LUFFY_LEADER)
    assert p.action() == "CONFIRM_OPTIONAL"
    p.answer(accept=True)
    assert NORA in p.ids("p2", "field") and _life(p, "p2") == 4


def test_luffy_leader_ignores_a_character_outside_the_limits():
    p = _battle_ko(GENZO, p2_leader=LUFFY_LEADER)   # 4000・《空島》でない
    assert p.action() == "MAIN_ACTION"
    assert GENZO in p.ids("p2", "trash")


# --- EB05-010 ニコ・ロビン（KO で誘発する限定） --------------------------------------------


def _ko_under_robin(victim):
    """p2 の手番に p2 の 8000 が p1（リーダー ロビン）のレストのキャラをアタックして KO する。"""
    s = Scenario(p1_leader=ROBIN_LEADER, turn=6)
    s.turn_player = "p2"
    s.put("p2", BIG, "field", don=5)
    s.put("p1", victim, "field", rest=True)
    p = s.build()
    p.attack("P2", p.uuid("p2", "field", BIG), p.uuid("p1", "field", victim))
    p.pass_battle()
    return p


def test_robin_leader_triggers_on_a_wisdom_non_blocker_ko():
    p = _ko_under_robin(GENZO)
    log = p.settle(accept=True)
    assert GENZO in p.ids("p1", "trash")
    assert _life(p, "p1") == 6, ("デッキの上 1 枚がライフに加わる", log)


@pytest.mark.parametrize("victim", [SHAKA, VANILLA], ids=["wisdom_blocker", "slash_attribute"])
def test_robin_leader_ignores_a_blocker_or_another_attribute(victim):
    p = _ko_under_robin(victim)
    p.settle(accept=True)
    assert victim in p.ids("p1", "trash")
    assert _life(p, "p1") == 5, "【ブロッカー】持ち・属性(知)以外の KO では誘発しない"
