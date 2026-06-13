"""コアルール修正のテスト: 召喚酔い/速攻 と 場のキャラ5体上限（強制トラッシュ）。

engine_helpers の make_game/make_instance/make_master を用い、実際の GameManager 上で
ルール挙動を検証する。Firestore 等の外部依存は不要。
"""
import conftest  # noqa: F401  (google スタブ注入 & sys.path 設定)
import pytest

from engine_helpers import make_game, make_instance, make_master, action
from opcg_sim.src.models.enums import CardType, Phase, ActionType, Zone


def _char(name="キャラ", owner="P1", **kw):
    return make_instance(make_master(card_id=f"C-{name}", name=name, type=CardType.CHARACTER, power=5000), owner=owner, **kw)


def _setup_main(gm, p1):
    """MAIN フェイズで p1 が手番、検証はバイパス。"""
    gm.turn_player = p1
    gm.opponent = gm.p2
    gm.phase = Phase.MAIN
    gm._validate_action = lambda *a, **k: True


# ──────────────── 召喚酔い / 速攻 ────────────────

def test_newly_played_character_cannot_attack():
    gm, p1, p2 = make_game()
    _setup_main(gm, p1)
    attacker = _char("新規")
    attacker.is_newly_played = True
    p1.field.append(attacker)
    with pytest.raises(ValueError, match="登場したターン"):
        gm.declare_attack(attacker, p2.leader)


def test_rush_keyword_allows_attack_on_play_turn():
    gm, p1, p2 = make_game()
    _setup_main(gm, p1)
    attacker = _char("速攻持ち")
    attacker.is_newly_played = True
    attacker.timed_keywords.add("速攻")
    p1.field.append(attacker)
    gm.declare_attack(attacker, p2.leader)
    assert gm.active_battle is not None
    assert attacker.is_rest is True


def test_non_newly_played_character_can_attack():
    gm, p1, p2 = make_game()
    _setup_main(gm, p1)
    attacker = _char("既存")
    attacker.is_newly_played = False
    p1.field.append(attacker)
    gm.declare_attack(attacker, p2.leader)
    assert gm.active_battle is not None


def test_leader_not_affected_by_summoning_sickness():
    gm, p1, p2 = make_game()
    _setup_main(gm, p1)
    # リーダーは is_newly_played=False。相手のレストキャラを攻撃可能。
    target = _char("的", owner="P2")
    target.is_rest = True
    p2.field.append(target)
    gm.declare_attack(p1.leader, target)
    assert gm.active_battle is not None


# ──────────────── 場のキャラ5体上限（強制トラッシュ） ────────────────

def _fill_field(player, n=5):
    cards = [_char(f"場{i}", owner=player.name) for i in range(n)]
    player.field.extend(cards)
    return cards


def test_field_limit_triggers_forced_trash_on_hand_play():
    gm, p1, p2 = make_game()
    _setup_main(gm, p1)
    _fill_field(p1, 5)
    newcard = _char("6体目")
    p1.hand.append(newcard)
    gm.play_card_action(p1, newcard)
    # 6体になったので強制トラッシュの選択要求が立つ
    ai = gm.active_interaction
    assert ai is not None
    assert ai["action_type"] == "FIELD_OVERFLOW_TRASH"
    assert ai["player_id"] == p1.name
    assert ai["constraints"] == {"min": 1, "max": 1}
    assert ai["can_skip"] is False
    assert len(ai["candidates"]) == 6  # 新規含む全キャラ
    # FE 向けには SEARCH_AND_SELECT として配信される
    pending = gm.get_pending_request()
    assert pending["action"] == "SEARCH_AND_SELECT"


def test_field_limit_resolution_trashes_selected():
    gm, p1, p2 = make_game()
    _setup_main(gm, p1)
    existing = _fill_field(p1, 5)
    newcard = _char("6体目")
    p1.hand.append(newcard)
    gm.play_card_action(p1, newcard)
    victim = existing[0]
    gm.resolve_interaction(p1, {"selected_uuids": [victim.uuid]})
    assert gm.active_interaction is None
    assert victim in p1.trash
    assert victim not in p1.field
    assert len(p1.field) == 5


def test_field_limit_can_trash_the_newly_played_card():
    gm, p1, p2 = make_game()
    _setup_main(gm, p1)
    _fill_field(p1, 5)
    newcard = _char("6体目")
    p1.hand.append(newcard)
    gm.play_card_action(p1, newcard)
    # 新規登場カード自身も候補に含まれ、選んでトラッシュできる
    assert newcard.uuid in gm.active_interaction["selectable_uuids"]
    gm.resolve_interaction(p1, {"selected_uuids": [newcard.uuid]})
    assert newcard in p1.trash
    assert len(p1.field) == 5


def test_field_at_exactly_five_no_overflow():
    gm, p1, p2 = make_game()
    _setup_main(gm, p1)
    _fill_field(p1, 4)
    newcard = _char("5体目")
    p1.hand.append(newcard)
    gm.play_card_action(p1, newcard)
    assert gm.active_interaction is None
    assert len(p1.field) == 5


def test_field_limit_via_effect_play():
    gm, p1, p2 = make_game()
    _setup_main(gm, p1)
    _fill_field(p1, 5)
    newcard = _char("効果登場")
    p1.hand.append(newcard)
    gm.apply_action_to_engine(p1, action(ActionType.PLAY_CARD), [newcard], 0)
    assert gm.active_interaction is not None
    assert gm.active_interaction["action_type"] == "FIELD_OVERFLOW_TRASH"
    assert len(p1.field) == 6
