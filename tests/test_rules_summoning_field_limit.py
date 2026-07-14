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
    """MAIN フェイズで p1 が手番、検証はバイパス。

    最初のターン(turn_count<=2)はアタック禁止のため、攻撃を伴う検証では
    通常進行中のターン(turn_count=3)を既定とする。
    """
    gm.turn_player = p1
    gm.opponent = gm.p2
    gm.phase = Phase.MAIN
    gm.turn_count = 3
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


# ──────────────── 最初のターンのアタック禁止 ────────────────

def test_first_player_cannot_attack_on_turn1():
    """先攻の最初のターン(turn_count=1)はリーダーでもアタックできない。"""
    gm, p1, p2 = make_game()
    _setup_main(gm, p1)
    gm.turn_count = 1
    with pytest.raises(ValueError, match="最初のターン"):
        gm.declare_attack(p1.leader, p2.leader)


def test_second_player_cannot_attack_on_turn2():
    """後攻の最初のターン(turn_count=2)もリーダー・キャラともにアタックできない。"""
    gm, p1, p2 = make_game()
    gm.turn_player = p2
    gm.opponent = p1
    gm.phase = Phase.MAIN
    gm.turn_count = 2
    gm._validate_action = lambda *a, **k: True
    with pytest.raises(ValueError, match="最初のターン"):
        gm.declare_attack(p2.leader, p1.leader)


def test_attack_allowed_from_turn3():
    """先攻の2ターン目(turn_count=3)以降はアタックできる。"""
    gm, p1, p2 = make_game()
    _setup_main(gm, p1)  # turn_count=3 を設定
    assert gm.turn_count == 3
    gm.declare_attack(p1.leader, p2.leader)
    assert gm.active_battle is not None


def test_first_turn_attack_excluded_from_legal_actions():
    """get_legal_actions は最初のターン(turn_count<=2)に ATTACK 手を列挙しない。"""
    for tc in (1, 2):
        gm, p1, p2 = make_game()
        gm.turn_player = p1 if tc == 1 else p2
        gm.opponent = p2 if tc == 1 else p1
        gm.phase = Phase.MAIN
        gm.turn_count = tc
        actor = gm.turn_player
        # MAIN フェイズ + 手番設定で get_pending_request() が MAIN_ACTION を返す。
        moves = gm.get_legal_actions(actor)
        assert all(m.get("action_type") != "ATTACK" for m in moves), \
            f"turn_count={tc} で ATTACK 手が列挙された"


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


# ──────────────── 押し出し（超過トラッシュ）と【登場時】の順序 ────────────────
# 実ルールでは6枚目のキャラは並存しない＝押し出しの確定が先、【登場時】等の効果解決は後。
# 押し出し選択で中断中の ON_PLAY は誘発待ち行列（_pending_triggers）へ積まれ、
# FIELD_OVERFLOW_TRASH の解決時に消化される。

def _onplay_draw_char(name="6体目"):
    """【登場時】カード1枚を引く を持つテストキャラ。"""
    from opcg_sim.src.models.effect_types import Ability, GameAction, ValueSource
    from opcg_sim.src.models.enums import TriggerType
    onplay = Ability(trigger=TriggerType.ON_PLAY,
                     effect=GameAction(type=ActionType.DRAW, value=ValueSource(base=1)),
                     raw_text="【登場時】カード1枚を引く。")
    master = make_master(card_id=f"C-{name}", name=name, type=CardType.CHARACTER,
                         power=5000, abilities=(onplay,))
    return make_instance(master)


def test_field_limit_prompt_precedes_on_play_effect():
    """手札からの登場: 押し出し選択が【登場時】より先に立ち、効果は押し出し確定後に解決される。"""
    gm, p1, p2 = make_game()
    _setup_main(gm, p1)
    existing = _fill_field(p1, 5)
    p1.deck.extend(_char(f"山{i}") for i in range(3))
    newcard = _onplay_draw_char()
    p1.hand.append(newcard)
    hand_before = len(p1.hand) - 1  # newcard がプレイで手札を離れた後の枚数
    gm.play_card_action(p1, newcard)
    # 押し出し選択が先。【登場時】ドローはまだ解決されていない（待ち行列に退避）。
    assert gm.active_interaction["action_type"] == "FIELD_OVERFLOW_TRASH"
    assert len(p1.hand) == hand_before
    assert len(gm._pending_triggers) == 1
    gm.resolve_interaction(p1, {"selected_uuids": [existing[0].uuid]})
    # 押し出し確定後に【登場時】が解決される。
    assert gm.active_interaction is None
    assert not gm._pending_triggers
    assert len(p1.hand) == hand_before + 1
    assert len(p1.field) == 5 and newcard in p1.field


def test_field_limit_prompt_precedes_on_play_effect_via_effect_play():
    """効果による登場（PLAY_CARD）でも押し出し選択が先、【登場時】は押し出し確定後。"""
    gm, p1, p2 = make_game()
    _setup_main(gm, p1)
    existing = _fill_field(p1, 5)
    p1.deck.extend(_char(f"山{i}") for i in range(3))
    newcard = _onplay_draw_char("効果登場6体目")
    p1.hand.append(newcard)
    hand_before = len(p1.hand) - 1
    gm.apply_action_to_engine(p1, action(ActionType.PLAY_CARD), [newcard], 0)
    assert gm.active_interaction["action_type"] == "FIELD_OVERFLOW_TRASH"
    assert len(p1.hand) == hand_before
    assert len(gm._pending_triggers) == 1
    gm.resolve_interaction(p1, {"selected_uuids": [existing[0].uuid]})
    assert not gm._pending_triggers
    assert len(p1.hand) == hand_before + 1
    assert len(p1.field) == 5 and newcard in p1.field
