"""エンジン実行系（apply_action_to_engine）の効果セマンティクステスト。

実行:
    OPCG_LOG_SILENT=1 python -m pytest tests/test_effects_engine.py -q -s
    または: OPCG_LOG_SILENT=1 python tests/test_effects_engine.py
"""
from engine_helpers import action, make_game, make_instance, make_master
from opcg_sim.src.models.effect_types import Ability, Condition, GameAction, ValueSource
from opcg_sim.src.models.enums import (
    ActionType,
    CardType,
    CompareOperator,
    ConditionType,
    Player,
    TriggerType,
    Zone,
)


def test_ramp_don_active():
    """RAMP_DON: ドン!!デッキからアクティブで追加。"""
    gm, p1, _ = make_game()
    assert len(p1.don_deck) == 10 and len(p1.don_active) == 0
    ok = gm.apply_action_to_engine(p1, action(ActionType.RAMP_DON, value=2), [], 2)
    assert ok
    assert len(p1.don_active) == 2
    assert len(p1.don_deck) == 8
    assert all(not d.is_rest for d in p1.don_active)


def test_ramp_don_rested():
    """RAMP_DON + status=RESTED: レストで追加（レスト状態でコストエリアへ）。"""
    gm, p1, _ = make_game()
    ok = gm.apply_action_to_engine(
        p1, action(ActionType.RAMP_DON, value=1, status="RESTED"), [], 1
    )
    assert ok
    assert len(p1.don_rested) == 1
    assert len(p1.don_active) == 0
    assert len(p1.don_deck) == 9
    assert p1.don_rested[0].is_rest is True


def test_return_don_from_field():
    """RETURN_DON: 場のドン!!をN枚ドン!!デッキへ戻す（ドン‼-N）。"""
    gm, p1, _ = make_game()
    # 場に active 2, rested 2 を用意（don_deck から移す）
    for _ in range(2):
        p1.don_active.append(p1.don_deck.pop(0))
    for _ in range(2):
        d = p1.don_deck.pop(0)
        d.is_rest = True
        p1.don_rested.append(d)
    assert len(p1.don_deck) == 6

    ok = gm.apply_action_to_engine(p1, action(ActionType.RETURN_DON, value=2), [], 2)
    assert ok
    # 場のドンが計4→2に、don_deck が 6→8 に戻る
    assert len(p1.don_active) + len(p1.don_rested) == 2
    assert len(p1.don_deck) == 8
    # 戻ったドンは非レストに正規化されている
    assert all(not d.is_rest for d in p1.don_deck)


def test_return_don_not_enough():
    """場のドンが要求数に満たない場合でも、ある分だけ戻して落ちない。"""
    gm, p1, _ = make_game()
    p1.don_active.append(p1.don_deck.pop(0))  # 場に1枚だけ
    ok = gm.apply_action_to_engine(p1, action(ActionType.RETURN_DON, value=3), [], 3)
    assert ok
    assert len(p1.don_active) + len(p1.don_rested) == 0
    assert len(p1.don_deck) == 10


def test_execute_main_effect_reinvokes_main():
    """EXECUTE_MAIN_EFFECT: トリガーが自身の ACTIVATE_MAIN 効果(ここでは DRAW2)を再発動。"""
    gm, p1, _ = make_game()
    # デッキを5枚用意
    for i in range(5):
        p1.deck.append(make_instance(make_master(card_id=f"D-{i}"), owner=p1.name))

    # 自身の【メイン】= カード2枚ドロー を持つカード
    main_ability = Ability(
        trigger=TriggerType.ACTIVATE_MAIN,
        effect=GameAction(type=ActionType.DRAW, value=ValueSource(base=2)),
    )
    master = make_master(card_id="E-001", name="再発動イベント", type=CardType.EVENT,
                         abilities=(main_ability,))
    source = make_instance(master, owner=p1.name)

    # トリガー能力: このカードの【メイン】効果を発動する
    trigger_ability = Ability(
        trigger=TriggerType.TRIGGER,
        effect=GameAction(type=ActionType.EXECUTE_MAIN_EFFECT),
    )

    assert len(p1.hand) == 0
    gm.resolve_ability(p1, trigger_ability, source_card=source)
    # 【メイン】の DRAW2 が再発動して手札2枚・デッキ3枚
    assert len(p1.hand) == 2
    assert len(p1.deck) == 3


def _make_field_char(player, name="戦士", power=5000):
    inst = make_instance(make_master(card_id=f"C-{name}", name=name, power=power), owner=player.name)
    player.field.append(inst)
    return inst


def test_battle_power_buff_expires_at_battle_end():
    """このバトル中のパワー増減はバトル終了で失効し、後続バトルへ持ち越さない。"""
    gm, p1, _ = make_game()
    card = _make_field_char(p1)
    base = card.get_power(True)

    gm.continuous.apply(card, "POWER", "THIS_BATTLE", amount=3000)
    assert card.get_power(True) == base + 3000

    gm.continuous.expire("BATTLE_END", gm.turn_count)
    assert card.get_power(True) == base  # 失効


def test_this_turn_restriction_expires_at_turn_end():
    """このターン中のアタック制限は、ターン終了で失効する。"""
    gm, p1, _ = make_game()
    card = _make_field_char(p1)

    gm.continuous.apply(card, "FLAG", "THIS_TURN", flag="ATTACK_DISABLE")
    assert "ATTACK_DISABLE" in card.timed_flags

    gm.continuous.expire("TURN_END", gm.turn_count)
    assert "ATTACK_DISABLE" not in card.timed_flags


def test_multi_turn_restriction_survives_one_turn_then_expires():
    """次の相手のターン終了時までの制限は、1回のターン終了を跨ぎ、その後失効する。"""
    gm, p1, _ = make_game()
    gm.turn_count = 5
    card = _make_field_char(p1)

    # turn 5 に適用 → expire_turn=6
    gm.continuous.apply(card, "FLAG", "UNTIL_NEXT_TURN_END", flag="ATTACK_DISABLE", expire_turn=6)

    # turn 5 の終了では失効しない
    gm.continuous.expire("TURN_END", 5)
    assert "ATTACK_DISABLE" in card.timed_flags

    # turn 6 の終了で失効する
    gm.continuous.expire("TURN_END", 6)
    assert "ATTACK_DISABLE" not in card.timed_flags


def test_reset_turn_status_keeps_timed_effects():
    """reset_turn_status は継続効果(timed_*)をクリアしない（ターン跨ぎ保持の要）。"""
    gm, p1, _ = make_game()
    card = _make_field_char(p1)
    gm.continuous.apply(card, "POWER", "UNTIL_NEXT_TURN_END", amount=1000, expire_turn=99)
    gm.continuous.apply(card, "FLAG", "UNTIL_NEXT_TURN_END", flag="ATTACK_DISABLE", expire_turn=99)
    card.power_buff = 500  # ターン境界で消えるべき通常バフ

    card.reset_turn_status()
    assert card.power_buff == 0           # 通常バフは消える
    assert card.timed_power == 1000        # 継続効果は残る
    assert "ATTACK_DISABLE" in card.timed_flags


def test_grant_keyword_adds_to_current_keywords():
    """GRANT_KEYWORD: status のキーワードを付与し、passive 再計算後も保持される。"""
    gm, p1, _ = make_game()
    card = _make_field_char(p1)
    assert not card.has_keyword("ブロッカー")

    ok = gm.apply_action_to_engine(
        p1, action(ActionType.GRANT_KEYWORD, status="ブロッカー"), [card], 0
    )
    assert ok
    assert card.has_keyword("ブロッカー")
    # _apply_passive_effects は current_keywords を master のコピーに戻すが、
    # 付与分は timed_keywords に保持されるため消えない（修正前は消えていた）。
    gm._apply_passive_effects(p1)
    assert card.has_keyword("ブロッカー")


def test_cost_reduction_this_turn_persists_and_expires():
    """COST_REDUCTION(THIS_TURN): passive 再計算で消えず、ターン終了で失効する。"""
    gm, p1, _ = make_game()
    card = make_instance(make_master(card_id="CC-1", cost=5), owner=p1.name)
    p1.field.append(card)

    gm.apply_action_to_engine(
        p1, action(ActionType.BUFF, value=-2, status="COST_REDUCTION", duration="THIS_TURN"),
        [card], -2)
    assert card.current_cost == 3
    gm._apply_passive_effects(p1)        # cost_buff をリセットするが timed_cost は残る
    assert card.current_cost == 3
    gm.continuous.expire("TURN_END", gm.turn_count)
    assert card.current_cost == 5


def test_grant_keyword_this_turn_expires_at_turn_end():
    """duration=THIS_TURN のキーワード付与はターン終了で失効する。"""
    gm, p1, _ = make_game()
    card = _make_field_char(p1)
    gm.apply_action_to_engine(
        p1, action(ActionType.GRANT_KEYWORD, status="速攻", duration="THIS_TURN"), [card], 0
    )
    assert card.has_keyword("速攻")
    gm.continuous.expire("TURN_END", gm.turn_count)
    assert not card.has_keyword("速攻")


def test_grant_keyword_dropped_when_leaving_field():
    """場を離れると付与キーワード（継続効果）は破棄される。"""
    gm, p1, _ = make_game()
    card = _make_field_char(p1)
    gm.apply_action_to_engine(
        p1, action(ActionType.GRANT_KEYWORD, status="ブロッカー"), [card], 0
    )
    assert card.has_keyword("ブロッカー")
    gm.move_card(card, Zone.TRASH, p1)
    assert not card.has_keyword("ブロッカー")


def test_life_recover_from_deck():
    """HEAL: デッキの上から value 枚をライフに加える（対象不要）。"""
    gm, p1, _ = make_game()
    for i in range(3):
        p1.deck.append(make_instance(make_master(card_id=f"D-{i}"), owner=p1.name))
    assert len(p1.life) == 0 and len(p1.deck) == 3

    ok = gm.apply_action_to_engine(p1, action(ActionType.HEAL, value=2), [], 2)
    assert ok
    assert len(p1.life) == 2
    assert len(p1.deck) == 1


def test_life_to_hand_move_card():
    """MOVE_CARD(dest=HAND): ライフのカードを手札へ移す。"""
    gm, p1, _ = make_game()
    life_card = make_instance(make_master(card_id="LF"), owner=p1.name)
    p1.life.append(life_card)
    assert len(p1.hand) == 0

    ok = gm.apply_action_to_engine(
        p1, action(ActionType.MOVE_CARD, destination=Zone.HAND), [life_card], 0
    )
    assert ok
    assert life_card in p1.hand
    assert life_card not in p1.life


def test_face_up_life_sets_flag():
    """FACE_UP_LIFE: status で is_face_up を切り替える。"""
    gm, p1, _ = make_game()
    life_card = make_instance(make_master(card_id="LF2"), owner=p1.name)
    p1.life.append(life_card)
    assert life_card.is_face_up is False

    ok = gm.apply_action_to_engine(
        p1, action(ActionType.FACE_UP_LIFE, status="UP"), [life_card], 0
    )
    assert ok
    assert life_card.is_face_up is True

    gm.apply_action_to_engine(
        p1, action(ActionType.FACE_UP_LIFE, status="DOWN"), [life_card], 0
    )
    assert life_card.is_face_up is False


def test_rest_don_by_value():
    """REST_DON: value 枚のアクティブドンをレストにする（枚数ベース）。"""
    gm, p1, _ = make_game()
    for _ in range(3):
        p1.don_active.append(p1.don_deck.pop(0))
    ok = gm.apply_action_to_engine(p1, action(ActionType.REST_DON, value=2), [], 2)
    assert ok
    assert len(p1.don_rested) == 2
    assert len(p1.don_active) == 1
    assert all(d.is_rest for d in p1.don_rested)


def test_active_don_by_value():
    """ACTIVE_DON: value 枚のレストドンをアクティブにする（枚数ベース, target=None）。"""
    gm, p1, _ = make_game()
    for _ in range(2):
        d = p1.don_deck.pop(0)
        d.is_rest = True
        p1.don_rested.append(d)
    ok = gm.apply_action_to_engine(p1, action(ActionType.ACTIVE_DON, value=1), [], 1)
    assert ok
    assert len(p1.don_active) == 1
    assert len(p1.don_rested) == 1
    assert p1.don_active[0].is_rest is False


def test_attach_don_rested_multiple():
    """ATTACH_DON: status=RESTED で value 枚のレストドンをレストのまま付与する。"""
    gm, p1, _ = make_game()
    for _ in range(2):
        d = p1.don_deck.pop(0)
        d.is_rest = True
        p1.don_rested.append(d)
    target = _make_field_char(p1)

    ok = gm.apply_action_to_engine(
        p1, action(ActionType.ATTACH_DON, value=2, status="RESTED"), [target], 2
    )
    assert ok
    assert target.attached_don == 2
    assert len(p1.don_attached_cards) == 2
    assert all(d.is_rest for d in p1.don_attached_cards)
    assert len(p1.don_rested) == 0


def test_return_don_opponent_via_status():
    """RETURN_DON + status=OPPONENT: 相手の場のドンをドンデッキへ戻す。"""
    gm, p1, p2 = make_game()
    p2.don_active.append(p2.don_deck.pop(0))  # 相手の場に1枚
    assert len(p2.don_active) == 1 and len(p2.don_deck) == 9

    ok = gm.apply_action_to_engine(
        p1, action(ActionType.RETURN_DON, value=1, status="OPPONENT"), [], 1
    )
    assert ok
    assert len(p2.don_active) == 0
    assert len(p2.don_deck) == 10
    # 実行者(p1)のドンは不変
    assert len(p1.don_active) == 0


def test_turn_limit_blocks_second_activation():
    """【ターン1回】: 同一ターンの2回目は不発。ターンリセット後は再発動可。"""
    gm, p1, _ = make_game()
    for i in range(5):
        p1.deck.append(make_instance(make_master(card_id=f"D-{i}"), owner=p1.name))
    ab = Ability(
        trigger=TriggerType.ACTIVATE_MAIN,
        condition=Condition(type=ConditionType.TURN_LIMIT, value=1),
        effect=GameAction(type=ActionType.DRAW, value=ValueSource(base=1)),
    )
    src = make_instance(make_master(card_id="TL-1", abilities=(ab,)), owner=p1.name)
    p1.field.append(src)

    gm.resolve_ability(p1, ab, source_card=src)
    assert len(p1.hand) == 1
    gm.resolve_ability(p1, ab, source_card=src)   # 2回目は制限で不発
    assert len(p1.hand) == 1
    src.reset_turn_status()                        # ターン境界でカウンタが戻る
    gm.resolve_ability(p1, ab, source_card=src)
    assert len(p1.hand) == 2


def test_turn_limit_enforced_via_parsed_ability():
    """パーサ→リゾルバ統合: 【ターン1回】の起動メインが2回目に不発になる。"""
    from opcg_sim.src.core.effects.parser_v2 import EffectParserV2
    gm, p1, _ = make_game()
    for i in range(5):
        p1.deck.append(make_instance(make_master(card_id=f"D-{i}"), owner=p1.name))
    abilities = tuple(EffectParserV2().parse_card_text("【起動メイン】【ターン1回】カード1枚を引く。"))
    src = make_instance(make_master(card_id="TL-2", abilities=abilities), owner=p1.name)
    p1.field.append(src)

    gm.resolve_ability(p1, abilities[0], source_card=src)
    gm.resolve_ability(p1, abilities[0], source_card=src)
    assert len(p1.hand) == 1


def test_leader_trait_bracket_condition_gates_effect():
    """『X』を含む特徴の LEADER_TRAIT 条件が正しく評価される（満たす/満たさない）。"""
    from opcg_sim.src.core.effects.parser_v2 import EffectParserV2
    gm, p1, _ = make_game()
    for i in range(5):
        p1.deck.append(make_instance(make_master(card_id=f"D-{i}"), owner=p1.name))
    abilities = tuple(EffectParserV2().parse_card_text(
        "【起動メイン】自分のリーダーが『白ひげ海賊団』を含む特徴を持つ場合、カード1枚を引く。"))
    src = make_instance(make_master(card_id="LT-1", abilities=abilities), owner=p1.name)
    p1.field.append(src)

    # リーダーが該当特徴を持たない → 不発
    p1.leader = make_instance(make_master(card_id="LD-A", type=CardType.LEADER, traits=[]), owner=p1.name)
    gm.resolve_ability(p1, abilities[0], source_card=src)
    assert len(p1.hand) == 0

    # リーダーが該当特徴を持つ → 発動
    p1.leader = make_instance(
        make_master(card_id="LD-B", type=CardType.LEADER, traits=["白ひげ海賊団"]), owner=p1.name)
    gm.resolve_ability(p1, abilities[0], source_card=src)
    assert len(p1.hand) == 1


def test_field_count_condition_counts_rested_chars():
    """FIELD_COUNT(レストのキャラが2枚以上いる)が target フィルタ込みで評価される。"""
    from opcg_sim.src.core.effects.parser_v2 import EffectParserV2
    gm, p1, _ = make_game()
    for i in range(5):
        p1.deck.append(make_instance(make_master(card_id=f"D-{i}"), owner=p1.name))
    abilities = tuple(EffectParserV2().parse_card_text(
        "【起動メイン】自分のレストのキャラが2枚以上いる場合、カード1枚を引く。"))
    src = make_instance(make_master(card_id="FC-1", abilities=abilities), owner=p1.name)
    p1.field.append(src)  # src はアクティブ（カウント対象外）
    rested = []
    for i in range(2):
        c = make_instance(make_master(card_id=f"R-{i}"), owner=p1.name)
        c.is_rest = True
        p1.field.append(c)
        rested.append(c)

    gm.resolve_ability(p1, abilities[0], source_card=src)  # レスト2枚 >= 2 → 発動
    assert len(p1.hand) == 1
    rested[0].is_rest = False                               # レスト1枚に減らす
    gm.resolve_ability(p1, abilities[0], source_card=src)  # 1 < 2 → 不発
    assert len(p1.hand) == 1


def test_leader_color_multicolor_condition():
    """LEADER_COLOR(多色): リーダーが2色以上のときのみ発動する。"""
    from opcg_sim.src.core.effects.parser_v2 import EffectParserV2
    from opcg_sim.src.models.enums import Color
    gm, p1, _ = make_game()
    for i in range(5):
        p1.deck.append(make_instance(make_master(card_id=f"D-{i}"), owner=p1.name))
    abilities = tuple(EffectParserV2().parse_card_text(
        "【起動メイン】自分のリーダーが多色の場合、カード1枚を引く。"))
    src = make_instance(make_master(card_id="LC-1", abilities=abilities), owner=p1.name)
    p1.field.append(src)

    # 単色リーダー（既定 [RED]）→ 不発
    gm.resolve_ability(p1, abilities[0], source_card=src)
    assert len(p1.hand) == 0
    # 多色化（colors は List のため frozen dataclass でも内容変更可）→ 発動
    p1.leader.master.colors.append(Color.BLUE)
    gm.resolve_ability(p1, abilities[0], source_card=src)
    assert len(p1.hand) == 1


def test_replacement_prevents_effect_ko_and_runs_alternative():
    """置換: 相手の効果KOを回避し、代わりに自分の手札を1枚捨てる。"""
    from opcg_sim.src.core.effects.parser_v2 import EffectParserV2
    gm, p1, p2 = make_game()
    abilities = tuple(EffectParserV2().parse_card_text(
        "このキャラがKOされる場合、代わりに自分の手札1枚を捨てる。"))
    target = make_instance(make_master(card_id="RP-1", abilities=abilities), owner=p1.name)
    p1.field.append(target)
    p1.hand.append(make_instance(make_master(card_id="H-1"), owner=p1.name))

    gm.apply_action_to_engine(p2, action(ActionType.KO), [target], 0)
    assert target in p1.field      # 置換でKO回避
    assert len(p1.hand) == 0       # 代わりに手札1枚を捨てた


def test_replacement_not_applied_when_alternative_impossible():
    """代わりの行動が取れない（手札なし）場合は置換不成立で本来のKOが起こる。"""
    from opcg_sim.src.core.effects.parser_v2 import EffectParserV2
    gm, p1, p2 = make_game()
    abilities = tuple(EffectParserV2().parse_card_text(
        "このキャラがKOされる場合、代わりに自分の手札1枚を捨てる。"))
    target = make_instance(make_master(card_id="RP-2", abilities=abilities), owner=p1.name)
    p1.field.append(target)  # 手札なし

    gm.apply_action_to_engine(p2, action(ActionType.KO), [target], 0)
    assert target not in p1.field
    assert target in p1.trash


def test_replacement_prevents_battle_ko():
    """置換: バトルKOを回避し、代わりに手札を捨てる。"""
    from opcg_sim.src.core.effects.parser_v2 import EffectParserV2
    gm, p1, p2 = make_game()
    p1.deck.append(make_instance(make_master(card_id="DK"), owner=p1.name))
    p2.deck.append(make_instance(make_master(card_id="DK2"), owner=p2.name))
    attacker = make_instance(make_master(card_id="ATK", power=6000), owner=p1.name)
    p1.field.append(attacker)
    abilities = tuple(EffectParserV2().parse_card_text(
        "このキャラは、このターン中、バトルでKOされる場合、代わりに自分の手札1枚を捨てる。"))
    target = make_instance(make_master(card_id="RB", power=5000, abilities=abilities), owner=p2.name)
    p2.field.append(target)
    p2.hand.append(make_instance(make_master(card_id="H-2"), owner=p2.name))

    gm.active_battle = {
        "attacker": attacker, "target": target,
        "attacker_owner": p1, "target_owner": p2, "counter_buff": 0,
    }
    gm.resolve_attack()
    assert target in p2.field       # バトルKOを置換で回避
    assert len(p2.hand) == 0        # 代わりに手札1枚を捨てた


def _prevent_leave_master(card_id, status, condition=None):
    ab = Ability(
        trigger=TriggerType.PASSIVE,
        condition=condition,
        effect=GameAction(type=ActionType.PREVENT_LEAVE, status=status),
    )
    return make_master(card_id=card_id, name=f"被保護{status}", abilities=(ab,))


def test_prevent_leave_blocks_opponent_effect_ko_when_condition_met():
    """トラッシュ7枚以上の場合、相手の効果KOで場を離れない。条件を満たさなければKOされる。"""
    gm, p1, p2 = make_game()
    cond = Condition(type=ConditionType.TRASH_COUNT, operator=CompareOperator.GE, value=7, player=Player.SELF)
    target = make_instance(_prevent_leave_master("PL-1", "LEAVE", cond), owner=p1.name)
    p1.field.append(target)

    # トラッシュ7枚 → 保護有効。相手(p2)の効果KOは通らない
    for i in range(7):
        p1.trash.append(make_instance(make_master(card_id=f"TR-{i}"), owner=p1.name))
    gm.apply_action_to_engine(p2, action(ActionType.KO), [target], 0)
    assert target in p1.field

    # トラッシュを6枚に減らす → 保護無効。相手の効果KOが通る
    p1.trash.pop()
    gm.apply_action_to_engine(p2, action(ActionType.KO), [target], 0)
    assert target not in p1.field
    assert target in p1.trash


def test_prevent_leave_does_not_block_own_effect():
    """「相手の効果で」場を離れない: 自分の効果による移動は防がない。"""
    gm, p1, _ = make_game()
    cond = Condition(type=ConditionType.TRASH_COUNT, operator=CompareOperator.GE, value=0, player=Player.SELF)
    target = make_instance(_prevent_leave_master("PL-2", "LEAVE", cond), owner=p1.name)
    p1.field.append(target)
    # 自分(p1)の効果で手札に戻す → 保護対象外なので戻る
    gm.apply_action_to_engine(p1, action(ActionType.BOUNCE), [target], 0)
    assert target not in p1.field
    assert target in p1.hand


def test_prevent_battle_ko_in_resolve_attack():
    """バトルでKOされない: パワー負けでも戦闘ではKOされない。"""
    gm, p1, p2 = make_game()
    p1.deck.append(make_instance(make_master(card_id="DK"), owner=p1.name))  # check_victory 回避
    p2.deck.append(make_instance(make_master(card_id="DK2"), owner=p2.name))
    attacker = make_instance(make_master(card_id="ATK", power=6000), owner=p1.name)
    p1.field.append(attacker)
    target = make_instance(_prevent_leave_master("PL-3", "BATTLE_KO"), owner=p2.name)
    p2.field.append(target)

    gm.active_battle = {
        "attacker": attacker, "target": target,
        "attacker_owner": p1, "target_owner": p2, "counter_buff": 0,
    }
    gm.resolve_attack()
    # 6000 >= 5000 だが BATTLE_KO 保護で場に残る
    assert target in p2.field


def test_trash_from_deck_mills_top():
    """TRASH_FROM_DECK: 自分のデッキ上から value 枚をトラッシュへ送る（mill）。"""
    gm, p1, _ = make_game()
    cards = [make_instance(make_master(card_id=f"M-{i}"), owner=p1.name) for i in range(5)]
    p1.deck.extend(cards)
    ok = gm.apply_action_to_engine(p1, action(ActionType.TRASH_FROM_DECK, value=2), [], 2)
    assert ok
    assert len(p1.deck) == 3
    assert len(p1.trash) == 2
    # 上から（先頭から）送られる
    assert p1.trash == cards[:2]


def test_trash_from_deck_stops_when_empty():
    """デッキが要求枚数に満たなくても、ある分だけ送って落ちない。"""
    gm, p1, _ = make_game()
    p1.deck.append(make_instance(make_master(card_id="M-only"), owner=p1.name))
    ok = gm.apply_action_to_engine(p1, action(ActionType.TRASH_FROM_DECK, value=3), [], 3)
    assert ok
    assert len(p1.deck) == 0
    assert len(p1.trash) == 1


def test_trash_from_deck_opponent():
    """status=OPPONENT で相手のデッキ上をトラッシュへ送る。"""
    gm, p1, p2 = make_game()
    p2.deck.extend(make_instance(make_master(card_id=f"O-{i}"), owner=p2.name) for i in range(3))
    ok = gm.apply_action_to_engine(
        p1, action(ActionType.TRASH_FROM_DECK, value=1, status="OPPONENT"), [], 1
    )
    assert ok
    assert len(p2.deck) == 2 and len(p2.trash) == 1
    assert len(p1.trash) == 0  # 自分のデッキは減らない


def test_trash_self_moves_to_trash_without_ko_trigger():
    """TRASH(target=SOURCE): このキャラをトラッシュへ。KO ではないので場から消えるだけ。"""
    gm, p1, _ = make_game()
    card = _make_field_char(p1, name="自壊")
    ok = gm.apply_action_to_engine(p1, action(ActionType.TRASH), [card], 0)
    assert ok
    assert card not in p1.field
    assert card in p1.trash


def test_active_sets_card_active():
    """ACTIVE(target=SOURCE): レストのキャラをアクティブに戻す。"""
    gm, p1, _ = make_game()
    card = _make_field_char(p1, name="再起")
    card.is_rest = True
    ok = gm.apply_action_to_engine(p1, action(ActionType.ACTIVE), [card], 0)
    assert ok
    assert card.is_rest is False


def test_bounce_moves_to_owner_hand():
    """BOUNCE: フィールドのカードを持ち主の手札へ戻す。"""
    gm, p1, p2 = make_game()
    target = _make_field_char(p2, name="被バウンス")
    assert target in p2.field
    ok = gm.apply_action_to_engine(p1, action(ActionType.BOUNCE), [target], 0)
    assert ok
    assert target not in p2.field
    assert target in p2.hand


def test_deck_bottom_sends_hand_card():
    """DECK_BOTTOM(zone=HAND): 手札のカードをデッキ下へ送る。"""
    gm, p1, _ = make_game()
    card = make_instance(make_master(card_id="H-001"), owner=p1.name)
    p1.hand.append(card)
    ok = gm.apply_action_to_engine(p1, action(ActionType.DECK_BOTTOM), [card], 0)
    assert ok
    assert card not in p1.hand
    assert p1.deck[-1] is card  # デッキの末尾（bottom）へ


def test_play_card_from_trash_reanimates():
    """PLAY_CARD: トラッシュのカードをフィールドへ出す（リアニメーション）。"""
    gm, p1, _ = make_game()
    card = make_instance(make_master(card_id="T-001", name="甦るキャラ"), owner=p1.name)
    p1.trash.append(card)
    assert card in p1.trash
    ok = gm.apply_action_to_engine(p1, action(ActionType.PLAY_CARD), [card], 0)
    assert ok
    assert card not in p1.trash
    assert card in p1.field
    assert card.is_newly_played is True


def test_play_card_rested_enters_rested():
    """PLAY_CARD + status=RESTED: レストで登場させる。"""
    gm, p1, _ = make_game()
    from opcg_sim.src.models.effect_types import GameAction, ValueSource
    card = make_instance(make_master(card_id="R-001", name="レスト登場"), owner=p1.name)
    p1.trash.append(card)
    act = GameAction(type=ActionType.PLAY_CARD, value=ValueSource(base=0), status="RESTED")
    ok = gm.apply_action_to_engine(p1, act, [card], 0)
    assert ok
    assert card in p1.field
    assert card.is_rest is True


def test_reveal_keeps_card_in_hand():
    """REVEAL: 手札のカードを公開しても盤面（手札）は動かず成功する。"""
    gm, p1, _ = make_game()
    card = make_instance(make_master(card_id="EV-1", name="公開イベント"), owner=p1.name)
    p1.hand.append(card)
    ok = gm.apply_action_to_engine(p1, action(ActionType.REVEAL), [card], 0)
    assert ok
    assert card in p1.hand  # 公開しても手札に残る
    assert len(p1.trash) == 0


def test_reveal_no_targets_is_noop_success():
    """REVEAL: 公開対象が無くても（任意公開）落ちずに成功扱い。"""
    gm, p1, _ = make_game()
    ok = gm.apply_action_to_engine(p1, action(ActionType.REVEAL), [], 0)
    assert ok


def test_deck_search_look_grab_remaining_flow():
    """サーチの一連フロー: LOOK(deck→temp) → MOVE_CARD(temp→hand) → DECK_BOTTOM(残りtemp→deck下)。

    デッキ上に [c0(cost2), c1(cost5), c2(cost1), c3(cost4)] を積み、コスト4以上を1枚取得。
    """
    from opcg_sim.src.models.effect_types import GameAction, TargetQuery, ValueSource
    gm, p1, _ = make_game()
    costs = [2, 5, 1, 4]
    deck = [make_instance(make_master(card_id=f"S-{i}", cost=c), owner=p1.name) for i, c in enumerate(costs)]
    # さらに底にダミーを積んでおく（LOOK が掘る範囲外）
    bottom = make_instance(make_master(card_id="BOT"), owner=p1.name)
    p1.deck = list(deck) + [bottom]

    # 1) LOOK 4 → temp に上から4枚
    assert gm.apply_action_to_engine(p1, action(ActionType.LOOK, value=4), [], 4)
    assert len(p1.temp_zone) == 4
    assert p1.deck == [bottom]

    # 2) MOVE_CARD(temp→hand): コスト5以上(=ここでは cost>=5 の c1)を手札へ
    grab = GameAction(
        type=ActionType.MOVE_CARD,
        target=TargetQuery(player=Player.SELF, zone=Zone.TEMP, cost_min=5, count=1, is_up_to=True),
        destination=Zone.HAND,
        value=ValueSource(base=0),
    )
    picked = [c for c in p1.temp_zone if c.master.cost >= 5]
    assert gm.apply_action_to_engine(p1, grab, picked, 0)
    assert deck[1] in p1.hand
    assert len(p1.temp_zone) == 3

    # 3) DECK_BOTTOM(残りの temp 全件→デッキ下)
    remaining = list(p1.temp_zone)
    db = GameAction(
        type=ActionType.DECK_BOTTOM,
        target=TargetQuery(player=Player.SELF, zone=Zone.TEMP, select_mode="REMAINING", count=-1),
        value=ValueSource(base=0),
    )
    assert gm.apply_action_to_engine(p1, db, remaining, 0)
    assert len(p1.temp_zone) == 0  # temp リーク無し
    # 残り3枚がデッキ下に積まれた（bottom の後ろ）
    assert p1.deck[0] is bottom
    assert all(c in p1.deck[1:] for c in remaining)
    assert len(p1.deck) == 4


def test_deck_reveal_play_from_temp_flow():
    """デッキ公開→登場の一連フロー: LOOK(deck→temp) → PLAY_CARD(temp→field) → 残りをデッキ下。

    デッキ上に [c0(cost2 キャラ), c1, c2] を積み、上1枚を公開して TEMP へ。
    公開した1枚を登場させ、登場しなかった分（このケースは0枚）は temp に残さない。
    """
    from opcg_sim.src.models.effect_types import GameAction, TargetQuery, ValueSource
    gm, p1, _ = make_game()
    top = make_instance(make_master(card_id="R-0", cost=2), owner=p1.name)
    rest = [make_instance(make_master(card_id=f"R-{i}", cost=3), owner=p1.name) for i in (1, 2)]
    p1.deck = [top] + rest
    field_before = len(p1.field)

    # 1) LOOK 1 → temp に上から1枚（公開）
    assert gm.apply_action_to_engine(p1, action(ActionType.LOOK, value=1), [], 1)
    assert p1.temp_zone == [top]
    assert p1.deck == rest

    # 2) PLAY_CARD(temp→field): 公開した cost<=2 のキャラを登場
    play = GameAction(
        type=ActionType.PLAY_CARD,
        target=TargetQuery(player=Player.SELF, zone=Zone.TEMP, cost_max=2, count=1, is_up_to=True),
        destination=Zone.FIELD,
        value=ValueSource(base=0),
    )
    assert gm.apply_action_to_engine(p1, play, [top], 0)
    assert top in p1.field
    assert len(p1.field) == field_before + 1
    assert len(p1.temp_zone) == 0  # 登場で temp から抜けた（リーク無し）


def test_deck_reveal_play_rested_status():
    """公開→レストで登場（status=RESTED）: 登場したキャラが is_rest=True になる。"""
    from opcg_sim.src.models.effect_types import GameAction, TargetQuery, ValueSource
    gm, p1, _ = make_game()
    top = make_instance(make_master(card_id="RR-0", cost=2), owner=p1.name)
    p1.deck = [top, make_instance(make_master(card_id="RR-1"), owner=p1.name)]

    assert gm.apply_action_to_engine(p1, action(ActionType.LOOK, value=1), [], 1)
    play = GameAction(
        type=ActionType.PLAY_CARD,
        target=TargetQuery(player=Player.SELF, zone=Zone.TEMP, cost_max=2, count=1, is_up_to=True),
        destination=Zone.FIELD,
        status="RESTED",
        value=ValueSource(base=0),
    )
    assert gm.apply_action_to_engine(p1, play, [top], 0)
    assert top in p1.field
    assert top.is_rest is True


def _reveal_then_play_ability():
    """LOOK(1) → Branch(REVEALED_CARD_TRAIT: cost<=4 キャラ) → PLAY_CARD(TEMP, RESTED)。"""
    from opcg_sim.src.models.effect_types import Branch, Sequence, TargetQuery
    cond = Condition(
        type=ConditionType.REVEALED_CARD_TRAIT,
        player=Player.SELF,
        value={"cost": 4, "cost_op": CompareOperator.LE, "card_type": "キャラ"},
    )
    play = GameAction(
        type=ActionType.PLAY_CARD,
        target=TargetQuery(player=Player.SELF, zone=Zone.TEMP, count=1, is_up_to=True),
        destination=Zone.FIELD,
        status="RESTED",
        value=ValueSource(base=0),
    )
    return Ability(
        trigger=TriggerType.ACTIVATE_MAIN,
        effect=Sequence(actions=[
            GameAction(type=ActionType.LOOK, value=ValueSource(base=1)),
            Branch(condition=cond, if_true=play),
        ]),
    )


def test_reveal_conditional_play_match():
    """公開→条件一致: 公開したデッキトップ(cost4 キャラ)を選んで登場（レスト）させると temp が空になる。

    「登場させてもよい」（任意・最大1枚）は対象選択で中断するため、公開カードを選択して再開する。
    """
    gm, p1, _ = make_game()
    top = make_instance(make_master(card_id="RV-0", cost=4, type=CardType.CHARACTER), owner=p1.name)
    p1.deck = [top, make_instance(make_master(card_id="RV-1"), owner=p1.name)]
    field_before = len(p1.field)

    gm.resolve_ability(p1, _reveal_then_play_ability(), source_card=p1.leader)
    # LOOK→条件一致→PLAY_CARD(TEMP) が対象選択で中断する
    assert gm.active_interaction is not None
    assert gm.active_interaction["action_type"] == "SELECT_TARGET"
    gm.resolve_interaction(p1, {"selected_uuids": [top.uuid]})

    assert top in p1.field          # 公開カードが登場
    assert top.is_rest is True       # レスト登場
    assert len(p1.field) == field_before + 1
    assert top not in p1.temp_zone   # temp リーク無し


def test_reveal_conditional_play_no_match():
    """公開→条件不一致: 公開したデッキトップ(cost8)は登場せず、場は変化しない。"""
    gm, p1, _ = make_game()
    top = make_instance(make_master(card_id="RV-8", cost=8, type=CardType.CHARACTER), owner=p1.name)
    p1.deck = [top, make_instance(make_master(card_id="RV-9"), owner=p1.name)]
    field_before = len(p1.field)

    gm.resolve_ability(p1, _reveal_then_play_ability(), source_card=p1.leader)
    assert top not in p1.field           # 条件不一致なので登場しない
    assert len(p1.field) == field_before
    assert top in p1.temp_zone           # 公開カードは temp に残る（後続の残り処理対象）


def test_hand_to_deck_bottom():
    """DECK_BOTTOM(zone=HAND): 手札1枚をデッキ下へ（hand_to_deck ルールの実行検証）。"""
    from opcg_sim.src.models.effect_types import TargetQuery
    gm, p1, _ = make_game()
    hand_card = make_instance(make_master(card_id="H-1"), owner=p1.name)
    p1.hand.append(hand_card)
    tq = TargetQuery(player=Player.SELF, zone=Zone.HAND, count=1)
    ok = gm.apply_action_to_engine(p1, action(ActionType.DECK_BOTTOM, target=tq), [hand_card], 1)
    assert ok
    assert hand_card not in p1.hand
    assert hand_card in p1.deck  # デッキ下に加わった


def test_freeze_keeps_character_rested_after_refresh():
    """FREEZE: フリーズされたキャラはリフレッシュフェイズでアクティブになれない。"""
    gm, p1, p2 = make_game()
    char = make_instance(make_master(card_id="F-1", type=CardType.CHARACTER), owner=p2.name)
    char.is_rest = True
    p2.field.append(char)
    # FREEZE フラグを直接付与（パーサーを経由せず engine 層だけを検証）
    ok = gm.apply_action_to_engine(p1, action(ActionType.FREEZE), [char], 0)
    assert ok
    assert "FREEZE" in char.flags
    # p2 のリフレッシュを再現（refresh_all は turn_player を引数に取る）
    gm.refresh_all(p2)
    assert char.is_rest is True  # FREEZE 済みなので依然レスト


def test_negate_effect_sets_ability_disabled():
    """NEGATE_EFFECT: ability_disabled=True になり能力発動がブロックされる。"""
    gm, p1, p2 = make_game()
    char = make_instance(make_master(card_id="N-1", type=CardType.CHARACTER), owner=p2.name)
    p2.field.append(char)
    ok = gm.apply_action_to_engine(p1, action(ActionType.NEGATE_EFFECT), [char], 0)
    assert ok
    assert char.ability_disabled is True
    # reset_turn_status で THIS_TURN の無効化は解除される
    char.reset_turn_status()
    assert char.ability_disabled is False


def test_attack_active_allows_attacking_active_character():
    """ATTACK_ACTIVE キーワード持ちはアクティブキャラにアタック可能。
    _validate_action はゲームフロー依存なのでパッチし、攻撃可否の条件だけを検証する。
    """
    gm, p1, p2 = make_game()
    master = make_master(card_id="ATK-1", type=CardType.CHARACTER)
    master.keywords.add("ATTACK_ACTIVE")
    attacker = make_instance(master, owner=p1.name)
    defender = make_instance(make_master(card_id="DEF-1", type=CardType.CHARACTER), owner=p2.name)
    p1.field.append(attacker)
    p2.field.append(defender)
    defender.is_rest = False  # アクティブ状態（通常は攻撃不可）
    # _validate_action はゲームフロー（pending_request）依存なのでバイパス
    gm._validate_action = lambda player, action_type: None
    try:
        gm.declare_attack(attacker, defender)
        success = True
    except ValueError as e:
        success = "レスト" in str(e)  # レスト制約エラーなら失敗
        if not success:
            raise
    assert success, "ATTACK_ACTIVE 持ちはアクティブキャラを攻撃できるはず"


def test_attack_active_not_granted_means_no_active_attack():
    """ATTACK_ACTIVE を持たないキャラはアクティブキャラへの攻撃で ValueError。"""
    gm, p1, p2 = make_game()
    attacker = make_instance(make_master(card_id="ATK-2", type=CardType.CHARACTER), owner=p1.name)
    defender = make_instance(make_master(card_id="DEF-2", type=CardType.CHARACTER), owner=p2.name)
    p1.field.append(attacker)
    p2.field.append(defender)
    defender.is_rest = False
    gm._validate_action = lambda player, action_type: None
    raised = False
    try:
        gm.declare_attack(attacker, defender)
    except ValueError as e:
        if "レスト" in str(e):
            raised = True
    assert raised, "ATTACK_ACTIVE なしはアクティブキャラへの攻撃で例外を出すべき"


# ===== 新条件タイプ（GENERIC 分類拡充）のエンジンテスト =====

def _check_cond(gm, player, condition, source):
    """EffectResolver._check_condition を呼ぶ薄いラッパ。"""
    from opcg_sim.src.core.effects.resolver import EffectResolver
    return EffectResolver(gm)._check_condition(player, condition, source)


def test_source_state_is_rested():
    """SOURCE_STATE / IS_RESTED: レスト状態のときだけ True。"""
    gm, p1, _ = make_game()
    src = make_instance(make_master(), owner=p1.name)
    p1.field.append(src)
    cond = Condition(type=ConditionType.SOURCE_STATE, value="IS_RESTED")

    src.is_rest = True
    assert _check_cond(gm, p1, cond, src) is True

    src.is_rest = False
    assert _check_cond(gm, p1, cond, src) is False


def test_source_state_is_active():
    """SOURCE_STATE / IS_ACTIVE: アクティブ状態のときだけ True。"""
    gm, p1, _ = make_game()
    src = make_instance(make_master(), owner=p1.name)
    p1.field.append(src)
    cond = Condition(type=ConditionType.SOURCE_STATE, value="IS_ACTIVE")

    src.is_rest = False
    assert _check_cond(gm, p1, cond, src) is True

    src.is_rest = True
    assert _check_cond(gm, p1, cond, src) is False


def test_source_state_entered_this_turn():
    """SOURCE_STATE / ENTERED_THIS_TURN: is_newly_played が True のときだけ True。"""
    gm, p1, _ = make_game()
    src = make_instance(make_master(), owner=p1.name)
    p1.field.append(src)
    cond = Condition(type=ConditionType.SOURCE_STATE, value="ENTERED_THIS_TURN")

    src.is_newly_played = True
    assert _check_cond(gm, p1, cond, src) is True

    src.is_newly_played = False
    assert _check_cond(gm, p1, cond, src) is False


def test_source_state_power_ge():
    """SOURCE_STATE / POWER_GE: パワーが閾値以上のときだけ True。"""
    gm, p1, _ = make_game()
    src = make_instance(make_master(power=6000), owner=p1.name)
    p1.field.append(src)
    cond = Condition(
        type=ConditionType.SOURCE_STATE,
        value=("POWER", 7000),
        operator=CompareOperator.GE,
    )

    assert _check_cond(gm, p1, cond, src) is False  # 6000 < 7000

    src.power_buff = 1000  # 6000 + 1000 = 7000 → True
    assert _check_cond(gm, p1, cond, src) is True


def test_field_all_trait_exact():
    """FIELD_ALL_TRAIT: 全キャラが特定の特徴を持つときのみ True。"""
    from opcg_sim.src.models.enums import Attribute
    gm, p1, _ = make_game()
    m1 = make_master(card_id="C1", traits=["天竜人"])
    m2 = make_master(card_id="C2", traits=["天竜人", "海軍"])
    m3 = make_master(card_id="C3", traits=["麦わらの一味"])
    p1.field.append(make_instance(m1, owner=p1.name))
    p1.field.append(make_instance(m2, owner=p1.name))
    cond = Condition(type=ConditionType.FIELD_ALL_TRAIT, value=("天竜人", False), player=Player.SELF)

    assert _check_cond(gm, p1, cond, p1.leader) is True  # 両方天竜人

    p1.field.append(make_instance(m3, owner=p1.name))
    assert _check_cond(gm, p1, cond, p1.leader) is False  # 麦わらの一味が混在


def test_has_character_present_and_absent():
    """HAS_CHARACTER: キャラの存在/不在を正しく判定する。"""
    gm, p1, _ = make_game()
    luffy = make_instance(make_master(card_id="LF", name="ルフィ"), owner=p1.name)
    p1.field.append(luffy)
    cond_present = Condition(
        type=ConditionType.HAS_CHARACTER, value="ルフィ", operator=CompareOperator.GE, player=Player.SELF
    )
    cond_absent = Condition(
        type=ConditionType.HAS_CHARACTER, value="ゾロ", operator=CompareOperator.EQ, player=Player.SELF
    )

    assert _check_cond(gm, p1, cond_present, p1.leader) is True   # ルフィがいる
    assert _check_cond(gm, p1, cond_absent, p1.leader) is True    # ゾロがいない

    zoro = make_instance(make_master(card_id="ZR", name="ゾロ"), owner=p1.name)
    p1.field.append(zoro)
    assert _check_cond(gm, p1, cond_absent, p1.leader) is False   # ゾロが登場


def test_leader_attribute():
    """LEADER_ATTRIBUTE: リーダーの属性が一致するときだけ True。"""
    from opcg_sim.src.models.enums import Attribute
    gm, p1, _ = make_game()
    # make_player はデフォルト SLASH リーダーを作成する
    cond_slash = Condition(
        type=ConditionType.LEADER_ATTRIBUTE, value="斬", player=Player.SELF
    )
    cond_strike = Condition(
        type=ConditionType.LEADER_ATTRIBUTE, value="打", player=Player.SELF
    )
    assert _check_cond(gm, p1, cond_slash, p1.leader) is True
    assert _check_cond(gm, p1, cond_strike, p1.leader) is False


def test_rested_count():
    """RESTED_COUNT: フィールド＋リーダー＋ドン!! のレスト総数を正しく数える。"""
    gm, p1, _ = make_game()
    c1 = make_instance(make_master(card_id="C1"), owner=p1.name)
    c2 = make_instance(make_master(card_id="C2"), owner=p1.name)
    p1.field.extend([c1, c2])
    c1.is_rest = True
    p1.leader.is_rest = True

    from opcg_sim.src.models.models import DonInstance
    don = DonInstance(owner_id=p1.name)
    don.is_rest = True
    p1.don_rested.append(don)

    cond_ge3 = Condition(
        type=ConditionType.RESTED_COUNT,
        operator=CompareOperator.GE,
        value=3,
        player=Player.SELF,
    )
    cond_ge4 = Condition(
        type=ConditionType.RESTED_COUNT,
        operator=CompareOperator.GE,
        value=4,
        player=Player.SELF,
    )
    # rested: c1(1) + leader(1) + don(1) = 3
    assert _check_cond(gm, p1, cond_ge3, p1.leader) is True
    assert _check_cond(gm, p1, cond_ge4, p1.leader) is False


def test_opponent_removal_condition_power_filter():
    """OPPONENT_REMOVAL: 元々のパワーが閾値以下のカードだけ置換が発動する。"""
    from opcg_sim.src.core.effects.parser_v2 import EffectParserV2
    gm, p1, p2 = make_game()

    # 置換効果を持つ「守護者」カードを p1 フィールドに置く
    protector_abilities = tuple(EffectParserV2().parse_card_text(
        "自分の元々のパワー7000以下のキャラが相手の効果で場を離れる場合、代わりに自分の手札1枚を捨てる。"
    ))
    protector = make_instance(make_master(card_id="PT-1", name="守護者", abilities=protector_abilities), owner=p1.name)
    p1.field.append(protector)

    # 対象1: パワー6000（閾値以下） → 置換が発動して手札を捨て、フィールドに残る
    weak = make_instance(make_master(card_id="WK-1", power=6000), owner=p1.name)
    p1.field.append(weak)
    p1.hand.append(make_instance(make_master(card_id="H-1"), owner=p1.name))

    gm.apply_action_to_engine(p2, action(ActionType.KO), [weak], 0)
    assert weak in p1.field,   "パワー6000は置換で残るべき"
    assert len(p1.hand) == 0,  "代わりに手札を捨てたはず"

    # 対象2: パワー8000（閾値超え） → 置換不発動、KO される
    strong = make_instance(make_master(card_id="ST-1", power=8000), owner=p1.name)
    p1.field.append(strong)

    gm.apply_action_to_engine(p2, action(ActionType.KO), [strong], 0)
    assert strong not in p1.field, "パワー8000は置換なしにKOされるべき"
    assert strong in p1.trash


def test_opponent_removal_condition_trait_filter():
    """OPPONENT_REMOVAL: 特徴フィルタが一致するカードだけ置換が発動する。"""
    from opcg_sim.src.core.effects.parser_v2 import EffectParserV2
    gm, p1, p2 = make_game()

    protector_abilities = tuple(EffectParserV2().parse_card_text(
        "自分の特徴《エッグヘッド》を持つキャラが相手の効果でKOされる場合、代わりに自分の手札1枚を捨てる。"
    ))
    protector = make_instance(make_master(card_id="PT-2", name="守護者2", abilities=protector_abilities), owner=p1.name)
    p1.field.append(protector)

    # 特徴一致 → 置換
    egghead = make_instance(make_master(card_id="EH-1", traits=["エッグヘッド"]), owner=p1.name)
    p1.field.append(egghead)
    p1.hand.append(make_instance(make_master(card_id="H-2"), owner=p1.name))

    gm.apply_action_to_engine(p2, action(ActionType.KO), [egghead], 0)
    assert egghead in p1.field
    assert len(p1.hand) == 0

    # 特徴不一致 → 置換なし、KO
    other = make_instance(make_master(card_id="OT-1", traits=["麦わらの一味"]), owner=p1.name)
    p1.field.append(other)

    gm.apply_action_to_engine(p2, action(ActionType.KO), [other], 0)
    assert other not in p1.field
    assert other in p1.trash


def test_field_count_compare():
    """FIELD_COUNT_COMPARE: 自分のキャラ数が相手より少ない場合のみ True。"""
    gm, p1, p2 = make_game()
    cond = Condition(
        type=ConditionType.FIELD_COUNT_COMPARE,
        operator=CompareOperator.LT,
        player=Player.SELF,
    )
    # p1: 1枚, p2: 2枚
    p1.field.append(make_instance(make_master(card_id="A1"), owner=p1.name))
    p2.field.extend([
        make_instance(make_master(card_id="B1"), owner=p2.name),
        make_instance(make_master(card_id="B2"), owner=p2.name),
    ])
    assert _check_cond(gm, p1, cond, p1.leader) is True   # 1 < 2
    assert _check_cond(gm, p2, cond, p2.leader) is False  # 2 < 1 = False


def test_has_character_rested_state():
    """HAS_CHARACTER + IS_RESTED: 指名キャラがレストのときだけ True。"""
    gm, p1, _ = make_game()
    uta = make_instance(make_master(card_id="UTA", name="ウタ"), owner=p1.name)
    p1.field.append(uta)
    cond = Condition(
        type=ConditionType.HAS_CHARACTER,
        value=("ウタ", "IS_RESTED"),
        operator=CompareOperator.GE,
        player=Player.SELF,
    )
    uta.is_rest = False
    assert _check_cond(gm, p1, cond, p1.leader) is False

    uta.is_rest = True
    assert _check_cond(gm, p1, cond, p1.leader) is True


def test_revealed_card_trait_match():
    """REVEALED_CARD_TRAIT: context に公開カードがセットされ、特徴が一致する場合 True。"""
    gm, p1, _ = make_game()
    cond = Condition(
        type=ConditionType.REVEALED_CARD_TRAIT,
        value={"trait": "白ひげ海賊団", "trait_contains": True},
        player=Player.SELF,
    )
    from opcg_sim.src.core.effects.resolver import EffectResolver
    resolver = EffectResolver(gm)

    revealed = make_instance(make_master(card_id="WB1", traits=["白ひげ海賊団"]), owner=p1.name)
    resolver.context["last_revealed_card"] = revealed
    assert resolver._check_condition(p1, cond, p1.leader) is True

    other = make_instance(make_master(card_id="OP1", traits=["麦わらの一味"]), owner=p1.name)
    resolver.context["last_revealed_card"] = other
    assert resolver._check_condition(p1, cond, p1.leader) is False


def test_revealed_card_trait_cost_and_type():
    """REVEALED_CARD_TRAIT: コスト条件 + カードタイプも正しく評価する。"""
    from opcg_sim.src.models.enums import CardType
    gm, p1, _ = make_game()
    cond = Condition(
        type=ConditionType.REVEALED_CARD_TRAIT,
        value={"trait": "王下七武海", "trait_contains": False, "cost": 4, "cost_op": CompareOperator.LE, "card_type": "キャラ"},
        player=Player.SELF,
    )
    from opcg_sim.src.core.effects.resolver import EffectResolver
    resolver = EffectResolver(gm)

    match = make_instance(
        make_master(card_id="SL1", cost=3, traits=["王下七武海"], type=CardType.CHARACTER), owner=p1.name
    )
    resolver.context["last_revealed_card"] = match
    assert resolver._check_condition(p1, cond, p1.leader) is True

    # コストオーバー
    over_cost = make_instance(
        make_master(card_id="SL2", cost=5, traits=["王下七武海"], type=CardType.CHARACTER), owner=p1.name
    )
    resolver.context["last_revealed_card"] = over_cost
    assert resolver._check_condition(p1, cond, p1.leader) is False


def test_prev_action_succeeded():
    """PREV_ACTION / SUCCEEDED: last_action_success=True のときのみ True。"""
    gm, p1, _ = make_game()
    cond = Condition(type=ConditionType.PREV_ACTION, value="SUCCEEDED")
    from opcg_sim.src.core.effects.resolver import EffectResolver
    resolver = EffectResolver(gm)

    resolver.context["last_action_success"] = True
    resolver.context["_last_had_targets"] = True
    assert resolver._check_condition(p1, cond, p1.leader) is True

    resolver.context["last_action_success"] = False
    assert resolver._check_condition(p1, cond, p1.leader) is False


def test_prev_action_skipped():
    """PREV_ACTION / SKIPPED: last_action_success=False のときのみ True。"""
    gm, p1, _ = make_game()
    cond = Condition(type=ConditionType.PREV_ACTION, value="SKIPPED")
    from opcg_sim.src.core.effects.resolver import EffectResolver
    resolver = EffectResolver(gm)

    resolver.context["last_action_success"] = False
    assert resolver._check_condition(p1, cond, p1.leader) is True

    resolver.context["last_action_success"] = True
    resolver.context["_last_had_targets"] = True
    assert resolver._check_condition(p1, cond, p1.leader) is False


def test_don_count_compare():
    """DON_COUNT_COMPARE: 自分のドン!!が相手より多い場合のみ True。"""
    gm, p1, p2 = make_game()
    cond = Condition(
        type=ConditionType.DON_COUNT_COMPARE,
        operator=CompareOperator.GT,
        player=Player.SELF,
    )
    # p1: active=3, p2: active=2
    from opcg_sim.src.models.models import DonInstance
    for _ in range(3):
        p1.don_active.append(DonInstance(owner_id=p1.name))
    for _ in range(2):
        p2.don_active.append(DonInstance(owner_id=p2.name))

    assert _check_cond(gm, p1, cond, p1.leader) is True   # 3 > 2
    assert _check_cond(gm, p2, cond, p2.leader) is False  # 2 > 3 = False


def test_leader_state_is_active():
    """LEADER_STATE / IS_ACTIVE: リーダーがアクティブのときだけ True。"""
    gm, p1, _ = make_game()
    cond = Condition(type=ConditionType.LEADER_STATE, value="IS_ACTIVE")

    p1.leader.is_rest = False
    assert _check_cond(gm, p1, cond, p1.leader) is True

    p1.leader.is_rest = True
    assert _check_cond(gm, p1, cond, p1.leader) is False


def test_leader_state_power_le():
    """LEADER_STATE / POWER_LE: リーダーのパワーが閾値以下のときだけ True。"""
    gm, p1, _ = make_game()
    cond = Condition(
        type=ConditionType.LEADER_STATE,
        value=("POWER", 5000),
        operator=CompareOperator.LE,
    )

    p1.leader = make_instance(make_master(card_id="L1", power=4000), owner=p1.name)
    assert _check_cond(gm, p1, cond, p1.leader) is True   # 4000 <= 5000

    p1.leader = make_instance(make_master(card_id="L2", power=6000), owner=p1.name)
    assert _check_cond(gm, p1, cond, p1.leader) is False  # 6000 > 5000


if __name__ == "__main__":
    import traceback

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
            print(f"PASS {t.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
    print(f"\n=== engine: {passed} passed, {failed} failed / {len(tests)} ===")
    raise SystemExit(1 if failed else 0)
