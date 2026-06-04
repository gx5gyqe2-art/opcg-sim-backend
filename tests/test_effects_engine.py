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
    """GRANT_KEYWORD: status のキーワードを対象の current_keywords に付与する。"""
    gm, p1, _ = make_game()
    card = _make_field_char(p1)
    assert "ブロッカー" not in card.current_keywords

    ok = gm.apply_action_to_engine(
        p1, action(ActionType.GRANT_KEYWORD, status="ブロッカー"), [card], 0
    )
    assert ok
    assert "ブロッカー" in card.current_keywords


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
