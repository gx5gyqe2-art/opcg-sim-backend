"""エンジン実行系（apply_action_to_engine）の効果セマンティクステスト。

実行:
    OPCG_LOG_SILENT=1 python -m pytest tests/test_effects_engine.py -q -s
    または: OPCG_LOG_SILENT=1 python tests/test_effects_engine.py
"""
from engine_helpers import action, make_game, make_instance, make_master
from opcg_sim.src.models.effect_types import Ability, GameAction, ValueSource
from opcg_sim.src.models.enums import ActionType, CardType, TriggerType


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
