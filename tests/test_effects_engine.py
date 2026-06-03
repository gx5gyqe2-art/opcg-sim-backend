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
