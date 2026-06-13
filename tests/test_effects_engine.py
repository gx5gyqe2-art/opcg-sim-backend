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


def test_return_don_cost_requires_enough_don():
    """「ドン!!-N」コスト: コストエリアのドン!!が N 枚未満なら能力は発動できない
    （払えないコストで効果だけ通ってしまう不具合の回帰ガード）。"""
    gm, p1, _ = make_game()
    for i in range(5):
        p1.deck.append(make_instance(make_master(card_id=f"D-{i}"), owner=p1.name))
    ab = Ability(
        trigger=TriggerType.ACTIVATE_MAIN,
        cost=GameAction(type=ActionType.RETURN_DON, value=ValueSource(base=1)),
        effect=GameAction(type=ActionType.DRAW, value=ValueSource(base=1)),
    )
    src = make_instance(make_master(card_id="RD-1", abilities=(ab,)), owner=p1.name)
    p1.field.append(src)

    # ドン!!が場に無い → コスト不成立。ドローしない。
    assert len(p1.don_active) == 0 and len(p1.don_rested) == 0
    gm.resolve_ability(p1, ab, source_card=src)
    assert len(p1.hand) == 0

    # ドン!!を1枚用意 → 発動可。どのドン!!を戻すか選択を求められ、選択後に返却＆ドロー。
    p1.don_active.append(p1.don_deck.pop(0))
    deck_before = len(p1.don_deck)
    gm.resolve_ability(p1, ab, source_card=src)
    assert gm.active_interaction is not None
    assert gm.active_interaction["action_type"] == "SELECT_RESOURCE"
    don_uuid = gm.active_interaction["candidates"][0].uuid
    gm.resolve_interaction(p1, {"selected_uuids": [don_uuid]})
    assert len(p1.hand) == 1
    assert len(p1.don_active) == 0
    assert len(p1.don_deck) == deck_before + 1


def test_return_don_player_selects_attached_don():
    """「ドン!!-N」で、コストエリアでなく付与中のドン!!を選んで戻せる。
    付与中を戻すと付与先キャラの attached_don（パワー上昇）も解除される。"""
    from opcg_sim.src.models.models import DonInstance
    gm, p1, _ = make_game()
    for i in range(5):
        p1.deck.append(make_instance(make_master(card_id=f"D-{i}"), owner=p1.name))
    # キャラに1枚付与、コストエリアにアクティブ1枚。
    host = make_instance(make_master(card_id="HOST"), owner=p1.name)
    p1.field.append(host)
    attached = DonInstance(owner_id=p1.name, attached_to=host.uuid)
    host.attached_don = 1
    p1.don_attached_cards.append(attached)
    active = DonInstance(owner_id=p1.name)
    p1.don_active.append(active)

    ab = Ability(
        trigger=TriggerType.ACTIVATE_MAIN,
        cost=GameAction(type=ActionType.RETURN_DON, value=ValueSource(base=1)),
        effect=GameAction(type=ActionType.DRAW, value=ValueSource(base=1)),
    )
    src = make_instance(make_master(card_id="RD-2", abilities=(ab,)), owner=p1.name)
    p1.field.append(src)

    deck_before = len(p1.don_deck)
    gm.resolve_ability(p1, ab, source_card=src)
    assert gm.active_interaction["action_type"] == "SELECT_RESOURCE"
    # 候補にはアクティブも付与中も含まれる。付与中のドン!!を選んで戻す。
    cand_uuids = [c.uuid for c in gm.active_interaction["candidates"]]
    assert attached.uuid in cand_uuids and active.uuid in cand_uuids
    gm.resolve_interaction(p1, {"selected_uuids": [attached.uuid]})

    assert len(p1.hand) == 1                       # 効果は解決
    assert attached not in p1.don_attached_cards   # 付与中ドンが外れた
    assert host.attached_don == 0                  # パワー上昇も解除
    assert active in p1.don_active                 # アクティブは温存（選ばなかった）
    assert len(p1.don_deck) == deck_before + 1


def test_trigger_event_goes_to_trash_when_activated():
    """ライフ公開【トリガー】を発動した場合、カードは手札に残らずトラッシュへ行く
    （発動しても手札に加わってしまう不具合の回帰ガード）。発動しなければ手札に残る。"""
    # 発動するケース → トラッシュ
    gm, p1, p2 = make_game()
    for i in range(5):
        p2.deck.append(make_instance(make_master(card_id=f"D-{i}"), owner=p2.name))
    trig = Ability(trigger=TriggerType.TRIGGER,
                   effect=GameAction(type=ActionType.DRAW, value=ValueSource(base=1)))
    ev = make_master(card_id="EV-TRIG", type=CardType.EVENT, abilities=(trig,))
    life_card = make_instance(ev, owner=p2.name)
    p2.life.insert(0, life_card)
    gm.apply_action_to_engine(
        p1, GameAction(type=ActionType.DEAL_DAMAGE, value=ValueSource(base=1)), [], 1)
    assert gm.active_interaction["action_type"] == "CONFIRM_TRIGGER"
    gm.resolve_interaction(p2, {"accepted": True})
    assert life_card not in p2.hand
    assert life_card in p2.trash

    # 発動しない（パス）ケース → 手札に残る
    gm2, q1, q2 = make_game()
    for i in range(5):
        q2.deck.append(make_instance(make_master(card_id=f"E-{i}"), owner=q2.name))
    trig2 = Ability(trigger=TriggerType.TRIGGER,
                    effect=GameAction(type=ActionType.DRAW, value=ValueSource(base=1)))
    ev2 = make_master(card_id="EV-TRIG2", type=CardType.EVENT, abilities=(trig2,))
    lc2 = make_instance(ev2, owner=q2.name)
    q2.life.insert(0, lc2)
    gm2.apply_action_to_engine(
        q1, GameAction(type=ActionType.DEAL_DAMAGE, value=ValueSource(base=1)), [], 1)
    gm2.resolve_interaction(q2, {"accepted": False})
    assert lc2 in q2.hand
    assert lc2 not in q2.trash


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


def test_execute_main_effect_falls_back_to_counter():
    """EXECUTE_MAIN_EFFECT(【トリガー】): ACTIVATE_MAIN が無ければ COUNTER 能力を展開する。
    効果が【カウンター】に書かれたトリガーイベント(OP01-028 等)が従来 no-op だった回帰。"""
    gm, p1, p2 = make_game()
    for i in range(5):
        p1.deck.append(make_instance(make_master(card_id=f"D-{i}"), owner=p1.name))
    # 【カウンター】= カード2枚ドロー を持つイベント（ACTIVATE_MAIN は無い）
    counter_ability = Ability(
        trigger=TriggerType.COUNTER,
        effect=GameAction(type=ActionType.DRAW, value=ValueSource(base=2)),
    )
    trigger_ability = Ability(
        trigger=TriggerType.TRIGGER,
        effect=GameAction(type=ActionType.EXECUTE_MAIN_EFFECT),
    )
    master = make_master(card_id="E-CNT", name="トリガーカウンター", type=CardType.EVENT,
                         abilities=(counter_ability, trigger_ability))
    source = make_instance(master, owner=p1.name)

    assert len(p1.hand) == 0
    gm.resolve_ability(p1, trigger_ability, source_card=source)
    assert len(p1.hand) == 2  # COUNTER の DRAW2 が展開・実行された


def _make_field_char(player, name="戦士", power=5000):
    inst = make_instance(make_master(card_id=f"C-{name}", name=name, power=power), owner=player.name)
    player.field.append(inst)
    return inst


def _c8_ability(ko_target):
    """C8: Sequence[DECLARE_COST, Branch(DECLARED_COST_MATCH → KO opponent)]。"""
    from opcg_sim.src.models.effect_types import Sequence as Seq, Branch, TargetQuery
    return Ability(
        trigger=TriggerType.ACTIVATE_MAIN,
        effect=Seq(actions=[
            GameAction(type=ActionType.DECLARE_COST),
            Branch(
                condition=Condition(type=ConditionType.DECLARED_COST_MATCH),
                if_true=GameAction(type=ActionType.KO,
                                   target=TargetQuery(player=Player.OPPONENT, zone=Zone.FIELD, is_up_to=True)),
            ),
        ]),
    )


def test_c8_declare_cost_match_executes_effect():
    """C8: 宣言コストが相手デッキトップのコストと一致 → 後続効果(KO)が実行される。"""
    gm, p1, p2 = make_game()
    # 相手デッキトップ = コスト5
    p2.deck = [make_instance(make_master(card_id="TOP", cost=5), owner=p2.name)]
    victim = make_instance(make_master(card_id="V", cost=3), owner=p2.name)
    p2.field.append(victim)
    src = _make_field_char(p1, name="OP11")

    gm.resolve_ability(p1, _c8_ability(victim), source_card=src)
    # DECLARE_COST で中断
    assert gm.active_interaction is not None
    assert gm.active_interaction["action_type"] == "DECLARE_COST"
    # コスト5を宣言（一致）→ KO の対象選択へ
    gm.resolve_interaction(p1, {"declared_value": 5})
    assert gm.active_interaction is not None
    assert gm.active_interaction["action_type"] == "SELECT_TARGET"
    # 対象(victim)を選択 → KO 実行
    gm.resolve_interaction(p1, {"selected_uuids": [victim.uuid]})
    assert victim not in p2.field  # KO された


def test_c8_declare_cost_mismatch_skips_effect():
    """C8: 宣言コストが不一致 → 後続効果は実行されない。"""
    gm, p1, p2 = make_game()
    p2.deck = [make_instance(make_master(card_id="TOP", cost=5), owner=p2.name)]
    victim = make_instance(make_master(card_id="V", cost=3), owner=p2.name)
    p2.field.append(victim)
    src = _make_field_char(p1, name="OP11")

    gm.resolve_ability(p1, _c8_ability(victim), source_card=src)
    gm.resolve_interaction(p1, {"declared_value": 2})  # 不一致
    assert victim in p2.field  # KO されない


def test_optional_effect_confirm_yes_executes():
    """任意効果(is_optional)は yes/no 確認を経て、yes でドロー実行。"""
    gm, p1, _ = make_game()
    for i in range(3):
        p1.deck.append(make_instance(make_master(card_id=f"D-{i}"), owner=p1.name))
    opt_draw = GameAction(type=ActionType.DRAW, value=ValueSource(base=1), is_optional=True)
    ability = Ability(trigger=TriggerType.ON_PLAY, effect=opt_draw)
    src = _make_field_char(p1, name="任意ドロー")

    gm.resolve_ability(p1, ability, source_card=src)
    assert gm.active_interaction is not None
    assert gm.active_interaction["action_type"] == "CONFIRM_OPTIONAL"
    assert len(p1.hand) == 0  # まだ引いていない
    gm.resolve_interaction(p1, {"accepted": True})
    assert len(p1.hand) == 1  # yes → ドロー実行


def test_optional_effect_confirm_no_skips():
    """任意効果を no で拒否するとスキップされる。"""
    gm, p1, _ = make_game()
    for i in range(3):
        p1.deck.append(make_instance(make_master(card_id=f"D-{i}"), owner=p1.name))
    opt_draw = GameAction(type=ActionType.DRAW, value=ValueSource(base=1), is_optional=True)
    ability = Ability(trigger=TriggerType.ON_PLAY, effect=opt_draw)
    src = _make_field_char(p1, name="任意ドロー")

    gm.resolve_ability(p1, ability, source_card=src)
    gm.resolve_interaction(p1, {"accepted": False})
    assert len(p1.hand) == 0  # no → スキップ


def test_c10_deckout_win_replacement():
    """C10: デッキアウトしたプレイヤーが「敗北する代わりに勝利する」PASSIVE を持つ場合、
    本人が勝利する。持たない相手がデッキアウトした場合は通常どおり相手が敗北する。"""
    win_ab = Ability(
        trigger=TriggerType.PASSIVE,
        condition=Condition(type=ConditionType.DECK_COUNT, operator=CompareOperator.LE, value=0),
        effect=GameAction(type=ActionType.VICTORY, status="REPLACE_DECKOUT_LOSS"),
    )
    gm, p1, p2 = make_game()
    # p1 のリーダーに勝敗置換能力を付与（CardMaster は frozen のため abilities 付きで構築）
    p1.leader = make_instance(
        make_master(card_id="L-NAMI", name="ナミ", type=CardType.LEADER, abilities=(win_ab,)),
        owner=p1.name)
    p1.deck = []          # p1 がデッキアウト
    p2.deck = [make_instance(make_master(card_id="D"), owner=p2.name)]
    gm.check_victory()
    assert gm.winner == p1.name  # 通常 p2 勝利のところ、置換で p1 勝利

    # 置換能力が無い場合は通常どおり（相手の勝利）
    gm2, q1, q2 = make_game()
    q1.deck = []
    q2.deck = [make_instance(make_master(card_id="D"), owner=q2.name)]
    gm2.check_victory()
    assert gm2.winner == q2.name


def test_power_equalize_snapshot_opp_leader():
    """C9 同値パワー: 相手リーダーのパワーを発動時スナップショットで自身に固定する。
    スナップショット後に相手リーダーのパワーが変動しても追随しない。"""
    gm, p1, p2 = make_game()
    src = _make_field_char(p1, name="ボン・クレー", power=3000)
    p2.leader.base_power_override = 6000  # 相手リーダーを 6000 に
    # 値解決（発動時スナップショット）→ POWER_OVERRIDE で適用
    val = gm.get_dynamic_value(
        p1, ValueSource(dynamic_source="REFERENCE_POWER", ref_id="opp_leader"), [src], {})
    assert val == 6000
    gm.apply_action_to_engine(
        p1, action(ActionType.BUFF, value=val, status="POWER_OVERRIDE", duration="THIS_TURN"),
        [src], val)
    assert src.get_power(True) == 6000
    # 相手リーダーが後で変動してもスナップショットは追随しない
    p2.leader.base_power_override = 1000
    assert src.get_power(True) == 6000
    # ターン終了で失効し元に戻る
    src.reset_turn_status()
    assert src.get_power(True) == 3000


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
    src.reset_turn_status(clear_usage=True)        # ターン境界でカウンタが戻る
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


def test_turn_limit_survives_midturn_reset():
    """【ターン1回】の使用回数は、ターン途中の reset_turn_status（clear_usage 無し＝戦闘終了
    や passive 再計算で呼ばれる）では戻らない。戻ると同一ターン内で複数回発動できてしまう
    （報告バグの回帰ガード）。"""
    gm, p1, _ = make_game()
    for i in range(5):
        p1.deck.append(make_instance(make_master(card_id=f"D-{i}"), owner=p1.name))
    ab = Ability(
        trigger=TriggerType.ACTIVATE_MAIN,
        condition=Condition(type=ConditionType.TURN_LIMIT, value=1),
        effect=GameAction(type=ActionType.DRAW, value=ValueSource(base=1)),
    )
    src = make_instance(make_master(card_id="TL-3", abilities=(ab,)), owner=p1.name)
    p1.field.append(src)

    gm.resolve_ability(p1, ab, source_card=src)
    assert len(p1.hand) == 1
    # 戦闘終了相当の途中リセット（gamestate の battle 終了は keep_don=True で呼ぶ）。
    src.reset_turn_status(keep_don=True)
    gm.resolve_ability(p1, ab, source_card=src)   # 使用回数は維持されるので不発のまま
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
    # 公開（公開し）はカードを動かさない＝デッキトップに留まる。解決完了時に temp 残留を
    # デッキトップへ回収するため、temp リークはなく公開カードはデッキ先頭に戻る。
    assert top not in p1.temp_zone       # temp リーク無し
    assert p1.deck[0] is top             # 公開カードはデッキトップに留まる


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
    """NEGATE_EFFECT: 継続効果(timed_flags)として無効化し、is_effect_negated=True になる。
    途中の reset_turn_status では解除されず（A-6）、ターン終了(continuous.expire)で失効する。"""
    gm, p1, p2 = make_game()
    char = make_instance(make_master(card_id="N-1", type=CardType.CHARACTER), owner=p2.name)
    p2.field.append(char)
    ok = gm.apply_action_to_engine(p1, action(ActionType.NEGATE_EFFECT), [char], 0)
    assert ok
    assert char.is_effect_negated is True
    # 途中のアクション（reset_turn_status）では解除されない（報告バグの回帰ガード）
    char.reset_turn_status()
    assert char.is_effect_negated is True, "途中で無効化が解除されてはいけない"
    # ターン終了で失効する
    gm.continuous.expire("TURN_END", gm.turn_count)
    assert char.is_effect_negated is False


def test_attack_active_allows_attacking_active_character():
    """ATTACK_ACTIVE キーワード持ちはアクティブキャラにアタック可能。
    _validate_action はゲームフロー依存なのでパッチし、攻撃可否の条件だけを検証する。
    """
    gm, p1, p2 = make_game()
    gm.turn_count = 3  # 最初のターンのアタック禁止を避ける（通常進行ターン）
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
    gm.turn_count = 3  # 最初のターンのアタック禁止を避ける（通常進行ターン）
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


# ===== レスト制限（PREVENT_REST）のエンジンテスト =====
def test_prevent_rest_sets_flag_and_survives_then_expires():
    """PREVENT_REST(UNTIL_NEXT_TURN_END): CANNOT_REST を timed_flags に立て、
    自ターン終了は跨ぎ、次の相手ターン終了で失効する。"""
    gm, p1, p2 = make_game()
    gm.turn_count = 3
    target = make_instance(make_master(card_id="PR-EXP", type=CardType.CHARACTER), owner=p2.name)
    p2.field.append(target)

    ok = gm.apply_action_to_engine(
        p1, action(ActionType.PREVENT_REST, duration="UNTIL_NEXT_TURN_END"), [target], 0
    )
    assert ok
    assert "CANNOT_REST" in target.timed_flags

    gm.continuous.expire("TURN_END", 3)   # 適用ターンの終了では失効しない
    assert "CANNOT_REST" in target.timed_flags
    gm.continuous.expire("TURN_END", 4)   # 次の相手ターンの終了で失効
    assert "CANNOT_REST" not in target.timed_flags


def test_prevent_rest_blocks_attack_declaration():
    """CANNOT_REST 持ちはアタック宣言（本体をレストにする操作）で ValueError。"""
    gm, p1, p2 = make_game()
    gm.turn_count = 3  # 最初のターンのアタック禁止を避ける（通常進行ターン）
    attacker = make_instance(make_master(card_id="PR-ATK", type=CardType.CHARACTER), owner=p2.name)
    defender = make_instance(make_master(card_id="PR-DEF", type=CardType.CHARACTER), owner=p1.name)
    p2.field.append(attacker)
    p1.field.append(defender)
    defender.is_rest = True  # 通常はレスト済みキャラへ攻撃可能
    gm.apply_action_to_engine(
        p2, action(ActionType.PREVENT_REST, duration="UNTIL_NEXT_TURN_END"), [attacker], 0
    )
    gm._validate_action = lambda player, action_type: None
    raised = False
    try:
        gm.declare_attack(attacker, defender)
    except ValueError as e:
        raised = "レスト" in str(e)
    assert raised, "CANNOT_REST 持ちはアタック宣言で例外を出すべき"


def test_prevent_rest_excludes_card_from_blocking():
    """CANNOT_REST 持ちの【ブロッカー】はブロック候補から除外される。"""
    gm, p1, p2 = make_game()
    blk_master = make_master(card_id="PR-BLK", type=CardType.CHARACTER)
    blk_master.keywords.add("ブロッカー")
    blocker = make_instance(blk_master, owner=p2.name)
    p2.field.append(blocker)
    assert gm.has_blocker(p2) is True

    gm.apply_action_to_engine(
        p1, action(ActionType.PREVENT_REST, duration="UNTIL_NEXT_TURN_END"), [blocker], 0
    )
    assert "CANNOT_REST" in blocker.timed_flags
    assert gm.has_blocker(p2) is False  # レスト不可＝ブロック不可


# ===== モーダル選択「以下から1つを選ぶ」のエンジン実行テスト =====
def test_modal_choice_executes_selected_option():
    """「以下から1つを選ぶ」: CHOICE で中断→選んだ選択肢(DRAW 2)が実行される。
    従来は options が空でサイレント no-op だった難所の回帰テスト。"""
    from opcg_sim.src.core.effects.parser_v2 import EffectParserV2
    gm, p1, p2 = make_game()
    for i in range(5):
        p1.deck.append(make_instance(make_master(card_id=f"D-{i}"), owner=p1.name))
    abilities = tuple(EffectParserV2().parse_card_text(
        "【登場時】以下から1つを選ぶ。 / ・相手のコスト4以下のキャラ1枚までを、KOする。 / ・カード2枚を引く。"))
    src = make_instance(make_master(card_id="MC-1", abilities=abilities), owner=p1.name)
    p1.field.append(src)

    gm.resolve_ability(p1, abilities[0], source_card=src)
    # CHOICE で中断し、選択肢ラベルが2件提示される
    assert gm.active_interaction is not None
    assert gm.active_interaction["action_type"] == "CHOICE"
    assert len(gm.active_interaction["options"]) == 2
    # index 1（カード2枚を引く）を選択 → 手札 +2
    assert len(p1.hand) == 0
    gm.resolve_interaction(p1, {"index": 1})
    assert len(p1.hand) == 2
    assert gm.active_interaction is None


def test_modal_choice_ko_option_targets_opponent():
    """選択肢 index 0（相手キャラ KO）を選ぶと、対象選択へ遷移し相手キャラを KO する。"""
    from opcg_sim.src.core.effects.parser_v2 import EffectParserV2
    gm, p1, p2 = make_game()
    victim = make_instance(make_master(card_id="V-1", type=CardType.CHARACTER, cost=3), owner=p2.name)
    p2.field.append(victim)
    abilities = tuple(EffectParserV2().parse_card_text(
        "【登場時】以下から1つを選ぶ。 / ・相手のコスト4以下のキャラ1枚までを、KOする。 / ・カード2枚を引く。"))
    src = make_instance(make_master(card_id="MC-2", abilities=abilities), owner=p1.name)
    p1.field.append(src)

    gm.resolve_ability(p1, abilities[0], source_card=src)
    gm.resolve_interaction(p1, {"index": 0})   # KO 選択肢
    # 対象選択（SELECT_TARGET）へ遷移し、相手キャラを選んで KO
    assert gm.active_interaction is not None
    gm.resolve_interaction(p1, {"selected_uuids": [victim.uuid]})
    assert victim not in p2.field
    assert any(c.uuid == victim.uuid for c in p2.trash)


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


# ----- 構造的難所 C7: ライフ scry（対話選択 Choice の suspend/resume フロー） --------
def _life_scry_ability():
    from opcg_sim.src.core.effects.parser_v2 import EffectParserV2
    text = "【登場時】自分か相手のライフの上から1枚までを見て、ライフの上か下に置く。"
    return EffectParserV2().parse_card_text(text)[0]


def _setup_scry_game():
    gm, p1, p2 = make_game()
    a = make_instance(make_master(card_id="LA", name="A"), owner=p1.name)
    b = make_instance(make_master(card_id="LB", name="B"), owner=p1.name)
    c = make_instance(make_master(card_id="LC", name="C"), owner=p1.name)
    p1.life = [a, b, c]
    oa = make_instance(make_master(card_id="OA", name="OA"), owner=p2.name)
    ob = make_instance(make_master(card_id="OB", name="OB"), owner=p2.name)
    p2.life = [oa, ob]
    return gm, p1, p2


def test_life_scry_self_to_bottom():
    """自分のライフ上(A)を見て下に置く: [A,B,C] → [B,C,A]、temp リーク無し。"""
    gm, p1, _ = _setup_scry_game()
    gm.resolve_ability(p1, _life_scry_ability(), source_card=p1.leader)
    # 1) どのライフを見るか（自分=index0）
    assert gm.active_interaction["action_type"] == "CHOICE"
    gm.resolve_interaction(p1, {"index": 0})
    # LOOK_LIFE 実行後: A が temp、life は [B,C]
    assert [c.master.name for c in p1.life] == ["B", "C"]
    assert [c.master.name for c in p1.temp_zone] == ["A"]
    # 2) 上か下か（下=index1）
    assert gm.active_interaction["action_type"] == "CHOICE"
    gm.resolve_interaction(p1, {"index": 1})
    assert [c.master.name for c in p1.life] == ["B", "C", "A"]
    assert p1.temp_zone == []
    assert gm.active_interaction is None


def test_life_scry_self_to_top_noop():
    """自分のライフ上(A)を見て上に戻す: 並びは [A,B,C] のまま、temp リーク無し。"""
    gm, p1, _ = _setup_scry_game()
    gm.resolve_ability(p1, _life_scry_ability(), source_card=p1.leader)
    gm.resolve_interaction(p1, {"index": 0})   # 自分
    gm.resolve_interaction(p1, {"index": 0})   # 上に戻す
    assert [c.master.name for c in p1.life] == ["A", "B", "C"]
    assert p1.temp_zone == []
    assert gm.active_interaction is None


def test_life_scry_opponent_to_bottom():
    """相手のライフ上(OA)を見て下に置く: 相手 [OA,OB] → [OB,OA]、相手 temp リーク無し。"""
    gm, p1, p2 = _setup_scry_game()
    gm.resolve_ability(p1, _life_scry_ability(), source_card=p1.leader)
    gm.resolve_interaction(p1, {"index": 1})   # 相手のライフを見る
    assert [c.master.name for c in p2.life] == ["OB"]
    assert [c.master.name for c in p2.temp_zone] == ["OA"]
    gm.resolve_interaction(p1, {"index": 1})   # 下に置く
    assert [c.master.name for c in p2.life] == ["OB", "OA"]
    assert p2.temp_zone == []
    assert gm.active_interaction is None


def test_life_scry_skip():
    """「1枚まで」= 任意 → 見ない（index2）を選ぶとライフは不変・temp リーク無し。"""
    gm, p1, p2 = _setup_scry_game()
    gm.resolve_ability(p1, _life_scry_ability(), source_card=p1.leader)
    gm.resolve_interaction(p1, {"index": 2})   # 見ない
    assert [c.master.name for c in p1.life] == ["A", "B", "C"]
    assert p1.temp_zone == [] and p2.temp_zone == []
    assert gm.active_interaction is None


# ----- 構造的難所: select断片（SELECT→ref_id の suspend/resume リンク） --------
def test_select_then_attack_disable_linked():
    """「相手のキャラ1枚までを選ぶ。選んだキャラはアタックできない」:
    SELECT で選んだ相手キャラに後続の ATTACK_DISABLE が ref_id 経由で適用される。"""
    from opcg_sim.src.core.effects.parser_v2 import EffectParserV2
    gm, p1, p2 = make_game()
    x = make_instance(make_master(card_id="SX", name="X", cost=3, type=CardType.CHARACTER), owner=p2.name)
    y = make_instance(make_master(card_id="SY", name="Y", cost=3, type=CardType.CHARACTER), owner=p2.name)
    p2.field = [x, y]
    ab = EffectParserV2().parse_card_text(
        "【登場時】相手のコスト6以下のキャラ1枚までを選ぶ。選んだキャラは、このターン中、アタックできない。"
    )[0]

    gm.resolve_ability(p1, ab, source_card=p1.leader)
    # 候補2枚 → 選択で中断
    assert gm.active_interaction is not None
    assert gm.active_interaction["action_type"] == "SELECT_TARGET"
    gm.resolve_interaction(p1, {"selected_uuids": [x.uuid]})

    # 選んだ X にのみ ATTACK_DISABLE（Y は無傷）
    assert "ATTACK_DISABLE" in x.timed_flags
    assert "ATTACK_DISABLE" not in y.timed_flags
    assert gm.active_interaction is None


def test_select_skip_none_selected():
    """「1枚まで」= 任意 → 何も選ばない（空選択）と後続も対象なしで no-op。"""
    from opcg_sim.src.core.effects.parser_v2 import EffectParserV2
    gm, p1, p2 = make_game()
    x = make_instance(make_master(card_id="SX2", name="X", cost=3, type=CardType.CHARACTER), owner=p2.name)
    y = make_instance(make_master(card_id="SY2", name="Y", cost=3, type=CardType.CHARACTER), owner=p2.name)
    p2.field = [x, y]
    ab = EffectParserV2().parse_card_text(
        "【登場時】相手のコスト6以下のキャラ1枚までを選ぶ。選んだキャラは、このターン中、アタックできない。"
    )[0]
    gm.resolve_ability(p1, ab, source_card=p1.leader)
    assert gm.active_interaction["action_type"] == "SELECT_TARGET"
    gm.resolve_interaction(p1, {"selected_uuids": []})  # 選ばない
    assert "ATTACK_DISABLE" not in x.timed_flags
    assert "ATTACK_DISABLE" not in y.timed_flags
    assert gm.active_interaction is None


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


def test_redirect_attack_changes_battle_target():
    """REDIRECT_ATTACK: 進行中バトルの対象をコントローラー側のキャラへ差し替える。"""
    gm, p1, p2 = make_game()
    attacker = make_instance(make_master(card_id="A-1", name="Attacker", type=CardType.CHARACTER), owner="P2")
    redirect = make_instance(make_master(card_id="R-1", name="Redirect", type=CardType.CHARACTER), owner="P1")
    p2.field.append(attacker)
    p1.field.append(redirect)
    # 進行中バトル: P2 の attacker が P1 リーダーを攻撃中
    gm.active_battle = {"attacker": attacker, "target": p1.leader,
                        "attacker_owner": p2, "target_owner": p1, "counter_buff": 0}
    ok = gm.apply_action_to_engine(p1, action(ActionType.REDIRECT_ATTACK), [redirect], 0)
    assert ok
    assert gm.active_battle["target"] is redirect
    assert gm.active_battle["target_owner"] is p1


def test_redirect_attack_no_battle_is_noop():
    """進行中バトルが無ければ REDIRECT_ATTACK は安全に no-op（落ちない）。"""
    gm, p1, _ = make_game()
    redirect = make_instance(make_master(card_id="R-2", name="Redirect", type=CardType.CHARACTER), owner="P1")
    p1.field.append(redirect)
    assert gm.active_battle is None
    ok = gm.apply_action_to_engine(p1, action(ActionType.REDIRECT_ATTACK), [redirect], 0)
    assert ok
    assert gm.active_battle is None


def test_move_attached_don_to_cost_area():
    """MOVE_ATTACHED_DON: 付与ドンN枚を外しレストでコストエリア(don_rested)へ。attached_don も減算。"""
    gm, p1, _ = make_game()
    char = make_instance(make_master(card_id="C-1", name="Char", type=CardType.CHARACTER), owner="P1")
    char.attached_don = 2
    p1.field.append(char)
    # 付与ドン2枚を用意（don_deck から移し、attached_to を char に向ける）
    for _ in range(2):
        d = p1.don_deck.pop(0)
        d.attached_to = char.uuid
        p1.don_attached_cards.append(d)
    assert len(p1.don_attached_cards) == 2 and len(p1.don_rested) == 0
    ok = gm.apply_action_to_engine(p1, action(ActionType.MOVE_ATTACHED_DON, value=2), [], 2)
    assert ok
    assert len(p1.don_attached_cards) == 0
    assert len(p1.don_rested) == 2
    assert all(d.is_rest and d.attached_to is None for d in p1.don_rested)
    assert char.attached_don == 0


def test_rested_play_passive_makes_chars_enter_rested():
    """「自分のキャラはレストで登場する」PASSIVE: 効果(PLAY_CARD)で出たキャラがレスト状態になる。"""
    from opcg_sim.src.models.effect_types import Ability, GameAction
    gm, p1, _ = make_game()
    # リーダーに RESTED_PLAY の PASSIVE を付与
    passive = Ability(trigger=TriggerType.PASSIVE,
                      effect=GameAction(type=ActionType.RESTRICTION, status="RESTED_PLAY"))
    p1.leader.master = make_master(card_id="L-RP", name="RestedLeader", type=CardType.LEADER,
                                   life=5, abilities=(passive,))
    assert gm._has_rested_play(p1) is True

    # 効果で手札のキャラを登場 → owner は所在(手札=p1)から決まり、PASSIVE でレスト化される。
    char = make_instance(make_master(card_id="C-RP", name="Char", type=CardType.CHARACTER), owner="P1")
    p1.hand.append(char)
    ok = gm.apply_action_to_engine(p1, action(ActionType.PLAY_CARD, destination=Zone.FIELD), [char], 0)
    assert ok
    assert char in p1.field
    assert char.is_rest is True

    # PASSIVE が無い player では通常どおりアクティブで登場する。
    gm2, q1, _ = make_game()
    assert gm2._has_rested_play(q1) is False
    char2 = make_instance(make_master(card_id="C-N", name="Char2", type=CardType.CHARACTER), owner="P1")
    q1.hand.append(char2)
    gm2.apply_action_to_engine(q1, action(ActionType.PLAY_CARD, destination=Zone.FIELD), [char2], 0)
    assert char2 in q1.field and char2.is_rest is False


def test_no_effect_play_passive_blocks_effect_play():
    """「手札のこのカードは効果で登場できない」PASSIVE: 効果(PLAY_CARD,手札源)で登場しない。"""
    from opcg_sim.src.models.effect_types import Ability, GameAction
    gm, p1, _ = make_game()
    passive = Ability(trigger=TriggerType.PASSIVE,
                      effect=GameAction(type=ActionType.RESTRICTION, status="NO_EFFECT_PLAY"))
    blocked = make_instance(make_master(card_id="C-NB", name="Blocked", type=CardType.CHARACTER,
                                        abilities=(passive,)), owner="P1")
    p1.hand.append(blocked)
    ok = gm.apply_action_to_engine(p1, action(ActionType.PLAY_CARD, destination=Zone.FIELD), [blocked], 0)
    assert ok  # アクション自体は成功扱い（対象がスキップされるだけ）
    assert blocked not in p1.field
    assert blocked in p1.hand  # 手札に残る

    # 制限の無いキャラは通常どおり効果で登場する。
    normal = make_instance(make_master(card_id="C-NN", name="Normal", type=CardType.CHARACTER), owner="P1")
    p1.hand.append(normal)
    gm.apply_action_to_engine(p1, action(ActionType.PLAY_CARD, destination=Zone.FIELD), [normal], 0)
    assert normal in p1.field


def test_life_to_deck_top_moves_top_life_card():
    """LIFE→DECK top: ライフ上から1枚をデッキトップへ（カード保全・隠しゾーン保護で上から取得）。"""
    from opcg_sim.src.models.effect_types import TargetQuery
    gm, p1, _ = make_game()
    life_cards = [make_instance(make_master(card_id=f"LF-{i}"), owner="P1") for i in range(3)]
    p1.life.extend(life_cards)
    p1.deck.append(make_instance(make_master(card_id="DK-0"), owner="P1"))
    top_life = p1.life[0]
    n_life, n_deck = len(p1.life), len(p1.deck)
    act = action(ActionType.MOVE_CARD, destination=Zone.DECK,
                 target=TargetQuery(zone=Zone.LIFE, player=Player.SELF, count=1))
    act.dest_position = "TOP"
    ok = gm.apply_action_to_engine(p1, act, [top_life], 0)
    assert ok
    assert len(p1.life) == n_life - 1
    assert len(p1.deck) == n_deck + 1
    assert p1.deck[0] is top_life      # デッキトップへ
    assert top_life not in p1.life     # ライフから消えた（カード保全, 計は不変）


def test_life_to_deck_reveal_select_suspends_and_resumes():
    """「ライフすべてを見て1枚をデッキ上へ」: REVEAL_SELECT で対話選択に中断し、
    プレイヤーが選んだライフがデッキトップへ移る（自動「上から取得」ではない）。"""
    from opcg_sim.src.core.effects.parser_v2 import EffectParserV2
    from opcg_sim.src.models.enums import TriggerType, CardType
    txt = '【登場時】自分のライフすべてを見て、1枚を自分のデッキの上に置き、ライフを好きな順番で置く。'
    abils = EffectParserV2().parse_card_text(txt)
    gm, p1, _ = make_game()
    for i in range(3):
        p1.life.append(make_instance(make_master(card_id=f'LF{i}'), owner=p1.name))
    for i in range(2):
        p1.deck.append(make_instance(make_master(card_id=f'DK{i}'), owner=p1.name))
    src = make_instance(make_master(card_id='SRC', type=CardType.CHARACTER, abilities=tuple(abils)), owner=p1.name)
    p1.field.append(src)
    on_play = [a for a in abils if a.trigger == TriggerType.ON_PLAY][0]
    gm.resolve_ability(p1, on_play, source_card=src)
    # 3枚あるので「上から自動取得」ではなく対話選択に中断する
    assert gm.active_interaction is not None
    assert gm.active_interaction.get("action_type") == "SELECT_TARGET"
    assert len(gm.active_interaction.get("candidates", [])) == 3
    # プレイヤーが LF1（トップでない）を選択 → デッキトップへ
    chosen = [c.uuid for c in gm.active_interaction["candidates"] if c.master.card_id == 'LF1']
    gm.resolve_interaction(p1, {"selected_uuids": chosen})
    assert p1.deck[0].master.card_id == 'LF1'
    assert 'LF1' not in [c.master.card_id for c in p1.life]
    assert len(p1.life) + len(p1.deck) == 5  # カード保全


def test_scoped_prevent_leave_protects_trait_characters():
    """RC-2: 「自分の特徴《X》を持つキャラすべては、相手の効果で場を離れない」
    範囲保護が他カード（特徴一致キャラ）を守り、不一致キャラは守らない。"""
    from opcg_sim.src.models.effect_types import Ability, TargetQuery
    gm, p1, p2 = make_game()
    protector_ab = Ability(
        trigger=TriggerType.PASSIVE,
        effect=GameAction(
            type=ActionType.PREVENT_LEAVE, status="LEAVE",
            target=TargetQuery(player=Player.SELF, zone=Zone.FIELD,
                               card_type=["CHARACTER"], traits=["科学者"],
                               count=-1, select_mode="ALL"),
        ),
    )
    protector = make_instance(
        make_master(card_id="P-PL", name="守護者", abilities=(protector_ab,)), owner="P1")
    scientist = make_instance(
        make_master(card_id="C-SCI", name="科学者A", traits=["科学者"]), owner="P1")
    other = make_instance(make_master(card_id="C-OTH", name="一般人"), owner="P1")
    p1.field.extend([protector, scientist, other])

    # 相手(P2)の効果による KO: 特徴一致キャラは守られる
    ok = gm.apply_action_to_engine(p2, action(ActionType.KO), [scientist], 0)
    assert scientist in p1.field, "特徴一致キャラは範囲保護で場に残るべき"
    # 不一致キャラは KO される
    gm.apply_action_to_engine(p2, action(ActionType.KO), [other], 0)
    assert other not in p1.field and other in p1.trash
    # 自分自身(P1)の効果による除去は保護対象外（「相手の効果で」）
    gm.apply_action_to_engine(p1, action(ActionType.KO), [scientist], 0)
    assert scientist in p1.trash


def test_timed_prevent_battle_ko_flag():
    """RC-2/RC-1複合: 期間付き PREVENT_LEAVE(BATTLE_KO) が継続フラグとして付与され、
    バトル KO を防ぎ、ターン境界で失効する。"""
    gm, p1, p2 = make_game()
    char = make_instance(make_master(card_id="C-PRV", name="不死身", power=1000), owner="P1")
    p1.field.append(char)

    ok = gm.apply_action_to_engine(
        p1, action(ActionType.PREVENT_LEAVE, status="BATTLE_KO",
                   duration="UNTIL_NEXT_TURN_END"), [char], 0)
    assert ok
    assert "PREVENT_BATTLE_KO" in char.timed_flags
    assert gm._active_protection(char, ("BATTLE_KO",)) is True
    assert gm._active_protection(char, ("LEAVE",)) is False

    # ターン経過で失効する（UNTIL_NEXT_TURN_END → turn_count+1 の TURN_END）
    gm.continuous.expire("TURN_END", gm.turn_count + 1)
    assert "PREVENT_BATTLE_KO" not in char.timed_flags
    assert gm._active_protection(char, ("BATTLE_KO",)) is False


def test_blocker_disable_respects_cost_cap():
    """RC-2: 「コスト5以下のキャラの【ブロッカー】を発動できない」がコスト上限で絞られる。"""
    from opcg_sim.src.core.effects.parser_v2 import EffectParserV2
    parser = EffectParserV2()
    abilities = parser.parse_card_text(
        "【アタック時】相手は、このバトル中、コスト5以下のキャラの【ブロッカー】を発動できない。")
    assert abilities, "解析失敗"
    ab = abilities[0]
    acts = [a for a in _walk_nodes(ab.effect)
            if getattr(a, "status", None) == "BLOCKER_DISABLE"]
    assert acts, "BLOCKER_DISABLE が生成されるべき"
    tq = acts[0].target
    assert tq.cost_max == 5, f"コスト上限が保持されるべき: {tq.cost_max}"
    assert tq.player == Player.OPPONENT
    assert tq.count == -1 and tq.select_mode == "ALL"


def _walk_nodes(node):
    from opcg_sim.src.models.effect_types import Sequence as _Seq, Branch as _Br, Choice as _Ch
    if node is None:
        return
    if isinstance(node, GameAction):
        yield node
    elif isinstance(node, _Seq):
        for a in node.actions:
            yield from _walk_nodes(a)
    elif isinstance(node, _Br):
        yield from _walk_nodes(node.if_true)
        if node.if_false:
            yield from _walk_nodes(node.if_false)
    elif isinstance(node, _Ch):
        for o in node.options:
            yield from _walk_nodes(o)


def test_passive_buff_does_not_stack_across_recalcs():
    """PASSIVE「パワー+1000」が _apply_passive_effects の再計算で累積しない。

    従来は power_buff に直接加算され、盤面操作のたびに +1000 ずつ際限なく
    増えていた（実プレイの「テキスト通り動かない」主要因の一つ）。"""
    from opcg_sim.src.models.effect_types import Ability, TargetQuery
    gm, p1, _ = make_game()
    ab = Ability(
        trigger=TriggerType.PASSIVE,
        effect=GameAction(type=ActionType.BUFF, target=TargetQuery(select_mode="SOURCE"),
                          value=ValueSource(base=1000)),
    )
    char = make_instance(make_master(card_id="C-PB", name="自己バフ", power=1000,
                                     abilities=(ab,)), owner="P1")
    p1.field.append(char)

    for _ in range(3):
        gm._apply_passive_effects(p1)
    assert char.get_power(True) == 2000, (
        f"PASSIVE +1000 は何度再計算しても +1000 のまま: {char.get_power(True)}")

    # 場を離れて戻れば再適用される（レイヤは再計算で再構築）
    gm._apply_passive_effects(p1)
    assert char.passive_power == 1000 and char.power_buff == 0


def test_per_n_count_query_buff_tracks_board():
    """RC-4: 「自分のトラッシュにあるイベント2枚につき、パワー+1000」が
    実数に追随する（4枚→+2000、2枚→+1000）。"""
    from opcg_sim.src.core.effects.parser_v2 import EffectParserV2
    from opcg_sim.src.models.enums import TriggerType as TT
    parser = EffectParserV2()
    abilities = parser.parse_card_text(
        "このキャラは、自分のトラッシュにあるイベント2枚につき、パワー+1000。")
    assert abilities
    ab = abilities[0]

    gm, p1, _ = make_game()
    char = make_instance(make_master(card_id="C-PN", name="スケーラ", power=1000,
                                     abilities=(ab,)), owner="P1")
    p1.field.append(char)
    for i in range(4):
        p1.trash.append(make_instance(
            make_master(card_id=f"E-{i}", name=f"イベ{i}", type=CardType.EVENT), owner="P1"))
    p1.trash.append(make_instance(
        make_master(card_id="C-X", name="キャラX", type=CardType.CHARACTER), owner="P1"))

    gm._apply_passive_effects(p1)
    assert char.get_power(True) == 3000, (
        f"イベント4枚 // 2 * 1000 = +2000 のはず: {char.get_power(True)}")

    # トラッシュからイベントが2枚減れば +1000 に追随する
    p1.trash = [c for c in p1.trash if c.master.card_id not in ("E-0", "E-1")]
    gm._apply_passive_effects(p1)
    assert char.get_power(True) == 2000, (
        f"イベント2枚 // 2 * 1000 = +1000 のはず: {char.get_power(True)}")


def test_per_n_rest_don_count():
    """RC-4: 「自分のレストのドン!!3枚につき、パワー+1000」がレストドン数で変わる。"""
    from opcg_sim.src.core.effects.parser_v2 import EffectParserV2
    parser = EffectParserV2()
    abilities = parser.parse_card_text(
        "このキャラは、自分のレストのドン!!3枚につき、パワー+1000。")
    assert abilities
    gm, p1, _ = make_game()
    char = make_instance(make_master(card_id="C-RD", name="ドン参照", power=1000,
                                     abilities=tuple(abilities)), owner="P1")
    p1.field.append(char)
    from opcg_sim.src.models.models import DonInstance
    p1.don_rested = [DonInstance(owner_id="P1", is_rest=True) for _ in range(7)]

    gm._apply_passive_effects(p1)
    assert char.get_power(True) == 3000, (
        f"レストドン7枚 // 3 * 1000 = +2000 のはず: {char.get_power(True)}")


def test_opponent_chooses_own_hand_discard():
    """RC-3: 「自分の手札1枚を相手が選び、捨てる」は自分の手札が対象で、
    選択者は相手プレイヤーになる（従来は相手の手札を捨てていた）。"""
    from opcg_sim.src.core.effects.parser_v2 import EffectParserV2
    parser = EffectParserV2()
    abilities = parser.parse_card_text("【KO時】自分の手札1枚を相手が選び、捨てる。")
    assert abilities
    gm, p1, p2 = make_game()
    src = make_instance(make_master(card_id="C-CHO", name="カン十郎"), owner="P1")
    p1.field.append(src)
    for i in range(3):
        p1.hand.append(make_instance(make_master(card_id=f"H-{i}", name=f"手札{i}"), owner="P1"))
    p2.hand.append(make_instance(make_master(card_id="H-OPP", name="相手手札"), owner="P2"))

    gm.resolve_ability(p1, abilities[0], src)
    ia = gm.active_interaction
    assert ia is not None and ia["action_type"] == "SELECT_TARGET"
    assert ia["player_id"] == "P2", f"選択者は相手のはず: {ia['player_id']}"
    cand_owners = {c.owner_id for c in ia["candidates"]}
    assert cand_owners == {"P1"}, f"候補は自分の手札のはず: {cand_owners}"

    # 相手が1枚選んで解決 → 自分の手札が減り、相手の手札は減らない
    chosen = ia["candidates"][0].uuid
    gm.resolve_interaction(p2, {"selected_uuids": [chosen], "index": 0})
    assert len(p1.hand) == 2
    assert len(p2.hand) == 1


def test_life_down_to_n_trash():
    """RC-6: 「自分のライフが1枚になるようにライフの上からトラッシュに置く」が
    残り1枚まで全てトラッシュする（従来は1枚だけ）。"""
    from opcg_sim.src.core.effects.parser_v2 import EffectParserV2
    parser = EffectParserV2()
    abilities = parser.parse_card_text(
        "【メイン】自分のライフが1枚になるようにライフの上からトラッシュに置く。")
    assert abilities
    gm, p1, _ = make_game()
    src = make_instance(make_master(card_id="E-RAI", name="雷迎", type=CardType.EVENT), owner="P1")
    for i in range(5):
        p1.life.append(make_instance(make_master(card_id=f"L-{i}", name=f"ライフ{i}"), owner="P1"))

    gm.resolve_ability(p1, abilities[0], src)
    assert gm.active_interaction is None
    assert len(p1.life) == 1, f"ライフは1枚になるはず: {len(p1.life)}"
    assert len(p1.trash) == 4


def test_trigger_executes_referenced_on_play_effect():
    """「【トリガー】このカードの【登場時】効果を発動する」が ON_PLAY 効果を展開する
    （従来は ACTIVATE_MAIN 固定で no-op だった）。"""
    from opcg_sim.src.core.effects.parser_v2 import EffectParserV2
    parser = EffectParserV2()
    main_abs = parser.parse_card_text("【登場時】カード1枚を引く。")
    trig = parser.parse_card_text("【トリガー】このカードの【登場時】効果を発動する。")
    # 効果(トリガー)は実パイプラインでも別フィールドとして個別に解析される
    trig = [ab for ab in trig if ab.effect is not None]
    assert main_abs and trig

    gm, p1, _ = make_game()
    src = make_instance(make_master(card_id="C-EXE", name="参照発動",
                                    abilities=tuple(main_abs + trig)), owner="P1")
    p1.field.append(src)
    p1.deck.extend(make_instance(make_master(card_id=f"D-{i}", name=f"山{i}"), owner="P1")
                   for i in range(3))
    hand_before = len(p1.hand)
    gm.resolve_ability(p1, trig[0], src)
    assert len(p1.hand) == hand_before + 1, "登場時効果(1ドロー)が発動するべき"


def test_untagged_reactive_ko_clause_maps_to_on_ko():
    """無タグ「このキャラが相手の効果でKOされた時、…」は PASSIVE ではなく ON_KO になり、
    passive 再計算で本体効果が実行されない。"""
    from opcg_sim.src.core.effects.parser_v2 import EffectParserV2
    parser = EffectParserV2()
    abs_ = parser.parse_card_text(
        "このキャラが相手の効果でKOされた時、自分のデッキの上から5枚を見て、"
        "コスト5以下のキャラカード1枚までを、登場させる。その後、残りを好きな順番でデッキの下に置く。")
    assert abs_
    assert abs_[0].trigger == TriggerType.ON_KO, f"ON_KO になるべき: {abs_[0].trigger}"


def test_passive_recalc_skipped_while_interaction_pending():
    """対話中断中の _apply_passive_effects はリセットだけ走って再適用できないため、
    丸ごとスキップされる（バフ消失防止）。"""
    from opcg_sim.src.models.effect_types import Ability, TargetQuery
    gm, p1, _ = make_game()
    ab = Ability(trigger=TriggerType.PASSIVE,
                 effect=GameAction(type=ActionType.BUFF, target=TargetQuery(select_mode="SOURCE"),
                                   value=ValueSource(base=1000)))
    char = make_instance(make_master(card_id="C-G", name="バフ持ち", power=1000, abilities=(ab,)), owner="P1")
    p1.field.append(char)
    gm._apply_passive_effects(p1)
    assert char.passive_power == 1000

    gm.active_interaction = {"player_id": "P1", "action_type": "SELECT_TARGET"}
    gm._apply_passive_effects(p1)
    assert char.passive_power == 1000, "中断中の再計算でバフが消えてはならない"
    gm.active_interaction = None


def test_extra_turn_keeps_turn_player():
    """「このターンの後に自分のターンを追加で得る」: 次のターンも自分が継続する。"""
    gm, p1, p2 = make_game()
    gm.turn_player, gm.opponent = p1, p2
    ok = gm.apply_action_to_engine(p1, action(ActionType.EXTRA_TURN), [], 0)
    assert ok
    gm.switch_turn()
    assert gm.turn_player is p1, "追加ターンで自分が継続するべき"
    gm.switch_turn()
    assert gm.turn_player is p2, "追加ターンは1回で消費される"


def test_base_power_reference_to_self_leader():
    """「このキャラの元々のパワーは、自分のリーダーの元々のパワーと同じパワーになる」。"""
    from opcg_sim.src.core.effects.parser_v2 import EffectParserV2
    parser = EffectParserV2()
    abs_ = parser.parse_card_text(
        "【登場時】このキャラの元々のパワーは、自分のリーダーの元々のパワーと同じパワーになる。")
    assert abs_
    gm, p1, _ = make_game()
    p1.leader.master = make_master(card_id="L-9", name="リーダー", type=CardType.LEADER,
                                   power=9000, life=5)
    char = make_instance(make_master(card_id="C-EQ", name="写し身", power=2000,
                                     abilities=tuple(abs_)), owner="P1")
    p1.field.append(char)
    gm.turn_player = p1
    gm.resolve_ability(p1, abs_[0], char)
    gm._apply_passive_effects(p1)  # 再計算で上書きが消えないこと（passive レイヤ分離）
    assert char.get_power(True) == 9000, f"リーダーの元々のパワーになるべき: {char.get_power(True)}"


def test_life_reveal_conditional_play_declined_returns_to_life():
    """「ライフの上から1枚を公開し、そのカードがコスト5の「サボ」の場合、登場させてもよい」:
    条件不成立なら公開カードはライフ上へ戻る（temp 回収先=ライフ）。"""
    from opcg_sim.src.core.effects.parser_v2 import EffectParserV2
    parser = EffectParserV2()
    abs_ = parser.parse_card_text(
        "【起動メイン】このキャラをトラッシュに置くことができる：自分のライフの上から1枚を公開し、"
        "そのカードがコスト5の「サボ」の場合、登場させてもよい。")
    assert abs_
    gm, p1, _ = make_game()
    src = make_instance(make_master(card_id="C-SAB", name="サボ起動"), owner="P1")
    p1.field.append(src)
    for i in range(3):
        p1.life.append(make_instance(make_master(card_id=f"LF-{i}", name=f"ライフ{i}"), owner="P1"))

    gm.resolve_ability(p1, abs_[0], src)
    # 自動確認が出る場合は受諾して流す
    guard = 0
    while gm.active_interaction and guard < 5:
        pl = gm.p1 if gm.p1.name == gm.active_interaction.get("player_id") else gm.p2
        gm.resolve_interaction(pl, {"selected_uuids": [], "index": 0})
        guard += 1
    assert len(p1.life) == 3, f"条件不成立: ライフは3枚のまま: {len(p1.life)}"
    assert len(p1.temp_zone) == 0
    assert len(p1.field) == 0 if src not in p1.field else True  # コストでトラッシュ


def test_life_reveal_conditional_play_matches_and_plays():
    """条件成立（コスト5の「サボ」）なら公開カードを登場できる。"""
    from opcg_sim.src.core.effects.parser_v2 import EffectParserV2
    parser = EffectParserV2()
    abs_ = parser.parse_card_text(
        "【起動メイン】このキャラをトラッシュに置くことができる：自分のライフの上から1枚を公開し、"
        "そのカードがコスト5の「サボ」の場合、登場させてもよい。")
    gm, p1, _ = make_game()
    src = make_instance(make_master(card_id="C-SAB", name="サボ起動"), owner="P1")
    p1.field.append(src)
    sabo = make_instance(make_master(card_id="ST13-007", name="サボ", cost=5, power=7000), owner="P1")
    p1.life.append(sabo)
    p1.life.append(make_instance(make_master(card_id="LF-1", name="ライフ1"), owner="P1"))

    gm.resolve_ability(p1, abs_[0], src)
    guard = 0
    while gm.active_interaction and guard < 6:
        ia = gm.active_interaction
        pl = gm.p1 if gm.p1.name == ia.get("player_id") else gm.p2
        if ia.get("action_type") == "SELECT_TARGET":
            cands = ia.get("selectable_uuids") or [c.uuid for c in ia.get("candidates", [])]
            gm.resolve_interaction(pl, {"selected_uuids": cands[:1], "index": 0})
        else:
            gm.resolve_interaction(pl, {"selected_uuids": [], "index": 0})
        guard += 1
    assert sabo in p1.field, "コスト5の「サボ」は登場できるべき"
    assert len(p1.temp_zone) == 0


def test_declare_cost_reveal_and_match():
    """C8/RC-7: 「任意のコストを宣言し、相手のデッキの上から1枚を公開する。
    公開したカードが宣言したコストと同じ場合、…パワー+5000」のエンドツーエンド。"""
    from opcg_sim.src.core.effects.parser_v2 import EffectParserV2
    parser = EffectParserV2()
    abs_ = parser.parse_card_text(
        "【カウンター】任意のコストを宣言し、相手のデッキの上から1枚を公開する。"
        "公開したカードが宣言したコストと同じ場合、自分のリーダーかキャラ1枚までを、このバトル中、パワー+5000。")
    assert abs_
    gm, p1, p2 = make_game()
    src = make_instance(make_master(card_id="E-DC", name="援護", type=CardType.EVENT), owner="P1")
    p1.hand.append(src)  # resume はソースカードを盤面/手札から uuid 解決する
    p2.deck.append(make_instance(make_master(card_id="D-3", name="山札3", cost=3), owner="P2"))

    gm.resolve_ability(p1, abs_[0], src)
    ia = gm.active_interaction
    assert ia is not None and ia["action_type"] == "DECLARE_COST"
    # 一致するコスト(3)を宣言 → リーダー対象選択 → +5000
    gm.resolve_interaction(p1, {"declared_value": 3})
    guard = 0
    while gm.active_interaction and guard < 4:
        ia = gm.active_interaction
        pl = gm.p1 if gm.p1.name == ia.get("player_id") else gm.p2
        cands = ia.get("selectable_uuids") or [c.uuid for c in ia.get("candidates", [])]
        gm.resolve_interaction(pl, {"selected_uuids": cands[:1], "index": 0})
        guard += 1
    assert p1.leader.get_power(True) >= 5000 + p1.leader.master.power - 1000, \
        f"宣言一致でバフが乗るべき: {p1.leader.get_power(True)}"

    # 不一致宣言ではバフされない
    gm2, q1, q2 = make_game()
    src2 = make_instance(make_master(card_id="E-DC2", name="援護2", type=CardType.EVENT), owner="P1")
    q1.hand.append(src2)
    q2.deck.append(make_instance(make_master(card_id="D-3b", name="山札3b", cost=3), owner="P2"))
    gm2.resolve_ability(q1, abs_[0], src2)
    gm2.resolve_interaction(q1, {"declared_value": 7})
    guard = 0
    while gm2.active_interaction and guard < 4:
        ia = gm2.active_interaction
        pl = gm2.p1 if gm2.p1.name == ia.get("player_id") else gm2.p2
        cands = ia.get("selectable_uuids") or [c.uuid for c in ia.get("candidates", [])]
        gm2.resolve_interaction(pl, {"selected_uuids": cands[:1], "index": 0})
        guard += 1
    assert q1.leader.timed_power == 0, "不一致宣言ではバフされない"


def test_on_block_ability_fires():
    """H-6: 【ブロック時】効果が handle_block で発動する（従来は未発火＝no-op）。"""
    from opcg_sim.src.models.effect_types import Ability, TargetQuery
    from opcg_sim.src.models.enums import Phase
    gm, p1, p2 = make_game()
    # p2 のブロッカーに ON_BLOCK「カード1枚を引く」を持たせる
    block_ab = Ability(trigger=TriggerType.ON_BLOCK,
                       effect=GameAction(type=ActionType.DRAW, value=ValueSource(base=1)))
    blocker = make_instance(make_master(card_id="C-BLK", name="ブロッカー", power=3000,
                                        abilities=(block_ab,)), owner="P2")
    blocker.current_keywords.add("ブロッカー")  # BLOCK_STEP に入る条件
    p2.field.append(blocker)
    p2.deck.append(make_instance(make_master(card_id="D-1", name="山1"), owner="P2"))
    attacker = make_instance(make_master(card_id="C-ATK", name="攻撃役", power=5000), owner="P1")
    p1.field.append(attacker)
    gm.turn_player, gm.opponent = p1, p2
    gm.turn_count = 3  # 最初のターンのアタック禁止を避ける（通常進行ターン）

    from opcg_sim.src.models.enums import Phase
    gm.phase = Phase.MAIN
    gm.declare_attack(attacker, p2.leader)
    cov_drain(gm)
    assert gm.phase == Phase.BLOCK_STEP, "ブロッカーがいれば BLOCK_STEP に入る"
    hand_before = len(p2.hand)
    gm.handle_block(blocker)
    cov_drain(gm)
    assert len(p2.hand) == hand_before + 1, "ON_BLOCK のドローが発動するべき"


def test_this_battle_buff_expires_on_resolve():
    """H-6: THIS_BATTLE のパワー増は resolve_attack（バトル終了）で失効する。"""
    gm, p1, p2 = make_game()
    char = make_instance(make_master(card_id="C-TB", name="戦士", power=5000), owner="P1")
    p1.field.append(char)
    gm.turn_player, gm.opponent = p1, p2
    gm.apply_action_to_engine(
        p1, action(ActionType.BUFF, value=2000, duration="THIS_BATTLE"), [char], 2000)
    assert char.get_power(True) == 7000
    gm.continuous.expire("BATTLE_END", gm.turn_count)
    assert char.get_power(True) == 5000, "THIS_BATTLE はバトル終了で失効するべき"


def test_until_next_turn_end_buff_expiry_boundary():
    """H-6: UNTIL_NEXT_TURN_END は付与ターンの end では消えず、次ターン end で消える。"""
    gm, p1, p2 = make_game()
    char = make_instance(make_master(card_id="C-UN", name="守人", power=4000), owner="P1")
    p1.field.append(char)
    gm.turn_player, gm.opponent = p1, p2
    gm.turn_count = 4
    gm.apply_action_to_engine(
        p1, action(ActionType.BUFF, value=2000, duration="UNTIL_NEXT_TURN_END"),
        [char], 2000)
    assert char.get_power(True) == 6000
    # 同ターン終了（turn_count 4）では消えない
    gm.continuous.expire("TURN_END", 4)
    assert char.get_power(True) == 6000, "付与ターンの終了では失効しない"
    # 次の相手ターン終了（turn_count 5 = expire_turn）で失効
    gm.continuous.expire("TURN_END", 5)
    assert char.get_power(True) == 4000, "次ターン終了で失効するべき"


def test_delayed_turn_end_action_defers_then_fires():
    """「このターン終了時、〜」: 解決時は即時実行せず、end_turn で発火する（OP03-005 系）。"""
    from opcg_sim.src.models.effect_types import Sequence as _Seq, TargetQuery as _TQ
    gm, p1, p2 = make_game()
    gm.turn_player, gm.opponent = p1, p2
    gm.turn_count = 2
    source = make_instance(make_master(card_id="C-DLY", name="遅延"), owner="P1")
    p1.field.append(source)
    src_tq = _TQ(select_mode="SOURCE")
    buff = GameAction(type=ActionType.BUFF, value=ValueSource(base=2000),
                      target=src_tq, duration="THIS_TURN")
    trash = GameAction(type=ActionType.TRASH, target=src_tq, delay="TURN_END")
    ab = Ability(trigger=TriggerType.ACTIVATE_MAIN, effect=_Seq(actions=[buff, trash]))
    gm.resolve_ability(p1, ab, source_card=source)
    # 即時: バフは適用、トラッシュは保留（場に残る）
    assert source in p1.field, "TRASH は即時実行されず保留されるべき"
    assert source.get_power(True) == 3000, "BUFF は即時適用される"
    assert len(gm.pending_end_of_turn) == 1
    # ターン終了で遅延 TRASH が発火する
    gm._flush_pending_end_of_turn()
    assert source not in p1.field, "end_turn で遅延 TRASH が発火するべき"
    assert source in p1.trash
    assert gm.pending_end_of_turn == []


def test_select_group_distribution_field():
    """§7-1 選択グループ分配: 「2枚を選び、1枚を-3000、残りを-2000」が
    選択集合の先頭1枚に -3000、残りに -2000 を適用する（OP08-118 系）。"""
    from opcg_sim.src.core.effects.parser_v2 import EffectParserV2
    gm, p1, p2 = make_game()
    gm.turn_player, gm.opponent = p1, p2
    gm.turn_count = 2
    e1 = make_instance(make_master(card_id="E1", name="敵1", power=5000), owner="P2")
    e2 = make_instance(make_master(card_id="E2", name="敵2", power=5000), owner="P2")
    p2.field += [e1, e2]
    src = make_instance(make_master(card_id="SD", name="分配"), owner="P1")
    p1.field.append(src)
    ab = EffectParserV2().parse_card_text(
        "【登場時】相手のキャラ2枚までを選び、次の相手のターン終了時まで、"
        "1枚をパワー-3000し、残りをパワー-2000。")[0]
    gm.resolve_ability(p1, ab, source_card=src)
    # 2枚を明示選択して分配を完走させる
    assert gm.active_interaction and gm.active_interaction.get("action_type") == "SELECT_TARGET"
    gm.resolve_interaction(p1, {"selected_uuids": [e1.uuid, e2.uuid], "index": 0})
    powers = sorted([e1.get_power(False), e2.get_power(False)])
    assert powers == [2000, 3000], f"先頭=-3000/残り=-2000 の分配を期待: {powers}"


def test_opp_turn_end_fires_at_end_turn():
    """§7-2 【相手のターン終了時】(OPP_TURN_END): ターンプレイヤーのターン終了で、
    非ターンプレイヤーの当該能力が自動発火する。"""
    gm, p1, p2 = make_game()
    gm.turn_player, gm.opponent = p1, p2
    gm.turn_count = 2
    fired = {"n": 0}
    ramp = GameAction(type=ActionType.RAMP_DON, value=ValueSource(base=1))
    ab = Ability(trigger=TriggerType.OPP_TURN_END, effect=ramp)
    watcher = make_instance(
        make_master(card_id="W", name="監視", abilities=(ab,)), owner="P2")
    p2.field.append(watcher)
    before = len(p2.don_active) + len(p2.don_rested)
    gm._fire_turn_end_triggers()  # p1 のターン終了 → p2 の【相手のターン終了時】が発火
    after = len(p2.don_active) + len(p2.don_rested)
    assert after == before + 1, "非ターンプレイヤーの OPP_TURN_END が発火するべき"


def test_prev_action_count_scaling():
    """§7-5 文脈依存「捨てたカード1枚につき、…パワー+N」: 直前アクションで対象にした
    枚数に比例してバフ値が決まる（P-051 系。2枚捨て→+2000）。"""
    from opcg_sim.src.core.effects.parser_v2 import EffectParserV2
    gm, p1, p2 = make_game()
    gm.turn_player, gm.opponent = p1, p2
    gm.turn_count = 2
    src = make_instance(make_master(card_id="P051", name="ボニー", power=5000), owner="P1")
    p1.field.append(src)
    for i in range(2):
        p1.hand.append(make_instance(make_master(card_id=f"H{i}", name=f"手{i}"), owner="P1"))
    ab = EffectParserV2().parse_card_text(
        "【アタック時】自分の手札を任意の枚数捨ててもよい。"
        "捨てたカード1枚につき、このキャラは、このバトル中、パワー+1000。")[0]
    gm.resolve_ability(p1, ab, source_card=src)
    gm.resolve_interaction(p1, {"accepted": True, "index": 0})       # 任意効果を発動
    gm.resolve_interaction(p1, {"selected_uuids": [c.uuid for c in p1.hand], "index": 0})
    assert src.get_power(True) == 7000, "2枚捨て → +2000 を期待"


def test_v2_is_active_by_default():
    from opcg_sim.src.utils.loader import make_parser
    assert type(make_parser()).__name__ == "EffectParserV2"


def cov_drain(gm):
    import effect_coverage as _cov
    _cov._smart_drain(gm, record={})
