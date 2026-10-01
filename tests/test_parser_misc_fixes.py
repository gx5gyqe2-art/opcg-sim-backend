"""カード効果監査 F_misc（2026-10-01）で直したパーサの類型の回帰テスト。

実行: python -m pytest tests/test_parser_misc_fixes.py -q
各テストは本文を実カードの文言のまま解析し、解析結果（AST）の形を確かめる。
エンジン側の挙動は rust/opcg_engine の `#[cfg(test)]`（cond.rs ほか）にある。
"""
import conftest  # noqa: F401  (google スタブ注入 & sys.path 設定)

from opcg_sim.src.effects.parser_v2 import EffectParserV2
from opcg_sim.src.models.effect_types import Branch, Choice, GameAction, Sequence
from opcg_sim.src.models.enums import ActionType, ConditionType, Player, TriggerType, Zone

parser = EffectParserV2()


def parse(text):
    return parser.parse_card_text(text)


def actions(node):
    """効果木の GameAction を前順に並べる。"""
    if node is None:
        return []
    if isinstance(node, GameAction):
        return [node]
    out = []
    if isinstance(node, Sequence):
        for a in node.actions:
            out += actions(a)
    elif isinstance(node, Branch):
        out += actions(node.if_true) + actions(node.if_false)
    elif isinstance(node, Choice):
        for o in node.options:
            out += actions(o)
    return out


def test_attack_active_grant_is_not_limited_to_active_grantee():
    """「アクティブのキャラにもアタックできる」の「アクティブの」は付与先ではなくアタック先（EB04-050）。"""
    ab = parse("【メイン】自分の特徴《SWORD》を持つ、リーダーかキャラ1枚までは、このターン中、アクティブのキャラにもアタックできる。")
    act = actions(ab[0].effect)[0]
    assert act.type == ActionType.GRANT_KEYWORD and act.status == "ATTACK_ACTIVE"
    assert act.target.is_rest is None


def test_trash_count_condition_counts_only_the_named_kind():
    """「トラッシュにイベントが4枚以上」はイベントだけを数える（OP12-063）。"""
    ab = parse("自分のトラッシュにイベントが4枚以上ある場合、このキャラのパワー+2000し、コスト+5。")
    cond = ab[0].condition
    assert cond.type == ConditionType.TRASH_COUNT
    assert cond.target is not None and cond.target.zone == Zone.TRASH
    assert cond.target.card_type == ["EVENT"]
    # 種類の無い枚数条件は従来どおり全枚数
    plain = parse("自分のトラッシュが5枚以上ある場合、このキャラのパワー+1000。")
    assert plain[0].condition.target is None


def test_previous_count_references():
    """「戻した／置いた／捨てた／引いた枚数分」は直前アクションの枚数を参照する。"""
    ab = parse("【登場時】自分の手札すべてをデッキに戻し、デッキをシャッフルする。その後、デッキに戻した枚数分カードを引く。")
    draw = [a for a in actions(ab[0].effect) if a.type == ActionType.DRAW][0]
    assert draw.value.dynamic_source == "PREV_ACTION_COUNT"

    ab = parse("【登場時】自分の特徴《海王類》を持つキャラ1枚につき、カード1枚を引く。その後、引いた枚数分自分の手札を捨てる。")
    discard = [a for a in actions(ab[0].effect) if a.type == ActionType.DISCARD][0]
    assert discard.target.count_dynamic == "PREV_ACTION_COUNT"

    ab = parse("【カウンター】自分の手札2枚までを捨てる。捨てた枚数と同じ枚数を、自分のデッキの上からトラッシュに置く。")
    mill = [a for a in actions(ab[0].effect) if a.type == ActionType.TRASH_FROM_DECK][0]
    assert mill.value.dynamic_source == "PREV_ACTION_COUNT"


def test_character_and_leader_both_receive_the_verb():
    """「キャラ2枚までとリーダーを」「このキャラと自分のリーダー1枚までを」は両方に効く（EB04-013・P-036）。"""
    ab = parse("【登場時】自分のリーダーが特徴《ミンク族》を持つ場合、自分の特徴《ミンク族》を持つキャラ2枚までとリーダーを、アクティブにする。")
    acts = actions(ab[0].effect)
    assert [a.type for a in acts] == [ActionType.ACTIVE, ActionType.ACTIVE]
    assert acts[0].target.card_type == ["CHARACTER"] and acts[0].target.count == 2
    assert acts[1].target.card_type == ["LEADER"]

    ab = parse("【アタック時】カード1枚を引く：このキャラと自分のリーダー1枚までを、このターン中、パワー+1000。")
    acts = [a for a in actions(ab[0].effect) if a.type == ActionType.BUFF]
    assert len(acts) == 2
    assert acts[0].target.select_mode == "SOURCE" and acts[1].target.card_type == ["LEADER"]


def test_per_n_scaling_carries_to_the_cost_clause():
    """「トラッシュ5枚につき、パワー+1000し、コスト+2」のコストも 5 枚ごと（EB04-048）。"""
    ab = parse("このキャラは、自分のトラッシュ5枚につき、パワー+1000し、コスト+2。")
    acts = actions(ab[0].effect)
    cost = [a for a in acts if a.status == "COST_REDUCTION"][0]
    assert cost.value.dynamic_source is not None and cost.value.divisor == 5 and cost.value.multiplier == 2


def test_dual_tier_play_rests_only_when_the_text_says_so():
    """「1枚ずつまでを、登場させる」は通常登場（EB03-049）。第2ティアが「1」のコスト丁度（OP14-084）。"""
    ab = parse("【メイン】自分の手札かトラッシュから特徴《スリラーバーク海賊団》を持つ、コスト6以下とコスト4以下のキャラカード1枚ずつまでを、登場させる。")
    acts = actions(ab[0].effect)
    assert len(acts) == 2 and all(a.status is None for a in acts)

    ab = parse("【登場時】自分のトラッシュから『B・W』を含む特徴を持つ、コスト4以下と1のキャラカード1枚ずつまでを、登場させる。")
    acts = actions(ab[0].effect)
    assert len(acts) == 2
    assert acts[0].target.cost_max == 4 and acts[0].target.cost_min is None
    assert acts[1].target.cost_min == 1 and acts[1].target.cost_max == 1


def test_each_named_card_is_its_own_play():
    """「「A」と「B」と「C」それぞれ1枚ずつまで」は名前ごとの登場（OP16-105）。"""
    ab = parse("【トリガー】自分のトラッシュからコスト4以下の、「アブサロム」と「ドクトル・ホグバック」と「ペローナ」それぞれ1枚ずつまでを、登場させる。")
    acts = actions(ab[0].effect)
    assert [a.target.names for a in acts] == [["アブサロム"], ["ドクトル・ホグバック"], ["ペローナ"]]
    assert all(a.target.cost_max == 4 for a in acts)


def test_this_effects_played_card_reference():
    """「この効果で登場させたキャラ」は直前の PLAY_CARD が登場させたカードへの厳密参照（OP15-086）。"""
    ab = parse("【登場時】自分のトラッシュからコスト7以下のキャラカード1枚までを、登場させる。この効果で登場させたキャラは、このターン中、【速攻】を得る。")
    acts = actions(ab[0].effect)
    assert acts[0].type == ActionType.PLAY_CARD and acts[0].target.save_id == "played_by_effect"
    assert acts[1].target.ref_id == "played_by_effect"


def test_optional_rest_with_dekiru():
    """「レストにできる」は任意（OP07-036 の「そうした場合」が常に成立しない）。"""
    ab = parse("【メイン】自分のコスト3以上のキャラ1枚をレストにできる。そうした場合、相手のコスト5以下のキャラ1枚までを、レストにする。")
    rests = [a for a in actions(ab[0].effect) if a.type == ActionType.REST]
    assert rests[0].is_optional and not rests[1].is_optional


def test_arrange_two_tiers_in_one_deck_bottom():
    """2 枚を「好きな順番で」置くときは、選んだ 2 枚をまとめて 1 回の並び替え配置にする（OP06-056）。"""
    ab = parse("【メイン】相手の、コスト2以下のキャラ1枚までとコスト1以下のキャラ1枚までを、持ち主のデッキの下に好きな順番で置く。")
    acts = actions(ab[0].effect)
    assert [a.type for a in acts] == [ActionType.SELECT, ActionType.SELECT, ActionType.DECK_BOTTOM]
    assert acts[2].status == "ARRANGE" and acts[2].target.ref_id == "_arr_a+_arr_b"


def test_top_or_bottom_choice_for_the_revealed_card():
    """「そのカードをデッキの上か下に置く」は上下をプレイヤーが選ぶ（OP08-049）。"""
    ab = parse("【登場時】自分のデッキの上から1枚を公開し、そのカードをデッキの上か下に置く。")
    put = [a for a in actions(ab[0].effect) if a.type == ActionType.DECK_BOTTOM][0]
    assert put.dest_position == "CHOOSE"


def test_target_swap_sentence_becomes_a_branch():
    """「…の代わりに…を選ぶ」は条件で対象の絞り込みが変わる 1 回の効果（OP12-096）。"""
    ab = parse("【メイン】相手のコスト4以下のキャラ1枚までを、KOする。自分のコスト8以上のキャラがいる場合、相手のコスト4以下のキャラの代わりに相手のコスト6以下のキャラを選ぶ。")
    node = ab[0].effect
    assert isinstance(node, Branch)
    assert node.if_true.target.cost_max == 6 and node.if_false.target.cost_max == 4


def test_leader_attack_scoped_blocker_lock():
    """「自分のリーダーがアタックする際、相手は【ブロッカー】を発動できない」は制限（OP13-057）。"""
    ab = parse("【メイン】自分のドン‼1枚をレストにできる：自分のライフが1枚以下の場合、相手は、このターン中、自分のリーダーがアタックする際【ブロッカー】を発動できない。")
    act = actions(ab[0].effect)[0]
    # リーダーのアタック時だけ封じる＝リーダーに【ブロック不可】をこのターン中付与する表現（A の方式）。
    assert act.type == ActionType.GRANT_KEYWORD and act.status == "ブロック不可"
    assert act.target.card_type == ["LEADER"]


def test_opponent_chooses_and_draws():
    """「相手は以下から1つを選ぶ。・カード2枚を引く」のドローは選んだ相手が引く（OP17-049）。"""
    ab = parse("【登場時】相手は以下から1つを選ぶ。\n・カード2枚を引く。\n・相手は自身の手札2枚を捨てる。")
    choice = ab[0].effect
    assert isinstance(choice, Choice) and choice.player == Player.OPPONENT
    assert choice.options[0].target.player == Player.OPPONENT


def test_action_before_choice_is_kept():
    """「カード1枚を引き、以下から1つを選ぶ」の前半（ドロー）を脱落させない（OP17-112）。"""
    ab = parse("【登場時】カード1枚を引き、以下から1つを選ぶ。\n・自分のデッキの上から1枚までを、ライフの上に加える。\n・相手のライフの上から1枚までを、持ち主の手札に加える。")
    acts = actions(ab[0].effect)
    assert acts[0].type == ActionType.DRAW


def test_cost_sum_limited_ko_has_no_card_cap():
    """「コストの合計が4以下になるようにKO」は枚数の上限なし・合計コスト上限（OP17-119）。"""
    ab = parse("【登場時】相手のキャラを、コストの合計が4以下になるようにKOする。")
    tq = actions(ab[0].effect)[0].target
    assert tq.power_sum_max == 4 and "SUM_COST" in tq.flags and tq.count == -1


def test_turn_context_condition_is_not_generic():
    """「相手のターン中の場合」は CONTEXT 条件（常に真の GENERIC ではない・OP17-119 の +3000）。"""
    ab = parse("このキャラのコスト+12し、相手のターン中の場合、このキャラのパワー+3000。")
    branches = [n for n in ab[0].effect.actions if isinstance(n, Branch)]
    assert branches and branches[0].condition.type == ConditionType.CONTEXT


def test_leader_only_buff_target():
    """「自分のリーダーは、…キャラ1枚につきパワー+1000」の付与先はリーダーだけ（P-024）。"""
    ab = parse("【メイン】自分のリーダーは、このターン中、自分のキャラ1枚につきパワー+1000。")
    assert actions(ab[0].effect)[0].target.card_type == ["LEADER"]


def test_unseparated_attack_ability_is_its_own_ability():
    """区切り無しで続く【アタック時】は別能力（P-075）。"""
    ab = parse("【登場時】自分のリーダーかキャラ1枚にレストのドン!!1枚までを、付与する。【アタック時】自分の場にコスト8以上のキャラがいる場合、カード1枚を引き、自分の手札1枚を捨てる。")
    assert [a.trigger for a in ab] == [TriggerType.ON_PLAY, TriggerType.ON_ATTACK]


def test_keyword_grant_with_renyou_form():
    """「【ブロッカー】を得て、コスト+4」はブロッカー付与とコスト増の 2 動作（P-105）。"""
    ab = parse("自分のリーダーが特徴《革命軍》を持つ場合、このキャラは【ブロッカー】を得て、コスト+4。")
    types = [a.type for a in actions(ab[0].effect)]
    assert ActionType.GRANT_KEYWORD in types and ActionType.BUFF in types


def test_opponent_life_left_this_turn_condition():
    """「相手のライフが離れているターン中」はこのターンの離脱事実（P-120。LIFE_COUNT ではない）。"""
    ab = parse("手札のこのカードは、相手のライフが離れているターン中、コスト-2。")
    cond = ab[0].condition
    assert cond.type == ConditionType.EVENT_THIS_TURN and cond.value[0] == "OPP_LIFE_LEFT"


def test_cost_qualifier_on_self_trash_cost():
    """「コスト20以上のこのキャラをトラッシュに置く」の修飾語は払える条件になる（OP16-084）。"""
    ab = parse("【起動メイン】コスト20以上のこのキャラをトラッシュに置くことができる：自分の場のドン!!が9枚以上ある場合、自分のトラッシュからコスト9の「光月モモの助」1枚までを、登場させる。")
    assert ab[0].cost.target.cost_min == 20


def test_battle_ko_trigger_is_on_ko():
    """「このキャラのバトルによって相手のキャラをKOした時」は起動メインではなく ON_KO（OP04-086）。"""
    ab = parse("【ドン!!×1】このキャラのバトルによって相手のキャラをKOした時、カード2枚を引き、自分の手札2枚を捨てる。")
    assert ab[0].trigger == TriggerType.ON_KO
