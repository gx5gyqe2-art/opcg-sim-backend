"""条件・対象の取りこぼし是正（WP B_cond）のパーサ出力を固定する小テスト。"""
import _bootstrap  # noqa: F401
from opcg_sim.src.effects.parser_v2 import EffectParserV2


def _abilities(text):
    return EffectParserV2().parse_card_text(text)


def _flat(node):
    if node is None:
        return []
    if hasattr(node, "actions"):
        return [x for a in node.actions for x in _flat(a)]
    return [node]


def test_leader_trait_contains_uses_has_operator():
    ab = _abilities("【登場時】自分のリーダーが『CP』を含む特徴を持つ場合、カード1枚を引く。")[0]
    assert ab.condition.operator.name == "HAS"
    ab = _abilities("【登場時】自分のリーダーが特徴《麦わらの一味》を持つ場合、カード1枚を引く。")[0]
    assert ab.condition.operator.name == "EQ"


def test_trash_count_keeps_card_type():
    ab = _abilities("自分のトラッシュにイベントが4枚以上ある場合、このキャラのパワー+2000。")[0]
    assert ab.condition.target.card_type == ["EVENT"]
    assert ab.condition.target.zone.name == "TRASH"


def test_face_up_life_exists_is_ge_1():
    ab = _abilities("【登場時】自分の表向きのライフがある場合、カード1枚を引く。")[0]
    c = ab.condition
    assert c.operator.name == "GE" and c.value == 1 and c.target.is_face_up is True


def test_draw_until_hand_size():
    ab = _abilities("【登場時】自分の手札が3枚になるようにカードを引く。")[0]
    v = _flat(ab.effect)[0].value
    assert v.dynamic_source == "HAND_TO_N" and v.base == 3


def test_blocker_grant_with_cost_buff():
    ab = _abilities("自分のライフが3枚以下の場合、このキャラは【ブロッカー】を得て、コスト+3。")[0]
    kinds = [a.type.name for a in _flat(ab.effect)]
    assert kinds == ["GRANT_KEYWORD", "BUFF"]


def test_ko_all_except_self_targets_both_sides():
    ab = _abilities("【アタック時】このキャラ以外のキャラすべてを、KOする。")[0]
    assert _flat(ab.effect)[0].target.player.name == "ALL"


def test_cost_pair_is_a_range():
    ab = _abilities("コスト3と4のキャラすべては、アタックできない。")[0]
    t = _flat(ab.effect)[0].target
    assert (t.cost_min, t.cost_max) == (3, 4)


def test_trait_partial_flag():
    ab = _abilities("【登場時】自分の手札から『ロックス海賊団』を含む特徴を持つカード1枚を捨てることができる:カード1枚を引く。")[0]
    assert "TRAIT_PARTIAL" in _flat(ab.cost)[0].target.flags


def test_multi_name_has_character_is_and():
    ab = _abilities("【起動メイン】自分の「サトリ」と「ホトリ」がいる場合、カード1枚を引く。")[0]
    assert ab.condition.type.name == "AND" and len(ab.condition.args) == 2


def test_hand_reveal_leading_gate_covers_then_clause():
    ab = _abilities("【メイン】自分のリーダーが特徴《赤髪海賊団》を持つ場合、相手のキャラ1枚までを、このターン中、パワー-3000。"
                    "その後、相手のパワー5000以上のキャラがいる場合、カード1枚を引く。")[0]
    assert ab.condition is not None and ab.condition.type.name == "LEADER_TRAIT"


def test_self_onplay_negation_is_a_passive_marker():
    ab = _abilities("自分の【登場時】効果は無効になる。")[0]
    assert ab.trigger.name == "PASSIVE" and ab.effect.status == "NEGATE_OWN_ONPLAY"


def test_face_down_life_cost_direction():
    ab = _abilities("【登場時】自分の表向きのライフ1枚を裏向きにできる:カード1枚を引く。")[0]
    assert _flat(ab.cost)[0].status == "DOWN"
