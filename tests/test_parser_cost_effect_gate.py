"""「コスト：〜の場合、効果」の条件はコストの後ろ＝効果側に残す（ユーザ決定 2026-10-01）。

公式ルールでは条件が偽でもコストは払え、効果だけが不発になる。パーサは「：」より後ろの条件を
能力全体（ability.condition）へ持ち上げず、効果ノードの Branch に残す。コストなしの能力・
条件が「：」の前にある能力・見出し（【ターン1回】等）の条件は従来どおり能力全体。
Rust 側の挙動（コストだけ払い効果が不発）は `rust/opcg_engine/src/rules/tests_cost_gate.rs`。
"""
import conftest  # noqa: F401  (google スタブ注入 & sys.path 設定)

import pytest

from opcg_sim.src.effects.parser_v2 import EffectParserV2
from opcg_sim.src.models.effect_types import Branch, Sequence
from opcg_sim.src.models.enums import ConditionType


def _first(text):
    abilities = EffectParserV2().parse_card_text(text)
    assert abilities, text
    return abilities[0]


def _has_branch(node, ctype):
    if isinstance(node, Branch):
        return node.condition is not None and node.condition.type == ctype
    if isinstance(node, Sequence):
        return any(_has_branch(a, ctype) for a in node.actions)
    return False


@pytest.mark.parametrize("text,ctype", [
    # OP05-082
    ("【起動メイン】このキャラをレストにし、自分のトラッシュのカード2枚を好きな順番でデッキの下に置くことができる:"
     "相手の手札が6枚以上ある場合、相手は自身の手札1枚を捨てる。", ConditionType.HAND_COUNT),
    # OP10-118
    ("【アタック時】自分のトラッシュからカード3枚を好きな順番でデッキの下に置くことができる:"
     "相手の手札が5枚以上ある場合、相手は自身の手札1枚を捨てる。", ConditionType.HAND_COUNT),
    # OP09-060
    ("【起動メイン】自分の手札2枚を好きな順番でデッキの下に置き、このステージをレストにできる:"
     "自分のリーダーが特徴《クロスギルド》を持つ場合、カード2枚を引く。", ConditionType.LEADER_TRAIT),
    # OP15-074（「その後」の B まで条件が係る）
    ("【メイン】ドン!!-1:自分のリーダーが「エネル」の場合、カード1枚を引く。"
     "その後、自分のキャラ1枚までを、次の相手のエンドフェイズ終了時まで、コスト+2。", ConditionType.LEADER_NAME),
])
def test_effect_side_condition_stays_in_the_effect(text, ctype):
    ab = _first(text)
    assert ab.cost is not None
    assert ab.condition is None, "効果側の条件を能力全体へ持ち上げない"
    assert _has_branch(ab.effect, ctype)


def test_op15_074_condition_covers_the_after_clause():
    ab = _first("【メイン】ドン!!-1:自分のリーダーが「エネル」の場合、カード1枚を引く。"
                "その後、自分のキャラ1枚までを、次の相手のエンドフェイズ終了時まで、コスト+2。")
    assert isinstance(ab.effect, Branch)
    assert isinstance(ab.effect.if_true, Sequence) and len(ab.effect.if_true.actions) == 2


def test_turn_limit_heading_stays_on_the_ability():
    ab = _first("【起動メイン】【ターン1回】2(コストエリアのドン!!を指定の数レストにできる):"
                "自分のキャラが5枚いる場合、自分のキャラ1枚を、持ち主の手札に戻す。")
    assert ab.condition is not None and ab.condition.type == ConditionType.TURN_LIMIT
    assert _has_branch(ab.effect, ConditionType.FIELD_COUNT)


def test_condition_before_the_colon_stays_on_the_ability():
    ab = _first("【起動メイン】自分のリーダーが「しらほし」の場合、このキャラをレストにできる:カード1枚を引く。")
    assert ab.condition is not None and ab.condition.type == ConditionType.LEADER_NAME


def test_ability_without_cost_still_lifts_the_gate():
    ab = _first("【登場時】自分のリーダーが特徴《海軍》を持つ場合、カード1枚を引く。")
    assert ab.cost is None
    assert ab.condition is not None and ab.condition.type == ConditionType.LEADER_TRAIT
