"""EB05 追加（2026-11-09）の厳格レビューで ng になった句のパーサ出力（`docs/reports/2026-11-09_card_fix_eb05.md`）。

**問い**: 本文の限定語・構造が解析結果に残るか（どれも句の脱落／OR の AND 化／余分な付与だった）。
"""
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: E402,F401  (sys.path 設定＋google スタブ)

from opcg_sim.src.effects.parser_v2 import EffectParserV2  # noqa: E402
from opcg_sim.src.models.effect_types import _nfc  # noqa: E402

P = EffectParserV2()


def _abilities(text):
    return P.parse_card_text(_nfc(text))


def _actions(node):
    if node is None:
        return []
    if hasattr(node, "actions"):
        return [a for x in node.actions for a in _actions(x)]
    if hasattr(node, "options"):
        return [a for x in node.options for a in _actions(x)]
    return [node]


def test_damage_replacement_is_an_optional_replace_effect():
    """EB05-052: ダメージの置換（status=DAMAGE・任意・このキャラ自身をトラッシュ）。"""
    ab = _abilities("自分がダメージを受ける場合、代わりにこのキャラをトラッシュに置くことができる。")[0]
    assert ab.trigger.name == "PASSIVE" and ab.condition is None
    assert ab.effect.type.name == "REPLACE_EFFECT" and ab.effect.status == "DAMAGE"
    sub = ab.effect.sub_effect
    assert sub.type.name == "TRASH" and sub.is_optional and sub.target.select_mode == "SOURCE"


def test_removed_by_the_opponent_covers_battle_ko():
    """EB05-061／OP15-098: 「相手によって場を離れる」は相手の効果と相手のアタック（バトル KO）。"""
    for text in (
        "【ターン1回】自分の元々のパワー6000以下のキャラが相手によって場を離れる場合、代わりに自分のライフの上から1枚を手札に加えることができる。",
        "自分の元々のパワー6000以上の特徴《空島》を持つキャラが相手によって場を離れる場合、代わりに自分のライフの上から1枚を手札に加えることができる。",
    ):
        ab = _abilities(text)[0]
        assert set(ab.effect.status.split(",")) == {"LEAVE", "BATTLE_KO"}, ab.effect.status
    # 「相手の効果で」は効果だけのまま。
    ab = _abilities("このキャラが相手の効果で場を離れる場合、代わりに自分の手札1枚を捨てることができる。")[0]
    assert ab.effect.status == "LEAVE"


def test_trait_and_trigger_both_required():
    """EB05-005: 「特徴《革命軍》と【トリガー】を両方持つ」= 特徴 AND トリガー所持。"""
    ab = _abilities("【登場時】自分の特徴《革命軍》と【トリガー】を両方持つキャラ3枚までを、このターン中、パワー+2000。")[0]
    tq = ab.effect.target
    assert tq.traits == ["革命軍"] and "HAS_TRIGGER" in tq.flags and "TRAIT_OR_TRIGGER" not in tq.flags


def test_unsided_character_buff_targets_both_sides():
    """EB05-006: 陣営の指定が無い「キャラ1枚までを、…パワー±N」は両陣営。"""
    ab = _abilities("【起動メイン】【ターン1回】自分のライフの上から1枚を手札に加えることができる：以下から1つを選ぶ。"
                    "\n・キャラ1枚までを、このターン中、パワー+3000。\n・キャラ1枚までを、このターン中、パワー-4000。")[0]
    buffs = [a for a in _actions(ab.effect) if a.type.name == "BUFF"]
    assert len(buffs) == 2 and all(b.target.player.name == "ALL" for b in buffs)
    # 「自分の…キャラ」は自分のまま。
    ab = _abilities("【登場時】自分のキャラ1枚までを、このターン中、パワー+1000。")[0]
    assert ab.effect.target.player.name == "SELF"


def test_event_or_trait_reveal_is_or():
    """EB05-007: 「イベントか特徴《パンクハザード》を持つカード」= 種類 OR 特徴。"""
    ab = _abilities("【登場時】自分の手札からイベントか特徴《パンクハザード》を持つカード合計3枚を公開できる：カード1枚を引く。")[0]
    tq = ab.cost.target
    assert tq.card_type == ["EVENT"] and tq.traits == ["パンクハザード"] and "TYPE_OR_TRAIT" in tq.flags


def test_negate_then_bounce_keeps_the_negation():
    """EB05-029: 「このターン中、効果を無効にし、持ち主の手札に戻す」= 無効 → 同じ対象を戻す。"""
    ab = _abilities("【メイン】自分の手札1枚を捨てることができる：相手のコスト6以下のキャラ1枚までを、このターン中、効果を無効にし、持ち主の手札に戻す。")[0]
    neg, bounce = _actions(ab.effect)
    assert neg.type.name == "NEGATE_EFFECT" and neg.duration == "THIS_TURN"
    assert neg.target.player.name == "OPPONENT" and neg.target.cost_max == 6 and neg.target.is_up_to
    assert bounce.type.name == "BOUNCE" and bounce.target.ref_id == neg.target.save_id


def test_attribute_or_chain():
    """EB05-057: 「属性(特)か(知)を持つ」の 2 つ目の属性も拾う。"""
    ab = _abilities("【起動メイン】【ターン1回】自分の属性(特)か(知)を持つ、リーダーかキャラ1枚にレストのドン!!1枚までを、付与する。")[0]
    assert ab.effect.target.attributes == ["特", "知"]


def test_rush_reminder_after_a_period_is_not_a_permanent_grant():
    """P-143: 句点の直後の注釈「(このカードは登場したターンにアタックできる)」は効果ではない。"""
    abs_ = _abilities("【登場時】コスト0のキャラがいる場合、このキャラは、このターン中、【速攻】を得る。(このカードは登場したターンにアタックできる)")
    grants = [a for ab in abs_ for a in _actions(ab.effect) if a.type.name == "GRANT_KEYWORD"]
    assert len(grants) == 1 and grants[0].duration == "THIS_TURN"
