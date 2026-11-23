"""OP18 追加（2026-11-22）の効果監査で ng になった句のパーサ出力（作業パッケージ p3）。

**問い**: 「両方持つ」「1枚ずつまで」「合計N枚」「AかB」「AとBを得る」の構造が解析結果に残るか
（どれも OR の AND 化・2 つ目の対象の脱落・1 対象への合算だった）。
"""
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: E402,F401  (sys.path 設定＋google スタブ)

from opcg_sim.src.effects.parser_v2 import EffectParserV2  # noqa: E402
from opcg_sim.src.models.effect_types import _nfc  # noqa: E402
from opcg_sim.src.models.enums import Player, Zone  # noqa: E402
from opcg_sim.src.utils.loader import _extract_static_keywords  # noqa: E402

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
    if hasattr(node, "if_true"):
        return _actions(node.if_true) + _actions(getattr(node, "if_false", None))
    return [node]


def test_both_traits_is_an_and_and_the_buff_inherits_the_all_subject():
    """OP18-001: 「《動物》と《アラバスタ王国》を両方持つキャラすべては、【速攻】を得て、パワー+1000」。"""
    acts = _actions(_abilities(
        "自分の特徴《動物》と《アラバスタ王国》を両方持つキャラすべては、【速攻】を得て、パワー+1000。")[0].effect)
    assert [a.type.name for a in acts] == ["GRANT_KEYWORD", "BUFF"]
    for a in acts:
        tq = a.target
        assert tq.select_mode == "ALL" and tq.count == -1 and tq.card_type == ["CHARACTER"]
        assert tq.traits == ["動物", "アラバスタ王国"] and "TRAITS_ALL" in tq.flags


def test_both_traits_up_to_one():
    """OP18-012: 「…を両方持つキャラ1枚までは」も AND。単一特徴・「か」の並びには立てない。"""
    a = _actions(_abilities(
        "【アタック時】自分の特徴《動物》と《アラバスタ王国》を両方持つキャラ1枚までは、このターン中、パワー+1000。")[0].effect)[0]
    assert "TRAITS_ALL" in a.target.flags and a.target.is_up_to
    b = _actions(_abilities(
        "【登場時】自分の特徴《革命軍》と【トリガー】を両方持つキャラ3枚までを、このターン中、パワー+2000。")[0].effect)[0]
    assert "TRAITS_ALL" not in b.target.flags


def test_each_tier_selects_two_distinct_cards():
    """OP18-051/064: 「AとBのキャラ1枚ずつまで」＝基準ごとに 1 枚まで（2 枚目は 1 枚目を除く）を選び、まとめて処理。"""
    acts = _actions(_abilities(
        "【登場時】自分のドン!!5枚をレストにできる：コスト6以下と5以下のキャラ1枚ずつまでを、"
        "持ち主が好きな順番でデッキの下に置く。")[0].effect)
    assert [a.type.name for a in acts] == ["SELECT", "SELECT", "DECK_BOTTOM"]
    a, b, put = acts
    assert (a.target.cost_max, b.target.cost_max) == (6, 5)
    assert a.target.is_up_to and b.target.is_up_to and a.target.player == Player.ALL
    assert "EXCLUDE_SAVED:_each_a" in b.target.flags
    assert put.status == "ARRANGE" and put.target.ref_id == "_each_a+_each_b"

    acts = _actions(_abilities(
        "【自分のターン中】【登場時】ドン!!-1：相手の元々の、パワー7000以下と5000以下のキャラ1枚ずつまでは、"
        "次の相手のエンドフェイズ終了時まで、アタックできない。")[0].effect)
    assert [x.type.name for x in acts] == ["SELECT", "SELECT", "ATTACK_DISABLE"]
    a, b, act = acts
    assert (a.target.power_max, b.target.power_max) == (7000, 5000)
    assert "ORIGINAL_POWER" in a.target.flags and "ORIGINAL_POWER" in b.target.flags
    assert a.target.player == Player.OPPONENT and a.target.is_up_to
    assert act.duration == "UNTIL_NEXT_TURN_END" and act.target.ref_id == "_each_a+_each_b"


def test_chars_and_filtered_leader_are_two_targets():
    """OP18-076: 「自分のキャラ2枚までとパワー5000以下のリーダー」＝キャラ 2 枚まで＋修飾つきリーダー。"""
    acts = _actions(_abilities(
        "【相手のアタック時】このステージをレストにできる：自分のキャラ2枚までとパワー5000以下のリーダーを、"
        "このターン中、パワー+1000。")[0].effect)
    assert [a.type.name for a in acts] == ["BUFF", "BUFF"]
    ch, ld = (a.target for a in acts)
    assert ch.card_type == ["CHARACTER"] and ch.count == 2 and ch.is_up_to and ch.power_max is None
    assert ld.card_type == ["LEADER"] and ld.count == 1 and ld.power_max == 5000


def test_leader_and_chars_total_is_one_target():
    """OP18-078: 「リーダーとキャラ合計4枚までにドン!!1枚ずつ」＝リーダーとキャラから合計 4 枚まで。"""
    acts = _actions(_abilities(
        "【起動メイン】このステージをレストにできる：自分のリーダーとキャラ合計4枚までにレストのドン!!1枚ずつまでを、"
        "付与する。")[0].effect)
    assert len(acts) == 1 and acts[0].type.name == "ATTACH_DON" and acts[0].value.base == 1
    tq = acts[0].target
    assert sorted(tq.card_type) == ["CHARACTER", "LEADER"] and tq.count == 4 and tq.is_up_to


def test_stage_card_or_trait_is_an_or():
    """OP18-061: 「ステージカードか特徴《W7》を持つカード」＝種類 OR 特徴（TYPE_OR_TRAIT）。"""
    acts = _actions(_abilities(
        "【登場時】自分のデッキの上から5枚を見て、ステージカードか特徴《W7》を持つカード合計2枚までを公開し、"
        "手札に加える。その後、残りを好きな順番でデッキの下に置く。")[0].effect)
    mv = [a for a in acts if a.type.name == "MOVE_CARD"][0]
    assert mv.target.card_type == ["STAGE"] and mv.target.traits == ["W7"]
    assert "TYPE_OR_TRAIT" in mv.target.flags


def test_trait_card_and_don_both_become_active():
    """OP18-031: 「特徴《W7》を持つカード1枚までとドン!!1枚までを、アクティブにする」。"""
    acts = _actions(_abilities(
        "【自分のターン終了時】自分の特徴《W7》を持つカード1枚までとドン!!1枚までを、アクティブにする。")[0].effect)
    assert [a.type.name for a in acts] == ["ACTIVE", "ACTIVE_DON"]
    assert acts[0].target.traits == ["W7"] and acts[0].target.is_up_to


def test_two_keywords_are_both_granted_and_not_static():
    """OP18-081: 「【ダブルアタック】と【ブロッカー】を得る」は両方の付与で、固定キーワードではない。"""
    text = "自分のキャラの「ゲッコー・モリア」がいる場合、このキャラは【ダブルアタック】と【ブロッカー】を得る。"
    acts = _actions(_abilities(text)[0].effect)
    assert [(a.type.name, a.status) for a in acts] == [
        ("GRANT_KEYWORD", "ダブルアタック"), ("GRANT_KEYWORD", "ブロッカー")]
    assert all(a.target.select_mode == "SOURCE" for a in acts)
    assert _extract_static_keywords(text) == set()


def test_unsided_all_chars_trash_and_deck_or_trash_play():
    """OP18-117: 「キャラすべてをトラッシュに置き」＝両者・「デッキかトラッシュから」＝両ゾーン。"""
    acts = _actions(_abilities(
        "【メイン】自分のライフすべてをトラッシュに置くことができる：自分のリーダーが「スパンダム」の場合、"
        "キャラすべてをトラッシュに置き、自分のデッキかトラッシュからカード名の異なる特徴《中将》を持つ"
        "キャラカード合計5枚までを、登場させる。その後、デッキをシャッフルする。")[0].effect)
    tr = [a for a in acts if a.type.name == "TRASH"][0]
    assert tr.target.player == Player.ALL and tr.target.select_mode == "ALL"
    pl = [a for a in acts if a.type.name == "PLAY_CARD"][0]
    assert pl.target.zone == [Zone.DECK, Zone.TRASH] and pl.target.count == 5 and pl.target.is_unique_name


def test_shuffle_after_the_gated_search_is_inside_the_gate():
    """OP18-117: 「〜の場合、…デッキかトラッシュから登場させる。その後、デッキをシャッフルする」の
    シャッフルは条件の内側（条件が偽なら何も見ていない＝シャッフルも起きない。2026-11-23）。"""
    eff = _abilities(
        "【メイン】自分のライフすべてをトラッシュに置くことができる：自分のリーダーが「スパンダム」の場合、"
        "キャラすべてをトラッシュに置き、自分のデッキかトラッシュからカード名の異なる特徴《中将》を持つ"
        "キャラカード合計5枚までを、登場させる。その後、デッキをシャッフルする。")[0].effect
    assert hasattr(eff, "if_true") and eff.condition is not None, eff        # 効果全体が 1 つの分岐
    assert [a.type.name for a in _actions(eff.if_true)][-1] == "SHUFFLE"
