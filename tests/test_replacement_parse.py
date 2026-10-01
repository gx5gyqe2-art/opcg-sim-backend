"""置換効果（「〜場合、代わりに〜」）と「KOされない」保護のパーサ出力（2026-10-01 カード効果監査）。

**問い**: 本文の限定語（誰が除去されるか・効果／バトル・任意）が解析結果に残るか。
- 置換の限定句は除去されるカードの TargetQuery（OPPONENT_REMOVAL.target）になる。
- 「代わりに〜できる／てもよい」は sub_effect が任意（エンジンが確認を先に挟む）。
- 「KOされない」は修飾なしなら効果KO・バトルKOの両方を防ぐ。
"""
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: E402,F401  (sys.path 設定＋google スタブ)

from opcg_sim.src.effects.parser_v2 import EffectParserV2  # noqa: E402
from opcg_sim.src.models.effect_types import _nfc  # noqa: E402

P = EffectParserV2()


def _ability(text):
    return P.parse_card_text(_nfc(text))[0]


def _conds(c):
    if c is None:
        return []
    if c.type.name == "AND":
        return [x for a in c.args for x in _conds(a)]
    return [c]


def _removal_query(ab):
    for c in _conds(ab.condition):
        if c.type.name == "OPPONENT_REMOVAL":
            return c.target
    return None


def test_replacement_scope_becomes_a_target_query():
    ab = _ability("【ターン1回】自分の元々のコスト5以下の黒のキャラが相手の効果でKOされる場合、"
                  "代わりに自分の手札1枚を捨てることができる。")
    q = _removal_query(ab)
    assert q is not None and q.cost_max == 5 and "ORIGINAL_COST" in q.flags and q.colors == ["黒"]
    assert ab.effect.status == "EFFECT_KO"  # 相手の効果でKO（バウンス等には効かない）
    assert ab.effect.sub_effect.is_optional


def test_other_than_this_character_keeps_the_exclusion_but_a_name_exclusion_does_not():
    ab = _ability("このキャラ以外の自分のコスト5以下の属性(斬)を持つキャラが相手の効果でKOされる場合、"
                  "代わりにこのキャラをレストにできる。")
    q = _removal_query(ab)
    assert "EXCLUDE_SOURCE" in q.flags and q.attributes == ["斬"]
    assert ab.effect.sub_effect.target.select_mode == "SOURCE"  # 「このキャラ」＝置換の持ち主
    ab = _ability("「サボ」以外の自分の元々のコスト7以下のキャラが相手の効果で場を離れる場合、"
                  "代わりにこのキャラを持ち主の手札に戻すことができる。")
    q = _removal_query(ab)
    assert q.exclude_names == ["サボ"] and "EXCLUDE_SOURCE" not in q.flags
    assert ab.effect.sub_effect.target.select_mode == "SOURCE"  # 持ち主句でも ALL にならない


def test_leader_clause_and_plain_ko_cover_battle_ko_too():
    ab = _ability("【ターン1回】自分のリーダーが特徴《王下七武海》を持ち、このキャラが相手の効果で場を離れる場合、"
                  "代わりに自分の、「ゲッコー・モリア」以外のキャラ1枚を持ち主のデッキの下に置いてもよい。")
    assert any(c.type.name == "LEADER_TRAIT" for c in _conds(ab.condition))
    assert ab.effect.status == "LEAVE"
    ab = _ability("このキャラがKOされる場合、代わりに自分のカード2枚をレストにできる。")
    assert ab.effect.status == "EFFECT_KO,BATTLE_KO"
    ab = _ability("このキャラがKOされるか相手の効果で場を離れる場合、代わりに自分の手札1枚を捨てることができる。")
    assert ab.effect.status == "EFFECT_KO,BATTLE_KO,LEAVE"


def test_compound_replacement_is_optional_as_a_whole():
    ab = _ability("【相手のターン中】自分の青の特徴《海軍》を持つキャラが相手の効果で場を離れる場合、"
                  "代わりにこのキャラをレストにし、自分の手札1枚を捨てることができる。")
    assert ab.effect.sub_effect.actions[0].is_optional


def test_that_character_refers_to_the_removed_card():
    ab = _ability("【ドン!!×1】【相手のターン中】【ターン1回】自分のパワー5000以上のキャラがKOされる場合、"
                  "そのキャラはKOされる代わりに、このターン中、パワー-1000できる。")
    assert ab.effect.sub_effect.target.ref_id == "removed_card"


def test_ko_protection_range_follows_the_text():
    def status(text):
        node = _ability(text).effect
        node = node.actions[0] if hasattr(node, "actions") else node
        return node.status

    assert status("このキャラは相手の効果でKOされず、【ブロッカー】を得る。") == "EFFECT_KO"
    assert status("このキャラは効果でKOされず、パワー+1000。") == "EFFECT_KO"
    assert status("相手の場にドン!!が10枚ある場合、このキャラはKOされない。") == "EFFECT_KO,BATTLE_KO"
    assert status("このキャラはリーダーとのバトルでKOされない。") == "BATTLE_KO"


def test_replace_alternative_conditions_and_target_swap():
    ab = _ability("【アタック時】自分のライフと手札の合計枚数が4枚以下の場合、カード1枚を引く。"
                  "自分のコスト8以上のキャラがいる場合、カード1枚を引く代わりに自分のデッキの上から1枚までを、ライフの上に加えることができる。")
    heal = ab.effect.actions[1]
    assert {c.type.name for c in _conds(heal.condition)} >= {"LIFE_HAND_SUM", "FIELD_COUNT"}
    ab = _ability("【メイン】相手のコスト4以下のキャラ1枚までを選び、KOする。自分のトラッシュが15枚以上ある場合、"
                  "相手のコスト4以下のキャラの代わりに相手のコスト6以下のキャラを選ぶ。")
    # 条件付きの対象差し替え（F の Branch 表現）: 満たせば cost6 以下、満たさなければ cost4 以下。
    br = ab.effect
    assert br.if_true.target.cost_max == 6
    assert br.if_false.target.cost_max == 4


def test_triggers_for_leave_and_rest_and_self_play_from_trash():
    ab = _ability("【相手のターン中】このステージをレストにできる:自分の特徴《麦わらの一味》を持つキャラが相手の効果で場を離れた時、"
                  "ドン!!デッキからドン!!1枚までを、レストで追加する。")
    assert ab.trigger.name == "ON_LEAVE"
    abs_ = P.parse_card_text(_nfc("【相手のターン中】【ターン1回】自分の特徴《ドレスローザ》を持つキャラがKOされた時か、"
                                  "相手の効果で場を離れた時、発動できる。カード1枚を引く。"))
    # 「KOされた時か〜離れた時」は ON_KO 1 本（離脱側はエンジンが句を読んで積む・OP10-042）。
    assert [a.trigger.name for a in abs_] == ["ON_KO"]
    ab = _ability("このキャラが相手の効果でレストになった時、発動できる。このキャラをトラッシュに置き、カード2枚を引くことができる。")
    assert ab.trigger.name == "ON_REST" and ab.effect.actions[0].is_optional
    ab = _ability("【相手のターン中】自分の手札1枚を捨てることができる:このキャラが相手の効果でKOされた時、"
                  "このキャラカードをトラッシュからレストで登場させる。")
    assert ab.effect.target.ref_id == "self"
