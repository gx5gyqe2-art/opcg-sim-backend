"""パーサ H1 群の是正（2026-10-01 カード効果監査・再修正）。

**問い**: 本文の限定語・条件・任意性が解析結果に残るか。
- OP02-002: 「ドン!!が付与された時」の誘発句は効果本体に残さない（残すと対象に自分のリーダーが混入する）。
- OP07-083: コストの「カードN枚をデッキの下に置くことができる」は厳密枚数（N 枚未満では払えない）。
- OP11-024: 「捨て、…してもよい」は一体の任意＝確認は 1 回、後句は前句が成立したときだけ。
- OP15-031: 「選んだキャラのコストが付与ドン!!数と同じ場合」は選んだキャラへの参照フィルタ。
"""
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: E402,F401  (sys.path 設定＋google スタブ)

from opcg_sim.src.effects.parser_v2 import EffectParserV2  # noqa: E402
from opcg_sim.src.models.effect_types import _nfc  # noqa: E402

P = EffectParserV2()


def _ability(text, i=0):
    return P.parse_card_text(_nfc(text))[i]


def _flat(node):
    out = []
    if node is None:
        return out
    out.append(node)
    for f in ("if_true", "if_false"):
        out += _flat(getattr(node, f, None))
    for a in getattr(node, "actions", None) or []:
        out += _flat(a)
    return out


def test_don_attached_trigger_clause_is_not_part_of_the_target():
    ab = _ability("【自分のターン中】このリーダーか自分のキャラにドン!!が付与された時、"
                  "相手のコスト7以下のキャラ1枚までを、このターン中、コスト-1。")
    act = ab.effect
    assert act.target.card_type == ["CHARACTER"]
    assert act.target.cost_max == 7


def test_deck_bottom_cost_requires_exact_count():
    ab = _ability("【起動メイン】自分のトラッシュの特徴《スリラーバーク海賊団》を持つカード4枚を"
                  "好きな順番でデッキの下に置くことができる:このキャラは、このターン中、【バニッシュ】を得て、パワー+1000。")
    assert ab.cost.target.is_strict_count is True
    assert ab.cost.target.count == 4


def test_discard_then_rest_don_is_one_optional_unit():
    ab = _ability("このキャラが相手の効果でKOされた時、自分の手札1枚を捨て、自分のドン!!1枚をレストにしてもよい。"
                  "そうした場合、自分の手札からコスト6以下の、特徴《魚人族》か《人魚族》を持つキャラカード1枚までを、登場させる。")
    nodes = _flat(ab.effect)
    types = [(type(n).__name__, getattr(getattr(n, "type", None), "name", None)) for n in nodes]
    discard = next(n for n in nodes if getattr(getattr(n, "type", None), "name", None) == "DISCARD")
    rest = next(n for n in nodes if getattr(getattr(n, "type", None), "name", None) == "REST_DON")
    assert discard.is_optional is True, types
    assert rest.is_optional is False, types
    # 後句（レスト）は「そうした」＝捨てたときだけ
    gates = [n for n in nodes if type(n).__name__ == "Branch" and n.condition is not None
             and n.condition.type.name == "PREV_ACTION" and rest in _flat(n.if_true)]
    assert gates, types


def test_selected_character_cost_equals_attached_don_is_a_ref_filter():
    ab = _ability("【登場時】相手のレストのキャラ1枚までを選ぶ。選んだキャラのコストがそのキャラに"
                  "付与されているドン!!の枚数と同じ場合、KOする。")
    nodes = _flat(ab.effect)
    ko = next(n for n in nodes if getattr(getattr(n, "type", None), "name", None) == "KO")
    assert ko.target.ref_id == "selected_card"
    assert "REF_COST_EQ_ATTACHED_DON" in ko.target.flags
    assert not any(type(n).__name__ == "Branch" and n.condition is not None
                   and n.condition.type.name == "DON_COUNT" for n in nodes)
