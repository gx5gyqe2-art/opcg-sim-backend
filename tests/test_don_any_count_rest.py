"""「ドン!!を任意の枚数レストにできる」（OP13-001・2026-10-01 カード効果監査 H3_count）。

**問い**: 枚数をプレイヤーが選べる（0..N 枚）形で解析され、「1枚につき」の倍率が選んだ枚数に
掛かるか。REST_DON の固定 1 枚ではなく COST_AREA の SELECT_TARGET（is_up_to）に乗る。
"""
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: E402,F401

from opcg_sim.src.effects.parser_v2 import EffectParserV2  # noqa: E402
from opcg_sim.src.models.effect_types import _nfc  # noqa: E402

TEXT = ("【ドン!!×1】【相手のアタック時】自分のアクティブのドン!!が5枚以下の場合、自分のドン!!を任意の枚数レストにできる。"
        "レストにしたドン!!1枚につき、このリーダーか自分の特徴《麦わらの一味》を持つキャラ1枚までを、"
        "このバトル中、パワー+2000。")


def test_any_count_don_rest_is_variable_selection():
    ab = EffectParserV2().parse_card_text(_nfc(TEXT))[0]
    seq = ab.effect
    acts = [a for a in (getattr(seq, "actions", None) or getattr(seq, "nodes", None) or [seq])]
    rest = acts[0]
    assert rest.type.name == "REST"
    tq = rest.target
    assert tq.zone.name == "COST_AREA" and tq.is_up_to and tq.count >= 10
    assert tq.is_rest is False
    buff = acts[1]
    assert buff.value.dynamic_source == "PREV_ACTION_COUNT" and buff.value.multiplier == 2000
