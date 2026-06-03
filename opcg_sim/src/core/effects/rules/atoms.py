"""原子アクションの宣言的ルール（シード実装）。

ここに登録されたルールが、新パーサ(EffectParserV2)の原子句解析を担う。
未対応の句はレガシー parser.py にフォールバックされ、診断ツールで
「未対応句ランキング」として可視化される。ランキング上位から本ファイルに
ルールを追加していくことで、段階的にカバレッジを burn down できる。

各ルールは小さく・独立・テスト可能であることを重視する。
"""
from __future__ import annotations

import re
from typing import Optional

from ....models.effect_types import GameAction, TargetQuery, ValueSource
from ....models.enums import ActionType, Player, Zone
from ..matcher import parse_target
from .base import ParseContext, rule, _nfc


def _first_int(text: str, default: int = 0) -> int:
    nums = re.findall(r"[+-]?\d+", text)
    return int(nums[0]) if nums else default


# ---------------------------------------------------------------------------
# ドロー: 「カードN枚を引く」
# ---------------------------------------------------------------------------
@rule("draw", priority=80)
def _draw(ctx: ParseContext) -> Optional[GameAction]:
    t = ctx.text
    if _nfc("引く") not in t and not re.search(_nfc(r"カード\d*枚?を?引き"), t):
        return None
    # 付与/その他の「引く」誤検知を避けるため、ドロー以外の強い動詞があれば見送る
    if _nfc("付与") in t or _nfc("KOする") in t:
        return None
    return GameAction(
        type=ActionType.DRAW,
        value=ValueSource(base=_first_int(t, 1)),
        raw_text=t,
    )


# ---------------------------------------------------------------------------
# KO: 「（対象）をKOする」
# ---------------------------------------------------------------------------
@rule("ko", priority=70)
def _ko(ctx: ParseContext) -> Optional[GameAction]:
    t = ctx.text
    if _nfc("KOする") not in t:
        return None
    tq = parse_target(t)
    if _nfc("まで") in t or re.search(r"\d+枚まで", t):
        tq.is_up_to = True
    return GameAction(type=ActionType.KO, target=tq, raw_text=t)


# ---------------------------------------------------------------------------
# 自己レスト（コスト): 「このキャラ／このリーダーをレストにできる」
# ---------------------------------------------------------------------------
@rule("rest_self_cost", priority=90)
def _rest_self_cost(ctx: ParseContext) -> Optional[GameAction]:
    if not ctx.is_cost:
        return None
    t = ctx.text
    if not re.search(_nfc(r"このキャラをレスト|このリーダーをレスト"), t):
        return None
    return GameAction(
        type=ActionType.REST,
        target=TargetQuery(
            player=Player.SELF,
            zone=Zone.FIELD,
            count=1,
            is_strict_count=True,
            ref_id="self",
        ),
        raw_text=t,
    )


# ---------------------------------------------------------------------------
# レスト: 「（対象）をレストにする」
# ---------------------------------------------------------------------------
@rule("rest", priority=40)
def _rest(ctx: ParseContext) -> Optional[GameAction]:
    t = ctx.text
    if not re.search(_nfc(r"レストに(する|し[、。])"), t):
        return None
    if re.search(_nfc(r"このキャラをレスト|このリーダーをレスト"), t):
        return None  # 自己レストは rest_self_cost が担当
    tq = parse_target(t)
    if _nfc("まで") in t:
        tq.is_up_to = True
    return GameAction(type=ActionType.REST, target=tq, raw_text=t)


# ---------------------------------------------------------------------------
# 手札を捨てる: 「自分の手札N枚を捨てる（ことができる）」
# ---------------------------------------------------------------------------
@rule("discard", priority=50)
def _discard(ctx: ParseContext) -> Optional[GameAction]:
    t = ctx.text
    if _nfc("捨てる") not in t:
        return None
    # 「デッキ…トラッシュに置く」等は別アクション。ここは手札の discard に限定する。
    if _nfc("手札") not in t:
        return None
    tq = parse_target(t)
    if _nfc("まで") in t:
        tq.is_up_to = True
    return GameAction(type=ActionType.DISCARD, target=tq, raw_text=t)


# ---------------------------------------------------------------------------
# パワー増減: 「（対象）を、…パワー±N」
# ---------------------------------------------------------------------------
@rule("power_buff", priority=60)
def _power_buff(ctx: ParseContext) -> Optional[GameAction]:
    t = ctx.text
    m = re.search(_nfc(r"パワー([+-]\d+)"), t)
    if not m:
        return None
    if _nfc("にする") in t:
        return None  # 「パワーをNにする」は base_power_override 系（別ルールで対応予定）
    tq = parse_target(t)
    return GameAction(
        type=ActionType.BUFF,
        target=tq,
        value=ValueSource(base=int(m.group(1))),
        raw_text=t,
    )


# 符号として使われ得る各種マイナス記号（ASCII / 全角 / 数学記号 / ハイフン）
_SIGN_RE = re.compile(r"コスト[ 　]*([+\-－−‐])[ 　]*(\d+)")


# ---------------------------------------------------------------------------
# コスト増減: 「（対象）を、…コスト±N」
#   従来は ActionType.OTHER に落ちて「解析できたが何もしない」状態だった。
#   resolver は BUFF + status="COST_REDUCTION" を cost_buff 加算として実行できる。
# ---------------------------------------------------------------------------
@rule("cost_change", priority=58)
def _cost_change(ctx: ParseContext) -> Optional[GameAction]:
    t = ctx.text
    m = _SIGN_RE.search(t)
    if not m:
        return None
    sign = -1 if m.group(1) in "-－−‐" else 1
    value = sign * int(m.group(2))
    tq = parse_target(t)
    return GameAction(
        type=ActionType.BUFF,
        target=tq,
        value=ValueSource(base=value),
        status="COST_REDUCTION",
        raw_text=t,
    )


# ---------------------------------------------------------------------------
# デッキシャッフル: 「デッキをシャッフルする」
#   従来は OTHER（legacy にシャッフル判定が無かった）。resolver は SHUFFLE を実行可。
# ---------------------------------------------------------------------------
@rule("shuffle", priority=85)
def _shuffle(ctx: ParseContext) -> Optional[GameAction]:
    if _nfc("シャッフル") not in ctx.text:
        return None
    return GameAction(type=ActionType.SHUFFLE, raw_text=ctx.text)


# ---------------------------------------------------------------------------
# 残りをデッキの下へ: 「残りを（好きな順番で）デッキの下に置く」
#   「置き、」で文分割されると「…デッキの下に」だけが残り OTHER 化していた。
#   置く有無に依らず、残り(TEMP)→デッキ下 として解釈する。
# ---------------------------------------------------------------------------
@rule("remaining_deck_bottom", priority=65)
def _remaining_deck_bottom(ctx: ParseContext) -> Optional[GameAction]:
    t = ctx.text
    if _nfc("残り") not in t or _nfc("デッキの下") not in t:
        return None
    return GameAction(
        type=ActionType.DECK_BOTTOM,
        target=TargetQuery(
            player=Player.SELF, zone=Zone.TEMP, select_mode="REMAINING", count=-1
        ),
        raw_text=t,
    )


# ---------------------------------------------------------------------------
# 自己登場: 「このカード／このキャラ／このリーダーを登場させる」
#   主にトリガー（ライフから自身を登場）で使われる。対象は自身(ref_id=self)。
#   従来は対象が汎用 FIELD/SELF になり、誤った対象を登場させていた。
# ---------------------------------------------------------------------------
@rule("play_self", priority=75)
def _play_self(ctx: ParseContext) -> Optional[GameAction]:
    t = ctx.text
    if _nfc("登場させる") not in t:
        return None
    if not re.search(_nfc(r"この(カード|キャラ|リーダー)を、?登場させる"), t):
        return None
    return GameAction(
        type=ActionType.PLAY_CARD,
        target=TargetQuery(player=Player.SELF, zone=Zone.FIELD, count=1, ref_id="self"),
        destination=Zone.FIELD,
        raw_text=t,
    )
