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
def _duration_of(text: str) -> str:
    if _nfc("このバトル中") in text:
        return "THIS_BATTLE"
    if _nfc("このターン中") in text:
        return "THIS_TURN"
    return "INSTANT"


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
        duration=_duration_of(t),
        raw_text=t,
    )


# ---------------------------------------------------------------------------
# キーワード付与: 「（対象）は【ブロッカー】を得る」
#   parser.py の構造分解が keyword タグを保持するようになり、原子句に
#   【ブロッカー】等が残る。これを GRANT_KEYWORD(status=キーワード名) に変換する。
#   従来は keyword が脱落して BUFF/OTHER に落ち、能力が付かなかった（既知バグ）。
#   「このキャラ／このリーダー／このカード」が主語なら対象は自身(SOURCE)。
# ---------------------------------------------------------------------------
_KEYWORD_GRANT_RE = re.compile(
    _nfc(r"【(ブロッカー|速攻[^】]*|ダブルアタック|バニッシュ|ブロック不可|貫通|シフト)】")
)


@rule("grant_keyword", priority=63)
def _grant_keyword(ctx: ParseContext) -> Optional[GameAction]:
    t = ctx.text
    if _nfc("得る") not in t:
        return None
    m = _KEYWORD_GRANT_RE.search(t)
    if not m:
        return None
    # パワー増減を伴う複合句は power_buff に委ねる（単一アクションでは両立不可）。
    if re.search(_nfc(r"パワー[+-]\d+"), t):
        return None
    keyword = m.group(1)
    if re.search(_nfc(r"この(カード|キャラ|リーダー)"), t):
        tq = TargetQuery(select_mode="SOURCE")
    else:
        tq = parse_target(t)
    return GameAction(
        type=ActionType.GRANT_KEYWORD,
        target=tq,
        status=keyword,
        duration=_duration_of(t),
        raw_text=t,
    )


# ---------------------------------------------------------------------------
# ライフ操作 ---------------------------------------------------------------
#   実カードに頻出するライフ周りの原子句をルール化する。従来は legacy へ
#   フォールバックし、一部は誤った destination（ライフ→手札なのに dest=LIFE）や
#   OTHER（表/裏向き）になっていた。
# ---------------------------------------------------------------------------


@rule("life_face", priority=72)
def _life_face(ctx: ParseContext) -> Optional[GameAction]:
    """「（自分の）ライフ…を表向き／裏向きにする」→ FACE_UP_LIFE(status=UP/DOWN)。"""
    t = ctx.text
    if _nfc("ライフ") not in t:
        return None
    if _nfc("表向き") in t:
        status = "UP"
    elif _nfc("裏向き") in t:
        status = "DOWN"
    else:
        return None
    tq = parse_target(t)
    tq.zone = Zone.LIFE
    return GameAction(type=ActionType.FACE_UP_LIFE, target=tq, status=status, raw_text=t)


@rule("life_recover", priority=71)
def _life_recover(ctx: ParseContext) -> Optional[GameAction]:
    """「（自分の）デッキの上から…ライフの上に加える」→ HEAL（デッキ上→ライフ）。

    エンジンの HEAL は対象を見ずデッキ上から value 枚をライフへ加えるため、
    対象選択待ちに陥らないよう target=None とする（legacy は target=LIFE で
    余計な選択を発生させていた）。
    """
    t = ctx.text
    if _nfc("デッキの上") not in t or _nfc("ライフ") not in t:
        return None
    if _nfc("加える") not in t and _nfc("置く") not in t:
        return None
    if _nfc("手札") in t:
        return None  # デッキ→手札 等は別アクション
    return GameAction(
        type=ActionType.HEAL,
        target=None,
        value=ValueSource(base=_first_int(t, 1)),
        raw_text=t,
    )


@rule("life_to_hand", priority=70)
def _life_to_hand(ctx: ParseContext) -> Optional[GameAction]:
    """「（自分／相手の）ライフの上（か下）から…手札に加える／戻す」→ MOVE_CARD(dest=HAND)。

    legacy は「上か下から」を destination=LIFE と誤判定していた（実質 no-op）。
    """
    t = ctx.text
    if _nfc("ライフの上") not in t and _nfc("ライフの下") not in t:
        return None
    if _nfc("手札に加える") not in t and _nfc("手札に戻す") not in t:
        return None
    tq = parse_target(t)
    tq.zone = Zone.LIFE
    return GameAction(
        type=ActionType.MOVE_CARD,
        target=tq,
        destination=Zone.HAND,
        raw_text=t,
    )


@rule("hand_to_life", priority=69)
def _hand_to_life(ctx: ParseContext) -> Optional[GameAction]:
    """「（自分の）手札…を、ライフの上／下に加える」→ MOVE_CARD(dest=LIFE)。"""
    t = ctx.text
    if _nfc("手札") not in t:
        return None
    if _nfc("ライフの上に加える") not in t and _nfc("ライフの下に加える") not in t:
        return None
    tq = parse_target(t)
    tq.zone = Zone.HAND
    return GameAction(
        type=ActionType.MOVE_CARD,
        target=tq,
        destination=Zone.LIFE,
        raw_text=t,
    )


@rule("life_to_trash", priority=68)
def _life_to_trash(ctx: ParseContext) -> Optional[GameAction]:
    """「（自分／相手の）ライフの上（か下）から…トラッシュに置く」→ TRASH。"""
    t = ctx.text
    if _nfc("ライフの上") not in t and _nfc("ライフの下") not in t:
        return None
    if _nfc("トラッシュ") not in t or _nfc("置く") not in t:
        return None
    tq = parse_target(t)
    tq.zone = Zone.LIFE
    return GameAction(type=ActionType.TRASH, target=tq, raw_text=t)


# ---------------------------------------------------------------------------
# ドン!! 操作 --------------------------------------------------------------
#   ドン!!は均質（どれを選んでも同じ）なため、対象を1枚ずつ選択させると
#   無意味な中断が起きる。そこで枚数(value)ベースで扱い、対象は持たない
#   （付与=ATTACH_DON のみ付与先キャラを対象に持つ）。プレイヤーは
#   status="OPPONENT" で相手のドンを指す（「相手は自身の…」用）。
# ---------------------------------------------------------------------------
_DON_COUNT_RE = re.compile(_nfc(r"ドン(?:!!|‼)?[ 　]*(\d+)[ 　]*枚"))


def _don_count(t: str) -> int:
    if _nfc("すべて") in t or _nfc("全て") in t:
        return 99  # エンジン側でプールが尽きるまで処理
    m = _DON_COUNT_RE.search(t)
    return int(m.group(1)) if m else 1


def _don_opponent(t: str) -> Optional[str]:
    return "OPPONENT" if (_nfc("相手") in t and _nfc("自分") not in t) else None


@rule("don_attach", priority=84)
def _don_attach(ctx: ParseContext) -> Optional[GameAction]:
    """「（自分の）リーダーかキャラ1枚に（レストの）ドン!!N枚までを付与する」→ ATTACH_DON。

    付与先（リーダー/キャラ）を対象に持つ。「レストのドン」は status="RESTED"。
    付与先は parse_target だと「レスト」で is_rest が立ってしまうため手動構築する。
    """
    t = ctx.text
    if _nfc("付与") not in t or _nfc("ドン") not in t:
        return None
    recipient = TargetQuery(player=Player.SELF, zone=Zone.FIELD, count=1)
    if _nfc("リーダー") in t:
        recipient.card_type.append("LEADER")
    if _nfc("キャラ") in t:
        recipient.card_type.append("CHARACTER")
    if not recipient.card_type:
        recipient.card_type.extend(["LEADER", "CHARACTER"])
    return GameAction(
        type=ActionType.ATTACH_DON,
        target=recipient,
        value=ValueSource(base=_don_count(t)),
        status="RESTED" if _nfc("レストのドン") in t else None,
        raw_text=t,
    )


@rule("don_set_active", priority=74)
def _don_set_active(ctx: ParseContext) -> Optional[GameAction]:
    """「（自分の）ドン!!N枚までを、アクティブにする」→ ACTIVE_DON（レスト→アクティブ）。"""
    t = ctx.text
    if _nfc("ドン") not in t:
        return None
    if _nfc("アクティブにする") not in t and _nfc("アクティブにできる") not in t:
        return None
    return GameAction(
        type=ActionType.ACTIVE_DON,
        target=None,
        value=ValueSource(base=_don_count(t)),
        status=_don_opponent(t),
        raw_text=t,
    )


@rule("don_set_rest", priority=74)
def _don_set_rest(ctx: ParseContext) -> Optional[GameAction]:
    """「（自分の）ドン!!N枚をレストにする/できる」→ REST_DON（アクティブ→レスト）。多くはコスト。"""
    t = ctx.text
    if _nfc("ドン") not in t:
        return None
    if not re.search(_nfc(r"レストに(する|できる|し[、。])"), t):
        return None
    if _nfc("アクティブ") in t:
        return None
    return GameAction(
        type=ActionType.REST_DON,
        target=None,
        value=ValueSource(base=_don_count(t)),
        status=_don_opponent(t),
        raw_text=t,
    )


@rule("don_return_deck", priority=83)
def _don_return_deck(ctx: ParseContext) -> Optional[GameAction]:
    """「（場の）ドン!!…をドン!!デッキに戻す」→ RETURN_DON。

    「ドン!!-N」記法は上位の don_return が処理するため、ここは明示的な
    「ドン!!デッキに戻す」表記のみを担う。
    """
    t = ctx.text
    if _nfc("ドン") not in t or _nfc("戻す") not in t:
        return None
    if not re.search(_nfc(r"ドン(?:!!|‼)?デッキ"), t):
        return None
    return GameAction(
        type=ActionType.RETURN_DON,
        target=None,
        value=ValueSource(base=_don_count(t)),
        status=_don_opponent(t),
        raw_text=t,
    )


# ---------------------------------------------------------------------------
# 除去保護: 「相手の効果で場を離れない」「（バトルで）KOされない」
#   保護マーカーを生成し、除去の瞬間に gamestate 側でライブ評価される。
#   多くは条件付き PASSIVE（例: トラッシュ7枚以上の場合）。
# ---------------------------------------------------------------------------
@rule("prevent_leave", priority=64)
def _prevent_leave(ctx: ParseContext) -> Optional[GameAction]:
    t = ctx.text
    if _nfc("場を離れない") in t:
        status = "LEAVE"
    elif _nfc("KOされない") in t:
        status = "BATTLE_KO"
    else:
        return None
    return GameAction(
        type=ActionType.PREVENT_LEAVE,
        target=TargetQuery(select_mode="SOURCE"),
        status=status,
        raw_text=t,
    )


# ---------------------------------------------------------------------------
# アタック制限: 「（このターン中／次の…まで）アタックできない」
#   継続効果として管理し、適切なタイミングで失効する（従来 OTHER）。
# ---------------------------------------------------------------------------
@rule("attack_disable", priority=62)
def _attack_disable(ctx: ParseContext) -> Optional[GameAction]:
    t = ctx.text
    if _nfc("アタックできない") not in t:
        return None
    tq = parse_target(t)
    duration = "UNTIL_NEXT_TURN_END" if _nfc("次の") in t else "THIS_TURN"
    return GameAction(type=ActionType.ATTACK_DISABLE, target=tq, duration=duration, raw_text=t)


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
        duration=_duration_of(t),
        raw_text=t,
    )


# ---------------------------------------------------------------------------
# ドン!!返却: 「ドン‼-N」（場のドン!!を N 枚ドン!!デッキへ戻す）
#   多くはコストとして登場。従来 OTHER（何もしない）だった。
# ---------------------------------------------------------------------------
_DON_RETURN_RE = re.compile(r"ドン(?:!!|‼)[ 　]*[-－−‐][ 　]*(\d+)")


@rule("don_return", priority=88)
def _don_return(ctx: ParseContext) -> Optional[GameAction]:
    m = _DON_RETURN_RE.search(ctx.text)
    if not m:
        return None
    return GameAction(
        type=ActionType.RETURN_DON,
        value=ValueSource(base=int(m.group(1))),
        raw_text=ctx.text,
    )


# ---------------------------------------------------------------------------
# ドン!!追加: 「ドン!!デッキからドン!!N枚までを、アクティブ/レストで追加する」
#   「レストで追加」は従来 OTHER。status=RESTED を付けて resolver に伝える。
# ---------------------------------------------------------------------------
@rule("don_add", priority=86)
def _don_add(ctx: ParseContext) -> Optional[GameAction]:
    t = ctx.text
    if _nfc("追加") not in t:
        return None
    is_active = _nfc("アクティブで追加") in t or _nfc("アクティブで加える") in t
    is_rested = _nfc("レストで追加") in t
    if not (is_active or is_rested):
        return None
    return GameAction(
        type=ActionType.RAMP_DON,
        value=ValueSource(base=_first_int(t, 1)),
        status="RESTED" if is_rested else None,
        raw_text=t,
    )


# ---------------------------------------------------------------------------
# 自己メイン再発動: 「このカードの【メイン】効果を発動する」
#   主にトリガー。resolver が自身の ACTIVATE_MAIN 効果を再実行する。
#   「（対象）を、発動する」（イベントのプレイ）とは "効果を発動" の有無で区別。
# ---------------------------------------------------------------------------
@rule("execute_main", priority=82)
def _execute_main(ctx: ParseContext) -> Optional[GameAction]:
    if _nfc("効果を発動") not in ctx.text:
        return None
    return GameAction(type=ActionType.EXECUTE_MAIN_EFFECT, raw_text=ctx.text)


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
