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

from ....models.effect_types import Choice, EffectNode, GameAction, Sequence, TargetQuery, ValueSource
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
    # 「引く代わりに」は REPLACE_EFFECT 文脈なので除外（life_recover 等が担当）
    if re.search(_nfc(r"引く代わりに"), t):
        return None
    # 「引くことができない」= ドロー制限であり、ドローアクションではない（self_cannot が担当）
    if re.search(_nfc(r"引くことができない"), t):
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
    # 「KOする／できる」に加え、Sequence 分割で末尾が連用形「KOし」になる句も対象
    # （例:「相手の…をKOし、このカードを手札に加える」→ 前段「…をKOし」）。
    if not re.search(_nfc(r"KO(する|できる|してもよい)"), t) and not re.search(_nfc(r"KOし(?:[、。]|$)"), t.strip()):
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
# 自己レスト（効果）: 「このキャラ／このリーダーをレストにする（できる）」(コスト以外)
# ---------------------------------------------------------------------------
@rule("rest_self", priority=76)
def _rest_self(ctx: ParseContext) -> Optional[GameAction]:
    """「このキャラ／このリーダーをレストにする（できる）」(効果文脈) → REST(SOURCE)。"""
    if ctx.is_cost:
        return None  # コスト文脈は rest_self_cost が担当
    t = ctx.text
    if not re.search(_nfc(r"このキャラをレスト|このリーダーをレスト"), t):
        return None
    return GameAction(
        type=ActionType.REST,
        target=TargetQuery(select_mode="SOURCE"),
        raw_text=t,
    )


# ---------------------------------------------------------------------------
# レスト: 「（対象）をレストにする」
# ---------------------------------------------------------------------------
@rule("rest", priority=40)
def _rest(ctx: ParseContext) -> Optional[GameAction]:
    t = ctx.text
    # 「レストにする／し／できる」を対象とする。従来は「できる」を取りこぼし、
    # 「このステージをレストにできる」等が OTHER に落ちていた。
    # 「レストにする／し、／し。」に加え、Sequence 分割で末尾が連用形「レストにし」になる
    # 句も対象（例:「相手の…をレストにし、このカードを手札に加える」→ 前段「…をレストにし」）。
    if not re.search(_nfc(r"レストに(する|できる|し[、。]|し$)"), t.strip()):
        return None
    if re.search(_nfc(r"このキャラをレスト|このリーダーをレスト"), t):
        return None  # 自己レストは rest_self_cost が担当
    # ドン!!が直接のレスト対象の場合のみ除外（「ドン!!が付与されている」等の修飾語は除外しない）
    if re.search(_nfc(r"ドン!![^がのは]*をレストに|コストエリア.*レストに"), t):
        return None  # ドン!!自体のレストは don_set_rest が担当
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
    # 連用形「捨て」も対象（「捨て」は「捨てる」「捨てて」「捨て（て形）」すべてを包含）
    if _nfc("捨て") not in t:
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
# パワー設定（上書き）: 「（対象）を、…パワーNにする／元々のパワーNにする」
#   power_buff(priority=60) は「±N」を担当し「にする」を明示除外している。
#   ここは静的な数値設定（base_power_override）のみを担当する。
#   エンジンは BUFF+status="POWER_OVERRIDE" で base_power_override をセットし、
#   reset_turn_status() で失効する（「このターン中」相当のセマンティクス）。
#   「相手のリーダーと同じパワーになる」「入れ替える」等の動的参照は C9 の別件として除外。
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# C9 同値パワー: 「このキャラの元々のパワーは、（このターン中、）
#   （相手のリーダー／選んだキャラ／アタックしているリーダーかキャラ）と同じパワーになる」
#   → BUFF+POWER_OVERRIDE。値は発動時スナップショット（dynamic_source=REFERENCE_POWER）。
#   参照は ref_id: opp_leader / selected / attacker。対象は自身(SOURCE)。
# ---------------------------------------------------------------------------
@rule("power_equalize", priority=62)
def _power_equalize(ctx: ParseContext) -> Optional[GameAction]:
    t = ctx.text
    if _nfc("同じパワー") not in t:
        return None
    # 参照先を判定（アタック文脈＞選択＞相手リーダー）
    if _nfc("アタックしている") in t:
        ref = "attacker"
    elif _nfc("選んだ") in t:
        ref = "selected"
    elif _nfc("相手のリーダー") in t:
        ref = "opp_leader"
    else:
        return None  # 未知の参照は set_power 等にフォールバック
    return GameAction(
        type=ActionType.BUFF,
        status="POWER_OVERRIDE",
        target=TargetQuery(select_mode="SOURCE"),
        value=ValueSource(dynamic_source="REFERENCE_POWER", ref_id=ref),
        duration=_duration_of(t),
        raw_text=t,
    )


@rule("power_swap", priority=61)
def _power_swap(ctx: ParseContext) -> Optional[GameAction]:
    """「選んだキャラそれぞれの元々のパワーを、このターン中/このバトル中、入れ替える」→ SWAP_POWER。"""
    t = ctx.text
    if _nfc("入れ替え") not in t:
        return None
    if _nfc("パワー") not in t:
        return None
    tq = parse_target(t)
    return GameAction(
        type=ActionType.SWAP_POWER,
        target=tq,
        duration=_duration_of(t),
        raw_text=t,
    )


@rule("set_power", priority=59)
def _set_power(ctx: ParseContext) -> Optional[GameAction]:
    t = ctx.text
    # 動的参照（同値・入れ替え）は対象外
    if _nfc("同じパワー") in t or _nfc("入れ替") in t:
        return None
    m = re.search(_nfc(r"パワー(?:を)?(\d+)に(?:なる|する)"), t)
    if not m:
        return None
    tq = parse_target(t)
    if _nfc("まで") in t:
        tq.is_up_to = True
    return GameAction(
        type=ActionType.BUFF,
        status="POWER_OVERRIDE",
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


@rule("prevent_leave_and_keyword", priority=70)
def _prevent_leave_and_keyword(ctx: ParseContext):
    """「このキャラは相手の効果で場を離れず（離れない）、【X】を得る」の複合。
    PREVENT_LEAVE と GRANT_KEYWORD の両方を Sequence で返す。従来は1原子句のため
    GRANT_KEYWORD のみが残り、トラッシュ7枚以上の除去保護(ナス寿郎/ウォーキュリー/マーズ)が
    脱落していた。条件(トラッシュ7枚以上)は ability 側に lift されるので原子句は分割しない。"""
    t = ctx.text
    if not (_nfc("場を離れない") in t or _nfc("場を離れず") in t):
        return None
    if _nfc("得る") not in t:
        return None
    m = _KEYWORD_GRANT_RE.search(t)
    if not m:
        return None
    keyword = m.group(1)
    src = TargetQuery(select_mode="SOURCE")
    prevent = GameAction(type=ActionType.PREVENT_LEAVE, target=TargetQuery(select_mode="SOURCE"),
                         status="LEAVE", raw_text=t)
    grant = GameAction(type=ActionType.GRANT_KEYWORD, target=src, status=keyword,
                       duration=_duration_of(t), raw_text=t)
    return Sequence(actions=[prevent, grant])


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
    # 「ライフの表向きのカードをトラッシュに置く」はトラッシュアクション（修飾語として使用）
    if _nfc("トラッシュ") in t:
        return None
    tq = parse_target(t)
    tq.zone = Zone.LIFE
    return GameAction(type=ActionType.FACE_UP_LIFE, target=tq, status=status, raw_text=t)


@rule("life_cards_to_trash", priority=74)
def _life_cards_to_trash(ctx: ParseContext) -> Optional[GameAction]:
    """「自分のライフの表向きのカードすべてをトラッシュに置く」→ TRASH(zone=LIFE, face-up filter)。"""
    t = ctx.text
    if _nfc("ライフ") not in t or _nfc("トラッシュ") not in t:
        return None
    if _nfc("表向き") not in t:
        return None
    tq = parse_target(t)
    tq.zone = Zone.LIFE
    tq.player = Player.SELF
    if _nfc("すべて") in t or _nfc("全て") in t:
        tq.count = -1
        tq.select_mode = "ALL"
    return GameAction(type=ActionType.TRASH, target=tq, raw_text=t)


# ---------------------------------------------------------------------------
# ライフ scry（C7）: 「（自分か相手の）ライフの上から1枚（まで）を見て、ライフの上か下に置く」
#   → 対話選択（Choice）で実装する。フロントは action_type="CHOICE"＋options を
#     既にボタン描画・index 返却まで対応済みなので、バックエンドで Choice ツリーを
#     生成すれば end-to-end で動く（resolver の suspend/resume に乗る）。
#   構造:
#     Choice[どのライフを見るか]
#       ├ 自分: Sequence[LOOK_LIFE(SELF,1) → Choice[上/下に置く（SELF）]]
#       ├ 相手: Sequence[LOOK_LIFE(OPP,1)  → Choice[上/下に置く（OPP）]]
#       └ （「まで」なら）見ない: Sequence[]（no-op）
#   LOOK_LIFE が対象ライフ上 1 枚を temp_zone へ移し、後続 Choice が temp→ライフ上/下へ戻す。
# ---------------------------------------------------------------------------
def _place_temp_to_life(target_player: Player, position: str, raw: str) -> GameAction:
    """temp_zone の公開カードを target_player のライフの上(TOP)/下(BOTTOM)へ戻す。"""
    return GameAction(
        type=ActionType.MOVE_CARD,
        target=TargetQuery(zone=Zone.TEMP, player=target_player, select_mode="ALL", count=1),
        destination=Zone.LIFE,
        dest_position=position,
        raw_text=raw,
    )


def _scry_one_life(target_player: Player, status: str, raw: str) -> Sequence:
    look = GameAction(type=ActionType.LOOK_LIFE, status=status, value=ValueSource(base=1), raw_text=raw)
    place = Choice(
        message="ライフの上か下に置く",
        option_labels=["ライフの上に置く", "ライフの下に置く"],
        options=[
            _place_temp_to_life(target_player, "TOP", raw),
            _place_temp_to_life(target_player, "BOTTOM", raw),
        ],
    )
    return Sequence(actions=[look, place])


# ---------------------------------------------------------------------------
# イベント発動: 「自分の手札から（条件）イベント1枚までを、発動する」
#   → EXECUTE_EVENT。エンジンは手札の該当イベントの効果を解決しトラッシュへ送る。
#   PLAY_CARD（キャラ登場）とは別概念。従来 OTHER（不発）。
#   「発動した時」(トリガー条件) とは "発動する"(終止) で区別する。
# ---------------------------------------------------------------------------
@rule("execute_event", priority=71)
def _execute_event(ctx: ParseContext) -> Optional[GameAction]:
    t = ctx.text
    if not re.search(_nfc(r"発動する"), t):
        return None
    if _nfc("イベント") not in t or _nfc("手札") not in t:
        return None
    tq = parse_target(t)
    tq.zone = Zone.HAND
    if _nfc("まで") in t:
        tq.is_up_to = True
    return GameAction(type=ActionType.EXECUTE_EVENT, target=tq, raw_text=t)


# ---------------------------------------------------------------------------
# ライフ並び替え: 「（自分/相手の）ライフすべてを見て、好きな順番で置く」
#   → ORDER_LIFE（ライフ内を任意順に並べ替え。対象選択を伴う）。従来 OTHER。
#   「ライフの下／デッキへ…好きな順番で置く」等の移動系は別ルールが担当するため、
#   ライフ内に留まる並べ替え（"デッキ" を含まない）に限定する。
# ---------------------------------------------------------------------------
@rule("order_life", priority=77)
def _order_life(ctx: ParseContext) -> Optional[GameAction]:
    t = ctx.text
    if not re.search(_nfc(r"ライフ.*好きな順番で置く"), t):
        return None
    if _nfc("デッキ") in t:  # 「デッキの下に好きな順番で置く」等は移動系（別ルール）
        return None
    status = "OPPONENT" if _nfc("相手のライフ") in t else None
    return GameAction(type=ActionType.ORDER_LIFE, status=status, raw_text=t)


@rule("life_scry_top", priority=73)
def _life_scry_top(ctx: ParseContext) -> Optional[Choice]:
    t = ctx.text
    if _nfc("ライフの上から") not in t or _nfc("見て") not in t:
        return None
    # 戻し先が「ライフの上か下」のもののみ（「ライフすべてを見て…」並び替えは別パターン）。
    if not (_nfc("ライフの上か下") in t or (_nfc("ライフの上") in t and _nfc("下に置く") in t)):
        return None
    both = _nfc("自分か相手") in t
    labels: list = []
    options: list = []
    if both or (_nfc("自分") in t and _nfc("相手") not in t):
        labels.append("自分のライフを見る")
        options.append(_scry_one_life(Player.SELF, "SELF", t))
    if both or (_nfc("相手") in t and _nfc("自分") not in t):
        labels.append("相手のライフを見る")
        options.append(_scry_one_life(Player.OPPONENT, "OPPONENT", t))
    if not options:
        return None
    if _nfc("まで") in t:  # 「1枚まで」= 任意 → 見ない選択肢
        labels.append("見ない")
        options.append(Sequence(actions=[]))
    return Choice(message="どちらのライフを見ますか", option_labels=labels, options=options)


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
    # 「加える」(終止) に加え連用形「加え」(「…ライフの上に加え、その後…」の分割後) も拾う。
    if _nfc("加え") not in t and _nfc("置く") not in t:
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
    「自分のライフN枚を手札に加えることができる」（上か下を明示しない形）も対応。
    """
    t = ctx.text
    has_life_pos = _nfc("ライフの上") in t or _nfc("ライフの下") in t
    has_life_count = bool(re.search(_nfc(r"ライフ\d*枚"), t))
    if not has_life_pos and not has_life_count:
        return None
    # 「加えてもよい」は「加える」を含まないため個別に対応する
    if (_nfc("手札に加える") not in t and _nfc("手札に戻す") not in t
            and _nfc("手札に加えてもよい") not in t):
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
    """「（自分の）手札…を、ライフの上／下に加える」→ MOVE_CARD(dest=LIFE, dest_position=TOP/BOTTOM)。"""
    t = ctx.text
    if _nfc("手札") not in t:
        return None
    if _nfc("ライフの上に加える") not in t and _nfc("ライフの下に加える") not in t:
        return None
    tq = parse_target(t)
    tq.zone = Zone.HAND
    dest_position = "TOP" if _nfc("ライフの上に加える") in t else "BOTTOM"
    return GameAction(
        type=ActionType.MOVE_CARD,
        target=tq,
        destination=Zone.LIFE,
        dest_position=dest_position,
        raw_text=t,
    )


@rule("life_to_trash", priority=68)
def _life_to_trash(ctx: ParseContext) -> Optional[GameAction]:
    """「（自分／相手の）ライフの上（か下）から…トラッシュに置く（もよい）」→ TRASH。"""
    t = ctx.text
    if _nfc("ライフの上") not in t and _nfc("ライフの下") not in t:
        return None
    # 「置く」「置いて」「置いてもよい」等の活用形に対応（トラッシュに置で統一）
    if _nfc("トラッシュ") not in t or not re.search(_nfc(r"トラッシュに置"), t):
        return None
    tq = parse_target(t)
    tq.zone = Zone.LIFE
    if _nfc("まで") in t or _nfc("もよい") in t:
        tq.is_up_to = True
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
    # 「付与されている」が対象フィルタ（キャラ/パワー等を修飾）ならドン付与ではない。
    # 「付与されているドン!!を...に付与する」のようにドン自体を移動するケースは除外しない。
    if re.search(_nfc(r"付与されている"), t) and not re.search(_nfc(r"に付与する"), t):
        return None
    # 付与先（に付与 の前の主語）が「相手の」なら OPPONENT、それ以外は SELF
    # 例: 「相手のキャラ1枚に相手のレストのドン!!1枚を付与する」→ recipient=OPPONENT
    # 例: 「自分のリーダーかキャラ1枚にドン!!を付与する」→ recipient=SELF
    ni_idx = t.find(_nfc("に付与")) if _nfc("に付与") in t else len(t)
    recipient_part = t[:ni_idx]
    player = Player.OPPONENT if re.search(_nfc(r"相手の[^でに]*?(キャラ|リーダー)"), recipient_part) else Player.SELF
    recipient = TargetQuery(player=player, zone=Zone.FIELD, count=1)
    if _nfc("リーダー") in recipient_part:
        recipient.card_type.append("LEADER")
    if _nfc("キャラ") in recipient_part:
        recipient.card_type.append("CHARACTER")
    if not recipient.card_type:
        recipient.card_type.extend(["LEADER", "CHARACTER"])
    return GameAction(
        type=ActionType.ATTACH_DON,
        target=recipient,
        value=ValueSource(base=_don_count(t)),
        status="RESTED" if _nfc("レストのドン") in t or _nfc("コストエリアのドン") in t else None,
        duration=_duration_of(t),
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


# ---------------------------------------------------------------------------
# ドン!!複合コストの前半断片: 「自分のドン‼N枚と」
#   「自分のドン‼N枚とこのキャラ／このリーダーをレストにできる」というコストは、レガシー構造分解で
#   「自分のドン‼N枚と」(断片) と「…をレストにできる」(REST self) に割れる。後半は rest_self_cost が
#   拾うが、前半の断片は従来 OTHER。これを REST_DON（ドン!!を N 枚レスト）として補完する。
#   「枚と」で終わる断片に限定し、効果文への誤爆を避ける（多くはコスト=ctx.is_cost）。
# ---------------------------------------------------------------------------
@rule("don_rest_cost_fragment", priority=76)
def _don_rest_cost_fragment(ctx: ParseContext) -> Optional[GameAction]:
    t = ctx.text.strip()
    m = re.match(_nfc(r"^(?:自分の)?ドン[‼!]*\s*(\d+)枚と$"), t)
    if not m:
        return None
    return GameAction(type=ActionType.REST_DON, value=ValueSource(base=int(m.group(1))), raw_text=ctx.text)


@rule("don_set_rest", priority=74)
def _don_set_rest(ctx: ParseContext) -> Optional[GameAction]:
    """「（自分の）ドン!!N枚をレストにする/できる」→ REST_DON（アクティブ→レスト）。多くはコスト。"""
    t = ctx.text
    if _nfc("ドン") not in t:
        return None
    # Sequence 分割で末尾が連用形「レストにし」になる句（例:「ドン‼1枚をレストにし、…捨てる」）も対象
    if not re.search(_nfc(r"レストに(する|できる|し[、。]|し$)"), t.strip()):
        return None
    if _nfc("アクティブ") in t:
        return None
    # 「ドン!!が付与されているキャラをレストにする」等、ドン!!が修飾語として使われている場合は
    # キャラを対象とする REST に委ねる（ドン!!自体をレストにするわけではない）
    if re.search(_nfc(r"ドン!!.*付与"), t):
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
    if _nfc("ドン") not in t or not re.search(_nfc(r"戻(す|して)"), t):
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
    # 「場を離れない」(終止形) と「場を離れず」(連用中止形、「…場を離れず、【X】を得る」の分割後)
    # の両方を保護マーカーとして拾う。
    if _nfc("場を離れない") in t or _nfc("場を離れず") in t:
        status = "LEAVE"
    elif _nfc("KOされない") in t:
        status = "BATTLE_KO"
    else:
        return None
    return GameAction(
        type=ActionType.PREVENT_LEAVE,
        target=TargetQuery(select_mode="SOURCE"),
        status=status,
        duration=_duration_of(t),
        raw_text=t,
    )


# ---------------------------------------------------------------------------
# 複合除去保護: 「（このキャラは相手の効果で、）KOされず（、）レストにされない」
#   除去保護(PREVENT_LEAVE) + レスト不可(PREVENT_REST) の複合（従来 OTHER）。
#   「相手の効果で」KO されない場合は除去（場を離れる）保護＝status="LEAVE"。
#   prevent_ko_and_rest を prevent_leave/rest_restrict より高優先で先に拾う。
# ---------------------------------------------------------------------------
@rule("prevent_ko_and_rest", priority=67)
def _prevent_ko_and_rest(ctx: ParseContext) -> Optional[EffectNode]:
    t = ctx.text
    if not (re.search(_nfc(r"KOされ(?:ず|ない)"), t) and _nfc("レストにされない") in t):
        return None
    # 「相手の効果で」KO されない＝効果除去保護（LEAVE）。明示が無ければバトルKO保護。
    ko_status = "LEAVE" if (_nfc("相手の効果") in t or _nfc("効果で") in t) else "BATTLE_KO"
    prevent_ko = GameAction(
        type=ActionType.PREVENT_LEAVE,
        target=TargetQuery(select_mode="SOURCE"),
        status=ko_status,
        duration=_duration_of(t),
        raw_text=t,
    )
    prevent_rest = GameAction(
        type=ActionType.PREVENT_REST,
        target=TargetQuery(select_mode="SOURCE"),
        duration=_duration_of(t),
        raw_text=t,
    )
    return Sequence(actions=[prevent_ko, prevent_rest])


# ---------------------------------------------------------------------------
# レスト不可保護: 「このキャラは相手の効果でレストにされない」（自身の静的保護, 従来 OTHER）。
#   rest_restrict(「相手の…キャラはレストにできない」=相手キャラへの制限) とは別物で、
#   こちらは自身(SOURCE)が相手効果でレストされないようにする保護。
# ---------------------------------------------------------------------------
@rule("prevent_rest_self", priority=66)
def _prevent_rest_self(ctx: ParseContext) -> Optional[GameAction]:
    t = ctx.text
    if _nfc("レストにされない") not in t:
        return None
    if not re.search(_nfc(r"この(カード|キャラ|リーダー)"), t):
        return None
    return GameAction(
        type=ActionType.PREVENT_REST,
        target=TargetQuery(select_mode="SOURCE"),
        duration=_duration_of(t),
        raw_text=t,
    )


# ---------------------------------------------------------------------------
# 効果ダメージ: 「相手に N ダメージを与える」「自分は N ダメージを受ける」
#   → DEAL_DAMAGE。エンジンは対象リーダーのライフ上 N 枚を手札へ移し、
#   ライフが尽きれば勝利（gamestate の DEAL_DAMAGE 実装）。従来 OTHER（不発）。
#   「与えてもよい」は登場/サーチ系の任意効果と同様、選択 UI 未実装のため実行扱い。
# ---------------------------------------------------------------------------
@rule("deal_damage", priority=55)
def _deal_damage(ctx: ParseContext) -> Optional[GameAction]:
    t = ctx.text
    m = re.search(_nfc(r"(\d+)\s*ダメージを(与え|受け)"), t)
    if not m:
        return None
    n = int(m.group(1))
    # 「自分は／自分に…受ける／与える」は自分が被ダメージ、それ以外は相手。
    is_self = m.group(2) == _nfc("受け") or _nfc("自分は") in t or _nfc("自分に") in t
    player = Player.SELF if is_self else Player.OPPONENT
    return GameAction(
        type=ActionType.DEAL_DAMAGE,
        target=TargetQuery(player=player),
        value=ValueSource(base=n),
        raw_text=t,
    )


# ---------------------------------------------------------------------------
# 相手デッキの覗き見: 「相手のデッキの上から N 枚を見る」（後続消費なしの純粋な公開）
#   → LOOK + status="OPPONENT"。盤面は不変（並びも変えない）。look_deck(自分・TEMP移動)
#   とは別経路。look_deck は「見て／公開し」(連用) を拾うため「見る」(終止) とは衝突しない。
# ---------------------------------------------------------------------------
@rule("look_opp_deck", priority=81)
def _look_opp_deck(ctx: ParseContext) -> Optional[GameAction]:
    t = ctx.text
    m = re.search(_nfc(r"相手のデッキの上から(\d+)枚(?:まで)?を見る"), t)
    if not m:
        return None
    return GameAction(
        type=ActionType.LOOK,
        status="OPPONENT",
        value=ValueSource(base=int(m.group(1))),
        raw_text=t,
    )


# ---------------------------------------------------------------------------
# 自分デッキトップの公開: 「（自分の）デッキの上からN枚を公開する」（終止形・後続条件用）
#   → LOOK（自分・TEMP へ移して公開）。後続の「公開したカードが…の場合」が
#   REVEALED_CARD_TRAIT 条件で temp[0] を参照し、未消費 temp は解決完了時にデッキトップへ戻る。
#   look_deck(「見て/公開し」連用) とは語尾で区別（「公開する」終止形を拾う）。
# ---------------------------------------------------------------------------
@rule("reveal_deck_top", priority=79)
def _reveal_deck_top(ctx: ParseContext) -> Optional[GameAction]:
    t = ctx.text
    if _nfc("相手のデッキ") in t:
        return None  # 相手デッキは look_opp_deck が担当
    m = re.search(_nfc(r"デッキの上から(\d+)枚(?:まで)?を公開する"), t)
    if not m:
        return None
    return GameAction(type=ActionType.LOOK, value=ValueSource(base=int(m.group(1))), raw_text=t)


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
# ---------------------------------------------------------------------------
# コスト絶対値セット: 「（対象）を、（このターン中、）コスト0にする」
#   → BUFF + status="COST_OVERRIDE"。エンジンは base_cost_override をセットし、
#   reset_turn_status() で失効する（set_power の COST 版）。
#   「コスト-N」等の増減は cost_change(priority=58) が担当（こちらは「Nにする」限定）。
# ---------------------------------------------------------------------------
@rule("set_cost", priority=60)
def _set_cost(ctx: ParseContext) -> Optional[GameAction]:
    t = ctx.text
    m = re.search(_nfc(r"コスト(?:を)?(\d+)に(?:なる|する)"), t)
    if not m:
        return None
    tq = parse_target(t)
    if _nfc("まで") in t:
        tq.is_up_to = True
    return GameAction(
        type=ActionType.BUFF,
        status="COST_OVERRIDE",
        target=tq,
        value=ValueSource(base=int(m.group(1))),
        duration=_duration_of(t),
        raw_text=t,
    )


@rule("cost_change", priority=58)
def _cost_change(ctx: ParseContext) -> Optional[GameAction]:
    t = ctx.text
    m = _SIGN_RE.search(t)
    if m:
        sign = -1 if m.group(1) in "-－−‐" else 1
        value = sign * int(m.group(2))
    else:
        # 「支払うコストはN少なくなる」パターン（コスト軽減）
        m2 = re.search(_nfc(r"コストは(\d+)少なくなる"), t)
        if not m2:
            return None
        value = -int(m2.group(1))
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
# 「ドン !!-1」のように ドン と !! の間にスペースが入る表記も許容する
_DON_RETURN_RE = re.compile(r"ドン[ 　]*(?:!!|‼)[ 　]*[-－−‐][ 　]*(\d+)")


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
@rule("look_deck", priority=80)
def _look_deck(ctx: ParseContext) -> Optional[GameAction]:
    """「（自分の）デッキの上からN枚（まで）を見て／公開し」→ LOOK（デッキ上 N 枚→TEMP）。

    parser.py が「デッキの上からN枚を見て、」「…を公開し、」を独立クローズに分割するため、
    LOOK を明示生成する。後続の「公開し手札に加える」(search_to_hand)・「登場させる」
    (play_from_temp)・「並び替えてデッキへ戻す」(temp_to_deck)・「残りを…」(remaining_*) が
    TEMP を消費する。「公開し」は本来 REVEAL（相手に開示）だが、登場/サーチ処理のため候補を
    TEMP に載せる点は「見て」と同じなので LOOK で扱う。
    """
    t = ctx.text
    m = re.search(_nfc(r"デッキの上から(\d+)枚(?:まで)?を(?:見て|公開し)"), t)
    n = int(m.group(1)) if m else None
    if n is None and re.search(_nfc(r"デッキの一番上を(?:見て|公開し)"), t):
        n = 1  # 「デッキの一番上を公開し」= 上から1枚
    if n is None:
        return None
    return GameAction(
        type=ActionType.LOOK,
        value=ValueSource(base=n),
        raw_text=t,
    )


@rule("search_to_hand", priority=54)
def _search_to_hand(ctx: ParseContext) -> Optional[GameAction]:
    """「（公開し、）（コスト/特徴/名前で絞った）カードM枚までを手札に加える」（サーチの取得）
    → MOVE_CARD(zone=TEMP, dest=HAND)。

    直前の look_deck が候補を TEMP に置いている前提で、TEMP からフィルタ一致を手札へ。
    明示的な別ソース（トラッシュ/ライフ/手札から/デッキ）がある句は対象外（既存ルールが担当）。
    """
    t = ctx.text
    # 「手札に加える」（辞書形）と「手札に加え」（連用形、split後の末尾）の両方を対象とする
    # 「手札に加えられない」（禁止節）は除外
    if not re.search(_nfc(r"手札に加え(?!られ)"), t):
        return None
    # 明示的なソースゾーンがある句は別ルール（life_to_hand 等）に委ねる。
    if any(_nfc(z) in t for z in ["トラッシュ", "ライフ", "手札から", "デッキ"]):
        return None
    tq = parse_target(t)
    tq.zone = Zone.TEMP
    tq.player = Player.SELF
    if _nfc("まで") in t:
        tq.is_up_to = True
    return GameAction(
        type=ActionType.MOVE_CARD,
        target=tq,
        destination=Zone.HAND,
        raw_text=t,
    )


@rule("search_deck_to_hand", priority=67)
def _search_deck_to_hand(ctx: ParseContext) -> Optional[GameAction]:
    """「自分のデッキから（条件）のカードN枚までを公開し、手札に加える/加え」→ MOVE_CARD(zone=DECK, dest=HAND)。

    デッキを直接検索して手札に加えるサーチ効果（LOOK 文脈ではなくデッキ直接参照）。
    """
    t = ctx.text
    if _nfc("デッキから") not in t:
        return None
    if not re.search(_nfc(r"手札に加え(?!られ)"), t):
        return None
    tq = parse_target(t)
    tq.zone = Zone.DECK
    tq.player = Player.SELF
    if _nfc("まで") in t:
        tq.is_up_to = True
    return GameAction(
        type=ActionType.MOVE_CARD,
        target=tq,
        destination=Zone.HAND,
        raw_text=t,
    )


@rule("trash_to_hand", priority=62)
def _trash_to_hand(ctx: ParseContext) -> Optional[GameAction]:
    """「自分のトラッシュの（種別/名前/色/コスト条件の）カード/キャラ/イベントN枚までを手札に加える」
    → MOVE_CARD(zone=TRASH, dest=HAND)。

    search_to_hand(54) より高優先で、トラッシュ明示のパターンを先取りする。
    """
    t = ctx.text
    if _nfc("手札に加える") not in t and _nfc("手札に加えてもよい") not in t:
        return None
    if _nfc("トラッシュ") not in t:
        return None
    tq = parse_target(t)
    tq.zone = Zone.TRASH
    tq.player = Player.SELF
    if _nfc("まで") in t:
        tq.is_up_to = True
    return GameAction(
        type=ActionType.MOVE_CARD,
        target=tq,
        destination=Zone.HAND,
        raw_text=t,
    )


@rule("temp_to_deck", priority=63)
def _temp_to_deck(ctx: ParseContext) -> Optional[GameAction]:
    """「（好きな順番に並び替え、）デッキの上か下に置く」「好きな順番で置く」（scry の戻し）
    → DECK_BOTTOM（TEMP 全件→デッキ下, 保守的）。

    look_deck の後、手札に取らなかった残りをデッキへ戻す。「残り」を含む句は
    remaining_* が担当するため除外。上下/並び替えの選択 UI は未実装のため下へ戻す。

    明示ソース（トラッシュ/ライフ）またはフィールドのキャラ対象（コスト/枚数フィルタ付き）は
    deck_bottom_general(priority=55) が担当するため除外。
    """
    t = ctx.text
    if _nfc("残り") in t or _nfc("手札") in t:
        return None  # remaining_* / search_to_hand が担当
    # 明示ソースゾーンがある場合は deck_bottom_general に委ねる
    if _nfc("トラッシュ") in t or _nfc("ライフ") in t:
        return None
    # キャラ対象（コスト/枚数フィルタ付き、すべて、以外）はフィールドターゲット → deck_bottom_general
    if re.search(_nfc(r"コスト\d+以[下上]|キャラ\d+枚|キャラすべて|キャラ以外"), t):
        return None
    has_arrange = _nfc("並び替え") in t or _nfc("並び変え") in t or _nfc("好きな順番") in t
    to_deck = _nfc("デッキの上か下") in t or _nfc("デッキの下") in t or _nfc("デッキの上") in t
    if not (has_arrange and (_nfc("置く") in t or _nfc("戻す") in t)):
        return None
    if not to_deck and _nfc("置く") not in t:
        return None
    return GameAction(
        type=ActionType.DECK_BOTTOM,
        target=TargetQuery(
            player=Player.SELF, zone=Zone.TEMP, select_mode="REMAINING", count=-1
        ),
        raw_text=t,
    )


@rule("reveal_hand", priority=59)
def _reveal_hand(ctx: ParseContext) -> Optional[GameAction]:
    """「自分の手札から（コスト/パワー/特徴で絞った）カードN枚を公開する/できる/することができる」
    → REVEAL（情報開示, zone=HAND）。

    公開は盤面を動かさず、指定枚数の手札を公開する（多くは条件成立の証明）。
    「公開し、手札に加える」（デッキを見てのサーチ）は別物（手札に加える/デッキを含む）なので除外。
    従来は OTHER（公開できない no-op）、または「パワーN…公開」が誤って BUFF に落ちていた。
    """
    t = ctx.text
    if _nfc("公開") not in t or _nfc("手札") not in t:
        return None
    # サーチ（デッキを見て公開し手札に加える/加え）系・手札に戻す系は対象外。
    if re.search(_nfc(r"手札に加え(?!られ)"), t) or _nfc("手札に戻す") in t:
        return None
    if _nfc("デッキの上") in t or _nfc("デッキの下") in t:
        return None
    tq = parse_target(t)
    tq.zone = Zone.HAND
    # 「できる」「ことができる」「まで」は任意（対象不在でも no-op 成功）。
    if _nfc("できる") in t or _nfc("まで") in t:
        tq.is_up_to = True
    return GameAction(type=ActionType.REVEAL, target=tq, raw_text=t)


@rule("active_target", priority=51)
def _active_target(ctx: ParseContext) -> Optional[GameAction]:
    """「（自分の/相手の）キャラ/リーダー1枚（まで）を、アクティブにする/できる」→ ACTIVE。

    active_self（「このキャラを…」priority=75）・don_set_active（「ドン」priority=74）
    が先に処理されるため、本ルールはそれ以外の対象指定アクティブを担う。
    「自分のキャラ1枚までを、アクティブにする」等が典型。
    """
    t = ctx.text
    if _nfc("ドン") in t:
        return None  # don_set_active が担当
    if not re.search(_nfc(r"アクティブに(する|できる)"), t):
        return None
    if re.search(_nfc(r"この(カード|キャラ|リーダー)を"), t):
        return None  # active_self が担当
    tq = parse_target(t)
    if _nfc("まで") in t:
        tq.is_up_to = True
    return GameAction(type=ActionType.ACTIVE, target=tq, raw_text=t)


@rule("blocker_disable", priority=61)
def _blocker_disable(ctx: ParseContext) -> Optional[GameAction]:
    """「（相手は、）（このバトル中、）【ブロッカー】を発動できない」
    → BUFF(status=BLOCKER_DISABLE, target=相手フィールド全体)。

    エンジンの BLOCKER_DISABLE ブランチが対象の flags に "BLOCKER_DISABLED" を立て、
    has_blocker() がブロック不可と判断する。flags はターン終了時にリセットされる。
    """
    t = ctx.text
    if _nfc("ブロッカー") not in t or _nfc("発動できない") not in t:
        return None
    tq = TargetQuery(
        player=Player.OPPONENT, zone=Zone.FIELD, count=-1, select_mode="ALL"
    )
    return GameAction(
        type=ActionType.BUFF,
        target=tq,
        status="BLOCKER_DISABLE",
        duration="THIS_BATTLE" if _nfc("このバトル中") in t else "THIS_TURN",
        raw_text=t,
    )


@rule("rush_natural", priority=61)
def _rush_natural(ctx: ParseContext) -> Optional[GameAction]:
    """「（このキャラは）登場したターンにキャラへアタックできる」
    → GRANT_KEYWORD("速攻", PERMANENT)。

    【速攻】タグを持たない自然言語表現からキーワード付与を生成する。
    登場したターン限定でなく PERMANENT（場を離れるまで）とする
    （= 常に速攻を持つ）のが実際の効果に近い。
    """
    t = ctx.text
    if not re.search(_nfc(r"登場した(ターン|時)に.*アタックできる"), t):
        return None
    return GameAction(
        type=ActionType.GRANT_KEYWORD,
        target=TargetQuery(select_mode="SOURCE"),
        status="速攻",
        duration="PERMANENT",
        raw_text=t,
    )


@rule("select_target", priority=58)
def _select_target(ctx: ParseContext) -> Optional[GameAction]:
    """「（対象）を選ぶ」（終止形・単独の選択句）→ SELECT（対象を選択して保存）。

    「…を選ぶ。選んだキャラは…」のように選択と効果が別文に分割されたケースで、
    選択句が動詞なしの OTHER に落ちていたのを是正する。選択結果は
    target.save_id="selected_card" に保存され、後続句の「選んだ／その
    （カード/キャラ/リーダー）」が ref_id で参照する（matcher.parse_target が付与）。

    除外: 「以下から（1つを）選ぶ」（Choice）、連用形「選び、」（legacy が同一句で
    save_id を付与する連結形）。
    """
    t = ctx.text
    if _nfc("を選ぶ") not in t:
        return None
    if _nfc("以下から") in t or _nfc("効果を選択") in t:
        return None
    tq = parse_target(t)
    tq.save_id = "selected_card"
    if _nfc("まで") in t:
        tq.is_up_to = True
    return GameAction(type=ActionType.SELECT, target=tq, raw_text=t)


@rule("bounce", priority=56)
def _bounce(ctx: ParseContext) -> Optional[GameAction]:
    """「（コストN以下の）（特徴X の）キャラ1枚（まで）を（持ち主の）手札に戻す（ことができる）」
    → BOUNCE（フィールド→手札）。

    OPTCG では「持ち主の手札に戻す」はほぼ相手カードを対象とするため、
    「自分の」が明示されていなければ OPPONENT をデフォルトとする。
    「手札から…手札に戻す」等の二段指示は除外（手札 source 文脈）。
    """
    t = ctx.text
    # 「手札に戻す」に加え、Sequence 分割で末尾が連用形「手札に戻し」になる句も対象
    # （例:「相手の…を持ち主の手札に戻し、このカードを手札に加える」→ 前段「…手札に戻し」）。
    if not re.search(_nfc(r"手札に戻す(ことができる)?"), t) \
            and not re.search(_nfc(r"手札に戻し(?:[、。]|てもよい|$)"), t.strip()):
        return None
    if _nfc("手札から") in t:
        return None  # 「手札から何かして手札に戻す」等の誤検知を避ける
    tq = parse_target(t)
    # 「自分の」明示がなければ OPPONENT（「持ち主の手札」→相手カードが多数派）。
    if tq.player != Player.OPPONENT and _nfc("自分の") not in t:
        tq.player = Player.OPPONENT
    if _nfc("まで") in t:
        tq.is_up_to = True
    return GameAction(type=ActionType.BOUNCE, target=tq, raw_text=t)


@rule("deck_bottom_general", priority=55)
def _deck_bottom_general(ctx: ParseContext) -> Optional[GameAction]:
    """「（対象）を（持ち主の/好きな順番で）デッキの下に置く/戻す」→ DECK_BOTTOM。

    remaining_deck_bottom（「残り」→TEMP→DECK）は priority=65 で先に処理される。
    temp_to_deck（scry 戻し、priority=63）がスキップした明示ソース付きパターンも担当。
    「持ち主のデッキの下」でプレイヤー未指定なら OPPONENT（相手キャラ対象が多い）。
    「戻す」も「置く」と同義として受け付ける。
    """
    t = ctx.text
    if _nfc("デッキの下") not in t:
        return None
    if _nfc("置く") not in t and _nfc("戻す") not in t and _nfc("置き") not in t:
        return None
    if _nfc("残り") in t:
        return None  # remaining_deck_bottom / remaining_deck_top_or_bottom が担当
    tq = parse_target(t)
    # 「持ち主のデッキの下」でプレイヤーがデフォルト(SELF)なら OPPONENT に補正。
    if _nfc("持ち主") in t and tq.player != Player.OPPONENT:
        tq.player = Player.OPPONENT
    if _nfc("まで") in t:
        tq.is_up_to = True
    return GameAction(type=ActionType.DECK_BOTTOM, target=tq, raw_text=t)


@rule("scry_place", priority=64)
def _scry_place(ctx: ParseContext) -> Optional[GameAction]:
    """「（そのカードを）デッキの上か下に置く」（公開後の単純配置）
    → DECK_BOTTOM(TEMP)。LOOK 直後に「上か下」を選んで置くパターン。

    「好きな順番」付きは temp_to_deck が担当。「残り」付きは
    remaining_deck_top_or_bottom が担当。選択 UI 未実装のため下に保守的フォールバック。
    """
    t = ctx.text
    if not re.search(_nfc(r"デッキの上か下に置く"), t):
        return None
    if _nfc("残り") in t or _nfc("好きな順番") in t or _nfc("並び替え") in t:
        return None  # 既存ルールへ委ねる
    if _nfc("手札") in t:
        return None  # hand_to_deck が担当
    return GameAction(
        type=ActionType.DECK_BOTTOM,
        target=TargetQuery(player=Player.SELF, zone=Zone.TEMP),
        raw_text=t,
    )


@rule("remaining_deck_top_or_bottom", priority=63)
def _remaining_deck_top_or_bottom(ctx: ParseContext) -> Optional[GameAction]:
    """「残りを（好きな順番に並び替え、）?デッキの上か下に置く」→ DECK_BOTTOM（保守的）。

    「上か下」を選ぶ UI は未実装のため、保守的にデッキ下扱い。
    「残りをデッキの下に置く」は remaining_deck_bottom(priority=65) が優先処理する。
    """
    t = ctx.text
    if _nfc("残り") not in t or _nfc("デッキ") not in t:
        return None
    if not re.search(_nfc(r"上か下|上か、下"), t):
        return None  # 「上か下」の択がある場合のみ
    return GameAction(
        type=ActionType.DECK_BOTTOM,
        target=TargetQuery(
            player=Player.SELF, zone=Zone.TEMP, select_mode="REMAINING", count=-1
        ),
        raw_text=t,
    )


@rule("play_card_from_zone", priority=52)
def _play_card_from_zone(ctx: ParseContext) -> Optional[GameAction]:
    """「（自分の）手札/トラッシュからコストN以下の...カード1枚（まで）を（レストで）登場させる」
    → PLAY_CARD（手札/トラッシュ→フィールド）。

    play_self（このカード/キャラ自身を登場させる）とは「このカード/キャラ」の有無で区別。
    「登場させてもよい」（任意）は選択 UI 未実装のため登場させる扱いにする。
    レスト登場（レストで登場させる）は status="RESTED" をエンジンに伝える。
    """
    t = ctx.text
    # 「登場させる/させてもよい/させることができる」に加え、短縮形「登場できる」も拾う
    # （例: OP05-111「手札から「コトリ」1枚を、登場できる」）。
    if not re.search(_nfc(r"登場(させ(る|てもよい|ることができる)|できる)"), t):
        return None
    if re.search(_nfc(r"この(カード|キャラ|リーダー)を"), t):
        return None  # play_self が担当
    has_hand = _nfc("手札") in t
    has_trash = _nfc("トラッシュ") in t
    if not has_hand and not has_trash:
        return None  # 手札/トラッシュ以外からの登場（プレイ自体）は対象外
    tq = parse_target(t)
    # parse_target は「手札から」「トラッシュから」を zone に反映するが、
    # フィールドキャラ系（「場のキャラを」）と混在する場合に備えて上書き。
    if has_trash:
        tq.zone = Zone.TRASH
    elif has_hand:
        tq.zone = Zone.HAND
    if _nfc("まで") in t:
        tq.is_up_to = True
    status = "RESTED" if re.search(_nfc(r"レストで(、)?登場"), t) else None
    return GameAction(
        type=ActionType.PLAY_CARD,
        target=tq,
        destination=Zone.FIELD,
        status=status,
        raw_text=t,
    )


# ---------------------------------------------------------------------------
# デッキから直接登場（サーチ→登場）:
#   「自分のデッキから（コスト/名前/特徴で絞った）キャラ1枚までを、（レストで）登場させる」
#   → PLAY_CARD(zone=DECK, dest=FIELD)。後続の「デッキをシャッフルする」は shuffle が担当。
#   「デッキの上から…公開/見て」(LOOK 文脈) とは「デッキから」(検索) で区別。
#   ドン!!/手札/トラッシュ/ライフ明示・「このキャラを」(play_self) は対象外。
# ---------------------------------------------------------------------------
@rule("play_from_deck", priority=53)
def _play_from_deck(ctx: ParseContext) -> Optional[GameAction]:
    t = ctx.text
    if _nfc("デッキから") not in t:
        return None
    # 「登場させる」断定・連用形「登場させ、」(split 後は末尾「登場」) の両方を拾う。
    if not re.search(_nfc(r"を、?(?:レストで)?登場(?:させ(?:る)?)?$"), t):
        return None
    if re.search(_nfc(r"この(カード|キャラ|リーダー)を"), t):
        return None  # play_self が担当
    if any(_nfc(z) in t for z in ["ドン", "手札", "トラッシュ", "ライフ"]):
        return None  # ドン操作・手札/トラッシュ/ライフからの登場は別ルール
    tq = parse_target(t)
    tq.zone = Zone.DECK
    tq.player = Player.SELF
    if _nfc("まで") in t:
        tq.is_up_to = True
    status = "RESTED" if re.search(_nfc(r"レストで(、)?登場"), t) else None
    return GameAction(
        type=ActionType.PLAY_CARD,
        target=tq,
        destination=Zone.FIELD,
        status=status,
        raw_text=t,
    )


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


# ---------------------------------------------------------------------------
# 自己トラッシュ: 「このキャラ／このカード／このリーダーをトラッシュに置く（ことができる）」
#   多くはコスト（このキャラをトラッシュして…）。KO ではなく単なる移動なので
#   ON_KO は誘発しない。対象は自身(SOURCE)。従来は OTHER に落ちる最頻出表現（49 件）。
#   「このキャラ以外の…をトラッシュ」を巻き込まないよう、直後が「を(、)?トラッシュ」の
#   ものに限定する（残り/デッキの上からの mill は別ルールが担当）。
# ---------------------------------------------------------------------------
@rule("trash_self", priority=67)
def _trash_self(ctx: ParseContext) -> Optional[GameAction]:
    t = ctx.text
    if _nfc("トラッシュ") not in t:
        return None
    # 「置く」なしの短縮形「このキャラをトラッシュに」「代わりにこのキャラをトラッシュに」も対応
    if not re.search(_nfc(r"この(カード|キャラ|リーダー)を、?トラッシュ"), t):
        return None
    return GameAction(
        type=ActionType.TRASH,
        target=TargetQuery(select_mode="SOURCE"),
        raw_text=t,
    )


# ---------------------------------------------------------------------------
# 選択型トラッシュ: 「（自分/相手の）（コストN以下／特徴X の）キャラ1枚（まで）を
#   トラッシュに置く（ことができる）」→ TRASH（選択したフィールドのキャラ→トラッシュ）。
#   trash_self(priority=67) が「このキャラ／このカード／このリーダー」主語の自己トラッシュを
#   先に担当するため、ここは *選択型*（自分/相手のキャラを選んでトラッシュ）を拾う。
#   残り(remaining_trash)・デッキ(mill_deck)・手札・ライフ等の別ソース文脈は除外。
#   エンジンの TRASH は選択ターゲットを既にトラッシュへ移動する（gamestate）。
# ---------------------------------------------------------------------------
@rule("trash_target", priority=57)
def _trash_target(ctx: ParseContext) -> Optional[GameAction]:
    t = ctx.text
    # 「トラッシュに置く」のほか、構造分割で動詞が落ちた「…をトラッシュに」(節末)も拾う。
    # 例: 五老星(OP13-082)「自分のキャラすべてをトラッシュに置き、…」は「置き、」で分割され
    # 「自分のキャラすべてをトラッシュに」となり、従来は OTHER に落ちて全体KOが不発だった。
    if not re.search(_nfc(r"トラッシュに置|トラッシュに$"), t):
        return None
    # 自己トラッシュ（このキャラ／このカード／このリーダーを、?トラッシュ）は trash_self が担当
    if re.search(_nfc(r"この(カード|キャラ|リーダー)を、?トラッシュ"), t):
        return None
    # 別ソース文脈は各専用ルールへ委ねる（残り/デッキ/手札/ライフ）
    if _nfc("残り") in t or _nfc("デッキ") in t or _nfc("手札") in t or _nfc("ライフ") in t:
        return None
    tq = parse_target(t)
    if _nfc("まで") in t:
        tq.is_up_to = True
    return GameAction(type=ActionType.TRASH, target=tq, raw_text=t)


# ---------------------------------------------------------------------------
# 自己手札回収: 「このカード／このキャラカードを手札に加える（ことができる）」
#   KO時/トリガー等で自カードを手札に戻す効果。SOURCE モードで自身を参照。
#   search_to_hand(priority=54) が zone=TEMP に誤設定するより先に処理する（priority=75）。
#   明示ソースゾーン（トラッシュから）がある場合は対象外（別途対応）。
# ---------------------------------------------------------------------------
@rule("self_to_hand", priority=75)
def _self_to_hand(ctx: ParseContext) -> Optional[GameAction]:
    t = ctx.text
    if not re.search(_nfc(r"この(カード|キャラカード)を"), t):
        return None
    if _nfc("手札に加える") not in t and _nfc("手札に加えてもよい") not in t:
        return None
    # 明示ソース（「トラッシュから」「ライフから」）は別ルールに委ねる
    if _nfc("トラッシュ") in t or _nfc("ライフ") in t:
        return None
    return GameAction(
        type=ActionType.MOVE_CARD,
        target=TargetQuery(select_mode="SOURCE"),
        destination=Zone.HAND,
        raw_text=t,
    )


# ---------------------------------------------------------------------------
# 自己アクティブ: 「このキャラ／このカード／このリーダーをアクティブにする/できる」
#   対象は自身(SOURCE)。ドン!!のアクティブ化（don_set_active）とは「ドン」の有無で区別。
#   従来は OTHER に落ちていた（27 件）。
# ---------------------------------------------------------------------------
@rule("active_self", priority=75)
def _active_self(ctx: ParseContext) -> Optional[GameAction]:
    t = ctx.text
    if _nfc("ドン") in t:
        return None  # ドン!!のアクティブ化は don_set_active が担当
    if not re.search(_nfc(r"この(カード|キャラ|リーダー)を、?アクティブに(する|できる)"), t):
        return None
    return GameAction(
        type=ActionType.ACTIVE,
        target=TargetQuery(select_mode="SOURCE"),
        raw_text=t,
    )


# ---------------------------------------------------------------------------
# デッキ上をトラッシュへ（mill）: 「（自分／相手の）デッキの上からN枚をトラッシュに置く」
#   デッキは並びが意味を持つため対象選択させず、枚数(value)ベースでデッキ上から
#   N 枚をトラッシュへ送る。「相手は…」は status="OPPONENT"。従来 OTHER（11 件）。
#   デッキ→ライフ(life_recover)は ライフ を含むため上位ルールが先に拾う。
# ---------------------------------------------------------------------------
@rule("mill_deck", priority=66)
def _mill_deck(ctx: ParseContext) -> Optional[GameAction]:
    t = ctx.text
    if _nfc("デッキの上") not in t:
        return None
    if _nfc("トラッシュ") not in t:
        return None
    # 「置く」「置き（連用形）」「置いて」「置いてもよい」など活用形に対応。
    # 「デッキの上から2枚をトラッシュに置き、シャッフルする」等の連鎖文も拾う。
    if not re.search(_nfc(r"トラッシュに置|トラッシュに$"), t):
        return None
    return GameAction(
        type=ActionType.TRASH_FROM_DECK,
        target=None,
        value=ValueSource(base=_first_int(t, 1)),
        status="OPPONENT" if (_nfc("相手") in t and _nfc("自分") not in t) else None,
        raw_text=t,
    )


# ---------------------------------------------------------------------------
# 残りをトラッシュへ: 「残りを（好きな順番で）?トラッシュに置く」
#   「見る／公開する」等で TEMP に出した残余をトラッシュへ送る。remaining_deck_bottom
#   （残り→デッキの下）のトラッシュ版。従来 OTHER（18 件）。
# ---------------------------------------------------------------------------
@rule("remaining_trash", priority=64)
def _remaining_trash(ctx: ParseContext) -> Optional[GameAction]:
    t = ctx.text
    if _nfc("残り") not in t:
        return None
    # 「置く」(終止) と Sequence 分割後の連用形「置き」(例:「残りをトラッシュに置き、…捨てる」) を拾う。
    if _nfc("トラッシュ") not in t or not re.search(_nfc(r"置(く|き)"), t):
        return None
    return GameAction(
        type=ActionType.TRASH,
        target=TargetQuery(
            player=Player.SELF, zone=Zone.TEMP, select_mode="REMAINING", count=-1
        ),
        raw_text=t,
    )


# ---------------------------------------------------------------------------
# 手札→デッキ上か下:
#   「自分の手札N枚を（好きな順番で並び替え、）デッキの上か下（/上/下）に置く」
#   → DECK_BOTTOM(zone=HAND)。
#   「並び替え」は UI 未実装のため無視（順序不定でデッキ下）。
#   「上か下」の選択 UI も未実装のため保守的にデッキ下扱い。
# ---------------------------------------------------------------------------
@rule("hand_to_deck", priority=64)
def _hand_to_deck(ctx: ParseContext) -> Optional[GameAction]:
    t = ctx.text
    if _nfc("手札") not in t:
        return None
    # 「デッキの上/下に置く」に加え、「（手札すべてを）デッキに戻す/戻し」(シャッフル前提) も拾う。
    if not re.search(_nfc(r"デッキの(上か下|上|下)に置(く|い)"), t) and not re.search(_nfc(r"デッキに戻"), t):
        return None
    # 「ライフ」「トラッシュ」「ドン」を含む場合は別ルールへ委ねる
    if _nfc("ライフ") in t or _nfc("トラッシュ") in t or _nfc("ドン") in t:
        return None
    tq = parse_target(t)
    tq.zone = Zone.HAND
    if _nfc("まで") in t:
        tq.is_up_to = True
    return GameAction(
        type=ActionType.DECK_BOTTOM,
        target=tq,
        raw_text=t,
    )


# ---------------------------------------------------------------------------
# 公開カードをそのまま登場させる:
#   「（レストで）登場させてもよい」— 明示ゾーン指定なし（デッキ公開→条件付き登場の文脈）。
#   play_card_from_zone(priority=52) が「手札/トラッシュ」明示の場合を先に担当するため、
#   ここは明示ゾーンなし・かつ「このカード/キャラ」指定なしの残余ケースを拾う。
#   UI 未実装のため任意(もよい)も即登場扱い。is_up_to=True で登場しない選択も可。
# ---------------------------------------------------------------------------
@rule("play_revealed", priority=40)
def _play_revealed(ctx: ParseContext) -> Optional[GameAction]:
    t = ctx.text
    if not re.search(_nfc(r"登場させてもよい"), t):
        return None
    if re.search(_nfc(r"この(カード|キャラ|リーダー)を"), t):
        return None  # play_self が担当
    if _nfc("手札") in t or _nfc("トラッシュ") in t:
        return None  # play_card_from_zone が担当（ゾーン明示）
    status = "RESTED" if re.search(_nfc(r"レストで(、)?登場"), t) else None
    # 「（デッキ/ライフの一番上を）公開し、そのカードが…の場合、登場させてもよい」の登場句。
    # 公開で候補が TEMP に載っている前提で、TEMP の1枚（=公開カード）を登場させる。
    # 条件側（REVEALED_CARD_TRAIT）がフィルタを担うため、ここはフィルタ無し1枚で足りる。
    tq = TargetQuery(player=Player.SELF, zone=Zone.TEMP, count=1, is_up_to=True)
    return GameAction(
        type=ActionType.PLAY_CARD,
        target=tq,
        destination=Zone.FIELD,
        status=status,
        raw_text=t,
    )


# ---------------------------------------------------------------------------
# 公開したデッキトップを登場させる（デッキ公開→条件付き登場）:
#   「（コスト/特徴/名前で絞った）キャラ（カード）1枚までを、（レストで）登場させる」
#   → PLAY_CARD(zone=TEMP, dest=FIELD)。
#   parser.py が「デッキの上からN枚を公開し、」を独立クローズに分割し、look_deck が候補を
#   TEMP に載せた後、本ルールが TEMP からフィルタ一致の1枚を登場させる。登場しなかった残りは
#   remaining_*（残り→デッキ）が戻す。明示ゾーン（手札/トラッシュ/ライフ）句や「このキャラを」
#   （play_self）、条件/トリガー文（「登場させた場合/時」）は対象外。
# ---------------------------------------------------------------------------
@rule("play_from_temp", priority=39)
def _play_from_temp(ctx: ParseContext) -> Optional[GameAction]:
    t = ctx.text
    # 「1枚までを登場させ(る)」「…を登場させ、」（連用形, させ、split 後は末尾「登場」）を対象。
    if not re.search(_nfc(r"を、?(?:レストで)?登場(?:させ(?:る)?)?$"), t):
        return None
    if re.search(_nfc(r"この(カード|キャラ|リーダー)を"), t):
        return None  # play_self が担当
    if any(_nfc(z) in t for z in ["手札", "トラッシュ", "ライフ", "デッキ"]):
        return None  # 明示ゾーンは play_card_from_zone 等が担当。「デッキから…登場」(直接登場)も
        #             公開→TEMP の文脈ではないため除外（分割後の登場句に デッキ は残らない）。
    if not re.search(_nfc(r"\d+枚"), t):
        return None  # 「N枚（まで）」の指定がある登場句に限定（条件/トリガー文を除外）
    tq = parse_target(t)
    tq.zone = Zone.TEMP
    tq.player = Player.SELF
    if _nfc("まで") in t:
        tq.is_up_to = True
    status = "RESTED" if re.search(_nfc(r"レストで(、)?登場"), t) else None
    return GameAction(
        type=ActionType.PLAY_CARD,
        target=tq,
        destination=Zone.FIELD,
        status=status,
        raw_text=t,
    )


# ---------------------------------------------------------------------------
# アクティブキャラへのアタック付与:
#   「（このターン中、）アクティブのキャラにもアタックできる」
#   → GRANT_KEYWORD("ATTACK_ACTIVE")。
#   通常はレストキャラしか攻撃できないが、このキーワードがあればアクティブも攻撃可。
#   単体ルール「このキャラは相手のアクティブのキャラにもアタックできる」は PERMANENT、
#   対象付き「リーダーかキャラ1枚までは、このターン中、〜」は THIS_TURN。
# ---------------------------------------------------------------------------
@rule("attack_active", priority=60)
def _attack_active(ctx: ParseContext) -> Optional[GameAction]:
    t = ctx.text
    if not re.search(_nfc(r"アクティブ.*キャラ.*アタックできる"), t):
        return None
    duration = "THIS_TURN" if _nfc("このターン中") in t else "PERMANENT"
    tq = parse_target(t)
    if re.search(_nfc(r"このキャラは"), t):
        tq = TargetQuery(select_mode="SOURCE")
    return GameAction(
        type=ActionType.GRANT_KEYWORD,
        target=tq,
        status="ATTACK_ACTIVE",
        duration=duration,
        raw_text=t,
    )


# ---------------------------------------------------------------------------
# フリーズ: 「（相手の）レストのキャラ1枚までは、次の相手のリフレッシュフェイズでアクティブにならない」
#   → FREEZE(target=相手のレストキャラ, is_up_to=True)。
#   エンジンの refresh_all が card.flags に "FREEZE" があればアクティブ化をスキップする。
# ---------------------------------------------------------------------------
@rule("freeze_target", priority=65)
def _freeze_target(ctx: ParseContext) -> Optional[GameAction]:
    t = ctx.text
    if not re.search(_nfc(r"アクティブにならない"), t):
        return None
    tq = parse_target(t)
    tq.player = Player.OPPONENT
    if _nfc("まで") in t:
        tq.is_up_to = True
    return GameAction(type=ActionType.FREEZE, target=tq, raw_text=t)


# ---------------------------------------------------------------------------
# レスト制限: 「（相手の）コストN以下のキャラM枚までは、次の相手の（ターン/エンドフェイズ）
#             終了時まで、レストにできない」
#   → PREVENT_REST(target=相手キャラ, duration)。
#   「レストにできない」＝そのキャラは（自身を）レストにできない＝アタックもブロックも
#   できない（どちらも本体をレストにする操作のため）。エンジンは timed_flags に
#   "CANNOT_REST" を立て、declare_attack / has_blocker でこのフラグを弾く。
#   freeze_target（アクティブにならない）とは逆向きのレスト制限。
# ---------------------------------------------------------------------------
@rule("rest_restrict", priority=66)
def _rest_restrict(ctx: ParseContext) -> Optional[GameAction]:
    t = ctx.text
    if not re.search(_nfc(r"レストにできない"), t):
        return None
    tq = parse_target(t)
    tq.player = Player.OPPONENT  # 全カード「相手の…キャラ」を対象にする
    if _nfc("まで") in t:
        tq.is_up_to = True
    # 「次の…終了時まで」は次の相手ターンを跨いで持続、それ以外は当ターン限り。
    duration = "UNTIL_NEXT_TURN_END" if _nfc("次の") in t else "THIS_TURN"
    return GameAction(
        type=ActionType.PREVENT_REST,
        target=tq,
        duration=duration,
        raw_text=t,
    )


# ---------------------------------------------------------------------------
# 効果無効: 「（相手の）リーダーかキャラ1枚までを、このターン中、効果を無効にする」
#   → NEGATE_EFFECT(target=相手リーダー/キャラ, duration=THIS_TURN)。
#   エンジンは ability_disabled=True を対象に設定し、能力発動をブロックする。
# ---------------------------------------------------------------------------
@rule("negate_effect", priority=65)
def _negate_effect(ctx: ParseContext) -> Optional[GameAction]:
    t = ctx.text
    if not re.search(_nfc(r"効果を無効にする"), t):
        return None
    tq = parse_target(t)
    if _nfc("まで") in t:
        tq.is_up_to = True
    return GameAction(
        type=ActionType.NEGATE_EFFECT,
        target=tq,
        duration="THIS_TURN",
        raw_text=t,
    )


# ---------------------------------------------------------------------------
# 自己効果無効: 「このキャラは、このターン中、効果が無効になる」
#   「効果が無効になる」（受け身/自動詞）→ DISABLE_ABILITY(target=self, THIS_TURN)。
#   「効果を無効にする」（他動詞、相手対象）は negate_effect(p65) が担当。
# ---------------------------------------------------------------------------
@rule("self_effect_disabled", priority=64)
def _self_effect_disabled(ctx: ParseContext) -> Optional[GameAction]:
    t = ctx.text
    # 「効果が無効になる」(自動詞・自身の効果が無効化される)。
    # 「（自分の/相手の）【登場時】効果は無効になる」等の "は" + 範囲修飾付きは意味が
    # 異なり（特定トリガーのみ無効・対象が相手）SOURCE 全無効では不正確なため対象外。
    if not re.search(_nfc(r"効果が無効になる"), t):
        return None
    return GameAction(
        type=ActionType.DISABLE_ABILITY,
        target=TargetQuery(select_mode="SOURCE"),
        duration=_duration_of(t),
        raw_text=t,
    )


# ---------------------------------------------------------------------------
# ルール処理: 「ルール上、このカードはカード名を「X」としても扱う」
#              「ルール上、このカードはデッキに何枚でも入れることができる」
#   → RULE_PROCESSING（エンジン no-op）。
#   ゲームエンジンには影響しないルール注記（デッキ構築ルール等）を吸収する。
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# C10 勝敗置換: 「（自分のデッキが0枚になった場合、）自分は敗北する代わりに勝利する」
#   → VICTORY + status="REPLACE_DECKOUT_LOSS"。エンジン check_victory が
#   デッキアウト時にこの PASSIVE を走査し、敗北を勝利へ置換する（OP03-040 ナミ等）。
#   "ルール上" を含むため rule_processing(p35) より高優先度で先に捕捉する。
# ---------------------------------------------------------------------------
@rule("win_on_deckout", priority=95)
def _win_on_deckout(ctx: ParseContext) -> Optional[GameAction]:
    t = ctx.text
    if _nfc("敗北する代わりに勝利") not in t and _nfc("敗北する代わりに、勝利") not in t:
        return None
    return GameAction(
        type=ActionType.VICTORY,
        status="REPLACE_DECKOUT_LOSS",
        raw_text=t,
    )


# ---------------------------------------------------------------------------
# 勝利宣言: 「（自分は）ゲームに勝利する」→ VICTORY（即時勝利）。従来 OTHER。
#   「敗北する代わりに勝利」(デッキアウト置換) は win_on_deckout(priority 95) が先に拾うため、
#   ここは無条件/条件付きの能動勝利のみ（status なし → エンジンが即 winner 設定）。
# ---------------------------------------------------------------------------
@rule("declare_victory", priority=22)
def _declare_victory(ctx: ParseContext) -> Optional[GameAction]:
    t = ctx.text
    if _nfc("ゲームに勝利") not in t:
        return None
    if _nfc("代わりに") in t:  # デッキアウト置換は win_on_deckout が担当
        return None
    return GameAction(type=ActionType.VICTORY, raw_text=t)


# ---------------------------------------------------------------------------
# C8 コスト宣言: 「任意のコストを宣言し、相手のデッキの上から1枚を公開する」
#   → DECLARE_COST。エンジンは数値入力インタラクションで宣言値を受け取り、相手デッキ
#   トップを公開して context に記録する。後続の「公開したカードが宣言したコストと同じ
#   場合、…」は DECLARED_COST_MATCH 条件の Branch として解釈される（OP11系6枚）。
# ---------------------------------------------------------------------------
@rule("declare_cost", priority=92)
def _declare_cost(ctx: ParseContext) -> Optional[GameAction]:
    t = ctx.text
    if _nfc("コストを宣言") not in t:
        return None
    return GameAction(type=ActionType.DECLARE_COST, raw_text=t)


@rule("rule_processing", priority=35)
def _rule_processing(ctx: ParseContext) -> Optional[GameAction]:
    t = ctx.text
    if not re.search(_nfc(r"ルール上"), t):
        return None
    return GameAction(type=ActionType.RULE_PROCESSING, raw_text=t)


# ---------------------------------------------------------------------------
# 自己制限: 「自分は、（このターン中、）...できない/られない」
#   → RULE_PROCESSING（エンジン no-op）。
#   「自分の効果でライフを手札に加えられない」「自分はキャラカードを登場できない」等。
#   制限チェックは未実装のため、解析だけ行いエンジンは何もしない（OTHER 脱出のみ）。
# ---------------------------------------------------------------------------
@rule("self_cannot", priority=33)
def _self_cannot(ctx: ParseContext) -> Optional[GameAction]:
    t = ctx.text
    if _nfc("自分は") not in t:
        return None
    if not re.search(_nfc(r"(できない|られない)"), t):
        return None
    # 制限自体はエンジン未実装(no-op)だが、「このターン中／このバトル中」の期間は
    # 正しく保持する（監査 DURATION の真値化。将来の enforce 時にそのまま使える）。
    return GameAction(type=ActionType.RULE_PROCESSING, duration=_duration_of(t), raw_text=t)
