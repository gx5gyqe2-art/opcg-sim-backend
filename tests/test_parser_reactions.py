"""反応型の誘発句のパーサ出力（誘発の欠落 WP D_trigger・2026-10-01）。

「相手がイベントを発動した時」「ライフが0枚になった時」「…とバトルしたバトル終了時」等は、
トリガー種別だけを正しく出し、誘発句そのもの（主語・要因・絞り込み）は能力の raw_text に残す。
エンジンが raw_text を読んで該当するイベントで待ち行列へ積む（`rust/opcg_engine/src/effects/triggers.rs`・
単体テストは `effects/tests_reactions.rs`）。ここで固定するのはパーサ側の約束だけ。
"""
import conftest  # noqa: F401  (google スタブ注入 & sys.path 設定)

import pytest

from opcg_sim.src.effects.parser_v2 import EffectParserV2


def _first(text):
    abilities = EffectParserV2().parse_card_text(text)
    assert abilities, text
    return abilities[0]


def _actions(node):
    if node is None:
        return []
    acts = getattr(node, "actions", None)
    if acts is not None:
        return [a for x in acts for a in _actions(x)]
    out = [node] if hasattr(node, "type") else []
    for k in ("if_true", "if_false", "sub_effect"):
        out += _actions(getattr(node, k, None))
    return out


@pytest.mark.parametrize("text,trigger", [
    ("【ドン!!×1】【自分のターン中】【ターン1回】相手がイベントを発動した時、カード1枚を引く。", "ON_EVENT_PLAY"),
    ("【ターン1回】相手が【ブロッカー】を発動した時、相手のパワー8000以下のキャラ1枚までを、KOする。", "ON_EVENT_PLAY"),
    ("【ターン1回】【トリガー】が発動した時、カード2枚を引き、自分の手札2枚を捨てる。", "ON_EVENT_PLAY"),
    ("【相手のターン中】【トリガー】が発動した時、このキャラは、このターン中、【ブロッカー】を得る。", "ON_EVENT_PLAY"),
    ("【ターン1回】自分が元々の効果のないキャラを手札から登場させた時、自分のドン!!2枚までを、アクティブにする。", "ON_OPP_PLAY"),
    ("【相手のターン中】【ターン1回】相手がキャラを登場させた時、相手のキャラ1枚までを、レストにする。", "ON_OPP_PLAY"),
    ("【自分のターン中】【ターン1回】相手のキャラが自分の効果で持ち主の手札に戻った時、カード1枚を引く。", "ON_LEAVE"),
    ("【自分のターン中】【ターン1回】キャラが自分の効果で場を離れた時、発動できる。カード1枚を引く。", "ON_LEAVE"),
    ("【ドン!!×1】【自分のターン中】【ターン1回】相手のライフが離れた時、カード2枚を引く。", "ON_LIFE_DECREASE"),
    ("【相手のターン中】【ターン1回】自分のライフが0枚になった時、カード1枚を引く。", "ON_LIFE_DECREASE"),
    ("【自分のターン中】【ターン1回】自分のライフが手札に加わった時、このキャラは、このターン中、パワー+2000。", "ON_LIFE_DECREASE"),
    ("【ドン!!×1】【ターン1回】このキャラのバトルによって相手のキャラをKOした時、このキャラをアクティブにする。", "ON_KO"),
    ("【ドン!!×1】このキャラが相手のキャラとバトルしたバトル終了時、バトルした相手のキャラをKOしてもよい。", "PASSIVE"),
    ("【ターン1回】自分の『ロックス海賊団』を含む特徴を持つリーダーがアタックした時かアタックされた時、"
     "自分の手札1枚を捨てて発動できる。自分のリーダーを、このバトル中、パワー+3000。", "PASSIVE"),
])
def test_reaction_clauses_get_a_reactive_trigger(text, trigger):
    ab = _first(text)
    assert ab.trigger.name == trigger
    # 誘発句は raw_text に残る（エンジンが読む）。
    assert ab.raw_text.startswith(text[:12]) or text[:12] in ab.raw_text


def test_timing_tag_is_kept_as_a_condition_for_reactive_triggers():
    ab = _first("【自分のターン中】【ターン1回】相手がイベントを発動した時、カード1枚を引く。")
    assert ab.trigger.name == "ON_EVENT_PLAY"
    assert "SELF_TURN" in repr(ab.condition)
    assert "TURN_LIMIT" in repr(ab.condition)


def test_trigger_clause_is_not_left_in_the_effect():
    ab = _first("【自分のターン中】【ターン1回】自分がドローフェイズ以外でカードを引いた時、このキャラは、このターン中、パワー+2000。")
    act = _actions(ab.effect)[0]
    assert act.type.name == "BUFF"
    assert act.target.select_mode == "SOURCE"


def test_inline_discard_cost_with_activation_phrase():
    ab = _first("【ターン1回】自分の『ロックス海賊団』を含む特徴を持つリーダーがアタックした時かアタックされた時、"
                "自分の手札1枚を捨てて発動できる。自分のリーダーを、このバトル中、パワー+3000。")
    assert ab.cost is not None and ab.cost_optional
    assert [a.type.name for a in _actions(ab.cost)] == ["DISCARD"]
    assert [a.type.name for a in _actions(ab.effect)] == ["BUFF"]


def test_battled_opponent_is_the_trigger_subject():
    ab = _first("【自分のターン中】このキャラが相手のコスト5以下のキャラとバトルしたバトル終了時、"
                "バトルした相手のキャラを持ち主のデッキの下に置く。")
    assert ab.trigger.name == "YOUR_TURN"
    act = _actions(ab.effect)[0]
    assert act.type.name == "DECK_BOTTOM" and act.target.ref_id == "trigger_subject"


def test_self_ko_and_self_deck_bottom_target_the_source():
    ab = _first("【ドン!!×1】このキャラが相手のキャラとバトルしたバトル終了時、バトルした相手のキャラをKOしてもよい。そうした場合、このキャラをKOする。")
    kos = [a for a in _actions(ab.effect) if a.type.name == "KO"]
    assert [k.target.select_mode for k in kos] == ["CHOOSE", "SOURCE"]
    assert kos[0].target.ref_id == "trigger_subject"


def test_this_battle_end_delay_is_battle_end():
    ab = _first("【ドン!!×1】【アタック時】自分の手札1枚を捨てることができる:コスト2以下のキャラ1枚までを、持ち主のデッキの下に置く。"
                "その後、このバトル終了時、このキャラを持ち主のデッキの下に置く。")
    acts = _actions(ab.effect)
    assert [a.delay for a in acts] == [None, "BATTLE_END"]
    assert acts[1].target.select_mode == "SOURCE"


def test_negate_then_ko_keeps_both_halves():
    ab = _first("【起動メイン】【ターン1回】ドン!!-1:このキャラが登場したターンの場合、"
                "相手のコスト6以下のキャラ1枚までを、このターン中、効果を無効にし、KOする。")
    acts = _actions(ab.effect)
    assert [a.type.name for a in acts] == ["NEGATE_EFFECT", "KO"]
    assert acts[1].target.ref_id == acts[0].target.save_id


def test_per_revealed_cost_scaling():
    ab = _first("相手がイベントか【ブロッカー】を発動した時、自分のライフの上から1枚までを公開する。"
                "公開したカードのコスト1につき、このキャラは、このターン中、パワー+1000。")
    buff = [a for a in _actions(ab.effect) if a.type.name == "BUFF"][0]
    assert buff.value.dynamic_source == "REVEALED_CARD_COST"
    assert buff.value.multiplier == 1000


def test_hand_discarded_this_turn_condition_is_an_event_not_a_hand_count():
    ab = _first("手札のこのカードは、効果で自分の手札が捨てられているターン中、コスト-3。")
    assert ab.condition.type.name == "EVENT_THIS_TURN"
    assert ab.condition.value == ("HAND_DISCARDED_BY_EFFECT_SEAT", 1)


# --- カード効果監査 WP=G2_trigger（誘発の読み） ------------------------------------------

def test_char_left_by_own_effect_event_is_recorded_per_seat():
    """OP07-038: 「キャラが自分の効果で場を離れた時」＝実行者の席ごとのターン内イベント。"""
    ab = _first("【自分のターン中】【ターン1回】キャラが自分の効果で場を離れた時、発動できる。"
                "自分の手札が5枚以下の場合、カード1枚を引く。")
    assert ab.trigger.name == "ON_LEAVE"

    def conds(c):
        if c is None:
            return []
        return [c] + [x for a in (getattr(c, "args", None) or []) for x in conds(a)]
    ev = [c for c in conds(ab.condition) if c.type.name == "EVENT_THIS_TURN"]
    assert ev and ev[0].value == ("CHAR_LEFT_BY_OWN_EFFECT_SEAT", 1)


def test_ko_or_leave_clause_stays_one_on_ko_ability():
    """OP10-042: 「KOされた時か、相手の効果で場を離れた時」は ON_KO 1 本（ON_LEAVE へ複製しない）。"""
    abs_ = EffectParserV2().parse_card_text(
        "【相手のターン中】【ターン1回】自分の特徴《ドレスローザ》を持つキャラがKOされた時か、"
        "相手の効果で場を離れた時、発動できる。自分の手札が5枚以下の場合、カード1枚を引く。")
    assert [a.trigger.name for a in abs_] == ["ON_KO"]


def test_navy_discard_draws_one_per_discarded_card():
    """OP12-040: 反応は捨てたカード 1 枚ごと＝「捨てた枚数分」は 1 枚分。"""
    ab = _first("自分の特徴《海軍》を持つカードの効果で自分の手札からカードが捨てられた時、"
                "捨てた枚数分カードを引く。")
    draw = [a for a in _actions(ab.effect) if a.type.name == "DRAW"][0]
    assert draw.value.dynamic_source is None and draw.value.base == 1


def test_optional_instead_branch_becomes_a_choice_with_the_original_action():
    """OP04-040: 「引く代わりに〜できる」を断ったとき通常のドローが残る＝択一にする。"""
    ab = _first("【ドン!!×1】【アタック時】自分のライフと手札の合計枚数が4枚以下の場合、カード1枚を引く。"
                "自分のコスト8以上のキャラがいる場合、カード1枚を引く代わりに自分のデッキの上から1枚までを、ライフの上に加えることができる。")
    seen = []

    def walk(n):
        if n is None:
            return
        if type(n).__name__ == "Choice":
            seen.append([o.type.name for o in n.options])
        for k in ("actions", "options"):
            for x in getattr(n, k, None) or []:
                walk(x)
        for k in ("if_true", "if_false"):
            walk(getattr(n, k, None))
    walk(ab.effect)
    assert seen == [["HEAL", "DRAW"]]


def test_protecting_replacement_moves_the_removed_character():
    """OP11-101: 他のキャラを守る置換の「ライフの上に加える」対象は離れる側のキャラ。"""
    ab = _first("【ターン1回】「カポネ・ベッジ」以外の自分の特徴《超新星》を持つキャラが相手の効果で場を離れる場合、"
                "代わりに自分のライフの上に裏向きで加えることができる。")
    sub = [a for a in _actions(ab.effect) if a.type.name == "MOVE_CARD"][0]
    assert sub.target.ref_id == "removed_card"
