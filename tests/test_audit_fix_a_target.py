"""カード効果監査（2026-10-01）で見つかった欠陥のうち WP「A_target」で直したパーサ類型の回帰テスト。

本文（カード DB の実文）を `EffectParserV2` に通し、解析結果の**構造**を直接見る小さなテスト群。
エンジン側（Rust）の挙動は `rust/opcg_engine` の `#[cfg(test)]` が受け持つ。
ゲームプレイの退行（誤った効果解決）を見逃さないための必須/標準テスト（cpu_infra ではない）。
"""
import sys

import pytest

import _bootstrap  # noqa: F401  (sys.path + google stub)
from opcg_sim.src.effects.parser_v2 import EffectParserV2
from opcg_sim.tools.export_effects_json import Stats, _encode


def _parse(text, as_trigger=False):
    return [_encode(a, Stats()) for a in EffectParserV2().parse_card_text(text, as_trigger=as_trigger)]


def _walk(o):
    if isinstance(o, dict):
        yield o
        for v in o.values():
            yield from _walk(v)
    elif isinstance(o, list):
        for v in o:
            yield from _walk(v)


def _actions(abs_, ty=None):
    out = [d for d in _walk(abs_) if d.get("node") == "GameAction"]
    return [d for d in out if ty is None or d["type"] == ty]


def _one(abs_, ty):
    acts = _actions(abs_, ty)
    assert len(acts) == 1, (ty, acts)
    return acts[0]


# --- 自身を指す句 ---------------------------------------------------------------------------

def test_bounce_self_is_source_not_a_free_choice():
    for text in ("【起動メイン】このキャラを持ち主の手札に戻す。",
                 "【アタック時】自分のアクティブのリーダー1枚を、このターン中、パワー-5000することができる:"
                 "このターン終了時、このキャラを持ち主の手札に戻す。"):
        b = _one(_parse(text), "BOUNCE")
        assert b["target"]["select_mode"] == "SOURCE"


def test_trash_self_play_and_recover_are_source_in_trash():
    play = _one(_parse("【KO時】自分の手札から『白ひげ海賊団』を含む特徴を持つカード1枚を捨てることができる:"
                       "自分のライフが2枚以下の場合、このキャラカードをトラッシュからレストで登場させる。"), "PLAY_CARD")
    # 自身を指す表現は 2 通り（A_target: select_mode=SOURCE／E_replace: ref_id=self）。どちらもエンジンが自身に解決する。
    def _is_self(t):
        return t["select_mode"] == "SOURCE" or t.get("ref_id") == "self"
    assert _is_self(play["target"]) and play["status"] == "RESTED"
    mv = _one(_parse("【KO時】このキャラカードをトラッシュから手札に加える。"), "MOVE_CARD")
    assert _is_self(mv["target"]) and mv["target"]["zone"] == "TRASH"
    assert mv["destination"] == "HAND"


def test_freeze_self_is_source():
    fz = _one(_parse("【起動メイン】このキャラをアクティブにする。その後、このキャラは、"
                     "次の自分のリフレッシュフェイズでアクティブにならない。"), "FREEZE")
    assert fz["target"]["select_mode"] == "SOURCE"


# --- 公開したカード・登場させたカード -----------------------------------------------------------

def test_revealed_card_goes_to_deck_bottom_from_temp_not_field():
    db = _one(_parse("【アタック時】自分のデッキの上から1枚を公開し、公開したカードがパワー6000以上のキャラカードだった場合、"
                     "このキャラは、このターン中、パワー+3000。その後、公開したカードをデッキの下に置く。"), "DECK_BOTTOM")
    assert db["target"]["zone"] == "TEMP"


def test_played_card_is_referenced_by_the_following_grant():
    abs_ = _parse("【メイン】自分のデッキの上から1枚を公開し、そのカードがコスト9以下の『白ひげ海賊団』を含む特徴を持つ"
                  "キャラカードの場合、登場させてもよい。登場させた場合、そのキャラは、このターン中、【速攻】を得る。")
    play, grant = _one(abs_, "PLAY_CARD"), _one(abs_, "GRANT_KEYWORD")
    assert play["target"]["save_id"] == "played_card"
    assert grant["target"]["ref_id"] == "played_card"


# --- キーワード付与 ------------------------------------------------------------------------

def test_blocker_grant_is_kept_next_to_cost_change():
    abs_ = _parse("自分のリーダーが特徴《黒ひげ海賊団》を持つ場合、このキャラは【ブロッカー】を得て、自分のトラッシュ4枚につき、コスト+1。")
    assert _one(abs_, "GRANT_KEYWORD")["status"] == "ブロッカー"
    assert _one(abs_, "BUFF")["status"] == "COST_REDUCTION"


def test_grant_to_others_and_self_keeps_both_subjects():
    abs_ = _parse("自分の「シュラ」すべてとこのキャラは【ブロック不可】を得る。")
    grants = _actions(abs_, "GRANT_KEYWORD")
    assert len(grants) == 2
    assert grants[0]["target"]["names"] == ["シュラ"] and grants[0]["target"]["count"] == -1
    assert grants[1]["target"]["select_mode"] == "SOURCE"
    # 「このキャラ以外の…すべて」は他のキャラ全て（自身を除く・コスト/色の絞り込みを保つ）。
    g = _one(_parse("このキャラ以外の自分のコスト3以上の赤のキャラすべては、【速攻】を得る。"), "GRANT_KEYWORD")
    assert g["target"]["count"] == -1 and "EXCLUDE_SOURCE" in g["target"]["flags"]
    assert g["target"]["colors"] == ["赤"] and g["target"]["cost_min"] == 3
    # 述語の「属性(斬)を得る」を対象の絞り込みに混ぜない。
    g = _one(_parse("【起動メイン】自分のキャラの「モンキー・D・ルフィ」1枚までは、このターン中、【速攻:キャラ】と属性(斬)を得る。"),
             "GRANT_KEYWORD")
    assert not g["target"].get("attributes")


def test_power_override_covers_leader_and_self():
    acts = _actions(_parse("【相手のアタック時】自分のリーダーとこのキャラを、このターン中、元々のパワー7000にする。"), "BUFF")
    assert [a["target"].get("card_type") or a["target"]["select_mode"] for a in acts] == [["LEADER"], "SOURCE"]


def test_blocker_disable_on_attacker_becomes_unblockable_grant():
    for text in ("【メイン】自分の「トラファルガー・ロー」1枚までを選び、このターン中、パワー+2000。"
                 "その後、相手は、このターン中、選んだカードがアタックする場合【ブロッカー】を発動できない。",
                 "【メイン】自分の特徴《麦わらの一味》を持つ、リーダーかキャラ1枚までを選ぶ。"
                 "相手は、このターン中、そのリーダーかキャラがアタックする場合【ブロッカー】を発動できない。"):
        g = _one(_parse(text), "GRANT_KEYWORD")
        assert g["status"] == "ブロック不可" and g["target"]["ref_id"] == "selected_card"
        assert not _actions(_parse(text), "BUFF") or all(a.get("status") != "BLOCKER_DISABLE"
                                                         for a in _actions(_parse(text), "BUFF"))
    # 相手のブロッカー側を縛る従来形は変えない。
    b = _one(_parse("【アタック時】相手は、このバトル中、【ブロッカー】を発動できない。"), "BUFF")
    assert b["status"] == "BLOCKER_DISABLE"


# --- 攻撃先の制限 ---------------------------------------------------------------------------

def test_only_attack_named_character_is_an_opponent_restriction():
    a = _one(_parse("このキャラがレストの場合、相手はキャラの「ユースタス・キッド」以外にアタックできない。"), "RULE_PROCESSING")
    assert a["status"] == "ATTACK_CHAR_ONLY:ユースタス・キッド"


def test_attack_target_bans_are_not_a_blanket_attack_disable():
    ban = _one(_parse("このキャラは、登場したターン中、リーダーにアタックできない。"), "ATTACK_DISABLE")
    assert ban["status"] == "ATTACK_BAN_LEADER" and ban["target"]["select_mode"] == "SOURCE"
    ban = _one(_parse("【起動メイン】その後、このリーダーは、このターン中、相手の元々のコスト7以下のキャラへアタックできない。"),
               "ATTACK_DISABLE")
    assert ban["status"] == "ATTACK_BAN_CHAR_OCOST_LE_7"


# --- 択一・二段 ---------------------------------------------------------------------------

def test_shared_target_in_suruka_choice_and_trailing_sentence():
    abs_ = _parse("【メイン】相手のコスト1以下のキャラ1枚までを、KOするか、持ち主の手札に戻すか、ライフの上か下に表向きで置く。"
                  "その後、自分の特徴《天竜人》を持つキャラがいる場合、カード1枚を引く。")
    for ty in ("KO", "BOUNCE", "MOVE_CARD"):
        for a in _actions(abs_, ty):
            assert a["target"]["player"] == "OPPONENT" and a["target"]["cost_max"] == 1
    # 後続の「その後」は選択肢の外（最終選択肢の中ではない）。
    draw = _one(abs_, "DRAW")
    life_opts = [d for d in _walk(abs_) if d.get("node") == "Choice"]
    assert all(draw not in list(_walk(c)) for c in life_opts)


def test_dual_tier_removal_with_second_side_word_and_original_cost():
    ko = _actions(_parse("【登場時】ドン!!-2:相手のコスト3以下のキャラ1枚までと、相手のコスト2以下のキャラ1枚までを、KOする。"), "KO")
    assert [a["target"]["cost_max"] for a in ko] == [3, 2]
    ko = _actions(_parse("【メイン】相手の元々の、コスト6以下のキャラ1枚までとコスト4以下のキャラ1枚までを、KOする。"), "KO")
    assert all("ORIGINAL_COST" in a["target"]["flags"] for a in ko)


def test_bounce_or_deck_bottom_is_a_choice_on_field_characters():
    abs_ = _parse("【アタック時】コスト2以下のキャラ1枚までを、持ち主の手札かデッキの下に戻す。")
    kinds = sorted(a["type"] for a in _actions(abs_))
    assert kinds == ["BOUNCE", "DECK_BOTTOM"]
    assert all(a["target"]["zone"] == "FIELD" for a in _actions(abs_))


def test_ko_after_negate_filters_the_selected_character():
    ko = _one(_parse("【登場時】相手のキャラ1枚までを、このターン中、効果を無効にする。その後、そのキャラのパワーが5000以下の場合、KOする。"), "KO")
    assert ko["target"]["ref_id"] == "selected_card" and ko["target"]["power_max"] == 5000
    assert "REF_FILTER" in ko["target"]["flags"]


# --- 対象の絞り込み -------------------------------------------------------------------------

def test_name_or_type_and_selector_or_and_exclusion():
    b = _one(_parse("【カウンター】自分のキャラか「シルバーズ・レイリー」1枚までを、このバトル中、パワー+2000。"), "BUFF")
    assert "NAME_OR_TYPE" in b["target"]["flags"]
    p = _one(_parse("【登場時】自分の手札から「ジュラキュール・ミホーク」以外で、コスト4以下の、特徴《シッケアール王国》か属性(斬)を持つ"
                    "キャラカード1枚までを、レストで登場させる。"), "PLAY_CARD")
    assert "SELECTOR_OR" in p["target"]["flags"]
    assert p["target"]["exclude_names"] == ["ジュラキュール・ミホーク"] and not p["target"].get("names")
    p = _one(_parse("【登場時】自分の手札からコスト5以下の、「ペローナ」か属性(斬)を持つキャラカード1枚までを、登場させる。"), "PLAY_CARD")
    assert "SELECTOR_OR" in p["target"]["flags"]


def test_color_for_event_and_cost_for_character_scopes():
    m = _one(_parse("【メイン】自分のデッキの上から4枚を見て、赤のイベントかコスト3以上のキャラカード1枚までを公開し、手札に加える。"
                    "その後、残りを好きな順番でデッキの下に置く。"), "MOVE_CARD")
    assert {"COLORS_ONLY_EVENT", "COST_ONLY_CHARACTER"} <= set(m["target"]["flags"])


def test_keyword_possession_and_single_color_and_lacks_trait_flags():
    k = _one(_parse("【アタック時】相手の【ブロッカー】を持つキャラ1枚までを、レストにする。"), "REST")
    assert "HAS_KEYWORD:ブロッカー" in k["target"]["flags"]
    b = _one(_parse("【登場時】自分の単色のリーダーを、次の相手のエンドフェイズ終了時まで、元々のパワー8000にする。"), "BUFF")
    assert "SINGLE_COLOR" in b["target"]["flags"]
    n = _one(_parse("自分の、リーダーと『ロジャー海賊団』を含む特徴を持たないキャラすべては、効果が無効になる。"), "NEGATE_EFFECT")
    assert "LACKS_TRAIT_PARTIAL:ロジャー海賊団" in n["target"]["flags"] and not n["target"].get("traits")
    assert n["target"]["count"] == -1


def test_rest_modifier_survives_a_comma():
    f = _one(_parse("【登場時】相手のレストの、リーダーとキャラ合計3枚までは、次の相手のリフレッシュフェイズでアクティブにならない。"), "FREEZE")
    assert f["target"]["is_rest"] is True
    f = _one(_parse("【自分のターン中】このキャラがレストになった時、自分のライフの上から1枚を手札に加えてもよい。"
                    "そうした場合、相手のレストの、キャラかステージ1枚までは、次の相手のリフレッシュフェイズでアクティブにならない。"), "FREEZE")
    assert f["target"]["is_rest"] is True


def test_leader_only_grant_does_not_include_characters():
    g = _one(_parse("【アタック時】その後、自分の特徴《海軍》を持つリーダー1枚までは、このターン中、アクティブのキャラにもアタックできる。"),
             "GRANT_KEYWORD")
    assert g["target"]["card_type"] == ["LEADER"]


def test_all_others_ko_is_both_sides_and_counts_per_side_are_isolated():
    ko = _one(_parse("【登場時】このキャラ以外のキャラすべてを、KOする。"), "KO")
    assert ko["target"]["player"] == "ALL" and "EXCLUDE_SOURCE" in ko["target"]["flags"]
    b = _one(_parse("【登場時】自分の場の特徴《麦わらの一味》を持つカード1枚につき、相手のキャラ1枚までを、このターン中、パワー-1000。"), "BUFF")
    assert not b["target"].get("traits") and b["target"]["player"] == "OPPONENT"
    b = _one(_parse("【登場時】自分の、「サン五郎」と「サンジ」すべてを、持ち主の手札に戻す。"), "BOUNCE")
    assert b["target"]["card_type"] == ["CHARACTER"]


def test_hand_wide_cost_reduction_and_trash_from_temp_and_reveal_hand():
    b = _one(_parse("【ドン!!×1】自分の手札の青のイベントを、コスト-1。"), "BUFF")
    assert b["target"]["select_mode"] == "ALL" and b["target"]["count"] == -1
    t = _one(_parse("【登場時】自分のデッキの上から5枚を見て、カード2枚までを、トラッシュに置く。その後、残りを好きな順番でデッキの下に置く。"), "TRASH")
    assert t["target"]["zone"] == "TEMP"
    r = _one(_parse("【登場時】相手は自身の手札を1枚捨て、手札を公開する。その後、相手はカード1枚を引く。"), "REVEAL")
    assert r["target"]["player"] == "OPPONENT" and r["target"]["count"] == -1
    m = _one(_parse("【起動メイン】自分のデッキの上から5枚を見て、コスト5のキャラカード1枚までを、ライフの上に表向きで加える。"
                    "その後、残りを好きな順番でデッキの下に置く。"), "MOVE_CARD")
    assert m["target"]["zone"] == "TEMP"


def test_each_named_card_is_played_one_by_one():
    plays = _actions(_parse("【登場時】自分の手札からコスト2の、「サボ」と「ポートガス・D・エース」と「モンキー・D・ルフィ」"
                            "それぞれ1枚ずつまでを、登場させる。"), "PLAY_CARD")
    assert [p["target"]["names"] for p in plays] == [["サボ"], ["ポートガス・D・エース"], ["モンキー・D・ルフィ"]]


def test_play_cost_sum_total_cap_is_recorded():
    p = _actions(_parse("【登場時】カード1枚を引き、自分の手札からカード名の異なる特徴《ロックス海賊団》を持つカード2枚までを、"
                        "コストの合計が9以下になるように登場させる。"), "PLAY_CARD")[0]
    assert "COST_SUM_MAX:9" in p["target"]["flags"]


def test_chooser_of_opponent_hand_discard_is_the_opponent():
    d = _one(_parse("【登場時】相手の手札が6枚以上ある場合、相手の手札2枚を捨てる。"), "DISCARD")
    assert d["target"]["chooser"] == "OPPONENT"


def test_same_name_as_the_discarded_card():
    abs_ = _parse("【メイン】自分の手札からパワー4000以下の特徴《ジェルマ66》を持つキャラカード1枚を捨てることができる:"
                  "自分の場のドン!!が相手の場のドン!!の枚数以下の場合、自分のトラッシュのパワー5000から7000の捨てたカードと同じ"
                  "カード名を持つキャラカード1枚までを、登場させる。")
    assert _one(abs_, "DISCARD")["target"]["save_id"] == "discarded_card"
    assert "SAME_NAME_AS:discarded_card" in _one(abs_, "PLAY_CARD")["target"]["flags"]


def test_trash_n_back_to_deck_then_shuffle_cost():
    abs_ = _parse("【アタック時】【ターン1回】自分のトラッシュのカード20枚をデッキに戻しシャッフルできる:このキャラは、このバトル中、パワー+10000。")
    mv = _one(abs_, "MOVE_CARD")
    assert mv["target"]["zone"] == "TRASH" and mv["target"]["count"] == 20 and mv["destination"] == "DECK"
    assert _actions(abs_, "SHUFFLE")


# --- 条件 ---------------------------------------------------------------------------------

def _cond(text):
    ab = _parse(text)[0]
    return ab.get("condition")


def test_negated_count_condition_is_inverted():
    c = _cond("相手の元々のパワー5000以上のキャラが2枚以上いない場合、このキャラはアタックできない。")
    assert c["type"] == "FIELD_COUNT" and c["operator"] == "LT" and c["value"] == 2


def test_other_than_self_in_has_character_condition():
    c = _cond("【起動メイン】【ターン1回】自分のリーダーが特徴《フォクシー海賊団》を持ち、自分の他の「イトミミズ」がいない場合、"
              "ドン!!デッキからドン!!1枚までを、レストで追加する。")
    has = [d for d in _walk(c) if d.get("type") == "HAS_CHARACTER"][0]
    assert "EXCLUDE_SOURCE" in has["target"]["flags"]


def test_and_of_names_leader_or_and_card_other_than_self():
    ab = _parse("【カウンター】自分のリーダーかキャラ1枚までを、このバトル中、パワー+3000。その後、自分の元々のパワー6000のキャラの、"
                "「ポートガス・D・エース」と「モンキー・D・ルフィ」がいる場合、カード1枚を引く。")
    ands = [d for d in _walk(ab) if d.get("type") == "AND"]
    assert ands and len([a for a in ands[0]["args"] if a["type"] == "FIELD_COUNT"]) == 2
    c = _cond("【登場時】自分のリーダーが「エドワード・ニューゲート」か特徴《ワノ国》を持つ場合、自分の手札からパワー6000以下のキャラカード1枚までを、登場させる。")
    assert c["type"] == "OR"
    c = _cond("このカード以外の特徴《山賊》を持つキャラがいる場合、このキャラは【ダブルアタック】を得る。")
    assert "EXCLUDE_SOURCE" in c["target"]["flags"]


def test_condition_after_self_removal_cost_is_evaluated_after_the_cost():
    ab = _parse("【起動メイン】このキャラをトラッシュに置くことができる:自分のトラッシュが15枚以上ある場合、"
                "自分のリーダーかキャラ1枚にレストのドン!!1枚までを、付与する。")[0]
    assert ab.get("condition") is None
    assert ab["effect"]["node"] == "Branch" and ab["effect"]["condition"]["type"] == "TRASH_COUNT"


def test_keyword_reminder_glued_to_the_next_ability_is_split():
    abs_ = _parse("【ブロッカー】(相手のアタックの後、このカードをレストにし、アタックの対象をこのカードにできる)"
                  "【自分のターン終了時】自分のレストの特徴《ODYSSEY》を持つキャラが2枚以上いる場合、このキャラをアクティブにする。")
    assert not _actions(abs_, "REST")
    assert _one(abs_, "ACTIVE")


def test_hand_self_counter_passive_and_no_counter_chars_condition():
    ab = _parse("手札のこのカードは、自分のキャラがカウンターを持たないキャラのみの場合、カウンター+2000を持つ。")[0]
    eff = ab["effect"]
    assert eff["type"] == "BUFF" and eff["status"] == "COUNTER" and "SELF_IN_HAND" in eff["target"]["flags"]
    assert ab["condition"]["type"] == "FIELD_COUNT" and "HAS_COUNTER" in ab["condition"]["target"]["flags"]
    assert ab["condition"]["operator"] == "LT" and ab["condition"]["value"] == 1


def test_opponent_dynamic_cost_cap_does_not_flip_the_hand_owner():
    p = _one(_parse("【相手のターン中】【KO時】ドン!!-1:自分の手札から「シャーロット・スムージー」以外の相手の場のドン!!の枚数以下のコストを持ち、"
                    "特徴《ビッグ・マム海賊団》を持つキャラカード1枚までを、登場させる。"), "PLAY_CARD")
    assert p["target"].get("player", "SELF") == "SELF" and p["target"]["zone"] == "HAND"
    assert p["target"]["cost_max_dynamic"] == "DON_COUNT_FIELD_OPPONENT"


# --- WP「G3_engine」: 対話・遅延・一回限りの割引・分配（エンジン側は rust/.../effects/tests_g3.rs）----------

def test_g3_delay_until_the_opponents_next_main_phase():
    rest = _one(_parse("【自分のターン中】【登場時】自分のリーダーが多色で、相手の場のドン!!が7枚以下の場合、"
                       "次の相手のメインフェイズ開始時、相手は自身のアクティブのドン!!1枚をレストにする。"), "REST_DON")
    assert rest["delay"] == "OPP_MAIN_START"


def test_g3_next_play_discount_is_marked_once():
    for text in ("【起動メイン】【ターン1回】自分のキャラが1枚以下の場合、このターン中、次に自分が手札から登場させる"
                 "コスト3以上の特徴《ワノ国》を持つキャラカードの支払うコストは1少なくなる。",
                 "【起動メイン】【ターン1回】ドン!!-1:このターン中、次に自分が手札から登場させるコスト4以上の"
                 "「トラファルガー・ロー」の支払うコストは2少なくなる。"):
        b = _one(_parse(text), "BUFF")
        assert b["status"] == "COST_REDUCTION" and "NEXT_PLAY_ONCE" in b["target"]["flags"]
    # 常時の軽減（「次に」なし）には付かない
    plain = _one(_parse("【自分のターン中】自分の手札の青のイベントは、コスト-1。"), "BUFF")
    assert "NEXT_PLAY_ONCE" not in plain["target"]["flags"]


def test_g3_two_tier_pick_is_a_group_distribution():
    abs_ = _parse("【登場時】自分のトラッシュのコスト4以下のキャラカード1枚までとコスト2以下のキャラカード1枚までを選び、"
                  "1枚を登場させ、残りをレストで登場させる。")
    sel = _actions(abs_, "SELECT")
    assert [s["target"]["save_id"] for s in sel] == ["_sel_a", "_sel_b"]
    assert "EXCLUDE_SAVED:_sel_a" in sel[1]["target"]["flags"]
    plays = _actions(abs_, "PLAY_CARD")
    assert plays[0]["target"]["select_mode"] == "GROUP_FIRST" and plays[0]["target"]["ref_id"] == "_sel_a+_sel_b"
    assert plays[1]["target"]["select_mode"] == "REMAINING" and plays[1]["status"] == "RESTED"


def test_g3_revealed_cards_are_played_from_the_saved_group():
    abs_ = _parse("【登場時】自分の手札から、特徴《ドレスローザ》を持つコスト7以下のキャラカード2枚までを、公開する。"
                  "公開したカードのうち1枚を登場させ、残りがコスト4以下ならレストで登場させる。")
    first, rest = _actions(abs_, "PLAY_CARD")
    assert first["target"]["select_mode"] == "GROUP_FIRST" and first["target"]["ref_id"] == "revealed_cards"
    assert rest["target"]["select_mode"] == "REMAINING" and rest["target"]["cost_max"] == 4


def test_g3_battled_this_turn_is_not_an_in_progress_battle():
    abs_ = _parse("【ドン!!×3】【起動メイン】【ターン1回】このターン中、このリーダーが相手のキャラとバトルしている場合、"
                  "このリーダーをアクティブにする。")
    conds = [d for d in _walk(abs_) if d.get("node") == "Condition" and d.get("type") == "SOURCE_STATE"]
    assert [c["value"] for c in conds] == ["BATTLED_CHAR_THIS_TURN"]


def test_g3_name_or_typed_name():
    abs_ = _parse("このキャラがKOされる場合、代わりに自分の、「魚人島」かリーダーの「しらほし」1枚を、レストにできる。")
    tqs = [d for d in _walk(abs_) if d.get("node") == "TargetQuery" and d.get("names")]
    assert tqs and all("NAME_OR_TYPED_NAME" in t["flags"] and "NAME_OR_TYPE" not in t["flags"] for t in tqs)
