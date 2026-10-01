"""
parser.py の単体テスト（ロードマップ フェーズ1〜4の検証）。

実行: python -m pytest tests/test_parser.py -v
   または: python tests/test_parser.py  (pytest 無し環境でも自走)
"""
import conftest  # noqa: F401  (google スタブ注入 & sys.path 設定)

from opcg_sim.src.effects.parser import EffectParser

parser = EffectParser()


def test_on_play_draw():
    ab = parser.parse_card_text("【登場時】カード1枚を引く。")
    assert len(ab) >= 1
    assert ab[0].trigger.name == "ON_PLAY"
    assert ab[0].effect is not None
    assert ab[0].effect.type.name == "DRAW"


def test_self_rest_cost():
    ab = parser.parse_card_text("【起動メイン】このキャラをレストにできる：カード1枚を引く。")
    assert len(ab) >= 1
    assert ab[0].cost is not None
    # コストの対象が self を指している
    tgt = getattr(ab[0].cost, "target", None)
    assert tgt is not None and tgt.ref_id == "self"


def test_multiple_abilities():
    ab = parser.parse_card_text("【登場時】カード1枚を引く。 / 【KO時】カード1枚を引く。")
    assert len(ab) == 2
    triggers = {a.trigger.name for a in ab}
    assert "ON_PLAY" in triggers
    assert "ON_KO" in triggers


def test_condition_operator_le():
    ab = parser.parse_card_text(
        "【自分のターン中】自分のライフが3枚以下の場合、このリーダーのパワー+1000。"
    )
    assert len(ab) >= 1
    cond = ab[0].condition
    assert cond is not None
    assert cond.operator.name == "LE"
    assert cond.value == 3


def test_condition_operator_ge():
    ab = parser.parse_card_text(
        "【自分のターン中】ドン!!が5枚以上ある場合、このリーダーのパワー+1000。"
    )
    assert len(ab) >= 1
    cond = ab[0].condition
    assert cond is not None
    assert cond.operator.name == "GE"
    assert cond.value == 5


def test_trigger_keyword():
    ab = parser.parse_card_text("【トリガー】カード1枚を引く。")
    assert len(ab) >= 1
    assert ab[0].trigger.name == "TRIGGER"


def test_as_trigger_flag():
    ab = parser.parse_card_text("カード1枚を引く。", as_trigger=True)
    assert len(ab) >= 1
    assert ab[0].trigger.name == "TRIGGER"


def test_empty_text():
    assert parser.parse_card_text("") == []
    assert parser.parse_card_text("なし") == []


# --- pytest 無し環境向けの自走ランナー ---

# --- ドン!!まわりの本文忠実化（WP C_don・原子句ルールは V2 パーサ）----------------------
from opcg_sim.src.effects.parser_v2 import EffectParserV2  # noqa: E402

parser2 = EffectParserV2()


def _cond_of(text):
    ab = parser2.parse_card_text(text)
    return ab[0].condition


def test_don_compare_direction_when_opponent_is_subject():
    """「相手の場のドン!!が自分より多い」は 自分 < 相手（LT）。エンジンは 自分⟨op⟩相手 で評価する。"""
    c = _cond_of("【登場時】相手の場のドン!!の枚数が自分の場のドン!!の枚数より多い場合、カード1枚を引く。")
    assert c.type.name == "DON_COUNT_COMPARE" and c.operator.name == "LT"
    # 主語が自分側のものは従来どおり（自分が相手より2枚以上少ない＝LE 2）。
    c = _cond_of("【登場時】自分の場のドン!!の枚数が相手より2枚以上少ない場合、カード1枚を引く。")
    assert c.operator.name == "LE" and c.value == 2


def test_leader_trait_and_don_compare_are_both_kept():
    c = _cond_of("自分のリーダーが特徴《ジェルマ66》でかつ、自分の場のドン !!の枚数が相手より2枚以上少ない場合、このキャラは【ブロッカー】を得る。")
    assert c.type.name == "AND"
    assert {a.type.name for a in c.args} == {"LEADER_TRAIT", "DON_COUNT_COMPARE"}


def test_leader_trait_or_attribute():
    c = _cond_of("【登場時】自分のリーダーが特徴《FILM》か属性 (打)を持つ場合、自分のドン!!1枚までを、アクティブにする。")
    assert c.type.name == "OR"
    assert {a.type.name for a in c.args} == {"LEADER_TRAIT", "LEADER_ATTRIBUTE"}


def test_don_x1_tag_applies_to_both_attack_and_block():
    ab = parser2.parse_card_text("【ドン!!×1】【アタック時】/【ブロック時】自分の手札が5枚以下の場合、カード1枚を引く。")
    assert [a.trigger.name for a in ab] == ["ON_ATTACK", "ON_BLOCK"]
    for a in ab:
        assert a.condition.type.name == "AND"
        assert "HAS_DON" in {x.type.name for x in a.condition.args}


def test_name_and_trait_target_is_a_union():
    ab = parser2.parse_card_text(
        "【登場時】自分の、「ドンキホーテ・ロシナンテ」と特徴《ハートの海賊団》を持つキャラすべてを、このターン中、パワー+1000。")
    tq = ab[0].effect.target
    assert "TRAIT_OR_NAME" in tq.flags


def test_active_don_return_cost_is_active_only():
    ab = parser2.parse_card_text("【起動メイン】自分のアクティブのドン!!8枚をドン!!デッキに戻すことができる：カード1枚を引く。")
    assert ab[0].cost.type.name == "RETURN_DON" and ab[0].cost.status == "ACTIVE"
    ab = parser2.parse_card_text("【起動メイン】自分のドン!!8枚をドン!!デッキに戻すことができる：カード1枚を引く。")
    assert ab[0].cost.status is None


def test_attach_self_per_target_and_owner_variants():
    ab = parser2.parse_card_text("【起動メイン】このキャラにレストのドン!!2枚までを付与する。")
    a = ab[0].effect
    assert a.target.select_mode == "SOURCE" and a.value.base == 2 and a.status == "RESTED"
    ab = parser2.parse_card_text("【起動メイン】自分の特徴《アラバスタ王国》を持つキャラすべてにレストのドン!!1枚ずつまでを、付与する。")
    assert ab[0].effect.value.base == 1
    ab = parser2.parse_card_text("【起動メイン】リーダーかキャラ1枚に持ち主のレストのドン!!1枚までを、付与する。")
    a = ab[0].effect
    assert a.status == "RESTED_OWNER" and a.target.player.name == "ALL"
    ab = parser2.parse_card_text("【登場時】相手のキャラ1枚に相手のコストエリアのドン!!1枚までを、付与する。")
    assert ab[0].effect.status == "OPP"


def test_opponent_ramp_don_status():
    ab = parser2.parse_card_text("【登場時】相手はドン!!デッキからドン!!1枚を、アクティブで追加してもよい。")
    assert ab[0].effect.type.name == "RAMP_DON" and ab[0].effect.status == "OPPONENT"


def test_don_returned_events_and_untagged_trigger():
    ab = parser2.parse_card_text("【ターン1回】自分の場のドン!!が2枚以上ドン!!デッキに戻された時、カード1枚を引く。")
    assert ab[0].trigger.name == "PASSIVE"
    ab = parser2.parse_card_text(
        "【相手のターン中】【ターン1回】自分の場のドン!!が自分の効果によってドン!!デッキに戻された時、カード1枚を引く。")
    names = []

    def walk(c):
        if c is None:
            return
        if c.type.name == "EVENT_THIS_TURN":
            names.append(c.value[0])
        for x in c.args or []:
            walk(x)
    walk(ab[0].condition)
    assert names == ["DON_RETURNED_OWN"]


def test_bare_number_cost_is_rest_don():
    ab = parser2.parse_card_text("【自分のターン終了時】1：このキャラをアクティブにする。")
    assert ab[0].cost.type.name == "REST_DON" and ab[0].cost.value.base == 1


def test_leader_or_don_rest_cost_is_a_choice():
    ab = parser2.parse_card_text("【登場時】自分の、属性(斬)を持つリーダーかドン!!1枚をレストにできる：カード2枚を引く。")
    cost = ab[0].cost
    assert cost.options[0].type.name == "REST" and cost.options[1].type.name == "REST_DON"


def test_this_character_or_don_active_is_a_choice():
    ab = parser2.parse_card_text("【自分のターン終了時】このキャラか自分のドン!!1枚までを、アクティブにする。")
    assert [o.type.name for o in ab[0].effect.options] == ["ACTIVE", "ACTIVE_DON"]


if __name__ == "__main__":
    import traceback

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except Exception:
            print(f"FAIL  {t.__name__}")
            traceback.print_exc()
            failed += 1
    print(f"\n=== {passed} passed, {failed} failed ===")
    raise SystemExit(1 if failed else 0)
