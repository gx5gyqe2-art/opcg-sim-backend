"""
parser.py の単体テスト（ロードマップ フェーズ1〜4の検証）。

実行: python -m pytest tests/test_parser.py -v
   または: python tests/test_parser.py  (pytest 無し環境でも自走)
"""
import conftest  # noqa: F401  (google スタブ注入 & sys.path 設定)

from opcg_sim.src.core.effects.parser import EffectParser

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
