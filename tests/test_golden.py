"""ゴールデンコーパス・ランナー。

新パーサ(EffectParserV2)の出力 summary を golden_cases の期待値と部分一致比較する。

実行:
    python -m pytest tests/test_golden.py -v
    または: python tests/test_golden.py   (pytest 無し環境でも自走)
"""
import conftest  # noqa: F401  (google スタブ注入 & sys.path 設定)

from opcg_sim.src.core.effects.parser_v2 import EffectParserV2
from tests.golden.golden_cases import CASES
from tests.golden.summarize import matches_expected, summarize_ability


def _run_case(case) -> list:
    """1ケースを実行し、不一致メッセージのリストを返す（空なら成功）。"""
    parser = EffectParserV2()
    abilities = parser.parse_card_text(case["text"], as_trigger=case.get("as_trigger", False))
    expect = case["expect"]

    problems = []
    if len(abilities) != len(expect):
        problems.append(
            f"ability 数 {len(abilities)} != 期待 {len(expect)}"
        )
        return problems

    for i, (ab, exp) in enumerate(zip(abilities, expect)):
        summary = summarize_ability(ab)
        diffs = matches_expected(summary, exp)
        for d in diffs:
            problems.append(f"ability[{i}].{d}")
    return problems


def _make_test(case):
    def _test():
        problems = _run_case(case)
        assert not problems, "\n".join(
            [f"[{case['id']}] {case['text']}"] + problems
        )

    _test.__name__ = f"test_golden_{case['id']}"
    return _test


# pytest 用にケースごとのテスト関数を動的生成
for _case in CASES:
    globals()[f"test_golden_{_case['id']}"] = _make_test(_case)


if __name__ == "__main__":
    passed = failed = 0
    for case in CASES:
        problems = _run_case(case)
        if problems:
            failed += 1
            print(f"FAIL [{case['id']}]")
            for p in problems:
                print(f"     {p}")
        else:
            passed += 1
            print(f"PASS [{case['id']}]")
    print(f"\n=== golden: {passed} passed, {failed} failed / {len(CASES)} ===")
    raise SystemExit(1 if failed else 0)
