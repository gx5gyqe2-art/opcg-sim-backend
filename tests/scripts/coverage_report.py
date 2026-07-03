"""
カバレッジ確認スクリプト。

全カード(opcg_cards.json)に parser.py を適用し、TriggerType.UNKNOWN 数を集計する。

実行: python tests/coverage_report.py
"""
import json
import os

import os as _os, sys as _sys  # noqa: E402  test bootstrap (sys.path + google stub)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap  # noqa: E402,F401

from opcg_sim.src.core.effects.parser import EffectParser
from opcg_sim.src.models.enums import TriggerType
from opcg_sim.src.models.effect_types import Ability

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "opcg_sim", "data")


def report_parser_coverage():
    path = os.path.join(DATA_DIR, "opcg_cards.json")
    with open(path, "r", encoding="utf-8") as f:
        cards = json.load(f)

    parser = EffectParser()
    total = with_text = empty_parse = unknown = ok = 0
    for c in cards:
        text = c.get("効果(テキスト)") or c.get("効果(テキスト)") or ""
        total += 1
        if not text or text.strip() in ("なし", "None", ""):
            continue
        with_text += 1
        try:
            abilities = parser.parse_card_text(text)
        except Exception:
            empty_parse += 1
            continue
        if not abilities:
            empty_parse += 1
            continue
        if any(a.trigger == TriggerType.UNKNOWN for a in abilities):
            unknown += 1
        else:
            ok += 1

    print("=== parser.py カバレッジ (opcg_cards.json) ===")
    print(f"総カード数        : {total}")
    print(f"効果テキスト有り  : {with_text}")
    print(f"パース結果0件     : {empty_parse}")
    print(f"UNKNOWN trigger   : {unknown}")
    print(f"全trigger判定OK   : {ok}")
    if with_text:
        print(f"trigger判定率     : {ok / with_text * 100:.1f}%")
    print()
    return {"total": total, "with_text": with_text, "unknown": unknown, "ok": ok}


if __name__ == "__main__":
    report_parser_coverage()
