"""
カバレッジ確認スクリプト。

1. 全カード(opcg_cards.json)に parser.py を適用し、TriggerType.UNKNOWN 数を集計
2. generated_effects.json の from_dict デシリアライズ成否を集計

実行: python tests/coverage_report.py
"""
import json
import os

import conftest  # noqa: F401  (google スタブ注入 & sys.path 設定)

from opcg_sim.src.core.effects.parser import EffectParser
from opcg_sim.src.models.enums import TriggerType
from opcg_sim.src.models.effect_types import Ability

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "opcg_sim", "data")


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


def report_generated_effects():
    path = os.path.join(DATA_DIR, "generated_effects.json")
    if not os.path.exists(path):
        print("generated_effects.json が見つかりません")
        return
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cards = 0
    eff_total = eff_ok = eff_fail = 0
    fail_samples = []
    for card_id, effects in data.items():
        if not isinstance(effects, list):
            continue
        cards += 1
        for eff in effects:
            eff_total += 1
            try:
                Ability.from_dict(eff)
                eff_ok += 1
            except Exception as e:
                eff_fail += 1
                if len(fail_samples) < 10:
                    fail_samples.append(f"{card_id}: {e}")

    print("=== generated_effects.json デシリアライズ ===")
    print(f"カード数          : {cards}")
    print(f"効果エントリ総数  : {eff_total}")
    print(f"from_dict 成功    : {eff_ok}")
    print(f"from_dict 失敗    : {eff_fail}")
    if eff_total:
        print(f"成功率            : {eff_ok / eff_total * 100:.1f}%")
    if fail_samples:
        print("失敗サンプル:")
        for s in fail_samples:
            print(f"  - {s}")
    print()
    return {"eff_total": eff_total, "eff_ok": eff_ok, "eff_fail": eff_fail}


if __name__ == "__main__":
    report_parser_coverage()
    report_generated_effects()
