"""レガシー parser.py と新 EffectParserV2 の出力差分を全カードで比較する。

V2 有効化前の安全確認用。V2 は「構造分解はレガシー流用 + 原子句だけルール化、
未対応はレガシーへフォールバック」なので、出力はレガシーと一致するか、
ルールが当たった箇所だけ変化する（基本は改善）はず。

分類:
  identical   : 完全一致
  improved    : レガシーに OTHER があり V2 で解消（改善）
  changed     : OTHER 以外の差異（要レビュー）
  regression  : レガシーに無かった OTHER が V2 で発生（あってはならない）

実行: OPCG_LOG_SILENT=1 python tests/compare_parsers.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os as _os, sys as _sys  # noqa: E402  test bootstrap (sys.path + google stub)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap  # noqa: E402,F401

from opcg_sim.src.core.effects.parser import EffectParser  # noqa: E402
from opcg_sim.src.core.effects.parser_v2 import EffectParserV2  # noqa: E402
from golden.summarize import summarize_ability  # noqa: E402

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "opcg_sim", "data"
)


def _summ_list(parser, text, as_trigger=False):
    return [summarize_ability(a) for a in parser.parse_card_text(text, as_trigger=as_trigger)]


def _has_other(summ_list):
    # summary には enum（CompareOperator 等）が混じることがあるため default=str で吸収する。
    blob = json.dumps(summ_list, ensure_ascii=False, default=str)
    return '"type": "OTHER"' in blob


def run(show=12):
    cards = json.load(open(os.path.join(DATA_DIR, "opcg_cards.json"), "r", encoding="utf-8"))
    legacy, v2 = EffectParser(), EffectParserV2()

    identical = improved = changed = regression = 0
    changed_samples, regression_samples = [], []

    for c in cards:
        text = c.get("効果(テキスト)") or ""
        trig = c.get("効果(トリガー)") or ""
        if (not text or text.strip() in ("なし", "None", "")) and not trig:
            continue
        l = _summ_list(legacy, text) + _summ_list(legacy, trig, True)
        v = _summ_list(v2, text) + _summ_list(v2, trig, True)
        if l == v:
            identical += 1
            continue
        l_other, v_other = _has_other(l), _has_other(v)
        if v_other and not l_other:
            regression += 1
            if len(regression_samples) < show:
                regression_samples.append((c.get("number"), text or trig))
        elif l_other and not v_other:
            improved += 1
        else:
            changed += 1
            if len(changed_samples) < show:
                changed_samples.append((c.get("number"), text or trig))

    total = identical + improved + changed + regression
    print("=== legacy vs V2 出力比較 ===")
    print(f"対象カード   : {total}")
    print(f"  完全一致   : {identical}")
    print(f"  改善(OTHER解消): {improved}")
    print(f"  その他差異(要確認): {changed}")
    print(f"  ★退行(新規OTHER): {regression}")
    print()
    if regression_samples:
        print("--- 退行サンプル ---")
        for cid, t in regression_samples:
            print(f"  {cid}: {(t or '')[:70]}")
        print()
    if changed_samples:
        print(f"--- その他差異サンプル（上位{show}） ---")
        for cid, t in changed_samples:
            print(f"  {cid}: {(t or '')[:70]}")
    return regression


if __name__ == "__main__":
    reg = run()
    # 退行が出たら非0で終了（CIで検知できるように）
    raise SystemExit(1 if reg else 0)
