"""効果パース診断ツール（改善策②の可視化基盤）。

全カード(opcg_cards.json)に新パーサ(EffectParserV2)を適用し、以下を集計する:
  1. ルール命中率 vs レガシーフォールバック率（=ルールレジストリのカバレッジ）
  2. 未対応(フォールバック)原子句の頻度ランキング → 次に作るルールの優先順位
  3. 生成された AST 中の ActionType.OTHER 数（=「解析できたが何もしない」サイレント失敗）

実行:
    python tests/effect_diagnostics.py            # サマリのみ
    python tests/effect_diagnostics.py --top 40   # 未対応句ランキングを40件表示
"""
import json
import os
import sys
from collections import Counter

import conftest  # noqa: F401  (google スタブ注入 & sys.path 設定)

from opcg_sim.src.core.effects.parser_v2 import EffectParserV2
from opcg_sim.src.models.effect_types import Branch, Choice, GameAction, Sequence

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "opcg_sim", "data"
)


def _walk_actions(node):
    """AST を走査して全 GameAction を yield する。"""
    if node is None:
        return
    if isinstance(node, GameAction):
        yield node
    elif isinstance(node, Sequence):
        for a in node.actions:
            yield from _walk_actions(a)
    elif isinstance(node, Branch):
        yield from _walk_actions(node.if_true)
        yield from _walk_actions(node.if_false)
    elif isinstance(node, Choice):
        for o in node.options:
            yield from _walk_actions(o)


def run(top: int = 25):
    path = os.path.join(DATA_DIR, "opcg_cards.json")
    with open(path, "r", encoding="utf-8") as f:
        cards = json.load(f)

    parser = EffectParserV2()
    action_counter = Counter()
    other_count = 0
    cards_with_text = 0
    cards_with_other = 0

    for c in cards:
        text = c.get("効果(テキスト)") or ""
        trig = c.get("効果(トリガー)") or ""
        if (not text or text.strip() in ("なし", "None", "")) and not trig:
            continue
        cards_with_text += 1
        abilities = []
        if text:
            abilities += parser.parse_card_text(text)
        if trig:
            abilities += parser.parse_card_text(trig, as_trigger=True)

        had_other = False
        for ab in abilities:
            for action in list(_walk_actions(ab.effect)) + list(_walk_actions(ab.cost)):
                action_counter[action.type.name] += 1
                if action.type.name == "OTHER":
                    other_count += 1
                    had_other = True
        if had_other:
            cards_with_other += 1

    total_atomic = len(parser.rule_hits) + len(parser.unmatched)
    hit_rate = (len(parser.rule_hits) / total_atomic * 100) if total_atomic else 0.0

    print("=== EffectParserV2 診断 (opcg_cards.json) ===")
    print(f"効果テキスト有りカード : {cards_with_text}")
    print(f"原子句 総数            : {total_atomic}")
    print(f"  ルール命中           : {len(parser.rule_hits)} ({hit_rate:.1f}%)")
    print(f"  レガシーフォールバック : {len(parser.unmatched)} ({100 - hit_rate:.1f}%)")
    print(f"ActionType.OTHER 数    : {other_count}  (サイレント失敗の疑い)")
    print(f"  OTHER を含むカード   : {cards_with_other}")
    print()

    print("--- ルール命中 内訳 ---")
    for name, cnt in Counter(parser.rule_hits).most_common():
        print(f"  {cnt:5d}  {name}")
    print()

    print("--- 生成 ActionType 分布 (上位) ---")
    for name, cnt in action_counter.most_common(15):
        print(f"  {cnt:5d}  {name}")
    print()

    print(f"--- 未対応(フォールバック)原子句 ランキング 上位{top} ---")
    print("   （頻度の高い表現からルール化すると効率的）")
    for clause, cnt in Counter(parser.unmatched).most_common(top):
        snippet = clause[:60].replace("\n", " ")
        print(f"  {cnt:4d}  {snippet}")


if __name__ == "__main__":
    top = 25
    if "--top" in sys.argv:
        try:
            top = int(sys.argv[sys.argv.index("--top") + 1])
        except (ValueError, IndexError):
            pass
    run(top)
