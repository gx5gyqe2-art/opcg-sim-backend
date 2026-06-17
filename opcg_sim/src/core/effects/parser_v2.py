"""EffectParserV2 — 合成ルールレジストリ方式の新パーサ（段階的移行版）。

設計（改善策⑥）:
  - 「いつ／コスト／条件／選択肢／逐次」といった *構造分解* は、既に十分機能している
    レガシー parser.py を再利用する（車輪の再発明を避ける）。
  - 刷新の主眼である *原子句の解釈* のみをルールレジストリに置き換える。
  - レジストリで未対応の句はレガシーの `_parse_atomic_action` にフォールバックし、
    その句を `self.unmatched` に記録する。これにより:
      * 本番は決して壊れない（常にレガシーが受け止める）
      * 未対応の表現が定量的に可視化され、ルール追加の TODO になる
      * ルールを足すたびにフォールバック率が下がる（burn down）

  インターフェースはレガシーと同一（parse_card_text / parse_ability）なので、
  loader 側の差し替えは1行で済み、resolver / gamestate は無改修。
"""
from __future__ import annotations

import re
from typing import List, Optional

from ...models.effect_types import EffectNode, GameAction, Sequence, _nfc
from ...models.enums import ActionType
from .parser import EffectParser
from .rules import ParseContext, RuleRegistry, default_registry


# 「このターン終了時、〜」「ターン終了時に〜」= 遅延実行（ターン終了フックで解決）。
# 「ターン終了時まで」は期間（duration）であって遅延ではないため除外する。
_DELAY_TURN_END_RE = re.compile(_nfc(r"ターン終了時(?!まで)[、にはのでも]"))


def _mark_delay(node: EffectNode, delay: str) -> None:
    """ノード木中の GameAction に遅延マーカーを付与する。"""
    if isinstance(node, GameAction):
        node.delay = delay
    elif isinstance(node, Sequence):
        for a in node.actions:
            _mark_delay(a, delay)


class EffectParserV2(EffectParser):
    def __init__(self, registry: Optional[RuleRegistry] = None) -> None:
        super().__init__()
        self.registry = registry or default_registry
        # 解析中に「ルール未対応でレガシーに落ちた」原子句を蓄積する。
        self.unmatched: List[str] = []
        self.rule_hits: List[str] = []
        # フォールバックの結果 ActionType.OTHER になった（=実行時に何もしない）原子句。
        # 「効果が動かない」問題の直接の標的リスト。
        self.fallback_other: List[str] = []

    def _parse_atomic_action(self, text: str, is_cost: bool):
        """原子句の解釈をレジストリ優先に置き換える。

        ルールが一致すればその結果を、なければレガシー実装にフォールバックする。
        """
        ctx = ParseContext(text=text, is_cost=is_cost)
        delayed = bool(_DELAY_TURN_END_RE.search(_nfc(text)))
        result = self.registry.apply(ctx)
        if result is not None:
            self.rule_hits.append(result.rule_name)
            if delayed:
                _mark_delay(result.node, "TURN_END")
            return result.node

        # フォールバック（=未対応として記録）
        self.unmatched.append(ctx.text)
        node = super()._parse_atomic_action(text, is_cost)
        if delayed and node is not None:
            _mark_delay(node, "TURN_END")
        if isinstance(node, GameAction) and node.type == ActionType.OTHER:
            self.fallback_other.append(ctx.text)
        return node

    def reset_stats(self) -> None:
        self.unmatched.clear()
        self.rule_hits.clear()
        self.fallback_other.clear()
