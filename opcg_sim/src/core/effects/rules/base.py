"""合成ルールレジストリの中核。

設計意図:
  従来の parser.py は `_detect_action_type()` の巨大な if 連鎖で原子句を解釈しており、
  「順序依存・サイレント失敗・テスト困難」という課題があった（改善策⑥）。

  本モジュールは原子句（=これ以上分割しない 1 アクション相当のテキスト）を
  「宣言的ルール（パターン→ASTビルダー）」の集合として表現し、優先度順に適用する。

  - 各ルールは独立しており単体テスト可能
  - どのルールも一致しなければ「未対応(unmatched)」として明示的に扱える（サイレント失敗の排除）
  - 新しい表現への対応はルールを1つ追加するだけ（コア無改修）
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from ....models.effect_types import EffectNode


def _nfc(text: str) -> str:
    if not text:
        return ""
    return unicodedata.normalize("NFC", text)


@dataclass
class ParseContext:
    """1つの原子句を解析するための入力。

    text          : 正規化済み（NFC）の原子句テキスト
    is_cost       : この句がコスト（コロンの左側）に由来するか
    parse_subclause: 入れ子の句を再帰解析するためのフック（任意）
    """

    text: str
    is_cost: bool = False
    parse_subclause: Optional[Callable[[str], Optional[EffectNode]]] = None

    def __post_init__(self) -> None:
        self.text = _nfc(self.text)


@dataclass
class MatchResult:
    node: EffectNode
    rule_name: str
    confidence: float = 1.0


class Rule:
    """宣言的解析ルールの基底クラス。

    サブクラス（または `@rule` デコレータ）で `name` / `priority` を定義し、
    `matches()` と `build()` を実装する。
    """

    name: str = "unnamed"
    priority: int = 0  # 数値が大きいほど先に試行される

    def matches(self, ctx: ParseContext) -> bool:  # pragma: no cover - interface
        raise NotImplementedError

    def build(self, ctx: ParseContext) -> Optional[EffectNode]:  # pragma: no cover
        raise NotImplementedError


class _FuncRule(Rule):
    """関数ベースのルール（`@rule` デコレータが生成する内部表現）。"""

    def __init__(
        self,
        name: str,
        priority: int,
        matches: Callable[[ParseContext], bool],
        build: Callable[[ParseContext], Optional[EffectNode]],
    ) -> None:
        self.name = name
        self.priority = priority
        self._matches = matches
        self._build = build

    def matches(self, ctx: ParseContext) -> bool:
        return self._matches(ctx)

    def build(self, ctx: ParseContext) -> Optional[EffectNode]:
        return self._build(ctx)


class RuleRegistry:
    """ルールを保持し、優先度順に適用するレジストリ。"""

    def __init__(self) -> None:
        self._rules: List[Rule] = []
        self._sorted = True

    def register(self, rule: Rule) -> Rule:
        self._rules.append(rule)
        self._sorted = False
        return rule

    def _ensure_sorted(self) -> None:
        if not self._sorted:
            # 優先度降順。同一優先度は登録順を維持（安定ソート）。
            self._rules.sort(key=lambda r: r.priority, reverse=True)
            self._sorted = True

    @property
    def rules(self) -> List[Rule]:
        self._ensure_sorted()
        return list(self._rules)

    def apply(self, ctx: ParseContext) -> Optional[MatchResult]:
        """最初に一致してノードを構築できたルールの結果を返す。

        一致するルールが無い、または build が None を返した場合は None。
        （None は「このレジストリでは未対応」を意味し、呼び出し側で
         フォールバックや未対応記録を行う。）
        """
        self._ensure_sorted()
        for rule in self._rules:
            try:
                if rule.matches(ctx):
                    node = rule.build(ctx)
                    if node is not None:
                        return MatchResult(node=node, rule_name=rule.name)
            except Exception:  # noqa: BLE001 - 1ルールの失敗で全体を止めない
                continue
        return None


# プロセス全体で共有する既定レジストリ。
default_registry = RuleRegistry()


def rule(name: str, priority: int = 0, registry: Optional[RuleRegistry] = None):
    """関数ベースでルールを登録するデコレータ。

    使い方::

        @rule("draw", priority=50)
        def _draw(ctx):
            if "引く" not in ctx.text:
                return None        # 不一致
            return GameAction(...)  # 一致して構築

    デコレート対象の関数は ctx を受け取り、
      - 不一致なら None
      - 一致したら EffectNode
    を返す。matches/build を兼ねる簡潔な形式。
    """

    target_registry = registry or default_registry

    def deco(fn: Callable[[ParseContext], Optional[EffectNode]]):
        holder = {"node": None}

        def _matches(ctx: ParseContext) -> bool:
            holder["node"] = fn(ctx)
            return holder["node"] is not None

        def _build(ctx: ParseContext) -> Optional[EffectNode]:
            return holder["node"]

        target_registry.register(
            _FuncRule(name=name, priority=priority, matches=_matches, build=_build)
        )
        return fn

    return deco
