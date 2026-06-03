"""合成ルールレジストリ（パーサ刷新の中核パッケージ）。

`from ...rules import default_registry, ParseContext` のように使う。
atoms をインポートすることで、シードルールが default_registry に登録される。
"""
from .base import (
    MatchResult,
    ParseContext,
    Rule,
    RuleRegistry,
    default_registry,
    rule,
)

# 副作用としてシードルールを default_registry に登録する。
from . import atoms  # noqa: E402,F401

__all__ = [
    "MatchResult",
    "ParseContext",
    "Rule",
    "RuleRegistry",
    "default_registry",
    "rule",
    "atoms",
]
