"""委譲shim: 本番 `opcg_sim.src.learned.mcts`（make/unmake版 TreeMCTS＝唯一の正）へ委譲する。

旧clone版は削除済み（make/unmake に一本化・2026-07）。tests/harness 側の bare import
（`import az_mcts_tree` / `from az_mcts_tree import TreeMCTS`）は無変更で動く
（sys.modules 差し替えは import 完了後の再取得で解決される）。
"""
import sys
from opcg_sim.src.learned import mcts as _m
sys.modules[__name__] = _m
