"""委譲shim: 本番 `opcg_sim.src.learned.mcts` を単一の正とする（重複解消）。

旧はほぼ同一コピーだったが本番側のみ `last_stats`（decide_learned のトレース用・無害な
superset）を持ち、2コピー間の自動整合チェックが無くドリフトし始めていた。本番を正として
委譲する＝tests/harness 側の bare import（`import az_mcts_tree` / `from az_mcts_tree import
TreeMCTS`）は変えずに動く（sys.modules 差し替えは import 完了後の再取得で解決される）。
"""
import sys
from opcg_sim.src.learned import mcts as _m
sys.modules[__name__] = _m
