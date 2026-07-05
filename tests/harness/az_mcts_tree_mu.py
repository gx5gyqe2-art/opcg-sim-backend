"""委譲shim: 本番 `opcg_sim.src.learned.mcts_mu` を単一の正とする（clone版 az_mcts_tree と対）。

make/unmake版MCTS の実体は本番パッケージへ移動済み。harness の bare import
（`from az_mcts_tree_mu import TreeMCTSMakeUnmake`）は無変更で動く。
"""
import sys
from opcg_sim.src.learned import mcts_mu as _m
sys.modules[__name__] = _m
