"""GATE B end-to-end（slow・CI除外）: OPCG で探索が健全＝more search = stronger。

評価器を固定し sims だけ増やすと勝率が明確に上がることを確認（不成立＝探索/PIMC統合の不具合）。
重い（数分）ため slow。アダプタ/MCTSの高速検証は test_opcg_adapter / test_az_mcts_tree（CI内）。
実行: OPCG_LOG_SILENT=1 PYTHONPATH=tests python -m pytest tests/test_gate_b.py -q -s -m slow
"""
import pytest

import conftest  # noqa: F401
from opcg_game import OPCGGame
from gate_b_opcg import run_monotonicity
from cpu_selfplay import _load_db


@pytest.mark.slow
def test_gate_b_search_monotonicity():
    game = OPCGGame()
    db = _load_db()
    ok, rates, _ = run_monotonicity(game, db, pairs=6, base=20, levels=(20, 180))
    assert rates[-1] > rates[0], f"探索を増やしても強くならない（rates={rates}）"
    assert ok, f"探索健全トレンド未達（rates={rates}）"
