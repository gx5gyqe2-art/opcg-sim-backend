"""GATE A の end-to-end 検証（slow・CIから除外＝-m "not slow"）。

AZループ機械が三目並べで既知最適へ収束＝RLループ実装が健全、を seed 固定で再現的に検証。
重い（数分）ため slow マーカ。部品の高速検証は test_az_components.py（CI内）が担う。
実行: OPCG_LOG_SILENT=1 PYTHONPATH=tests python -m pytest tests/test_gate_a.py -q -s -m slow
"""
import pytest

import conftest  # noqa: F401
from gate_a_tictactoe import run_gate


@pytest.mark.slow
def test_gate_a_converges_to_optimal():
    ok, summary = run_gate(gens=8, games=200, sims=60, eval_sims=100, seed=0)
    assert summary["g0_perf"]["a_loss"] > 0, "gen0 が既に最適＝改善幅を測れない（前提崩れ）"
    assert summary["vs_perf"]["a_loss"] == 0, f"完全プレイに敗北＝最適未収束: {summary['vs_perf']}"
    assert summary["vs_rand"]["a_loss"] == 0, f"ランダムに敗北: {summary['vs_rand']}"
    assert ok
