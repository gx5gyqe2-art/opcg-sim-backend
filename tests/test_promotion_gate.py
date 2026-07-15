"""昇格ゲート（v6 柱①・`tests/scripts/promotion_gate.py`）の判定ロジックの高速単体テスト。

実対局 arena は重いので回さない＝**段階式判定の純関数**（stage1_decision / final_decision）だけを
固定する（`test_perf_gate.py` と同じ思想）。しきい値の意味:
  - stage1: 勝ち越し（>50%）で継続・五分以下は棄却（少局数の粗いふるい）
  - final : 累計勝率 ≥ 0.55 で昇格（AlphaZero evaluator 水準・61/100 まで要求しない）
"""
import conftest  # noqa: F401  (google スタブ注入 & sys.path 設定)
import pytest

import promotion_gate as PG

pytestmark = pytest.mark.cpu_infra


def test_stage1_majority_continues():
    """24局で勝ち越し（13勝〜）のみ stage2 へ進む。"""
    assert PG.stage1_decision(13, 24) == "continue"
    assert PG.stage1_decision(24, 24) == "continue"


def test_stage1_even_or_worse_rejects():
    """五分（12/24）以下は即棄却＝壊れかけの candidate に100局を使わない。"""
    assert PG.stage1_decision(12, 24) == "reject"
    assert PG.stage1_decision(11, 24) == "reject"
    assert PG.stage1_decision(0, 24) == "reject"


def test_final_threshold_at_55pct():
    """累計 55/100 ちょうどで昇格・54 は棄却（境界は昇格側に含む）。"""
    assert PG.final_decision(55, 100)
    assert not PG.final_decision(54, 100)


def test_final_respects_custom_frac():
    """frac を変えるとしきい値が追従する（例: 0.60 なら 60/100）。"""
    assert PG.final_decision(60, 100, frac=0.60)
    assert not PG.final_decision(59, 100, frac=0.60)


def test_final_float_boundary_is_stable():
    """浮動小数の境界（0.55*偶数局）で棄却側に誤爆しない。"""
    for games in (20, 40, 60, 80, 100, 124):
        need = int(games * PG.STAGE2_FRAC + 0.5 - 1e-9)
        # need 勝ちちょうどが「wr >= frac」を満たすときは必ず昇格側
        if need / games >= PG.STAGE2_FRAC - 1e-9:
            assert PG.final_decision(need, games), (need, games)
