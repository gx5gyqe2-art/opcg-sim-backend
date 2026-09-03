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


def test_anchor_requires_non_regression():
    """アンカー判定（v7・血統過適合の検出）: 固定アンカーに勝率 ≥ 0.5 で OK・未満は NG。

    実測根拠: v6 で対best 連鎖3段昇格の r99 が対gen5 直接 8/24（0.333）＝この判定が
    あれば弾けていた。五分（12/24）は非退行として許容（アンカー超えまでは要求しない）。"""
    assert PG.anchor_decision(12, 24)          # 五分＝非退行 OK
    assert PG.anchor_decision(13, 24)
    assert not PG.anchor_decision(11, 24)      # 負け越し NG
    assert not PG.anchor_decision(8, 24)       # r99 実測ケース
    assert not PG.anchor_decision(14, 24, frac=0.6)   # 14/24=0.583 < 0.6
    assert PG.anchor_decision(14, 24, frac=0.55)      # 0.583 ≥ 0.55（frac 可変）


def test_init_pool_applies_engine_options_to_candidate_only(monkeypatch):
    """席別 seam（v35）: `cand_kw` は候補席のエンジンにだけ渡る（best は既定のまま）。

    機構（戦闘窓の箱読み出し・静止探索）をグローバル定数で切り替えると**両席に同時に効き**、
    「新機構つき候補 vs 現行本番」を測れない。arena がこの seam を通していることを固定する。
    """
    import sys
    import types

    seen = []

    class _FakeEngine:
        def __init__(self, value_path=None, policy_path=None, **kw):
            seen.append({"value_path": value_path, **kw})

    fake_arena = types.ModuleType("cpu_arena")
    fake_arena._load_db = lambda: "db"
    monkeypatch.setitem(sys.modules, "cpu_arena", fake_arena)
    import opcg_sim.src.core.cpu_learned as CL
    monkeypatch.setattr(CL, "LearnedEngine", _FakeEngine)

    PG._init_pool("cand_value.npz", "", {"box_battle": True, "quiesce": True})
    assert seen[0]["value_path"] == "cand_value.npz"
    assert seen[0]["box_battle"] is True and seen[0]["quiesce"] is True
    assert "box_battle" not in seen[1] and "quiesce" not in seen[1], \
        "基準席にも機構が渡っている（A/B が成立しない）"


def test_init_pool_without_options_is_unchanged():
    """`cand_kw` 未指定は従来と同一＝既存の呼び出し側を壊さない（後方互換）。"""
    import inspect
    sig = inspect.signature(PG._init_pool)
    assert sig.parameters["cand_kw"].default is None
