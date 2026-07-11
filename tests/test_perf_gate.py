"""CPU 性能ゲート（perf_gate.py・perf計画 A2）の判定ロジックの高速単体テスト。

実対局（learned arena）は重いので CI では回さない＝**判定関数 `evaluate_gate` の純ロジック**と
モデルハッシュの健全性だけを高速に固定する。実強度の本走は `perf_gate.py --full` の手動/定期運用。
"""
import conftest  # noqa: F401  (google スタブ注入 & sys.path 設定)
import perf_gate as PG


def _lat(median=300.0, mx=800.0):
    return {"median_ms": median, "max_ms": mx, "n": 24.0}


def _ci(elo_lo=200.0, elo=400.0, elo_hi=600.0, pairs=20):
    return {"win_rate": 0.9, "elo": elo, "elo_lo": elo_lo, "elo_hi": elo_hi, "pairs": pairs}


def test_pass_when_strong_fast_and_no_failures():
    v = PG.evaluate_gate(_ci(), _lat(), failed_games=0, min_elo_lo=0.0, max_latency_ms=1200.0)
    assert v["passed"] and v["reasons"] == []


def test_fail_when_not_significantly_stronger():
    """elo_lo が閾値以下＝hard に有意勝ちできていない（退行疑い）で FAIL。"""
    v = PG.evaluate_gate(_ci(elo_lo=-5.0), _lat(), 0, min_elo_lo=0.0, max_latency_ms=1200.0)
    assert not v["passed"]
    assert any("強度不足" in r for r in v["reasons"])


def test_fail_when_latency_over_budget():
    v = PG.evaluate_gate(_ci(), _lat(median=1500.0), 0, min_elo_lo=0.0, max_latency_ms=1200.0)
    assert not v["passed"]
    assert any("レイテンシ超過" in r for r in v["reasons"])


def test_fail_when_games_failed():
    v = PG.evaluate_gate(_ci(), _lat(), failed_games=3, min_elo_lo=0.0, max_latency_ms=1200.0)
    assert not v["passed"]
    assert any("失敗局" in r for r in v["reasons"])


def test_fail_when_no_valid_pairs():
    assert not PG.evaluate_gate(None, _lat(), 0, 0.0, 1200.0)["passed"]
    assert not PG.evaluate_gate(_ci(pairs=0), _lat(), 0, 0.0, 1200.0)["passed"]


def test_multiple_reasons_accumulate():
    v = PG.evaluate_gate(_ci(elo_lo=-10.0), _lat(median=2000.0), failed_games=2,
                         min_elo_lo=0.0, max_latency_ms=1200.0)
    assert not v["passed"] and len(v["reasons"]) == 3


def test_model_hash_is_stable_and_present():
    """gen2/gen3 の npz が同梱され、ハッシュが安定（2回一致）。gen3=本番既定（2026-07-11採用）。"""
    h1, h2 = PG.model_hash(), PG.model_hash()
    assert h1 == h2
    assert h1.get("gen2_value.npz") and h1["gen2_value.npz"] != "<missing>"
    assert h1.get("gen3_value.npz") and h1["gen3_value.npz"] != "<missing>"
    assert h1.get("gen3_policy.npz") and h1["gen3_policy.npz"] != "<missing>"
