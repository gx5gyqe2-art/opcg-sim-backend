"""再開可能アリーナ（`tests/scripts/arena_resume.py`）の純関数テスト。

実対局は回さない。固定する性質:
  - 台帳の読み戻し＋残り seed 抽出＝再開が計画順・重複なしで進む（10分制限下の分割実行の土台）
  - 最終判定は全ペア消化後にのみ出る（部分結果で promoted を出さない）
  - 判定規約（ペア水準0/0.5/1・wr≥frac かつ CI下限>0.50）が arena_gate と同値
"""
import json

import conftest  # noqa: F401  (google スタブ注入 & sys.path 設定)
import pytest

import arena_resume as AR

pytestmark = pytest.mark.cpu_infra


def test_ledger_roundtrip_and_remaining(tmp_path):
    p = tmp_path / "led.jsonl"
    p.write_text(json.dumps({"seed": 71000, "score": 2.0}) + "\n"
                 + json.dumps({"seed": 71002, "score": 1.0}) + "\n")
    done = AR.load_ledger(str(p))
    assert done == {71000: 2.0, 71002: 1.0}
    assert AR.remaining_seeds([71000, 71001, 71002, 71003], done) == [71001, 71003]


def test_final_is_none_until_complete():
    assert AR.final_result([1, 2], {1: 2.0}) is None


def test_final_decision_matches_arena_gate_rule():
    """全ペア candidate 2勝 → wr=1.0・CI下限>0.5 → promoted。全ペア五分（1.0）→ wr=0.5 で否。
    正規化（勝ち数0..2→0/0.5/1）を経ることも同時に固定する（2026-07-27 ヌル対照の教訓）。"""
    planned = list(range(30))
    win_all = AR.final_result(planned, {s: 2.0 for s in planned})
    assert win_all["promoted"] and win_all["wr"] == pytest.approx(1.0)
    even = AR.final_result(planned, {s: 1.0 for s in planned})
    assert not even["promoted"] and even["wr"] == pytest.approx(0.5)
