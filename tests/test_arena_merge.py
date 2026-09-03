"""分散アリーナ台帳マージの集計規約テスト（基盤健全性）。

なぜ基盤健全性か: 対象は学習パイプラインの計測集計であり、ゲームプレイの正しさには触れない
（誤った効果解決・カード消失・API契約破壊は検出しない）。したがって `cpu_infra`。

守るべき性質は3つ——(1) シャードを跨いだ合算が単一台帳の判定と一致する、(2) void を母数から
外して件数を残す、(3) **seed 衝突を黙って畳まない**（二重計上は CI を不当に狭める）。
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401
sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))

import arena_merge  # noqa: E402

pytestmark = pytest.mark.cpu_infra


def _ledger(tmp_path, name, rows):
    p = tmp_path / name
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return str(p)


def test_shards_merge_equals_single_ledger(tmp_path):
    """同じペア集合なら「2シャードに割った合算」＝「1台帳」（判定も CI も一致）。"""
    rows = [{"seed": 1000 + i, "score": (i % 3) / 2.0 * 2} for i in range(12)]
    single = _ledger(tmp_path, "all.jsonl", rows)
    a = _ledger(tmp_path, "a.jsonl", rows[:6])
    b = _ledger(tmp_path, "b.jsonl", rows[6:])

    one = arena_merge.summarize(arena_merge.read_ledger(single))
    merged, dups, _ = arena_merge.merge_ledgers([a, b])
    two = arena_merge.summarize(merged)
    assert dups == []
    assert one["pairs"] == two["pairs"] == 12
    assert one["wr"] == two["wr"] and one["ci95"] == two["ci95"]


def test_void_excluded_from_denominator_but_counted(tmp_path):
    """void は勝率の母数から外し、件数は必ず残す（落としたぶんを隠さない）。"""
    p = _ledger(tmp_path, "v.jsonl", [{"seed": 1, "score": 2.0}, {"seed": 2, "score": 2.0},
                                      {"seed": 3, "score": None}])
    r = arena_merge.summarize(arena_merge.read_ledger(p))
    assert r["pairs"] == 2 and r["games"] == 4 and r["void"] == 1
    assert r["wr"] == 1.0


def test_duplicate_seeds_are_reported_not_silently_deduped(tmp_path):
    """同一 seed が複数シャードに現れたら衝突として報告する（帯設計のミスを検出）。"""
    a = _ledger(tmp_path, "a.jsonl", [{"seed": 7, "score": 2.0}, {"seed": 8, "score": 0.0}])
    b = _ledger(tmp_path, "b.jsonl", [{"seed": 8, "score": 0.0}, {"seed": 9, "score": 1.0}])
    merged, dups, _ = arena_merge.merge_ledgers([a, b])
    assert dups == [8]
    assert len(merged) == 3


def test_promoted_requires_both_frac_and_ci_lower_bound(tmp_path):
    """promoted は wr≥frac かつ CI下限>0.50（arena_resume と同規約）。"""
    # 全勝ペア6本＝wr 1.0・分散0 → 昇格
    win = _ledger(tmp_path, "w.jsonl", [{"seed": i, "score": 2.0} for i in range(6)])
    assert arena_merge.summarize(arena_merge.read_ledger(win))["promoted"] is True
    # 引き分けばかり＝wr 0.5 → 非昇格
    draw = _ledger(tmp_path, "d.jsonl", [{"seed": i, "score": 1.0} for i in range(6)])
    assert arena_merge.summarize(arena_merge.read_ledger(draw))["promoted"] is False
