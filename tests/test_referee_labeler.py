"""レフェリー再ラベルの純ロジック（v9 フェーズ1・`referee_labeler.py`）。

  1. 採掘候補の選抜（`select_candidates`）: 飽和負け（捲りラベル）優先・同一連鎖の間引き・上限
  2. policy 教師の構築（`plan_teacher_visit`）: 同価値バンド上位プランの初手 multi-hot＝
     バンド外プランの初手は 0（劣る手を明示的に教える）・合法手に初手が無ければ None
     （黙って誤教師を作らない）
生成・ラベルの重い経路は回さない（純関数のみ＝高速）。基盤健全性＝cpu_infra。
"""
import os
import sys

import conftest  # noqa: F401  (google スタブ注入 & sys.path 設定)
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "tests", "scripts"))
from referee_labeler import plan_teacher_visit, select_candidates  # noqa: E402

pytestmark = pytest.mark.cpu_infra


def test_select_candidates_sat_first_and_dedupe():
    """飽和負けが効率盲点より優先・隣接 index（差<2）は間引き・上限で切る。"""
    cands = [(10, "blind", 0.02), (40, "sat", -0.95), (41, "sat", -0.9),
             (70, "blind", 0.01), (90, "sat", -0.85)]
    picked = select_candidates(cands, 3)
    idxs = [c[0] for c in picked]
    assert 40 in idxs and 41 not in idxs, "隣接する飽和候補が間引かれていない"
    assert 90 in idxs
    assert len(picked) == 3
    assert 70 in idxs, "残り枠は spread 最小の効率盲点で埋める"


def test_select_candidates_cap_and_order():
    """返り値は index 昇順・max_per_game を超えない。"""
    cands = [(i, "blind", 0.01 * i) for i in range(0, 40, 5)]
    picked = select_candidates(cands, 4)
    assert len(picked) == 4
    assert [c[0] for c in picked] == sorted(c[0] for c in picked)


def _entry(first_key, outcomes, lifem):
    return {"keys": [first_key, ("PASS",)], "outcomes": dict(enumerate(outcomes)),
            "lifem": lifem, "wins": float(sum(outcomes)), "ok": len(outcomes)}


def test_plan_teacher_visit_band_multi_hot():
    """バンド上位2プラン（初手A/B）→ A,B に均等重み・バンド外 C は 0。"""
    A, B, C = ("ATTACK", "x"), ("ATTACH_DON", "y"), ("TURN_END", None)
    entries = [_entry(A, [1, 1, 1, 0], 0.5),      # best
               _entry(B, [1, 1, 0, 1], 0.4),      # 正味不一致1＝同価値
               _entry(C, [0, 0, 0, 0], -2.0)]     # 正味3＝バンド外
    legal_keys = [C, A, B]
    visit = plan_teacher_visit(legal_keys, entries, band=0.5)
    assert visit is not None
    assert visit[0] == 0.0
    assert visit[1] == pytest.approx(0.5) and visit[2] == pytest.approx(0.5)
    assert visit.sum() == pytest.approx(1.0)


def test_plan_teacher_visit_dedup_same_first_move():
    """同じ初手のプランが複数バンド内でも合法手側は1本に集約される（重み合算）。"""
    A = ("ATTACK", "x")
    entries = [_entry(A, [1, 1, 1, 1], 1.0), _entry(A, [1, 1, 1, 1], 0.9)]
    visit = plan_teacher_visit([A, ("TURN_END", None)], entries, band=0.5)
    assert visit[0] == pytest.approx(1.0) and visit[1] == 0.0


def test_plan_teacher_visit_unmatched_returns_none():
    """バンド上位の初手が合法手キーに見つからない → None（誤教師を作らない）。"""
    entries = [_entry(("ATTACK", "x"), [1, 1, 1, 1], 1.0)]
    assert plan_teacher_visit([("TURN_END", None)], entries, band=0.5) is None
