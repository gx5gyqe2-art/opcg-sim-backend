"""`kappa_vector.py`（T121）の集計の算術（順位の AUC）。段 7（2026-10-07）: `κ` の勾配・軸・腕の積み上げは Rust だけになり、
その式の試験は Python の写しと一緒に消した。
"""
import os
import sys

import pytest

pytestmark = pytest.mark.cpu_infra

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bootstrap  # noqa: F401,E402

_SCRIPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import kappa_vector as KV  # noqa: E402


def test_auc_is_the_rank_statistic_with_ties_at_one_half():
    assert KV.auc_of([1.0, 2.0, 3.0, 4.0], [0, 0, 1, 1]) == pytest.approx(1.0)
    assert KV.auc_of([4.0, 3.0, 2.0, 1.0], [0, 0, 1, 1]) == pytest.approx(0.0)
    assert KV.auc_of([1.0, 1.0, 1.0, 1.0], [0, 0, 1, 1]) == pytest.approx(0.5)


def test_auc_is_none_without_both_labels():
    assert KV.auc_of([1.0, 2.0], [1, 1]) is None
