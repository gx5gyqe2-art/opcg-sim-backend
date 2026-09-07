"""捲りモードの相手不完全性モデル（v8・`counterfactual_referee._sample_by_visits`）。

飽和負け局面（相手も temp0 なら捲り率が構造的に 0）で「相手のミスと引きの偏りを
どう最大化するか＝捲り率」を測るための、相手手番の訪問数比例サンプル（p ∝ N^(1/τ)）:
  - 低温 τ→0 は argmax（temp0 と一致＝連続性）
  - τ>0 は訪問数の多い手ほど高頻度・固定 rng で決定論（CRN の再現性を壊さない）
  - 訪問ゼロ/1手のみの縮退ケース
基盤健全性＝cpu_infra。純関数のみ＝高速。
"""
import os
import sys
from collections import Counter

import conftest  # noqa: F401  (google スタブ注入 & sys.path 設定)
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "tests", "scripts"))
from counterfactual_referee import _sample_by_visits  # noqa: E402

pytestmark = pytest.mark.cpu_infra

LEGAL = ["a", "b", "c"]


def test_low_temp_is_argmax():
    """τ→0 は最多訪問手を確定的に選ぶ（temp0 との連続性）。"""
    rng = np.random.default_rng(0)
    for _ in range(20):
        assert _sample_by_visits(LEGAL, [1, 50, 3], 1e-6, rng) == "b"


def test_temp_samples_proportionally_and_deterministic():
    """τ=1 は訪問数比例（多い手が多く出る）・同じ seed なら同じ列（CRN 再現性）。"""
    rng = np.random.default_rng(7)   # 1つの rng を共有（毎回作り直すと同じ初回ドローになる）
    c = Counter(_sample_by_visits(LEGAL, [10, 80, 10], 1.0, rng) for _ in range(200))
    assert c["b"] > c["a"] and c["b"] > c["c"]
    assert c["a"] > 0 or c["c"] > 0, "温度があるのに argmax 以外が一度も出ない"
    r1, r2 = np.random.default_rng(7), np.random.default_rng(7)
    s1 = [_sample_by_visits(LEGAL, [10, 80, 10], 1.0, r1) for _ in range(50)]
    s2 = [_sample_by_visits(LEGAL, [10, 80, 10], 1.0, r2) for _ in range(50)]
    assert s1 == s2


def test_degenerate_cases():
    """訪問ゼロ→先頭・1手のみ→その手・空→None。"""
    rng = np.random.default_rng(0)
    assert _sample_by_visits(LEGAL, [0, 0, 0], 0.7, rng) == "a"
    assert _sample_by_visits(["x"], [5], 0.7, rng) == "x"
    assert _sample_by_visits([], [], 0.7, rng) is None
