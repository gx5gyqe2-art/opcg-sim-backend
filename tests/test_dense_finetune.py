"""密ラベル追い学習（v16・`tests/scripts/dense_finetune.py`）のラベル生成の純関数テスト。

実学習は回さない。この計器の存在理由は「gen2→gen5 を支えた密度レジームの学習仕様
（混合ラベル y=α·z+(1−α)·q_root ＋残りターン補助）を gen6 以降で再現していない」ことなので、
固定する性質はラベルが**その仕様どおりに作られる**ことと、q_root 欠損時に安全側
（勝敗単独）へ退化することの2点。
"""
import conftest  # noqa: F401  (google スタブ注入 & sys.path 設定)
import numpy as np
import pytest

import dense_finetune as DF

pytestmark = pytest.mark.cpu_infra


def _vdata(value, q_root, turns_left):
    return {"value": np.array(value, np.float32), "q_root": np.array(q_root, np.float32),
            "turns_left": np.array(turns_left, np.float32)}


def test_label_is_alpha_blend_of_outcome_and_q_root():
    y, _ = DF.build_labels(_vdata([1.0, -1.0], [0.0, 0.0], [4.0, 4.0]), alpha=0.5)
    assert y == pytest.approx([0.5, -0.5])


def test_alpha_one_degenerates_to_outcome_only():
    """α=1 でレフェリー教師と同じ扱い（勝敗単独）＝仕様の両端が地続きであること。"""
    y, _ = DF.build_labels(_vdata([1.0, -1.0], [0.2, 0.9], [4.0, 4.0]), alpha=1.0)
    assert y == pytest.approx([1.0, -1.0])


def test_non_finite_q_root_falls_back_to_outcome():
    """q_root=NaN の行（L1 席など探索 root を持たない生成）は勝敗単独へ退化する
    ＝NaN がラベルへ伝播して学習を壊さない。"""
    y, _ = DF.build_labels(_vdata([1.0, -1.0], [np.nan, np.inf], [4.0, 4.0]), alpha=0.5)
    assert np.isfinite(y).all() and y == pytest.approx([1.0, -1.0])


def test_aux_is_clipped_and_normalized():
    """aux は [0,1] へ正規化され、スケール超過は 1.0 に飽和する。NaN は欠損のまま通す
    （`ValueNet.backward` 側が欠損として補助損失から除外する契約）。"""
    _, aux = DF.build_labels(_vdata([0.0] * 3, [0.0] * 3, [0.0, 8.0, np.nan]), turns_scale=8.0)
    assert aux[0] == pytest.approx(0.0) and aux[1] == pytest.approx(1.0)
    assert np.isnan(aux[2])
