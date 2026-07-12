"""v4 混合ラベル・batch スキーマ v2 の後方互換（pd_batch_common・docs/cpu_v4_plan.md §4-2）の単体検証。

learner が混ぜる純ロジック（normalize_batch_v2 / mixed_value_label / ring_append の v2 キー対応）を
git 非依存で固定する。旧バッチ（v1）混在時の退化規則＝「q_root←value（勝敗単独へ退化）・
turns_left←NaN（補助損失から除外）」が学習の安全弁。
"""
import numpy as np
import pytest

import conftest  # noqa: F401
import pd_batch_common as C

pytestmark = pytest.mark.cpu_infra   # 基盤健全性（学習パイプライン部品）


def _arrays(n=5, v2=True, seed=0):
    rng = np.random.default_rng(seed)
    a = {"scalars": rng.standard_normal((n, 46)).astype(np.float32),
         "field": rng.standard_normal((n, 10, 8)).astype(np.float32),
         "card_idx": rng.integers(0, 50, (n, 24)).astype(np.int32),
         "value": rng.choice([-1.0, 1.0], n).astype(np.float32)}
    if v2:
        a["q_root"] = rng.uniform(-1, 1, n).astype(np.float32)
        a["turns_left"] = rng.integers(0, 15, n).astype(np.float32)
    return a


def test_normalize_v1_fills_degraded_defaults():
    """v1 バッチ → q_root=value（混合が勝敗単独へ退化）・turns_left=NaN（補助から除外）。"""
    a = C.normalize_batch_v2(_arrays(v2=False))
    assert np.array_equal(a["q_root"], a["value"])
    assert np.isnan(a["turns_left"]).all()


def test_normalize_v2_passthrough():
    a0 = _arrays(v2=True)
    a = C.normalize_batch_v2(a0)
    assert a["q_root"] is a0["q_root"] and a["turns_left"] is a0["turns_left"]


def test_mixed_label_math():
    """y = α·z + (1−α)·q。α=1 で勝敗単独と一致・α=0 で q_root と一致・中間は線形補間。"""
    z = np.array([1.0, -1.0, 1.0], np.float32)
    q = np.array([0.2, -0.5, 0.8], np.float32)
    assert np.array_equal(C.mixed_value_label(z, q, 1.0), z)
    assert np.allclose(C.mixed_value_label(z, q, 0.0), q)
    assert np.allclose(C.mixed_value_label(z, q, 0.5), 0.5 * z + 0.5 * q)


def test_ring_append_with_v2_keys_and_v1_mix():
    """v2 キーを含むバッファ連結＋cap 切り。v1 バッチも normalize 後なら混在できる。"""
    b1 = C.normalize_batch_v2(_arrays(n=4, v2=True, seed=1))
    b2 = C.normalize_batch_v2(_arrays(n=3, v2=False, seed=2))
    buf = C.ring_append(None, b1, cap=6)
    buf = C.ring_append(buf, b2, cap=6)
    assert len(buf["value"]) == 6                        # 4+3 → cap 6 で末尾切り
    assert set(buf) == set(b1)
    assert np.isnan(buf["turns_left"][-3:]).all()        # v1 由来の末尾3件は NaN
    assert np.isfinite(buf["turns_left"][:3]).all()      # v2 由来（先頭は1件切られ3件残る）
