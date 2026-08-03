"""蒸留アンカー付き順位微調整（v33・`ref_finetune_smoke.rank_finetune_anchored`）の検証。

v32 の負の結果（3回再現）: アンカー無しの順位ヒンジは共有 value を歪め、順位が上がるほど
別の較正（防御窓の「素通しが正」＝m2@12/58）が先に壊れる。本機構は順位バッチと
「アンカー盤面で base 予測へ引き戻す蒸留バッチ」を交互に流し、既存挙動を錘で固定したまま
順位だけを動かす。

固定する性質（合成小ネット・実盤面不使用の pure 検証）:
  - **錘の効果そのもの**: 同一の順位ペアで学習したとき、アンカー有りはアンカー盤面の
    予測ドリフトがアンカー無しより小さい（これが崩れたら機構が死んでいる）
  - 順位はそれでも学習される（学習後の pair_acc ≥ 学習前）
  - anchor_scale=0 / アンカー空は素の rank_finetune と同じ自由度（ドリフトを妨げない）
  - dead_weighted_pairs: 負け側が不発PLAYのペアだけが k 倍・k=1 は恒等（pure）
"""
import conftest  # noqa: F401
import numpy as np
import pytest

from ref_finetune_smoke import (build_rank_pairs, dead_weighted_pairs, pair_acc,
                                rank_finetune, rank_finetune_anchored)
from opcg_sim.src.learned.value_net import ValueNet

pytestmark = pytest.mark.cpu_infra   # 基盤健全性（学習機構）

FEAT = 6
K_IDX = 24


def _net(seed=5):
    return ValueNet(vocab_size=8, d_emb=4, hidden=12, feat_dim=FEAT, seed=seed)


def _rows(n, rng):
    return {"scalars": rng.normal(size=(n, FEAT)).astype(np.float32),
            "field": np.zeros((n, 0), np.float32),
            "card_idx": rng.integers(0, 8, size=(n, K_IDX)).astype(np.int64)}


def _mk_data(rng):
    """順位教師（2群×3子）とアンカー盤面（別分布）を作る。"""
    child = _rows(6, rng)
    child["value"] = np.array([1.0, 0.0, -1.0, 0.9, 0.0, -0.9], np.float32)
    child["group"] = np.array([0, 0, 0, 1, 1, 1], np.int64)
    anchor = _rows(64, rng)
    return child, anchor


def _drift(net, base, anchor):
    pa = net.predict(anchor)
    pb = base.predict(anchor)
    return float(np.abs(pa - pb).mean())


def test_anchor_reduces_drift_and_still_learns_rank():
    rng = np.random.default_rng(0)
    child, anchor = _mk_data(rng)
    pairs = build_rank_pairs(child, delta=0.25)
    assert pairs
    base = _net()
    y_anchor = base.predict(anchor)

    free = rank_finetune(_net(), child, pairs, epochs=30, lr=5e-3)
    tied = rank_finetune_anchored(_net(), child, pairs, anchor, y_anchor,
                                  epochs=30, lr=5e-3, anchor_scale=2.0)
    d_free = _drift(free, base, anchor)
    d_tied = _drift(tied, base, anchor)
    assert d_tied < d_free, f"錘が効いていない: tied={d_tied:.4f} >= free={d_free:.4f}"
    assert pair_acc(tied, child, pairs) >= pair_acc(base, child, pairs), \
        "アンカー付きでも順位が学習できていない"


def test_zero_scale_matches_free_training():
    """anchor_scale=0 は素の rank_finetune と同一系列（錘ゼロ＝挙動互換）。"""
    rng = np.random.default_rng(1)
    child, anchor = _mk_data(rng)
    pairs = build_rank_pairs(child, delta=0.25)
    base = _net()
    y_anchor = base.predict(anchor)
    a = rank_finetune(_net(), child, pairs, epochs=10, lr=5e-3)
    b = rank_finetune_anchored(_net(), child, pairs, anchor, y_anchor,
                               epochs=10, lr=5e-3, anchor_scale=0.0)
    ra = a.predict({k: child[k] for k in ("scalars", "field", "card_idx")})
    rb = b.predict({k: child[k] for k in ("scalars", "field", "card_idx")})
    assert np.allclose(ra, rb), "scale=0 が素の順位学習と一致しない（rng 消費経路の乖離）"


def test_dead_weighted_pairs_duplicates_only_dead_losers():
    pairs = [(0, 1, 10), (2, 3, 11), (4, 5, 12)]
    dead = np.array([0, 1, 0, 0, 0, 1], np.float32)   # idx1 と idx5 が不発行
    out = dead_weighted_pairs(pairs, dead, k=3)
    assert out.count((0, 1, 10)) == 3                  # 負け側 idx1 が不発 → 3 倍
    assert out.count((2, 3, 11)) == 1                  # 負け側 idx3 は通常 → 1 倍
    assert out.count((4, 5, 12)) == 3
    assert dead_weighted_pairs(pairs, dead, k=1) == pairs
