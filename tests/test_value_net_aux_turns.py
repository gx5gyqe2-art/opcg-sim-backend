"""ValueNet 残りターン補助ヘッド（W2t/b2t・v4・docs/cpu_v4_plan.md §4-2）の単体検証。

補助ヘッドは value 出力経路から独立（A1→線形のみ）＝**恒等温スタート**（旧 npz ロード・
構造拡張のいずれでも value 出力不変）を崩さないことが最重要の契約。解析勾配は数値微分と
一致（既存 value_net テスト群と同じ流儀）。
"""
import os
import tempfile

import numpy as np
import pytest

import conftest  # noqa: F401
from opcg_sim.src.learned.value_net import ValueNet, train

pytestmark = pytest.mark.cpu_infra   # 基盤健全性（学習部品の機構）

_V3 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "opcg_sim", "data", "learned", "gen3_value.npz")


def _batch(rng, n=4, feat=126, k=24, vocab=50):
    return {"scalars": rng.standard_normal((n, feat - 80)),
            "field": rng.standard_normal((n, 10, 8)),
            "card_idx": rng.integers(0, vocab, (n, k))}


def _net(seed=0):
    return ValueNet(vocab_size=50, d_emb=8, hidden=16, feat_dim=126, seed=seed)


def test_aux_head_does_not_affect_value_output():
    """W2t/b2t をどう変えても value 予測は不変（補助ヘッドの独立性＝恒等温スタートの根拠）。"""
    rng = np.random.default_rng(0)
    net = _net(); b = _batch(rng)
    p0 = net.predict(b)
    net.W2t[:] = rng.standard_normal(net.W2t.shape)
    net.b2t[:] = 1.7
    assert np.array_equal(p0, net.predict(b))


def test_legacy_npz_loads_with_zero_aux():
    """旧 npz（現本番 v3=gen3_value.npz）ロードで aux はゼロ＝出力恒等・保存すれば aux も往復。"""
    net = ValueNet.load(_V3)
    rng = np.random.default_rng(1)
    b = {"scalars": rng.standard_normal((3, 46)), "field": rng.standard_normal((3, 10, 8)),
         "card_idx": rng.integers(0, 100, (3, 24))}
    assert np.allclose(net.predict_aux(b), 0.0)
    net.W2t[:] = 0.01
    f = tempfile.mktemp(suffix=".npz")
    try:
        net.save(f)
        net2 = ValueNet.load(f)
        assert np.allclose(net2.W2t, net.W2t)
        assert np.array_equal(net2.predict(b), net.predict(b))
        assert np.allclose(net2.predict_aux(b), net.predict_aux(b))
    finally:
        os.remove(f)


def test_aux_gradients_match_numeric():
    """解析勾配＝数値微分（W2t/b2t と、補助損失の共有層への寄与 W1。NaN ラベルはマスク）。"""
    rng = np.random.default_rng(2)
    net = _net(3)
    net.W2t = rng.standard_normal(net.W2t.shape) * 0.1
    b = _batch(rng, n=3)
    y = np.array([0.5, -0.2, 0.9]); ya = np.array([0.3, np.nan, 0.8]); aw = 0.7
    _, cache = net.forward(b)
    grads = net.backward(cache, y, y_aux=ya, aux_weight=aw)

    def loss():
        p, c = net.forward(b)
        t = net.aux_from_cache(c)
        m = np.isfinite(ya)
        return float(((p - y) ** 2).mean()) + aw * float(((t[m] - ya[m]) ** 2).mean())

    eps = 1e-6
    for name, ij in [("W2t", (5, 0)), ("b2t", (0,)), ("W1", (10, 3)), ("W2", (7, 0))]:
        p = getattr(net, name); orig = p[ij]
        p[ij] = orig + eps; lp = loss()
        p[ij] = orig - eps; lm = loss()
        p[ij] = orig
        num = (lp - lm) / (2 * eps)
        assert abs(num - grads[name][ij]) < 1e-6 * max(1.0, abs(num)), \
            f"{name}{ij}: 数値={num} 解析={grads[name][ij]}"


def test_structural_copies_carry_aux():
    """expanded / to_leader_conditioned / to_v3 / widened が aux ヘッドを引き継ぐ（widened は行拡張）。"""
    rng = np.random.default_rng(4)
    net = _net(5)
    net.W2t = rng.standard_normal(net.W2t.shape) * 0.1
    net.b2t[:] = 0.3
    ex = net.expanded(10, 2)
    assert np.array_equal(ex.W2t, net.W2t) and np.array_equal(ex.b2t, net.b2t)
    lc = net.to_leader_conditioned()
    assert np.array_equal(lc.W2t, net.W2t)
    eff = lc.to_v3(np.ones((51, 6)), eff_proj=4)
    assert np.array_equal(eff.W2t, net.W2t)
    wd = net.widened(24)
    assert wd.W2t.shape == (24, 1)
    assert np.array_equal(wd.W2t[:16], net.W2t) and np.allclose(wd.W2t[16:], 0.0)
    b = _batch(rng)
    assert np.allclose(wd.predict_aux(b), net.predict_aux(b)), "widened で aux 予測が変わった"


def test_train_learns_aux_target():
    """合成データ（残りターン≒scalars の1成分）で補助 mse が学習により下がる。NaN 混在でも学習可。"""
    rng = np.random.default_rng(6)
    n = 256
    data = {"scalars": rng.standard_normal((n, 46)), "field": rng.standard_normal((n, 10, 8)),
            "card_idx": rng.integers(1, 50, (n, 24)),
            "value": rng.choice([-1.0, 1.0], n).astype(np.float32)}
    aux = np.clip(0.5 + 0.4 * data["scalars"][:, 0], 0, 1)
    aux[::10] = np.nan                     # 旧スキーマ由来の欠損を混ぜる
    data["aux"] = aux
    net = _net(7)
    m = np.isfinite(aux)
    def aux_mse():
        return float(((net.predict_aux(data)[m] - aux[m]) ** 2).mean())
    before = aux_mse()
    train(net, data, epochs=8, lr=3e-3, batch=64, val_frac=0.1, aux_weight=1.0)
    after = aux_mse()
    assert after < before * 0.6, f"補助ターゲットが学習されない: {before:.4f}→{after:.4f}"
