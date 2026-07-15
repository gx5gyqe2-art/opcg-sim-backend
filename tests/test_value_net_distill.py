"""ValueNet 忘却抑制の value 蒸留（教師アンカー・v5 §4-4b）の単体検証。

凍結 v4 教師の value 予測へ引く MSE アンカーを value ヘッドに加算する（KL蒸留の回帰版）。
value ラベル MSE と同じ tanh 経路の追加項＝解析勾配が数値微分と一致すること・distill_weight=0 で
従来と完全一致すること・蒸留が予測を教師へ引き寄せることを固定する。
"""
import numpy as np
import pytest

import conftest  # noqa: F401
from opcg_sim.src.learned.value_net import ValueNet, train

pytestmark = pytest.mark.cpu_infra   # 基盤健全性（学習部品の機構）


def _batch(rng, n=4, feat=126, k=24, vocab=50):
    return {"scalars": rng.standard_normal((n, feat - 80)),
            "field": rng.standard_normal((n, 10, 8)),
            "card_idx": rng.integers(0, vocab, (n, k))}


def _net(seed=0):
    return ValueNet(vocab_size=50, d_emb=8, hidden=16, feat_dim=126, seed=seed)


def test_distill_weight_zero_is_identity():
    """distill_weight=0（既定）は y_distill を渡しても素の value MSE 勾配と完全一致（挙動不変ゲート）。"""
    rng = np.random.default_rng(1)
    net = _net(1)
    b = _batch(rng, n=3)
    y = np.array([0.5, -0.2, 0.9])
    yd = np.array([0.1, 0.1, 0.1])
    _, cache = net.forward(b)
    g0 = net.backward(cache, y)
    g1 = net.backward(cache, y, y_distill=yd, distill_weight=0.0)
    for k in g0:
        assert np.allclose(g0[k], g1[k]), f"{k} が distill_weight=0 で変化した"


def test_distill_gradients_match_numeric():
    """解析勾配＝数値微分（value MSE + 蒸留 MSE の合成損失・全パラメータ経路）。"""
    rng = np.random.default_rng(2)
    net = _net(3)
    b = _batch(rng, n=3)
    y = np.array([0.5, -0.2, 0.9])
    yd = np.array([-0.4, 0.3, 0.7])
    dw = 0.6
    _, cache = net.forward(b)
    grads = net.backward(cache, y, y_distill=yd, distill_weight=dw)

    def loss():
        p, _ = net.forward(b)
        return float(((p - y) ** 2).mean()) + dw * float(((p - yd) ** 2).mean())

    eps = 1e-6
    for name, ij in [("W2", (7, 0)), ("b2", (0,)), ("W1", (10, 3)), ("Emb", (5, 2))]:
        p = getattr(net, name); orig = p[ij]
        p[ij] = orig + eps; lp = loss()
        p[ij] = orig - eps; lm = loss()
        p[ij] = orig
        num = (lp - lm) / (2 * eps)
        assert abs(num - grads[name][ij]) < 1e-6 * max(1.0, abs(num)), \
            f"{name}{ij}: 数値={num} 解析={grads[name][ij]}"


def test_distill_pulls_prediction_toward_teacher():
    """教師予測を一定値に固定して蒸留のみ学習すると、生徒の予測がその値へ寄る（アンカーが効く）。"""
    rng = np.random.default_rng(5)
    n = 64
    data = _batch(rng, n=n)
    data["value"] = rng.standard_normal(n) * 0.1        # ラベルは弱ノイズ
    teacher_val = 0.6
    data["distill"] = np.full(n, teacher_val)
    net = _net(7)
    before = float(net.predict(data).mean())
    # 蒸留を強めに掛けて学習（value ラベルは 0 付近・教師は 0.6）。
    train(net, data, epochs=40, lr=5e-3, batch=32, val_frac=0.1, distill_weight=5.0)
    after = float(net.predict(data).mean())
    assert after > before + 0.1, f"予測が教師({teacher_val})方向へ寄っていない: {before:.3f}→{after:.3f}"
    assert after < teacher_val + 0.2   # ラベル(≈0)とのバランスで教師を超えて暴走しない
