"""value ネット(D-4)の検証：tiny集合を過学習できる＝forward/backprop/Adam/Embedding勾配が正しい。"""
import numpy as np

import conftest  # noqa: F401
import rl_encoder as E
import rl_net as N


def _synthetic(n=60, vocab=40, seed=1):
    rng = np.random.default_rng(seed)
    return {
        "scalars": rng.standard_normal((n, 14)).astype(np.float32),
        "field": rng.standard_normal((n, 2 * E.MAX_FIELD, E.PER_CHAR)).astype(np.float32),
        "card_idx": rng.integers(0, vocab + 1, size=(n, 2 + 2 * E.MAX_FIELD + E.MAX_HAND)).astype(np.int32),
        "value": rng.choice([-1.0, 1.0], size=n).astype(np.float32),
    }


def test_overfits_tiny():
    data = _synthetic(n=60, vocab=40)
    net = N.ValueNet(vocab_size=40, d_emb=16, hidden=64, feat_dim=E.feature_dim(), seed=0)
    # 同一集合で train=val（過学習能力の確認）。
    tm, _ = N.train(net, data, epochs=300, lr=3e-3, batch=60, val_frac=0.0001, seed=0)
    assert tm < 0.05, f"tiny集合を過学習できない（train_mse={tm:.3f}）＝学習ループにバグ"


def test_embedding_actually_used():
    """カードID Embedding が出力に効く（idx を変えると予測が変わる）＝Embedding経路が生きている。"""
    data = _synthetic(n=8, vocab=40)
    net = N.ValueNet(vocab_size=40, d_emb=16, hidden=32, feat_dim=E.feature_dim(), seed=2)
    N.train(net, data, epochs=50, lr=3e-3, batch=8, val_frac=0.0001)
    p1 = net.predict(data)
    d2 = dict(data); d2["card_idx"] = (data["card_idx"] * 0 + 1).astype(np.int32)  # 全部別カードへ
    p2 = net.predict(d2)
    assert float(np.abs(p1 - p2).mean()) > 1e-3, "idx を変えても出力不変＝Embedding が効いていない"
