"""ValueNet の lead_slots（リーダー条件付け専用枠）検証（docs/reports/lc_value_net_plan_20260708.md）。

`to_leader_conditioned()` は自/相手リーダーの Embedding を平均プールから薄めず専用枠として直結する。
追加行はゼロ初期化＝拡張直後は恒等。ここでは (1) 恒等性 (2) save/load 往復・旧形式後方互換
(3) 勾配の解析値=数値微分一致 (4) リーダーIDだけで決まる合成ターゲットを lead_slots=2 のみ fit できる
ことを確認する。
"""
import os
import tempfile

import numpy as np

import conftest  # noqa: F401
from opcg_sim.src.learned.value_net import ValueNet, train


FIELD_DIM = 80   # 10*8（field.reshape 後の flatten 次元・_rand_batch と feat_dim 計算で共有）


def _rand_batch(rng, n, vocab_size, feat_dim, k_idx=22):
    """feat_dim = scalars次元 + FIELD_DIM（呼び出し側は ValueNet の feat_dim と揃える）。"""
    scalars_dim = feat_dim - FIELD_DIM
    return {
        "scalars": rng.standard_normal((n, scalars_dim)).astype(np.float32),
        "field": rng.standard_normal((n, 10, 8)).astype(np.float32),
        "card_idx": rng.integers(0, vocab_size + 1, size=(n, k_idx)).astype(np.int32),
    }


def test_to_leader_conditioned_is_identity():
    """拡張直後（追加行=ゼロ）は旧ネットと予測が一致する（恒等温スタート）。"""
    rng = np.random.default_rng(0)
    feat_dim = 16 + 10 * 8
    net = ValueNet(vocab_size=50, d_emb=8, hidden=32, feat_dim=feat_dim, seed=1)
    lc = net.to_leader_conditioned()
    assert lc.lead_slots == 2
    assert lc.feat_dim == net.feat_dim, "feat_dim（X次元）は拡張で変わらない"
    batch = _rand_batch(rng, 64, 50, feat_dim)
    p_old = net.predict(batch)
    p_new = lc.predict(batch)
    assert np.allclose(p_old, p_new, atol=1e-12), "リーダー専用枠の追加直後は恒等でないといけない"


def test_to_leader_conditioned_rejects_double_apply():
    net = ValueNet(vocab_size=10, d_emb=4, hidden=8, feat_dim=16 + 80, seed=0)
    lc = net.to_leader_conditioned()
    import pytest
    with pytest.raises(ValueError):
        lc.to_leader_conditioned()


def test_save_load_roundtrip_preserves_lead_slots():
    rng = np.random.default_rng(2)
    feat_dim = 16 + 80
    net = ValueNet(vocab_size=30, d_emb=6, hidden=16, feat_dim=feat_dim, seed=3).to_leader_conditioned()
    batch = _rand_batch(rng, 20, 30, feat_dim)
    pred_before = net.predict(batch)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "lc.npz")
        net.save(path)
        loaded = ValueNet.load(path)
    assert loaded.lead_slots == 2
    assert loaded.feat_dim == net.feat_dim
    assert np.allclose(loaded.predict(batch), pred_before, atol=1e-12)


def test_load_legacy_npz_without_lead_slots_key_defaults_to_zero():
    """旧形式（lead_slots キー無し）npz は lead_slots=0 として読める＝出荷netの後方互換。"""
    net = ValueNet(vocab_size=10, d_emb=4, hidden=8, feat_dim=16 + 80, seed=0)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "legacy.npz")
        # 旧 save() 相当（lead_slots キーを含めない）を素の savez で再現。
        np.savez(path, Emb=net.Emb, W1=net.W1, b1=net.b1, W2=net.W2, b2=net.b2,
                 d_emb=np.array(net.d_emb))
        loaded = ValueNet.load(path)
    assert loaded.lead_slots == 0
    assert loaded.feat_dim == net.feat_dim


def test_expanded_preserves_lead_slots():
    """scalars 版拡張（expanded）は lead_slots を引き継ぐ（enc版温スタートと直交して併用できる）。"""
    feat_dim = 14 + 80
    net = ValueNet(vocab_size=10, d_emb=4, hidden=8, feat_dim=feat_dim, seed=0).to_leader_conditioned()
    grown = net.expanded(insert_at=14, n_new=2)   # v1(14)→v2(16) 相当のscalars拡張
    assert grown.lead_slots == 2
    assert grown.feat_dim == feat_dim + 2


def test_backward_gradient_matches_numerical_finite_difference():
    """lead枠経由の勾配（Emb・W1双方）が数値微分と一致することを小型ネットで確認。"""
    rng = np.random.default_rng(5)
    vocab_size, d_emb, hidden, feat_dim = 6, 3, 5, 4
    net = ValueNet(vocab_size=vocab_size, d_emb=d_emb, hidden=hidden, feat_dim=feat_dim,
                   seed=1, lead_slots=2)
    n = 5
    batch = {
        "scalars": rng.standard_normal((n, feat_dim)).astype(np.float64),
        "field": np.zeros((n, 0), dtype=np.float64),
        "card_idx": rng.integers(1, vocab_size + 1, size=(n, 4)).astype(np.int64),
    }
    y = rng.uniform(-1, 1, size=n)
    pred, cache = net.forward(batch)
    grads = net.backward(cache, y)

    def loss():
        p = net.predict(batch)
        return float(((p - y) ** 2).mean())

    eps = 1e-6
    # Emb の一部行・W1 の一部要素を数値微分でチェック。
    for r, c in [(0, 0), (2, 1)]:
        orig = net.Emb[r, c]
        net.Emb[r, c] = orig + eps; lp = loss()
        net.Emb[r, c] = orig - eps; lm = loss()
        net.Emb[r, c] = orig
        num = (lp - lm) / (2 * eps)
        assert abs(num - grads["Emb"][r, c]) < 1e-4, f"Emb[{r},{c}] 勾配不一致: num={num} analytic={grads['Emb'][r,c]}"
    for r, c in [(0, 0), (feat_dim, 2)]:   # feat_dim行目 = pooled/lead枠の先頭
        orig = net.W1[r, c]
        net.W1[r, c] = orig + eps; lp = loss()
        net.W1[r, c] = orig - eps; lm = loss()
        net.W1[r, c] = orig
        num = (lp - lm) / (2 * eps)
        assert abs(num - grads["W1"][r, c]) < 1e-4, f"W1[{r},{c}] 勾配不一致: num={num} analytic={grads['W1'][r,c]}"


def test_lead_slots_guard_logic_catches_legacy_under_lc_expectation():
    """p3_run の LC ガード相当ロジック: OPCG_P3_LEAD_SLOTS=2 期待で legacy(0) net を読むと停止判定。

    2026-07-08 の事故（LCコード枝に居ないワーカーが checkpoint を掃除→legacy を silently 訓練）の
    再発防止。ガード本体は `tests/scripts/p3_run.load_nets` にあるが、その分岐条件をここで固定する。
    """
    def guard_stops(net, want_lead):
        return want_lead is not None and int(getattr(net, "lead_slots", 0)) != want_lead

    legacy = ValueNet(vocab_size=10, d_emb=4, hidden=8, feat_dim=16 + 80, seed=0)
    lc = legacy.to_leader_conditioned()
    assert guard_stops(legacy, 2) is True, "legacy net を LC 期待で読んだら停止すべき"
    assert guard_stops(lc, 2) is False, "正しい LC net は通す"
    assert guard_stops(legacy, None) is False, "OPCG_P3_LEAD_SLOTS 未設定の従来 run は無影響"


def test_leader_conditioned_net_fits_leader_only_target_legacy_cannot():
    """リーダーIDだけで決まる合成ターゲットを lead_slots=2 は fit でき、legacy(0)は fit できない。

    LC-ValueNet の存在意義そのものの回帰テスト＝「平均プールで薄まる情報を専用枠で拾えるか」。
    """
    rng = np.random.default_rng(11)
    vocab_size, d_emb, hidden, feat_dim = 12, 6, 24, 8
    n = 400
    leaders = rng.integers(1, vocab_size + 1, size=n)
    opp = rng.integers(1, vocab_size + 1, size=n)
    # ターゲット=自リーダーIDの偶奇だけで決まる（他の入力とは無相関のノイズ）。
    y = np.where(leaders % 2 == 0, 0.8, -0.8).astype(np.float32)
    scalars = rng.standard_normal((n, feat_dim)).astype(np.float32)
    field = np.zeros((n, 0), dtype=np.float32)
    card_idx = np.zeros((n, 4), dtype=np.int32)
    card_idx[:, 0] = leaders; card_idx[:, 1] = opp
    card_idx[:, 2] = rng.integers(1, vocab_size + 1, size=n)
    card_idx[:, 3] = rng.integers(1, vocab_size + 1, size=n)
    data = {"scalars": scalars, "field": field, "card_idx": card_idx, "value": y}

    legacy = ValueNet(vocab_size=vocab_size, d_emb=d_emb, hidden=hidden, feat_dim=feat_dim, seed=0)
    lc = ValueNet(vocab_size=vocab_size, d_emb=d_emb, hidden=hidden, feat_dim=feat_dim,
                  seed=0, lead_slots=2)
    _, mse_legacy = train(legacy, data, epochs=60, lr=5e-3, batch=64, val_frac=0.2, seed=0)
    _, mse_lc = train(lc, data, epochs=60, lr=5e-3, batch=64, val_frac=0.2, seed=0)
    assert mse_lc < 0.3 * mse_legacy, (
        f"lead_slots=2 はリーダー限定ターゲットを大きく良く fit できるはず "
        f"(legacy val_mse={mse_legacy:.4f} lc val_mse={mse_lc:.4f})")
