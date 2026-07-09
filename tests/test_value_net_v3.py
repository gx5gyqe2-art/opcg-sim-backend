"""ValueNet v3（EffFeat組み込み）の検証（docs/reports/effect_semantics_v3_plan_20260708.md §2/§5）。

恒等温スタート連鎖（scalars拡張→LC→to_v3→widened で出力完全一致）・勾配の数値微分一致（W_eff含む）・
save/load往復・順序ガード・**ゼロショット回帰**（効果特徴だけで決まるターゲットを未見リーダーへ汎化
できるのは v3 のみ＝LCの埋め込みでは不可）・encoder v3（scalars46/card_idx24）との結線。
"""
import os
import tempfile

import numpy as np

import conftest  # noqa: F401
from opcg_sim.src.learned.value_net import ValueNet, train

FIELD = 80


def _batch(rng, n, vocab_size, scalars_dim, idx_w=22):
    return {
        "scalars": rng.standard_normal((n, scalars_dim)).astype(np.float32),
        "field": rng.standard_normal((n, 10, 8)).astype(np.float32),
        "card_idx": rng.integers(0, vocab_size + 1, size=(n, idx_w)).astype(np.int32),
    }


def _eff_table(rng, vocab_size, F):
    t = (rng.random((vocab_size + 1, F)) < 0.3).astype(np.float32)
    t[0] = 0.0
    return t


def test_identity_chain_scalars_lc_v3_widened():
    """恒等連鎖: legacy → scalars拡張 → LC → to_v3 → widened の各段で予測が完全一致。"""
    rng = np.random.default_rng(0)
    vocab, F, P = 50, 12, 4
    base = ValueNet(vocab_size=vocab, d_emb=8, hidden=16, feat_dim=16 + FIELD, seed=1)
    b22 = _batch(rng, 32, vocab, 16)
    p0 = base.predict(b22)

    grown = base.expanded(insert_at=16, n_new=30)          # scalars 16→46 相当
    b46 = dict(b22, scalars=np.concatenate(
        [b22["scalars"], rng.standard_normal((32, 30)).astype(np.float32)], axis=1))
    assert np.allclose(grown.predict(b46), p0, atol=1e-12), "scalars拡張が恒等でない"

    lc = grown.to_leader_conditioned()
    assert np.allclose(lc.predict(b46), p0, atol=1e-12), "LC化が恒等でない"

    v3 = lc.to_v3(_eff_table(rng, vocab, F), eff_proj=P, seed=2)
    assert np.allclose(v3.predict(b46), p0, atol=1e-12), "to_v3（22幅idx）が恒等でない"
    b46s = dict(b46, card_idx=np.concatenate(
        [b46["card_idx"], rng.integers(0, vocab + 1, size=(32, 2)).astype(np.int32)], axis=1))
    assert np.allclose(v3.predict(b46s), p0, atol=1e-12), "to_v3（24幅idx・ステージ枠）が恒等でない"

    wide = v3.widened(48)
    assert np.allclose(wide.predict(b46s), p0, atol=1e-12), "widened が恒等でない"
    assert wide.feat_dim == 46 + FIELD and wide.eff_dim == F and wide.lead_slots == 2


def test_order_guards():
    rng = np.random.default_rng(1)
    vocab, F = 10, 6
    net = ValueNet(vocab_size=vocab, d_emb=4, hidden=8, feat_dim=16 + FIELD, seed=0)
    import pytest
    with pytest.raises(ValueError):
        net.to_v3(_eff_table(rng, vocab, F))               # LC前のto_v3は拒否
    lc = net.to_leader_conditioned()
    v3 = lc.to_v3(_eff_table(rng, vocab, F))
    with pytest.raises(ValueError):
        v3.to_v3(_eff_table(rng, vocab, F))                # 二重適用は拒否
    with pytest.raises(ValueError):
        v3.to_leader_conditioned()                          # eff後のLC化は拒否
    with pytest.raises(ValueError):
        v3.widened(v3.W1.shape[1])                          # 拡張方向のみ


def test_gradients_match_numerical_including_w_eff():
    rng = np.random.default_rng(2)
    vocab, F, P, feat = 6, 5, 3, 4
    net = ValueNet(vocab_size=vocab, d_emb=3, hidden=5, feat_dim=feat, seed=1,
                   lead_slots=2, eff_table=_eff_table(rng, vocab, F), eff_proj=P)
    n = 5
    batch = {
        "scalars": rng.standard_normal((n, feat)).astype(np.float64),
        "field": np.zeros((n, 0), dtype=np.float64),
        "card_idx": rng.integers(1, vocab + 1, size=(n, 24)).astype(np.int64),
    }
    y = rng.uniform(-1, 1, size=n)
    _, cache = net.forward(batch)
    grads = net.backward(cache, y)

    def loss():
        return float(((net.predict(batch) - y) ** 2).mean())

    eps = 1e-6
    for pname, coords in (("W_eff", [(0, 0), (2, 1)]), ("Emb", [(1, 0)]),
                          ("W1", [(0, 0), (feat + 3, 1)])):
        M = getattr(net, pname)
        for r, c in coords:
            orig = M[r, c]
            M[r, c] = orig + eps; lp = loss()
            M[r, c] = orig - eps; lm = loss()
            M[r, c] = orig
            num = (lp - lm) / (2 * eps)
            assert abs(num - grads[pname][r, c]) < 1e-4, \
                f"{pname}[{r},{c}] 勾配不一致: num={num} analytic={grads[pname][r, c]}"


def test_save_load_roundtrip_and_backward_compat():
    rng = np.random.default_rng(3)
    vocab, F = 20, 7
    net = ValueNet(vocab_size=vocab, d_emb=4, hidden=8, feat_dim=16 + FIELD, seed=0)\
        .to_leader_conditioned().to_v3(_eff_table(rng, vocab, F), eff_proj=4, seed=1)
    batch = _batch(rng, 12, vocab, 16, idx_w=24)
    p = net.predict(batch)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "v3.npz")
        net.save(path)
        loaded = ValueNet.load(path)
    assert loaded.eff_dim == F and loaded.eff_proj == 4 and loaded.lead_slots == 2
    assert np.allclose(loaded.predict(batch), p, atol=1e-9)
    # 旧形式（EffF無し）が eff_dim=0 で読めること＝出荷net後方互換（LCテストと同じ規約）。
    legacy = ValueNet(vocab_size=vocab, d_emb=4, hidden=8, feat_dim=16 + FIELD, seed=0)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "legacy.npz")
        legacy.save(path)
        loaded = ValueNet.load(path)
    assert loaded.eff_dim == 0 and loaded.EffF is None


def test_zero_shot_generalization_to_unseen_leaders():
    """効果特徴だけで決まるターゲットを**未見リーダー**へ汎化できるのは v3 のみ。

    LC（埋め込み条件付け）は見たリーダーの暗記はできるが、未見リーダーの埋め込みは初期値のまま
    ＝汎化不能。EffFeat は意味（効果ビット）で転移する＝v3 の存在意義そのもの（OP03ナミ問題の回帰）。
    """
    rng = np.random.default_rng(4)
    vocab, F, P, feat = 40, 6, 4, 8
    table = np.zeros((vocab + 1, F), dtype=np.float32)
    table[1:, :] = (rng.random((vocab, F)) < 0.3).astype(np.float32)
    table[1:, 0] = (np.arange(1, vocab + 1) % 2 == 0)      # f0=リーダーIDの偶奇（=意味特徴）
    n = 600
    train_leaders = rng.integers(1, 31, size=n)            # 学習はリーダー1..30のみ
    test_leaders = rng.integers(31, vocab + 1, size=200)   # 検証は未見リーダー31..40
    def make(leaders):
        m = len(leaders)
        idx = np.zeros((m, 24), dtype=np.int32)
        idx[:, 0] = leaders
        idx[:, 1] = rng.integers(1, vocab + 1, size=m)
        return {"scalars": rng.standard_normal((m, feat)).astype(np.float32),
                "field": np.zeros((m, 0), dtype=np.float32), "card_idx": idx}
    def target(leaders):
        return np.where(table[leaders, 0] > 0, 0.8, -0.8).astype(np.float32)
    tr = make(train_leaders); tr["value"] = target(train_leaders)
    te = make(test_leaders); yte = target(test_leaders)

    lc = ValueNet(vocab_size=vocab, d_emb=6, hidden=24, feat_dim=feat, seed=0, lead_slots=2)
    v3 = ValueNet(vocab_size=vocab, d_emb=6, hidden=24, feat_dim=feat, seed=0, lead_slots=2,
                  eff_table=table, eff_proj=P)
    train(lc, tr, epochs=60, lr=5e-3, batch=64, val_frac=0.1, seed=0)
    train(v3, tr, epochs=60, lr=5e-3, batch=64, val_frac=0.1, seed=0)
    mse_lc = float(((lc.predict(te) - yte) ** 2).mean())
    mse_v3 = float(((v3.predict(te) - yte) ** 2).mean())
    assert mse_v3 < 0.3 * mse_lc, \
        f"未見リーダーへの汎化で v3 が LC を大差で上回るはず (LC={mse_lc:.4f} v3={mse_v3:.4f})"


def test_encoder_v3_dims_and_net_wiring():
    """encoder v3（scalars46・card_idx24・ステージ末尾）→ v3 net の e2e 結線（実局面）。"""
    import random
    random.seed(5)
    from cpu_selfplay import build_deck, _load_db
    from opcg_sim.src.core.gamestate import GameManager, Player
    from opcg_sim.src.learned import encoder as PROD_E
    from opcg_sim.src.learned import effect_features as EF
    db = _load_db()
    l1, c1 = build_deck(db, "p1"); l2, c2 = build_deck(db, "p2")
    m = GameManager(Player("p1", c1, l1), Player("p2", c2, l2)); m.start_game()
    vocab = PROD_E.build_vocab(db)
    enc = PROD_E.encode(m, "p1", vocab, version=3)
    assert enc["scalars"].shape == (46,), "SCALARS_V3=46"
    assert enc["card_idx"].shape == (24,), "v3 は card_idx 24（末尾2=ステージ）"
    assert PROD_E.encode(m, "p1", vocab, version=2)["card_idx"].shape == (22,), "v2 は不変"
    table = EF.build_efffeat(db, vocab)
    net = ValueNet(vocab_size=len(vocab), d_emb=8, hidden=16, feat_dim=PROD_E.feature_dim(3),
                   seed=0, lead_slots=2, eff_table=table, eff_proj=8)
    batch = {k: enc[k][None, ...] for k in ("scalars", "field", "card_idx")}
    p = net.predict(batch)
    assert p.shape == (1,) and np.isfinite(p).all()
