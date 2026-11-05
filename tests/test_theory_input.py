"""NRel の入力の束に理論の列を足す切替（§7.82・WP `theory-input`・`--theory-in`）の契約。

基盤健全性（`cpu_infra`）: 学習パイプラインの内部機構（ネットの入力の切替・sidecar の受け口・既定の不変）であり、
ゲームプレイの正しさには触れない（Rust の forward は触っていない＝golden 2 本がそのまま担保する）。

守る性質:
  1. **既定（切）は 1 ビットも同じ**: 同梱の r28x4 を読むと切のまま、価値は切替を入れる前の式とビット一致し、
     保存した npz に新しい鍵も meta も出ない。
  2. **切替を入れて新しい重み 0 なら r28x4 と同じ出力**（理論の列を何にしても・ビット一致）。1 回の更新の後も、
     共有の重みは切のときとビット一致し、新しい重みだけが 0 から動く。
  3. **勾配が正しい**: 新しい重みの勾配が数値微分と合う。
  4. **保存と読み込み**: 新しい重みと列の名前は別鍵で往復し、読んだネットは同じ値を返す。
  5. **sidecar の受け口**: 行の並びは pack と同じ・seed／step の合わない sidecar と無いシャードは止める・
     `ScTheory` は sc の後ろに列を足した束として切り出せる。
  6. **torch 経路**（入っていれば）: forward が numpy と一致し、1 回の更新の後の新しい重みも一致する。
"""
import os

import numpy as np
import pytest

import conftest  # noqa: F401
import _bootstrap  # noqa: F401

from opcg_sim.learned import n_rel as NL
from opcg_sim.learned import n_rel_feat as NR
from opcg_sim.learned.train import n_rel_train as NT
from opcg_sim.learned.train import theory_labels as TL

pytestmark = pytest.mark.cpu_infra

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R28X4 = os.path.join(ROOT, "opcg_sim", "data", "learned", "nrel_r28x4.npz")


@pytest.fixture(scope="module")
def tables():
    from opcg_sim.learned.train.n_eff_feat import build_eff_tables
    return build_eff_tables()[:5]


def _batch(net, B=12, seed=3, theory=False):
    r = np.random.default_rng(seed)
    n_card = len(net.card_table())
    ci = r.integers(0, n_card, (B, NL.N_TOK)).astype(np.int64)
    ci[:, 0] = np.maximum(ci[:, 0], 1)
    sc = r.standard_normal((B, NL.D_SC + (NL.D_TH if theory else 0))).astype(np.float32)
    tok = r.standard_normal((B, NL.N_TOK, NR.S_DIM)).astype(np.float32)
    rom = np.zeros((B, NR.N_OWN, NR.N_OPP, NR.R_DIM), np.float32)
    roo = np.zeros((B, NR.N_OWN, NR.N_OWN, NR.R_DIM), np.float32)
    zt = r.choice([-1.0, 1.0], B).astype(np.float32)
    return sc, ci, tok, rom, roo, zt


def _old_value(net, sc, ci, tok, rom, roo):
    """切替を入れる前の式（`body` の旧い本体をそのまま写したもの）。"""
    tab = net.card_table()
    h, present = net.tokens_forward(ci, tok, rom, roo, tab)
    sc = net.mask_sc(sc)
    n = np.maximum(present.sum(1, keepdims=True), 1.0)
    mean = h.sum(1) / n
    h_masked = np.where(present[:, :, None] > 0, h, -1e9)
    mx = h_masked.max(1); mx = np.where(mx < -1e8, 0.0, mx)
    z = np.concatenate([sc, mean, mx], 1)
    r1 = np.maximum(z @ net.W1 + net.b1, 0.0)
    e = np.maximum(r1 @ net.W2 + net.b2, 0.0)
    return np.tanh((e @ net.Wv + net.bv)[:, 0])


def test_default_off_is_bit_identical(tables, tmp_path):
    net = NT.NRelNet.load(R28X4, tables)
    assert net.theory_in is False
    b = _batch(net)
    assert np.array_equal(net.value(*b[:5]), _old_value(net, *b[:5]))
    out = tmp_path / "off.npz"
    net.save(str(out))
    with np.load(out, allow_pickle=True) as d:
        assert NL.TH_KEY + NL.TH_PARAM not in d.files
        assert "theory" not in str(d["meta"])
    assert NL.TH_PARAM not in net._trainable()


def test_on_with_zero_weight_matches_r28x4(tables):
    off = NT.NRelNet.load(R28X4, tables)
    on = NT.NRelNet.load(R28X4, tables)
    on.theory_in = True
    assert not np.any(on.W1t)
    b = _batch(off, B=16, seed=11, theory=True)
    narrow = b[0][:, :NL.D_SC]
    assert np.array_equal(on.value(b[0], *b[1:5]), off.value(narrow, *b[1:5]))
    v1, a1, t1 = on.value_with_aux(b[0], *b[1:5])
    v0, a0, t0 = off.value_with_aux(narrow, *b[1:5])
    assert np.array_equal(v1, v0) and np.array_equal(a1, a0) and np.array_equal(t1, t0)
    # 列の幅が合わなければ止める（黙って切り詰めない）
    with pytest.raises(ValueError):
        on.value(narrow, *b[1:5])
    # 1 回の更新: 共有の重みは切とビット一致・新しい重みだけが動く
    off.value_step(narrow, *b[1:], lr=2e-4)
    on.value_step(*b, lr=2e-4)
    for p in off.params:
        assert np.array_equal(getattr(on, p), getattr(off, p)), p
    assert np.abs(on.W1t).max() > 0


def _loss(net, b):
    v = net.value(*b[:5])
    return 0.5 * float(np.mean((v.astype(np.float64) - b[5]) ** 2))


def test_gradients_match_finite_differences(tables):
    net = NT.NRelNet.load(R28X4, tables)
    net.theory_in = True
    r = np.random.default_rng(1)
    net.W1t = (r.standard_normal((NL.D_TH, net.W1.shape[1])) * 0.05).astype(np.float32)
    b = _batch(net, B=8, seed=5, theory=True)
    grads = {}
    net.step = lambda g, lr=None, **_k: grads.update(g)
    net.value_step(*b, lr=0.0)
    eps = 1e-2
    for p, idx in ((NL.TH_PARAM, (3, 7)), (NL.TH_PARAM, (NL.D_TH - 1, 100)), ("W1", (300, 7))):
        W = getattr(net, p)
        old = W[idx]
        W[idx] = old + eps; lp = _loss(net, b)
        W[idx] = old - eps; lm = _loss(net, b)
        W[idx] = old
        num = (lp - lm) / (2 * eps)
        ana = float(np.asarray(grads[p])[idx])
        assert abs(num - ana) <= 2e-3 * max(1e-3, abs(num)) + 2e-5, (p, idx, num, ana)


def test_save_load_roundtrip(tables, tmp_path):
    net = NT.NRelNet.load(R28X4, tables)
    net.theory_in = True
    net.W1t = (np.random.default_rng(2).standard_normal((NL.D_TH, net.W1.shape[1])) * 0.01).astype(np.float32)
    out = tmp_path / "on.npz"
    net.save(str(out))
    with np.load(out, allow_pickle=True) as d:
        assert NL.TH_KEY + NL.TH_PARAM in d.files and "theory_cols" in str(d["meta"])
    back = NT.NRelNet.load(str(out), tables)
    assert back.theory_in is True
    assert np.array_equal(back.W1t, net.W1t)
    b = _batch(net, theory=True)
    assert np.array_equal(back.value(*b[:5]), net.value(*b[:5]))
    assert not np.array_equal(back.value(*b[:5]), _old_value(back, b[0][:, :NL.D_SC], *b[1:5]))


def _shard(path, seed, step, n_sc=NL.D_SC):
    n = len(seed)
    np.savez(path, z=np.ones(n, np.float32), seed=np.asarray(seed, np.int64), step=np.asarray(step, np.int64),
             scalars=np.zeros((n, n_sc), np.float16), tokens=np.zeros((n, 2, 2), np.float16),
             card_idx=np.zeros((n, 24), np.int16), who=np.zeros(n, np.int64), turn=np.ones(n, np.int64), kind=np.zeros(n, np.int64))


def test_sidecar_loader_and_view(tmp_path):
    d1, d2 = tmp_path / "w1", tmp_path / "w2"
    d1.mkdir(); d2.mkdir()
    f1, f2 = str(d1 / "n_record_00000.npz"), str(d2 / "n_record_00000.npz")
    _shard(f1, [1, 1, 2], [0, 1, 0]); _shard(f2, [3, 3], [0, 1])
    th1 = np.arange(3 * NL.D_TH, dtype=np.float32).reshape(3, NL.D_TH) / 10
    th2 = -np.arange(2 * NL.D_TH, dtype=np.float32).reshape(2, NL.D_TH) / 10
    TL.write_sidecar(f1, th1, [1, 1, 2], [0, 1, 0])
    # sidecar はシャードの一覧に紛れ込まない
    from opcg_sim.learned.train import dump_io as DIO
    assert DIO.shard_files(str(d1)) == [f1]
    # 無いシャードがあれば止める（列 0 で埋めない）
    with pytest.raises(ValueError, match="無い"):
        TL.load_theory([str(d1)], [str(d2)], 5)
    TL.write_sidecar(f2, th2, [3, 3], [0, 1])
    th = TL.load_theory([str(d1)], [str(d2)], 5)
    assert th.shape == (5, NL.D_TH) and th.dtype == np.float16
    assert np.allclose(th[:3], th1.astype(np.float16)) and np.allclose(th[3:], th2.astype(np.float16))
    # 行の並びが違う sidecar は止める
    TL.write_sidecar(f2, th2, [3, 4], [0, 1])
    with pytest.raises(ValueError, match="seed"):
        TL.load_theory([str(d1)], [str(d2)], 5)
    # 束の view: sc の後ろに列が付く・切り出しは ndarray
    sc = np.ones((5, NL.D_SC), np.float16)
    v = TL.ScTheory(sc, th)
    assert v.shape == (5, NL.D_SC + NL.D_TH) and len(v) == 5
    got = v[np.array([4, 0])]
    assert got.shape == (2, NL.D_SC + NL.D_TH)
    assert np.array_equal(got[:, :NL.D_SC], sc[[4, 0]]) and np.array_equal(got[:, NL.D_SC:], th[[4, 0]])
    assert np.array_equal(v[1], np.concatenate([sc[1], th[1]]))
    assert np.array_equal(np.asarray(v)[:, NL.D_SC:], th)


def test_torch_path_agrees(tables):
    torch = pytest.importorskip("torch", reason="torch は任意依存")
    from opcg_sim.learned.train import n_rel_torch as TT
    net = NT.NRelNet.load(R28X4, tables)
    net.theory_in = True
    net.W1t = (np.random.default_rng(4).standard_normal((NL.D_TH, net.W1.shape[1])) * 0.02).astype(np.float32)
    b = _batch(net, B=10, seed=9, theory=True)
    tr = TT.TorchTrainer(net, lr=2e-4, threads=1)
    with torch.no_grad():
        vt = tr.tn.value(*(torch.from_numpy(x) for x in (b[0], b[1], b[2])), None, None).numpy()
    assert np.allclose(vt, net.value(*b[:5]), atol=1e-5)
    off = NT.NRelNet.load(R28X4, tables)
    assert not hasattr(TT.TorchTrainer(off, lr=2e-4, threads=1).tn, "W1t")
    ref = NT.NRelNet.load(R28X4, tables)
    ref.theory_in = True
    ref.W1t = net.W1t.copy()
    w0 = net.W1t.copy()
    tr.value_step(b[0], b[1], b[2], None, None, b[5], 2e-4)
    tr.sync_to_numpy()
    ref.value_step(*b, lr=2e-4)
    assert np.allclose(net.W1t, ref.W1t, atol=1e-6)
    assert not np.array_equal(net.W1t, w0)
