"""NRel の価値の出口の形の切替（入力の束 → 出口へのまっすぐな道・§7.78・WP `value-skip`）の契約。

基盤健全性（`cpu_infra`）: 学習パイプラインの内部機構（ネットの形の切替・保存形式・既定の不変）であり、
ゲームプレイの正しさには触れない（Rust の forward は触っていない＝golden 2 本がそのまま担保する）。

守る性質:
  1. **既定（切）は 1 ビットも同じ**: 同梱の r28x4 を読むと切のまま、価値は「共有表現 × 出口の重み ＋ 切片」の
     tanh とビット一致し、保存した npz に新しい鍵が出ない。
  2. **切替を入れて新しい重み 0 なら r28x4 と同じ出力**（ビット一致）。1 回の更新の後も、共有の重みは
     切のときとビット一致し（新しい重みが 0 なので戻る勾配も 0）、新しい重みだけが 0 から動く。
  3. **勾配が正しい**: 新しい重みと、まっすぐな道から入力の束へ戻る分の勾配が数値微分と合う。
  4. **保存と読み込み**: 新しい重みは別鍵で往復し、読んだネットは同じ値を返す。
  5. **torch 経路**（入っていれば）: forward が numpy と一致し、1 回の更新の後の新しい重みも一致する。
"""
import os

import numpy as np
import pytest

import conftest  # noqa: F401
import _bootstrap  # noqa: F401

from opcg_sim.learned import n_rel as NL
from opcg_sim.learned import n_rel_feat as NR
from opcg_sim.learned.train import n_rel_train as NT

pytestmark = pytest.mark.cpu_infra

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R28X4 = os.path.join(ROOT, "opcg_sim", "data", "learned", "nrel_r28x4.npz")


@pytest.fixture(scope="module")
def tables():
    from opcg_sim.learned.train.n_eff_feat import build_eff_tables
    return build_eff_tables()[:5]


def _batch(net, B=12, seed=3):
    r = np.random.default_rng(seed)
    n_card = len(net.card_table())
    ci = r.integers(0, n_card, (B, NL.N_TOK)).astype(np.int64)
    ci[:, 0] = np.maximum(ci[:, 0], 1)
    sc = r.standard_normal((B, NL.D_SC)).astype(np.float32)
    tok = r.standard_normal((B, NL.N_TOK, NR.S_DIM)).astype(np.float32)
    rom = np.zeros((B, NR.N_OWN, NR.N_OPP, NR.R_DIM), np.float32)
    roo = np.zeros((B, NR.N_OWN, NR.N_OWN, NR.R_DIM), np.float32)
    zt = r.choice([-1.0, 1.0], B).astype(np.float32)
    return sc, ci, tok, rom, roo, zt


def _old_value(net, sc, ci, tok, rom, roo):
    """切替を入れる前の式（`value` の旧い本体をそのまま写したもの）。"""
    tab = net.card_table()
    h, present = net.tokens_forward(ci, tok, rom, roo, tab)
    e = net.body(sc, h, present)
    return np.tanh((e @ net.Wv + net.bv)[:, 0])


def test_default_off_is_bit_identical(tables, tmp_path):
    net = NT.NRelNet.load(R28X4, tables)
    assert net.value_skip is False
    b = _batch(net)
    v = net.value(*b[:5])
    assert np.array_equal(v, _old_value(net, *b[:5]))
    out = tmp_path / "off.npz"
    net.save(str(out))
    with np.load(out, allow_pickle=True) as d:
        assert NL.SKIP_KEY + NL.SKIP_PARAM not in d.files
        assert "value_skip" not in str(d["meta"])
    assert NL.SKIP_PARAM not in net._trainable()


def test_on_with_zero_weight_matches_r28x4(tables):
    off = NT.NRelNet.load(R28X4, tables)
    on = NT.NRelNet.load(R28X4, tables)
    on.value_skip = True
    assert not np.any(on.Wvz)
    b = _batch(off, B=16, seed=11)
    assert np.array_equal(on.value(*b[:5]), off.value(*b[:5]))
    v1, a1, t1 = on.value_with_aux(*b[:5])
    v0, a0, t0 = off.value_with_aux(*b[:5])
    assert np.array_equal(v1, v0) and np.array_equal(a1, a0) and np.array_equal(t1, t0)
    # 1 回の更新: 共有の重みは切とビット一致・新しい重みだけが動く
    off.value_step(*b, lr=2e-4)
    on.value_step(*b, lr=2e-4)
    for p in off.params:
        assert np.array_equal(getattr(on, p), getattr(off, p)), p
    assert np.abs(on.Wvz).max() > 0


def _loss(net, b):
    v = net.value(*b[:5])
    return 0.5 * float(np.mean((v.astype(np.float64) - b[5]) ** 2))


def test_gradients_match_finite_differences(tables):
    net = NT.NRelNet.load(R28X4, tables)
    net.value_skip = True
    r = np.random.default_rng(1)
    net.Wvz = (r.standard_normal((NL.D_Z, 1)) * 0.05).astype(np.float32)
    b = _batch(net, B=8, seed=5)
    # 勾配だけを取る（更新を止める）
    grads = {}
    net.step = lambda g, lr=None, **_k: grads.update(g)
    net.value_step(*b, lr=0.0)
    eps = 1e-2
    for p, idx in ((NL.SKIP_PARAM, (3, 0)), (NL.SKIP_PARAM, (200, 0)), ("Wt", (70, 5)), ("W1", (300, 7))):
        W = getattr(net, p)
        old = W[idx]
        W[idx] = old + eps; lp = _loss(net, b)
        W[idx] = old - eps; lm = _loss(net, b)
        W[idx] = old
        num = (lp - lm) / (2 * eps)
        ana = float(np.asarray(grads[p])[idx])
        # 許容は相対 2% ＋ 2e-5（2026-11-09: カード DB に 75 枚足すと引く盤面が変わり、勾配 0.002 の点で
        # 単精度の中心差分が相対 1.4% ずれた＝式の誤りではなく差分の丸め。式の誤りなら桁で外れる）。
        assert abs(num - ana) <= 2e-2 * max(1e-3, abs(num)) + 2e-5, (p, idx, num, ana)


def test_save_load_roundtrip(tables, tmp_path):
    net = NT.NRelNet.load(R28X4, tables)
    net.value_skip = True
    net.Wvz = (np.random.default_rng(2).standard_normal((NL.D_Z, 1)) * 0.01).astype(np.float32)
    out = tmp_path / "on.npz"
    net.save(str(out))
    back = NT.NRelNet.load(str(out), tables)
    assert back.value_skip is True
    assert np.array_equal(back.Wvz, net.Wvz)
    b = _batch(net)
    assert np.array_equal(back.value(*b[:5]), net.value(*b[:5]))
    assert not np.array_equal(back.value(*b[:5]), _old_value(back, *b[:5]))


def test_torch_path_agrees(tables):
    torch = pytest.importorskip("torch", reason="torch は任意依存")
    from opcg_sim.learned.train import n_rel_torch as TT
    net = NT.NRelNet.load(R28X4, tables)
    net.value_skip = True
    net.Wvz = (np.random.default_rng(4).standard_normal((NL.D_Z, 1)) * 0.02).astype(np.float32)
    b = _batch(net, B=10, seed=9)
    tr = TT.TorchTrainer(net, lr=2e-4, threads=1)
    with torch.no_grad():
        vt = tr.tn.value(*(torch.from_numpy(x) for x in (b[0], b[1], b[2])), None, None).numpy()
    assert np.allclose(vt, net.value(*b[:5]), atol=1e-5)
    # 切（既定）の torch は新しい重みを持たない
    off = NT.NRelNet.load(R28X4, tables)
    assert not hasattr(TT.TorchTrainer(off, lr=2e-4, threads=1).tn, "Wvz")
    # 1 回の更新の後、新しい重みが numpy の Adam と同じ向きに同じだけ動く
    ref = NT.NRelNet.load(R28X4, tables)
    ref.value_skip = True
    ref.Wvz = net.Wvz.copy()
    w0 = net.Wvz.copy()
    tr.value_step(b[0], b[1], b[2], None, None, b[5], 2e-4)
    tr.sync_to_numpy()
    ref.value_step(*b, lr=2e-4)
    assert np.allclose(net.Wvz, ref.Wvz, atol=1e-6)
    assert not np.array_equal(net.Wvz, w0)
