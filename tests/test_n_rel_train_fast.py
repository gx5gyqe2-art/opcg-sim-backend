"""訓練の高速化の切替（`--fast-pairs`／`--fast-max`／`--fast-index`・`docs/reports/2026-10-17_train_speed.md`）。

既定（切）の式と、同じ重み・同じバッチで比べる:
- `--fast-max`・`--fast-index` は forward（価値・方策の logits）が**ビット一致**。
- `--fast-pairs`（対のネットを重みの区切りごとに掛けて足す）と 3 つ全部は、forward が 1e-5・勾配が相対 1e-5 の内
  （式は同じで足す順だけが違う＝丸めの範囲）。全部のパラメータに勾配が出ること。
"""
import numpy as np
import pytest

import _bootstrap  # noqa: F401

torch = pytest.importorskip("torch")

from test_n_rel_train_torch import _pargs, _vargs, env  # noqa: E402,F401

pytestmark = pytest.mark.cpu_infra


def _run(env, fp, fm, fi):
    from opcg_sim.learned.train import n_rel_torch as TT
    net = env["net"]
    tn = TT.TorchNRel(net)
    tn.factor_pairs, tn.fast_max, tn.fast_index = fp, fm, fi
    sc, ci, tok, rom, roo, z = _vargs(env)
    v = tn.value(TT._f32(sc), TT._i64(ci), TT._f32(tok), TT._f32(rom), TT._f32(roo))
    ((v - TT._f32(z)) ** 2).mean().backward()
    psc, pci, ptok, prom, proo, seg, si, ti, idx, budget, _pi = _pargs(env)
    lo = TT.torch_policy_logits(tn, net, psc, pci, ptok, prom, proo, seg, si, ti, env["P"]["C"], idx, budget)
    g = {n: p.grad.numpy().copy() for n, p in tn.named_parameters() if p.grad is not None}
    return v.detach().numpy(), np.asarray(lo), g


def test_fast_switches_match_default(env):
    v0, l0, g0 = _run(env, False, False, False)
    for fp, fm, fi, exact in ((False, True, False, True), (False, False, True, True),
                              (True, False, False, False), (True, True, True, False)):
        v, lo, g = _run(env, fp, fm, fi)
        assert set(g) == set(g0)
        if exact:
            assert np.array_equal(v, v0) and np.array_equal(lo, l0)
        assert np.max(np.abs(v - v0)) < 1e-5 and np.max(np.abs(lo - l0)) < 1e-5
        for k in g0:
            assert np.max(np.abs(g[k] - g0[k])) <= 1e-5 * (np.max(np.abs(g0[k])) + 1e-12), k
