"""中間を広げた warm-start（§7.73 段 7・`n_rel_widen.py`）は**広げた直後は同じ関数**——Rust の対局が 1 ビットも変わらない。

基盤健全性（`cpu_infra`）: 値を変えない移植（幅を広げる）の一致の見張り。容量の試験を閉じたら道具と一緒に消す。
"""
import os
import tempfile

import numpy as np
import pytest

import conftest  # noqa: F401
import _bootstrap  # noqa: F401

from opcg_sim.learned.train import n_rel_widen as W
from opcg_sim.loop import record_gen as G

pytestmark = pytest.mark.cpu_infra

SRC = "opcg_sim/data/learned/nrel_r28x4.npz"


def _play(net, seed=930028):
    G._init_worker(16, net, 0.25, 4, "synth_roles", True, 0.0, 0.0, None, 0.0, None, None, None)
    return G.play_one(seed)


def test_widened_net_plays_bit_identically():
    with tempfile.TemporaryDirectory() as d:
        dst = os.path.join(d, "w.npz")
        h0, h1 = W.widen(SRC, dst, 256)
        assert (h0, h1) == (192, 256)
        z = np.load(dst, allow_pickle=True)
        assert z["W1"].shape[1] == 256 and z["W2"].shape[0] == 256 and not z["W2"][192:].any()
        a, b = _play(SRC), _play(dst)
        for k in sorted(set(a) - {"_meta"}):
            assert np.array_equal(a[k], b[k]), f"広げた直後に列 {k} が変わっている"
        assert a["_meta"]["winner"] == b["_meta"]["winner"]
