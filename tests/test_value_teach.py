"""WP `value-teach`（§7.71）の訓練の足し（案 A＝資源の先読みの頭・案 B＝対照の対）の契約。

基盤健全性（`cpu_infra`）: 学習パイプラインの内部機構（教師の sidecar・既定の不変）であり、ゲームプレイの
正しさには触れない。盤面は合成（`test_time_heads` と同じ作り）。

守る性質:
  1. **教師**（`value_teach.label_game`）: 手作りの 1 局で、各行の「次の自分のターン開始時」「2 つ先」の盤面が
     規則どおりに付く（今のターンは入れない・2 つ先が無ければ 1 つ先・次が無い行とターン 0 は mask=0）。
  2. **既定は 1 ビットも変わらない**: `--res-weight 0`（sidecar のある波でも）・`--pair-in` 無しは、足しを入れる前と
     同じ重み（numpy／torch の両方）。
"""
import os

import numpy as np
import pytest

import conftest  # noqa: F401
import _bootstrap  # noqa: F401

from opcg_sim.learned import n_rel as NL
from opcg_sim.learned.train import n_rel_train as NT
from opcg_sim.learned.train import value_teach as VT
from test_time_heads import _shard  # 合成のシャード（同じ作りを使い回す）

pytestmark = pytest.mark.cpu_infra


def test_label_game_handcrafted():
    # 席 0 は奇数ターン・席 1 は偶数ターン。行: (who, turn, kind)
    seq = [(0, 0, 0), (0, 1, 0), (1, 1, 1), (1, 2, 0), (0, 2, 1), (0, 3, 0), (0, 3, 0), (1, 4, 0)]
    n = len(seq)
    rows = {"who": np.array([s[0] for s in seq]), "turn": np.array([s[1] for s in seq]),
            "kind": np.array([s[2] for s in seq]), "z": np.ones(n), "seed": np.zeros(n, np.int64),
            "step": np.arange(n), "pol_len": np.zeros(n, np.int64)}
    snaps = np.arange(n, dtype=np.float32)[:, None] * np.ones((1, VT.D_RES // 2), np.float32)
    res, mask = VT.label_game(rows, snaps, list(range(n)))
    h = VT.D_RES // 2
    assert mask.tolist() == [0, 1, 1, 1, 1, 0, 0, 0]
    # 行 1（席 0・ターン 1・自分の手番）: 次＝ターン 3 の最初の main（行 5）・2 つ先は無い → 行 5
    assert res[1, :h].tolist() == [5.0] * h and res[1, h:].tolist() == [5.0] * h
    # 行 2（席 1・ターン 1・相手の手番）: 次＝ターン 2（行 3）・2 つ先＝ターン 4（行 7）
    assert res[2, 0] == 3.0 and res[2, h] == 7.0
    # 行 4（席 0・ターン 2）: 次＝ターン 3（行 5・同じターンの 2 本目の main 行 6 ではない）
    assert res[4, 0] == 5.0
    assert len(VT.RES_COLS) == VT.D_RES == 20


@pytest.fixture(scope="module")
def wave(tmp_path_factory):
    root = tmp_path_factory.mktemp("vtdump")
    rng = np.random.default_rng(5)
    d = os.path.join(root, "w")
    os.makedirs(d)
    for i, (s, _t) in enumerate([_shard(rng, 90, 4000), _shard(rng, 60, 5000)]):
        np.savez_compressed(os.path.join(d, f"n_record_{i:05d}.npz"), **s)
    return str(root), d


class _Args:
    def __init__(self, src, out, cache_dir, backend, res_weight=0.0):
        self.src = list(src); self.zsrc = []
        self.epochs = 1; self.bs_v = 16; self.bs_p = 8
        self.lr = 2e-3; self.seed = 7; self.hidden = 24; self.holdout_mod = 7
        self.warm_start = None; self.ablate = "rel"; self.backend = backend
        self.cache_dir = cache_dir; self.threads = 1; self.out = out
        self.aux_weight = 0.0; self.plan_weight = 0.0; self.time_weight = 0.0
        self.res_weight = res_weight; self.pair_in = []


def _train(root, d, tag, backend, res_weight=0.0):
    out = os.path.join(root, f"{tag}.npz")
    NT.train(_Args([d], out, os.path.join(root, f"cache_{tag}"), backend, res_weight))
    with np.load(out, allow_pickle=True) as z:
        return {k: z[k].copy() for k in NL.NRelNet.PARAMS}


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_weight0_is_bit_identical(wave, backend):
    if backend == "torch":
        pytest.importorskip("torch", reason="torch は任意依存")
    root, d = wave
    base = _train(root, d, f"base_{backend}", backend)
    # sidecar を足しても重み 0 なら同じ
    for f in sorted(os.listdir(d)):
        if f.endswith(".npz") and not f.endswith(VT.SIDECAR_SUFFIX):
            VT.build_sidecar(os.path.join(d, f), force=True)
    w0 = _train(root, d, f"w0_{backend}", backend, 0.0)
    for k in NL.NRelNet.PARAMS:
        assert np.array_equal(base[k], w0[k]), k
    for f in os.listdir(d):
        if f.endswith(VT.SIDECAR_SUFFIX):
            os.remove(os.path.join(d, f))
