"""価値の教師の受け口（`n_rel_train --z-mode {z,omni,mix} --omni-in`・`docs/search_efficiency_plan.md` §7.16・WP `omni-value`）。

基盤健全性（学習パイプラインの内部機構だけを見る）。2 本だけ（CLAUDE.md の最小限の規約）:
- 既定（`--z-mode z`）は教師を**同じオブジェクトのまま**返す＝今までの訓練と 1 bit も変わらない。
- `omni`／`mix` は (seed, step) が一致する行だけを `v_omni`／`0.5 z + 0.5 v_omni` に差し替え、他は z のまま。
"""
import numpy as np
import pytest

import _bootstrap  # noqa: F401

from opcg_sim.learned.train import n_rel_train as NRT

pytestmark = pytest.mark.cpu_infra


def test_z_mode_default_is_identity(monkeypatch):
    seen = {}
    monkeypatch.setattr(NRT, "train", lambda args: seen.setdefault("args", args) and 0)
    NRT.main(["train", "--in", "d", "--out", "o.npz"])
    assert seen["args"].z_mode == "z" and seen["args"].omni_in == []
    z = np.array([1, -1, 1], np.float16)
    out, n = NRT.value_teacher(z, np.array([5, 5, 6]), np.array([0, 1, 0]),
                               (np.array([5]), np.array([0])), np.array([0.3], np.float32), "z")
    assert out is z and n == 0


def test_omni_and_mix_replace_only_matching_rows():
    z = np.array([1, -1, 1, -1], np.float16)
    seed, step = np.array([5, 5, 6, 7]), np.array([0, 1, 0, 0])
    key = (np.array([6, 5, 9]), np.array([0, 1, 0]))          # 行 2・行 1 に当たる（9 は無い局）
    vo = np.array([0.25, 0.5, 0.9], np.float32)
    out, n = NRT.value_teacher(z, seed, step, key, vo, "omni")
    assert n == 2 and out.tolist() == [1.0, 0.5, 0.25, -1.0]
    out, n = NRT.value_teacher(z, seed, step, key, vo, "mix")
    assert n == 2 and out.tolist() == [1.0, -0.25, 0.625, -1.0]
