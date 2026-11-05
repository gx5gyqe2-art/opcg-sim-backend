"""価値の教師の受け口 `--z-mode rollout`（打ち続けの平均の sidecar `*.roll.npz`・`docs/search_efficiency_plan.md` §7.80・WP `rollout-teach`）。

基盤健全性（学習パイプラインの内部機構だけを見る）。2 本だけ（CLAUDE.md の最小限の規約）:
- 既定（`--z-mode z`）は sidecar があっても読まない・教師は同じオブジェクト＝今までの訓練と 1 bit も変わらない。
  sidecar はシャードの一覧（`dump_io.shard_files`）に紛れ込まない（pack の行が変わらない）。
- `rollout` は sidecar に行がある行だけを平均に差し替え、行の並びは pack と同じ（シャードの順・`--z-in` は後ろ）。
  seed・step がシャードの行と合わない sidecar は止める。
"""
import numpy as np
import pytest

import _bootstrap  # noqa: F401

from opcg_sim.learned.train import dump_io as DIO
from opcg_sim.learned.train import n_rel_train as NRT
from opcg_sim.learned.train import rollout_labels as RL

pytestmark = pytest.mark.cpu_infra


def _shard(path, z, seed, step):
    np.savez(path, z=np.array(z, np.float16), seed=np.array(seed), step=np.array(step))


def test_default_ignores_sidecar(tmp_path, monkeypatch):
    d = tmp_path / "w"
    d.mkdir()
    _shard(d / "n_record_00000.npz", [1, -1], [5, 5], [0, 1])
    RL.write_sidecar(str(d / "n_record_00000.npz"), [1], [0.25], [16], [5], [1])
    assert DIO.shard_files(str(d)) == [str(d / "n_record_00000.npz")]
    seen = {}
    monkeypatch.setattr(NRT, "train", lambda args: seen.setdefault("args", args) and 0)
    NRT.main(["train", "--in", str(d), "--out", "o.npz"])
    assert seen["args"].z_mode == "z"
    called = []
    monkeypatch.setattr(RL, "load_rollout", lambda *a, **k: called.append(1))
    z = np.array([1, -1], np.float16)
    out, n = NRT.value_teacher(z, np.array([5, 5]), np.array([0, 1]), (np.array([], int), np.array([], int)),
                               np.array([], np.float32), "z")
    assert out is z and n == 0 and not called


def test_rollout_replaces_only_rows_in_sidecar_in_pack_order(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    _shard(a / "n_record_00000.npz", [1, -1, 1], [5, 5, 6], [0, 1, 0])
    _shard(a / "n_record_00001.npz", [-1], [7], [0])
    _shard(b / "n_record_00000.npz", [1, 1], [8, 8], [0, 1])               # z 専用（--z-in）は後ろに並ぶ
    RL.write_sidecar(str(a / "n_record_00000.npz"), [2, 0], [-0.5, 0.125], [16, 15], [6, 5], [0, 0])
    RL.write_sidecar(str(b / "n_record_00000.npz"), [1], [0.75], [16], [8], [1])
    mean, mask = RL.load_rollout([str(a)], [str(b)], 6)
    assert mask.tolist() == [True, False, True, False, False, True]
    out, n = RL.apply_teacher(np.array([1, -1, 1, -1, 1, 1], np.float16), mean, mask)
    assert n == 3 and out.tolist() == [0.125, -1.0, -0.5, -1.0, 1.0, 0.75]
    RL.write_sidecar(str(a / "n_record_00001.npz"), [0], [0.5], [16], [7], [3])   # step が合わない
    with pytest.raises(ValueError):
        RL.load_rollout([str(a)], [str(b)], 6)
