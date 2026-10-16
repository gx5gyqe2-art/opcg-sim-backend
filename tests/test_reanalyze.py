"""再分析の器（`opcg_sim/loop/reanalyze.py`・`docs/reports/2026-10-16_reanalysis_pilot.md` §1.1）の差し替えの段。

小さな記録（雑音なし・温度なし・1 局・8 sims）を今のエンジンで作り、`--mult 1` で再分析すると:
- 全部の選んだ点が対応付く（記録と同じ候補の集合・`unmapped` 0）。雑音なしなので訪問・値も記録と一致する。
- 出力の行の列と候補の列は、記録の同じ行のコピーで、`pol_n`／`pol_q`／`pol_v0` だけが深い探索の値になる。

`tests/fixtures/f_identity/rec` は今のエンジンでは手が分岐する（デッキ `user` の組み方が当時と違う）ので、記録はこの場で作る。
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

import _bootstrap  # noqa: F401

pytestmark = pytest.mark.cpu_infra

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NET = os.path.join(ROOT, "opcg_sim", "data", "learned", "nrel_r3.npz")


def _run(args, tmp):
    env = dict(os.environ, OPCG_LOG_SILENT="1", PYTHONPATH=ROOT)
    subprocess.run([sys.executable] + args, cwd=ROOT, env=env, check=True, capture_output=True, timeout=600)


def test_reanalyze_copies_rows_and_replaces_only_visits_values(tmp_path):
    pytest.importorskip("opcg_engine")
    rec, out = str(tmp_path / "rec"), str(tmp_path / "re")
    _run(["-m", "opcg_sim.loop.record_gen", "--games", "1", "--seed-base", "990401", "--workers", "1", "--sims", "8",
          "--net", NET, "--out", rec, "--dirichlet-eps", "0", "--temp-turns", "0", "--decks", "synth_roles"], tmp_path)
    _run([os.path.join("opcg_sim", "loop", "reanalyze.py"), "--in", rec, "--out", out, "--mult", "1", "--quota", "4,4,2,2"],
         tmp_path)
    a = np.load(os.path.join(rec, "n_record_00000.npz"))
    files = sorted(f for f in os.listdir(out) if f.startswith("n_record_") and f.endswith(".npz"))
    assert files == ["n_record_990401.npz"]                      # 局ごとに 1 本（再開の単位）
    b = np.load(os.path.join(out, files[0]))
    assert set(a.files) == set(b.files)
    meta = json.load(open(os.path.join(out, "meta_n_record.json"), encoding="utf-8"))
    assert meta["reanalyzed"]["unmapped"] == 0 and meta["reanalyzed"]["diverged"] == 0
    assert meta["reanalyzed"]["points"] == len(b["seed"]) > 0
    ptr_a = np.concatenate([[0], np.cumsum(a["pol_len"])])
    ptr_b = np.concatenate([[0], np.cumsum(b["pol_len"])])
    at = {(int(s), int(t)): i for i, (s, t) in enumerate(zip(a["seed"], a["step"]))}
    pol_cols = [k for k in a.files if k.startswith("pol_") and k not in ("pol_len", "pol_chosen", "pol_v0")]
    row_cols = [k for k in a.files if k not in pol_cols]
    for j in range(len(b["seed"])):
        i = at[(int(b["seed"][j]), int(b["step"][j]))]
        for k in row_cols:
            if k != "pol_v0":
                assert np.array_equal(a[k][i], b[k][j]), k
        for k in pol_cols:
            va, vb = a[k][ptr_a[i]:ptr_a[i + 1]], b[k][ptr_b[j]:ptr_b[j + 1]]
            # 雑音なしの記録を mult 1 で読み直す＝訪問・値も記録と同じ（差し替えた列の対応が正しいことの確かめ）
            assert np.allclose(va, vb) if k in ("pol_n", "pol_q") else np.array_equal(va, vb), k
        n, q = b["pol_n"][ptr_b[j]:ptr_b[j + 1]], b["pol_q"][ptr_b[j]:ptr_b[j + 1]]
        assert abs(float(b["pol_v0"][j]) - float((n * q).sum() / n.sum())) < 1e-2
