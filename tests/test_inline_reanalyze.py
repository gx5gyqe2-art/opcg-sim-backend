"""生成の中での再分析（`record_gen --reanalyze-mult`・`docs/reports/2026-10-17_compute_shortening.md` §1.1）の不変量。

CLAUDE.md「値を変えない移植の一致の見張り」に当たる（打ち直しの器 `reanalyze.py` を使わなくなったら一緒に消す）:
- (i) `--reanalyze-*` を付けても、通常の記録の列は同じ seed でビット一致する。
- (ii) その場の再分析の点（選び方・n・q・v0）は、同じ記録を `reanalyze.py --mult 4` で打ち直した値と一致する。

雑音・温度は生成の既定のまま（再分析の器は記録と同じ設定で打ち直し、深い探索だけ雑音なし）。小さな局（sims 8・2 局）で回す。
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
GEN = ["-m", "opcg_sim.loop.record_gen", "--games", "2", "--seed-base", "990411", "--workers", "1", "--sims", "8",
       "--net", NET, "--decks", "synth_roles", "--shard-games", "1"]


def _run(args):
    env = dict(os.environ, OPCG_LOG_SILENT="1", PYTHONPATH=ROOT)
    subprocess.run([sys.executable] + args, cwd=ROOT, env=env, check=True, capture_output=True, timeout=900)


def _load(d):
    fs = sorted(f for f in os.listdir(d) if f.startswith("n_record_") and f.endswith(".npz"))
    zs = [np.load(os.path.join(d, f)) for f in fs]
    return {k: np.concatenate([z[k] for z in zs]) for k in zs[0].files}


def _points(d):
    out = {}
    for line in open(os.path.join(d, "points.jsonl"), encoding="utf-8"):
        x = json.loads(line)
        if "game" not in x:
            out[(int(x["seed"]), int(x["step"]))] = x
    return out


def test_inline_reanalysis_keeps_record_and_matches_reanalyze(tmp_path):
    pytest.importorskip("opcg_engine")
    plain, inl, rean = str(tmp_path / "plain"), str(tmp_path / "inl"), str(tmp_path / "re")
    _run(GEN + ["--out", plain])
    _run(GEN + ["--out", inl, "--reanalyze-mult", "4", "--reanalyze-quota", "4,4,2,2", "--reanalyze-per-game", "12"])
    a, b = _load(plain), _load(inl)
    assert set(a) == set(b)
    for k in a:                                                  # (i) 通常の記録はビット一致
        assert a[k].dtype == b[k].dtype and np.array_equal(a[k], b[k]), k

    _run([os.path.join("opcg_sim", "loop", "reanalyze.py"), "--in", plain, "--out", rean, "--mult", "4",
          "--quota", "4,4,2,2"])
    pi, pr = _points(inl + "_rean"), _points(rean)
    assert set(pi) == set(pr) and len(pi) > 0                   # (ii) 同じ点を選ぶ
    for key, x in pi.items():
        y = pr[key]
        assert x["rule"] == y["rule"] and bool(x.get("unmapped")) == bool(y.get("unmapped")), key
        if not x.get("unmapped"):
            assert x["n"] == y["n"] and x["q"] == y["q"] and x["v0"] == y["v0"], key
    c, d = _load(inl + "_rean"), _load(rean)                     # 差し替えた行の dump も同じ（並びは seed・step で揃える）
    oc = np.lexsort((c["step"], c["seed"]))
    od = np.lexsort((d["step"], d["seed"]))
    for k in ("seed", "step", "pol_len", "pol_v0", "z"):
        assert np.array_equal(c[k][oc], d[k][od]), k
    assert np.isclose(c["pol_n"].sum(), d["pol_n"].sum()) and np.isclose(c["pol_q"].sum(), d["pol_q"].sum())
    m = json.load(open(os.path.join(inl + "_rean", "meta_n_record.json"), encoding="utf-8"))
    assert m["reanalyzed"]["inline"] is True and m["reanalyzed"]["points"] == len(c["seed"])
