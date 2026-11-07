"""対の差の損失 `--pair-diff`（`docs/search_efficiency_plan.md` §7.88・WP `pair-diff-loss`）。

基盤健全性（学習パイプラインの内部機構だけを見る）。2 本だけ（CLAUDE.md の最小限の規約）:
- 既定（`--pair-diff` 無し）は対を読まず、価値の更新に何も渡さない＝今までの訓練と 1 bit も変わらない。
  torch の価値の更新は `pd=None` なら損失・勾配とも従来とビット一致。
- 対の読み込み（取り置き＝局 seed % 7・形で絞るのは訓練の対だけ）・1 周に 1 回ずつの割り振り・倍率
  （1 周で対の損失の合計＝それ以外の価値の損失の合計）・足した項の値と勾配（手で書いた式と一致）。
"""
import numpy as np
import pytest

import _bootstrap  # noqa: F401

from opcg_sim.learned.train import n_rel_train as NRT
from opcg_sim.learned.train import pair_diff as PD

pytestmark = pytest.mark.cpu_infra


def test_default_off_reads_nothing(tmp_path, monkeypatch):
    d = tmp_path / "w"
    d.mkdir()
    seen = {}
    monkeypatch.setattr(NRT, "train", lambda args: seen.setdefault("args", args) and 0)
    NRT.main(["train", "--in", str(d), "--out", "o.npz"])
    assert seen["args"].pair_diff == []
    src = open(NRT.__file__, encoding="utf-8").read()
    # 対は「--pair-diff があれば」だけ読まれ、価値の更新に渡るのは割り振りがあるときだけ（既定の経路は従来どおり）
    assert "if pd_paths:" in src and 'if pd_sched is not None and len(pd_sched[iv]):\n                        j_ = pd_sched[iv]' in src
    # torch の価値の更新: pd=None は従来の損失と同じ勾配（pd の項は足されない）
    torch = pytest.importorskip("torch")
    from opcg_sim.learned.train import n_rel_torch as TT
    tsrc = open(TT.__file__, encoding="utf-8").read()
    assert "        if pd is not None:\n" in tsrc
    v = torch.tensor([0.1, -0.2, 0.3, 0.05], requires_grad=True)
    d = PD.diff_terms(v, 2, torch.tensor([0.2, -0.1]))
    # 少ない側 [0.1, −0.2]・多い側 [0.3, 0.05] → 差 [0.2, 0.25] − 教師 [0.2, −0.1] = [0, 0.35]
    assert np.allclose(d.detach().numpy(), [0.0, 0.35 ** 2], atol=1e-6)
    (0.5 * d.sum()).backward()
    assert np.allclose(v.grad.numpy(), [0.0, -0.35, 0.0, 0.35], atol=1e-6)


def test_load_schedule_factor(tmp_path):
    n = 10
    rng = np.random.default_rng(0)
    seed = np.array([7, 8, 9, 14, 15, 16, 17, 18, 21, 22], np.int64)          # 7・14・21 が取り置き
    shape = np.array(["head", "close", "hold", "oppend", "head", "close", "hold", "oppend", "mid", "head"])
    kw = {"sc_l": rng.standard_normal((n, 3)).astype(np.float16), "ci_l": np.zeros((n, 22), np.int16),
          "tok_l": np.zeros((n, 22, 2), np.float16), "sc_m": rng.standard_normal((n, 3)).astype(np.float16),
          "ci_m": np.zeros((n, 22), np.int16), "tok_m": np.zeros((n, 22, 2), np.float16),
          "d": np.linspace(-0.5, 0.5, n).astype(np.float32), "seed": seed, "shape": shape}
    np.savez(tmp_path / "p.npz", **kw)
    P = PD.load_pairs([str(tmp_path / "p.npz")])
    assert P["va"].tolist() == [0, 3, 8] and P["tr"].tolist() == [1, 2, 4, 5, 6, 7, 9]
    Ph = PD.load_pairs([str(tmp_path / "p.npz")], shapes=["head"])
    assert Ph["tr"].tolist() == [4, 9] and Ph["va"].tolist() == [0, 3, 8]      # 取り置きは全部の形のまま
    sc, ci, tok = PD.take(P, np.array([1, 2]))
    assert sc.shape == (4, 3) and sc.dtype == np.float32 and ci.dtype == np.int64
    assert np.array_equal(sc[:2], kw["sc_l"][[1, 2]].astype(np.float32)) and np.array_equal(sc[2:], kw["sc_m"][[1, 2]].astype(np.float32))
    # 1 周に 1 回ずつ・価値の更新 3 回に均等に
    sch = PD.schedule(P["tr"], 3, np.random.default_rng(1))
    assert len(sch) == 3 and sorted(np.concatenate(sch).tolist()) == P["tr"].tolist()
    assert max(map(len, sch)) - min(map(len, sch)) <= 1
    # 倍率: 価値の行 1,000・損失 0.4／対 7・損失 0.1 → 1 周の対の合計 7×0.1×倍率 ＝ 1,000×0.4
    f, fc = PD.balanced_factor(1000, 7, 0.4, 0.1)
    assert np.isclose(7 * 0.1 * f, 1000 * 0.4) and np.isclose(fc, 1000 / 7)
    with pytest.raises(ValueError):
        PD.balanced_factor(1000, 0, 0.4, 0.1)
    st = PD.fit_stats([0.1, 0.2, 0.3], [0.2, 0.4, 0.6])
    assert st["slope"] == 0.5 and st["corr"] == 1.0
