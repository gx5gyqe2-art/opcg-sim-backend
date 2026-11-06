"""ドンだけが違う行の重み `--jitter-weight balanced`（`docs/search_efficiency_plan.md` §7.83・WP `don-weight`）。

基盤健全性（学習パイプラインの内部機構だけを見る）。2 本だけ（CLAUDE.md の最小限の規約）:
- 既定（`--jitter-weight off`）は `don_jitter` を読まず、価値の更新に重みを渡さない＝今までの訓練と 1 bit も変わらない。
- `balanced` の印は「ドンを動かした局の、その手番の手番の席の最初の main の行の step 以後（両席）」・動かしていない局と
  ふつうの波は印なし・行の並びは pack と同じ。重みは訓練の行で「印の行の合計＝それ以外の合計」・平均 1・倍率＝行数の比。
"""
import numpy as np
import pytest

import _bootstrap  # noqa: F401

from opcg_sim.learned.train import n_rel_train as NRT

pytestmark = pytest.mark.cpu_infra


def _shard(path, rows, dj=None):
    """rows: (seed, who, step, kind, turn) の並び。dj: 行ごとの (手番, 席, 符号)（局ごとに同じ値）を seed → 値で。"""
    cols = list(zip(*rows))
    kw = dict(z=np.zeros(len(rows), np.float16), seed=np.array(cols[0], np.int64), who=np.array(cols[1], np.int8),
              step=np.array(cols[2], np.int32), kind=np.array(cols[3], np.int8), turn=np.array(cols[4], np.int16))
    if dj is not None:
        kw["don_jitter"] = np.array([dj[s] for s in cols[0]], np.int16)
    np.savez(path, **kw)


def test_default_off_reads_nothing(tmp_path, monkeypatch):
    d = tmp_path / "w"
    d.mkdir()
    _shard(d / "n_record_00000.npz", [(5, 0, 0, 0, 1)])
    seen = {}
    monkeypatch.setattr(NRT, "train", lambda args: seen.setdefault("args", args) and 0)
    NRT.main(["train", "--in", str(d), "--out", "o.npz"])
    assert seen["args"].jitter_weight == "off"
    src = open(NRT.__file__, encoding="utf-8").read()
    # 重みは「切でなければ」だけ作られ、価値の更新に渡るのは重みがあるときだけ（既定の経路は従来どおり）
    assert 'if jw != "off":' in src and 'if v_w is not None:\n                        kw["vw"]' in src


def test_balanced_mask_and_weights(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    # 局 5: 手番 4 に席 1 のドンを動かした。手番 4 の席 1 は窓（step 3）の後に main（step 4）＝起点は step 4。
    # 行の並びは step 順でなくてよい。局 6: 動かしていない（符号 0）。局 7: 手番 9 を選んだが手番 9 の前に決着（main 無し）。
    rows = [(5, 0, 6, 0, 5),    # 以後（相手の席も）
            (5, 1, 4, 0, 4),    # 起点
            (5, 0, 2, 0, 3),    # 前
            (5, 1, 3, 1, 4),    # 同じ手番の窓・起点より前
            (5, 0, 5, 1, 4),    # 以後（相手の席の窓）
            (6, 0, 0, 0, 1), (6, 1, 1, 0, 2),
            (7, 0, 0, 0, 1)]
    dj = {5: (4, 1, -1), 6: (3, 0, 0), 7: (9, 0, 1)}
    _shard(a / "n_record_00000.npz", rows, dj)
    _shard(b / "n_record_00000.npz", [(8, 0, 0, 0, 1), (8, 1, 1, 0, 2)])        # ふつうの波（列なし）
    m, info = NRT.load_jitter_mask([str(a), str(b)])
    assert m.tolist() == [True, True, False, False, True, False, False, False, False, False]
    assert info == {"shards_with_don_jitter": 1}
    # 訓練の行＝0〜8（9 は取り置き）: 印 3 行・それ以外 6 行 → 倍率 2・重みの平均 1・合計が釣り合う
    w, wi = NRT.balanced_weights(m, np.arange(9))
    tr = np.arange(9)
    assert wi["ratio"] == 2.0 and wi["rows_jitter_train"] == 3 and wi["rows_other_train"] == 6
    assert np.isclose(w[tr][m[tr]].sum(), w[tr][~m[tr]].sum())
    assert np.isclose(w[tr].mean(), 1.0) and np.isclose(wi["w_jitter"] / wi["w_other"], 2.0)
    with pytest.raises(ValueError):
        NRT.balanced_weights(np.zeros(4, bool), np.arange(4))
