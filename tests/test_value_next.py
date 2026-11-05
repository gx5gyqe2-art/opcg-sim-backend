"""価値の教師の受け口 `--z-mode next_v0`（次の自分のメインの判断点の根の値・`docs/search_efficiency_plan.md` §7.81・WP `value-next`）。

基盤健全性（学習パイプラインの内部機構だけを見る）。2 本だけ（CLAUDE.md の最小限の規約）:
- 既定（`--z-mode z`）は `pol_v0` を読まず、教師は同じオブジェクト＝今までの訓練と 1 bit も変わらない。
- `next_v0` は「同じ局・同じ席の、その行の step より後の最初の根の推定のある main の行」の `pol_v0` に差し替え、
  無い行（局の最後・根の推定が無い main しか残らない行）は勝敗のまま。印の立つ行自身も自分の値ではなく次の行の値。
  行の並びは pack と同じ（シャードの順・`--z-in` は後ろ）。局は席ごとに引く（相手の席の行は飛ばす）。
"""
import numpy as np
import pytest

import _bootstrap  # noqa: F401

from opcg_sim.learned.train import n_rel_train as NRT

pytestmark = pytest.mark.cpu_infra


def _shard(path, rows, pol_n):
    """rows: (z, seed, who, step, kind, turn, pol_v0, pol_len) の並び。pol_n は候補ごとの訪問（pol_len の合計ぶん）。"""
    cols = list(zip(*rows))
    np.savez(path, z=np.array(cols[0], np.float16), seed=np.array(cols[1], np.int64), who=np.array(cols[2], np.int8),
             step=np.array(cols[3], np.int32), kind=np.array(cols[4], np.int8), turn=np.array(cols[5], np.int16),
             pol_v0=np.array(cols[6], np.float16), pol_len=np.array(cols[7], np.int32), pol_n=np.array(pol_n, np.float32))


def test_default_does_not_read_pol_v0(tmp_path, monkeypatch):
    d = tmp_path / "w"
    d.mkdir()
    _shard(d / "n_record_00000.npz", [(1, 5, 0, 0, 0, 1, 0.5, 1), (1, 5, 0, 1, 0, 1, 0.25, 1)], [3, 3])
    seen = {}
    monkeypatch.setattr(NRT, "train", lambda args: seen.setdefault("args", args) and 0)
    NRT.main(["train", "--in", str(d), "--out", "o.npz"])
    assert seen["args"].z_mode == "z"
    called = []
    monkeypatch.setattr(NRT, "load_next_v0", lambda *a, **k: called.append(1))
    z = np.array([1, 1], np.float16)
    out, n = NRT.value_teacher(z, np.array([5, 5]), np.array([0, 1]), (np.array([], int), np.array([], int)),
                               np.array([], np.float32), "z")
    assert out is z and n == 0 and not called


def test_next_v0_takes_next_own_main_root_value(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    # 局 5: 席 0 の main（step 0・根あり）→ 席 0 の窓（step 1）→ 席 1 の main（step 2・根あり）→ 席 0 の main
    # （step 3・候補 1 つ・訪問 0＝根の推定なし）→ 席 0 の main（step 4・根あり・次の手番）→ 席 1 の続き（step 5）。
    # 行の並びは step 順でなくてもよい（鍵で引く）。
    rows = [(1, 5, 0, 4, 0, 3, 0.75, 2),      # 席 0・main・根あり（自分の値 0.75 は使わない・後に無い→ z のまま）
            (1, 5, 0, 0, 0, 1, 0.5, 2),       # 席 0・main・根あり → 次は step 4 の 0.75（step 3 は根の推定なし）
            (1, 5, 0, 1, 1, 2, 0.0, 0),       # 席 0・窓（相手の手番）→ 0.75
            (-1, 5, 1, 2, 0, 2, -0.5, 1),     # 席 1・main・根あり → 席 1 の後の main は無い → z のまま
            (1, 5, 0, 3, 0, 3, 0.0, 1),       # 席 0・main・候補 1 つ・訪問 0 → 0.75
            (-1, 5, 1, 5, 2, 3, 0.0, 0)]      # 席 1・続き → 後に席 1 の main は無い → z のまま
    _shard(a / "n_record_00000.npz", rows, [4, 4, 4, 4, 2, 0])
    _shard(a / "n_record_00001.npz", [(-1, 7, 1, 0, 0, 1, -0.25, 1), (-1, 7, 1, 1, 0, 2, 0.125, 1)], [1, 1])
    _shard(b / "n_record_00000.npz", [(1, 8, 0, 0, 1, 1, 0.0, 0), (1, 8, 0, 1, 0, 1, 0.375, 1)], [5])  # --z-in は後ろ
    t, m, info = NRT.load_next_v0([str(a)], [str(b)])
    assert m.tolist() == [False, True, True, False, True, False, True, False, True, False]
    assert t.tolist() == [1.0, 0.75, 0.75, -1.0, 0.75, -1.0, 0.125, -1.0, 0.375, 1.0]
    assert info["replaced"] == 5 and info["kept"] == 5 and info["rows"] == 10
    assert info["same_turn"] == 2 and info["later_turn"] == 3        # 局 5 の step 3 と局 8 の窓は同じ手番の main が教師
    assert info["by_kind"] == {"main": 3, "window": 2, "commit": 0}
    # 鍵で引く器そのもの: 局・席をまたがない・step が後の行だけ
    out, mask, src = NRT.next_v0_teacher([1, 1, -1], [5, 5, 5], [0, 0, 1], [10, 2, 3], [0.5, 0.25, -0.5],
                                         [True, True, True])
    assert mask.tolist() == [False, True, False] and out.tolist() == [1.0, 0.5, -1.0] and src.tolist() == [-1, 0, -1]
