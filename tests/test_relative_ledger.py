"""`relative_ledger.py`（T122）の**当てはめゼロの較正**（`calib_of`）の定義の検算。段 7（2026-10-07）: 時計・`z`・`w`・`K`・
不変量の行ごとの計算は Rust だけになり、その式の試験は Python の写しと一緒に消した。
"""
import os
import sys

import pytest

pytestmark = pytest.mark.cpu_infra

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bootstrap  # noqa: F401,E402

_SCRIPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import relative_ledger as RL  # noqa: E402


# --------------------------------------------------------------------------- 6. 較正と κ の物差し
def test_calibration_is_one_and_zero_for_a_perfect_ledger():
    """`Σ = z − W₀` を厳密に満たす帳簿なら `sep = 1`・`bias = 0`（定義の検算）。

    **`W₀` が勝ち負けで釣り合っていること**が `sep = 1` の条件なので、そう作る。"""
    zs = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    w0 = [0.6, 0.6, 0.45, 0.45, 0.5, 0.5]
    xs = [z - w for z, w in zip(zs, w0)]
    c = RL.calib_of(xs, zs, w0)
    assert c["w0_gap"] == pytest.approx(0.0)
    assert c["sep"] == pytest.approx(1.0)
    assert c["bias"] == pytest.approx(0.0)


def test_calibration_sep_shifts_exactly_by_the_w0_gap():
    """**`sep` の理想は `1 − w0_gap`**——釣り合いが崩れた分だけ動く（読むときに必ず `w0_gap` を見る）。"""
    zs = [1.0, 0.0, 1.0, 0.0]
    w0 = [0.7, 0.3, 0.7, 0.3]                      # 勝った側の `W₀` が +0.4 高い
    xs = [z - w for z, w in zip(zs, w0)]
    c = RL.calib_of(xs, zs, w0)
    assert c["w0_gap"] == pytest.approx(0.4)
    assert c["sep"] == pytest.approx(1.0 - 0.4)


def test_calibration_sep_scales_with_the_ledger():
    """**帳簿を c 倍すれば `sep` も c 倍**（尺度に騙されない＝傾きと違って当てはめが要らない）。"""
    zs = [1.0, 0.0, 1.0, 0.0]; w0 = [0.5] * 4
    xs = [z - w for z, w in zip(zs, w0)]
    assert RL.calib_of([3.0 * x for x in xs], zs, w0)["sep"] == pytest.approx(3.0)


def test_calibration_is_none_without_both_labels():
    assert RL.calib_of([1.0, 2.0], [1.0, 1.0], [0.5, 0.5])["sep"] is None


def test_kappa_table_w_bar_was_measured_with_the_tables_own_sigma():
    """**`κ = w(D)/w̄` の分子と分母が同じ表の同じ実測から来る**（2026-10-08・ユーザ決定「後で実測値が変わった場合にずれない
    ような仕組み」）——既定の式の検算（値は表から読む・数を固定しない）。

    1. 出所の検算: 表の `w_bar_provenance` が覚えている入力（`σ_T`・`σ_rel`・輪郭）と `w̄` が今の表と一致する。
    2. 定義上の不変量: 測った実行の `Σκ/n × (その実行の分母) = E[w(D)]` を `w̄` で割ると 1（4 桁の丸めの内）
       ＝**測った記録の上では `κ` の平均が 1**。
    3. 1 つの読み込み: `κ` を使う器（`theory_bridge`・`relative_ledger`・`kappa_vector`）が引く組は、実の記録なら
       `σ_D = √2·σ_T[合成]`・`w̄[合成]`（別のセットの規約）。
    4. ずれの検出: `σ_T` だけ書き換えた表は、`w̄` を測り直すまで落ちる。"""
    import copy
    import json as _json
    import math
    import crossing_bridge as CB
    import theory_rs as TR
    with open(CB.HARM_PROFILE_PATH, encoding="utf-8") as fh:
        tbl = _json.load(fh)
    body = CB.THETA_BODY_MODE
    assert CB.check_w_bar_provenance(tbl)
    for key in ("real", "syn"):
        prov = tbl["w_bar_provenance"][body][key]
        wb = float(tbl["w_bar"][body][key])
        assert prov["inputs"]["sigma_t"] == tbl["sigma_t"][body][CB._other(key)]
        assert prov["sigma_turn_in_run"] == prov["inputs"]["sigma_t"]
        mean_kappa = prov["kappa_sum"] / prov["kappa_n"] * prov["w_bar_in_run"] / wb
        assert abs(mean_kappa - 1.0) <= 0.5e-4 / wb + 1e-12
    rec = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "f_identity", "rec")
    old = dict(TR.CLOCK)
    try:
        assert CB.record_kind([rec]) == "real"
        kc = CB.kappa_clock([rec])
        assert kc["sigma_t"] == tbl["sigma_t"][body]["syn"] and kc["w_bar"] == tbl["w_bar"][body]["syn"]
        assert TR.CLOCK["SIGMA_D"] == math.sqrt(2.0) * tbl["sigma_t"][body]["syn"]
        assert TR.CLOCK["W_BAR"] == tbl["w_bar"][body]["syn"]
    finally:
        TR.CLOCK.clear()
        TR.CLOCK.update(old)
    moved = copy.deepcopy(tbl)
    moved["sigma_t"][body]["real"] = float(moved["sigma_t"][body]["real"]) + 0.001
    with pytest.raises(ValueError, match="w_bar を測り直す"):
        CB.check_w_bar_provenance(moved)
