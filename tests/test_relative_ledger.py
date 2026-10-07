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
