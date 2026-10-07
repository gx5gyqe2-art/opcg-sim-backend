"""`theory_bridge.py`（理論と勝敗を繋ぐ橋・T28）の**局の突き合わせと集計**の符号を固める。

**符号が 1 つ狂うと結論が反転する器**なので、向きを値で押さえる。段 7（2026-10-07）: 行の価格（`s`・`g`）と守りの窓は
Rust だけになり、その式の試験は Python の写しと一緒に消した。残るのは Python の集計（`pair_games`・`summarise`・`calibration`）。
"""
import os
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.cpu_infra

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bootstrap  # noqa: F401,E402

_SCRIPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import theory_bridge as B  # noqa: E402


def _seat(s_atk=0.0, s_grd=0.0, n_atk=5, n_grd=5, z=1.0, silent=0):
    return {"z": z, "s_atk": s_atk, "s_grd": s_grd, "n_atk": n_atk, "n_grd": n_grd,
            "n_silent": silent, "v0": [0.1]}


def test_the_pairing_is_my_loss_minus_the_opponents():
    """`ΔS > 0` は**自席の方が取りこぼしが少ない**＝理論に近い打ち方をした、の向き。"""
    per = {(1, 0): _seat(s_atk=-0.1, z=1.0), (1, 1): _seat(s_atk=-0.5, z=0.0)}
    got = B.pair_games(per)
    assert len(got) == 1 and got[0]["dS"] == pytest.approx(0.4)
    assert got[0]["z"] == 1.0


def test_the_per_row_version_divides_each_seat_by_its_own_count():
    """**手数で正規化した版**——先手は手番が 1 つ多いので生の和は席順を拾いうる。"""
    per = {(1, 0): _seat(s_atk=-1.0, n_atk=10, n_grd=0),
           (1, 1): _seat(s_atk=-1.0, n_atk=5, n_grd=0, z=0.0)}
    got = B.pair_games(per)[0]
    assert got["dS"] == pytest.approx(0.0)              # 生の和では差が無い
    assert got["dS_per_row"] == pytest.approx(-0.1 + 0.2)   # 1 手あたりでは差が出る
    assert got["dn"] == 5


def test_the_verdict_reads_the_normalised_slope():
    """**判定は正規化した版**で出す（生の和は手数の差に汚染されうる）。"""
    rng = np.random.default_rng(0)
    pairs = []
    for i in range(200):
        d = float(rng.normal())
        pairs.append({"seed": i, "z": 1.0 if d > 0 else 0.0, "dS": d, "dS_per_row": d,
                      "dS_atk": d, "dS_grd": d, "n": 10, "dn": 0, "silent": 0, "v0": 0.1})
    assert B.summarise(pairs, reps=50)["verdict"] == "bridge_holds"
    for p in pairs:                                     # 向きだけ反転させる
        p["dS_per_row"] = -p["dS_per_row"]
    assert B.summarise(pairs, reps=50)["verdict"] == "bridge_inverted"


def test_the_pairing_of_gain_is_my_gain_minus_the_opponents_per_row():
    a = dict(_seat(n_atk=4, n_grd=4, z=1.0), g_atk=0.40, g_grd=-0.20)
    b = dict(_seat(n_atk=2, n_grd=2, z=1.0), g_atk=0.10, g_grd=-0.10)
    per = {(1, 0): a, (1, 1): b}
    p = B.pair_games(per)[0]
    assert p["dG"] == pytest.approx(0.20 - 0.0)
    assert p["dG_atk"] == pytest.approx(0.30) and p["dG_grd"] == pytest.approx(-0.10)
    assert p["dG_per_row"] == pytest.approx(0.20 / 8 - 0.0 / 4)
    # 旧い席（`g` を持たない）でも落ちない＝0 として扱う
    old = {(2, 0): _seat(z=1.0), (2, 1): _seat(z=0.0)}
    assert B.pair_games(old)[0]["dG"] == 0.0


def test_the_calibration_table_is_quantiles_with_their_raw_win_rate():
    """**較正表は当てはめない**——等分位ごとの実勝率をそのまま並べる。"""
    rng = np.random.default_rng(0)
    pairs = []
    for _ in range(200):
        x = float(rng.normal())
        z = 1.0 if rng.random() < 1.0 / (1.0 + np.exp(-3.0 * x)) else 0.0
        pairs.append({"dG_per_row": x, "z": z})
    cal = B.calibration(pairs, "dG_per_row", bins=5)
    assert len(cal["bins"]) == 5 and sum(r["n"] for r in cal["bins"]) == 200
    assert cal["bins"][0]["x_mean"] < cal["bins"][-1]["x_mean"]
    assert cal["monotone"] is True and cal["spread"] > 0.5
    assert B.calibration(pairs[:5], "dG_per_row", bins=5) is None      # 局数が足りなければ出さない
