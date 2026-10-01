"""`nu_ledger.py`（T44・`ν` の中身を因子ごとに実測と突き合わせる）の算術を固める。

**組み直しは実測の因子だけで作る**（攻撃 1 回の実現 × 攻撃の率 × `R` × 生存）——式 `nu_of` は
比較の相手としてしか出てこない。その約束を値で押さえる。
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

import nu_ledger as NL  # noqa: E402
import theory_order as T  # noqa: E402


def test_the_bands_are_the_measured_nu_bands():
    """帯は `nu_measure`／`price_realised` と同じ（リーダー未満／飽和まで／飽和超え）。"""
    assert NL.band_of(3000.0, 5000.0) == "lt_leader"
    assert NL.band_of(5000.0, 5000.0) == "leader_to_sat"
    assert NL.band_of(7000.0, 5000.0) == "leader_to_sat"
    assert NL.band_of(7000.0 + 2 * NL.PWR_EPS, 5000.0) == "over_sat"
    assert set(NL.BANDS) == set(NL.NU_MEAS)


def _atk(**per_band):
    d = {b: {"price": [], "real": []} for b in NL.BANDS}
    d["leader"] = {"price": [], "real": []}
    for b, (price, real, n) in per_band.items():
        d[b]["price"] = [price] * n
        d[b]["real"] = [real] * n
    return d


def _body(**per_band):
    d = {b: {"present": 0, "attacked": 0, "can": 0, "nu_formula": [], "ko_p": []} for b in NL.BANDS}
    for b, (present, attacked, nu_f, kop) in per_band.items():
        d[b].update(present=present, attacked=attacked, can=present,
                    nu_formula=[nu_f] * present, ko_p=[kop] * present)
    return d


def test_the_composition_multiplies_the_measured_factors_only():
    """組み直し＝攻撃 1 回の実現 × 攻撃の率 × `R` × (1 − ko_p)。式の値は比較の相手にしか使わない。"""
    atk = _atk(leader_to_sat=(0.0634, 0.0577, 50))
    body = _body(leader_to_sat=(100, 89, 0.2031, 0.274))
    out = NL.summarise(atk, body)
    c = out["compose"]["leader_to_sat"]
    flow = 0.0577 * 0.89
    assert c["flow_per_turn"] == pytest.approx(flow, abs=1e-4)
    assert c["composed_nu_const_kop"] == pytest.approx(flow * NL.R_TURNS * (1 - NL.KO_P), abs=1e-4)
    assert c["composed_nu_curve_kop"] == pytest.approx(flow * NL.R_TURNS * (1 - 0.274), abs=1e-4)
    assert c["nu_measured"] == NL.NU_MEAS["leader_to_sat"]
    assert c["nu_formula"] == pytest.approx(0.2031)
    # 式の値を変えても組み直しは動かない（式に依らない）
    body2 = _body(leader_to_sat=(100, 89, 0.9, 0.274))
    assert NL.summarise(atk, body2)["compose"]["leader_to_sat"]["composed_nu_const_kop"] == \
        pytest.approx(c["composed_nu_const_kop"])


def test_per_attack_ratio_and_attack_rate_are_reported_per_band():
    atk = _atk(lt_leader=(0.02, 0.022, 20), over_sat=(0.07, 0.093, 20))
    body = _body(lt_leader=(50, 35, 0.05, 0.25), over_sat=(40, 34, 0.23, 0.2))
    out = NL.summarise(atk, body)
    assert out["per_attack"]["lt_leader"]["ratio"] == pytest.approx(1.1)
    assert out["per_attack"]["over_sat"]["ratio"] == pytest.approx(0.093 / 0.07, rel=1e-3)
    assert out["per_body_turn"]["lt_leader"]["attack_rate"] == pytest.approx(0.7)
    assert out["per_body_turn"]["over_sat"]["attack_rate"] == pytest.approx(0.85)
    # 標本が足りない帯は出さない（`leader_to_sat` は空）
    assert "leader_to_sat" not in out["per_attack"] and "leader_to_sat" not in out["compose"]


def test_the_formula_assumes_an_attack_every_turn_which_the_ledger_can_contradict():
    """式の攻撃項は毎ターン殴る前提（率 1）——台帳の率がそれより低ければ式は過大になる。"""
    pw, olp = 6000.0, 5000.0
    v = T.nu_of(pw, olp, NL.R_TURNS, is_blocker=False, mode="base")   # 攻撃項だけ（身代わり・ブロックなし）
    per_turn = T.attack_value(pw, olp, True)
    assert v == pytest.approx(per_turn * NL.R_TURNS * (1 - T.KO_P), rel=1e-6)   # 率 1 が入っている
