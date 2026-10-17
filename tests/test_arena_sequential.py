"""アリーナの逐次判定（`opcg_sim.loop.arena.sequential_stop`・`docs/reports/2026-10-17_arena_sequential.md` §1.3）。

手で導ける例: σ = 1/2・α = 0.05・ρ = 1 で全勝なら、平均 1・半幅 u(t/4)/t、u(v) = √((v+1)·ln((v+1)/0.0025))。
下限 1 − u(t/4)/t > 0.50 ⇔ u(t/4) < t/2。t = 10: √(3.5·ln 1400) = 5.036 > 5（続ける）、t = 11: √(3.75·ln 1500) = 5.237 < 5.5
（止める）。全敗は線の反対側で同じ位置に止まる。五分の並び（1, 0, 1, 0, …）は線を跨いだまま止まらない。
"""
import math

import pytest

import _bootstrap  # noqa: F401

from opcg_sim.loop import arena as A

pytestmark = pytest.mark.cpu_infra


def test_sequential_stop_position_matches_hand_calculation():
    assert math.sqrt(3.5 * math.log(3.5 / 0.0025)) > 5.0 > 0
    assert math.sqrt(3.75 * math.log(3.75 / 0.0025)) < 5.5
    up = A.sequential_stop([1.0] * 50, planned=50, line=0.50, strict=True, rho=1.0)
    assert (up["stopped"], up["side"]) == (11, "above")
    down = A.sequential_stop([0.0] * 50, planned=50, line=0.50, strict=True, rho=1.0)
    assert (down["stopped"], down["side"]) == (11, "below")
    even = A.sequential_stop([1.0, 0.0] * 100, planned=200, line=0.50, strict=True)
    assert even["stopped"] is None
    # ρ は計画のペア数で半幅が最小になる点（両隣より狭い）
    r = A.cs_rho(192)
    assert A.cs_halfwidth(192, r) <= min(A.cs_halfwidth(192, r * 1.05), A.cs_halfwidth(192, r / 1.05))
