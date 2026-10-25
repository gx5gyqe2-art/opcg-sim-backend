"""相手の集団で教材を作る（`record_gen --opp-pool`・WP `league-gen`・`search_efficiency_plan.md` §7.56）。

基盤健全性（`cpu_infra`）: 教材の生成の内部機構だけを見る（ゲームプレイの正しさには触れない）。
エンジンを回さない（引きの規則だけ）。実際の記録の列・既定で 1 bit も変わらないことは報告の試しで確かめた
（`docs/reports/2026-10-25_league_gen.md`）。
"""
import os
from collections import Counter

import pytest

import conftest  # noqa: F401
import _bootstrap  # noqa: F401

from opcg_sim.loop import record_gen as R

pytestmark = pytest.mark.cpu_infra


def test_parse_opp_pool_names_and_weights():
    pool = R.parse_opp_pool("r24:2,r23:1,/x/y/nrel_foo.npz:1")
    assert [n for n, _p, _w in pool] == ["r24", "r23", "nrel_foo"]
    assert pool[0][1] == os.path.join(R.POOL_NET_DIR, "nrel_r24.npz")
    assert pool[2][1] == "/x/y/nrel_foo.npz"
    assert [w for _n, _p, w in pool] == [0.5, 0.25, 0.25]
    for bad in ("r24", "r24:0", "r24:1,r24:1", ""):
        with pytest.raises(ValueError):
            R.parse_opp_pool(bad)


def test_opp_draw_is_seeded_alternating_and_weighted():
    pool = R.parse_opp_pool("r24:0.5,r23:0.2,r22:0.15,r13d:0.15")
    seeds = range(200000, 220000)
    draws = [R.opp_draw(s, pool) for s in seeds]
    assert draws == [R.opp_draw(s, pool) for s in seeds]          # seed だけで決まる
    assert all(seat == ("p1" if s % 2 == 0 else "p2") for s, (seat, _o) in zip(seeds, draws))
    c = Counter(o for _s, o in draws)
    for name, _p, w in pool:
        assert abs(c[name] / len(draws) - w) < 0.015, (name, c[name])
