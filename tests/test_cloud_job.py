"""Cloud Run ジョブの入口（`opcg_sim/loop/cloud_job.py`・`docs/cloud_run_jobs.md`）のシャードの割り当て。

タスク番号 → アリーナの seed の区間が、帯からはみ出さず・タスクの間で重ならず・全タスクで計画と一致すること
（同じ対局を二重に打つと CI が不当に狭まり、抜けると判定が出ない）。写した `plan_bands` が元と同じ式であること。
"""
import pytest

import _bootstrap  # noqa: F401

from opcg_sim.loop import arena as A
from opcg_sim.loop import cloud_job as CJ

pytestmark = pytest.mark.cpu_infra


@pytest.mark.parametrize("pairs,bands,n", [(400, 4, 4), (400, 4, 7), (10, 4, 4), (3, 4, 8), (2800, 4, 64), (5, 1, 2)])
def test_arena_runs_cover_plan_once_within_bands(pairs, bands, n):
    plan = A.plan_bands(pairs, bands, 71000)
    assert CJ.plan_bands(pairs, bands, 71000) == plan
    band_of = {s: b for b, xs in enumerate(plan) for s in xs}
    got = []
    for i in range(n):
        for start, length in CJ.arena_runs(pairs, bands, 71000, i, n):
            seeds = list(range(start, start + length))
            assert {band_of.get(s) for s in seeds} in ({band_of[start]},)     # 区間は 1 つの帯の中
            got += seeds
    assert sorted(got) == sorted(s for xs in plan for s in xs)                 # 重ならず・抜けない
    assert len(got) == len(set(got))
    sizes = [sum(ln for _s, ln in CJ.arena_runs(pairs, bands, 71000, i, n)) for i in range(n)]
    assert max(sizes) - min(sizes) <= 1                                         # タスクの間で偏らない
