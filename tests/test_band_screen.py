"""`band_screen.py`（帯に入れるべき変数を洗う）の算術を固める。

**順位が変わると「次に何を帯に入れるか」が変わる**ので、採点の定義を固定する。
記録もエンジンも要らない。
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

import band_screen as B  # noqa: E402
from opcg_sim.learned import n_rel_feat as F  # noqa: E402

WIDTH = 94 + len(F.EXTRA_COLS)


def _make(n=600, seed=0):
    """帯 1 つ・列 3 本の人工データ。

    **`X` を 2 つの独立な成分に分けるのが要点**——`u`（z も動かす＝交絡の源）と
    `x_free`（z と無関係）。こうしないと「X とだけ相関する列」が作れない
    （`z` が `X` 全体に依ると、`X` と相関する列は必ず `z` とも相関してしまう）。

    | 列 | 作り | 期待 |
    |---|---|---|
    | 0 | `u` 由来 | **交絡**（`r_z` も `r_x` も立つ）＝最上位 |
    | 1 | `z` にだけ効く雑音 | `r_x ≈ 0` ＝下位 |
    | 2 | `x_free` 由来 | `r_z ≈ 0` ＝下位 |
    """
    rng = np.random.default_rng(seed)
    u = rng.normal(size=n)                             # 交絡の源
    x_free = rng.normal(size=n)                        # z と無関係な X の成分
    x = u + x_free
    z_only = rng.normal(size=n)
    z = u + z_only                                     # z は u から（x_free からではない）
    sc = np.zeros((n, WIDTH))
    sc[:, 0] = u + rng.normal(scale=0.2, size=n)       # 交絡
    sc[:, 1] = z_only + rng.normal(scale=0.2, size=n)  # z とだけ
    sc[:, 2] = x_free + rng.normal(scale=0.2, size=n)  # X とだけ
    xs = np.stack([x, np.zeros(n), np.zeros(n)], 1)
    return ["b"] * n, sc, xs, z


def test_the_score_needs_both_halves_of_a_confounder():
    """**交絡は「z を予測する」かつ「説明変数と相関する」の両方が要る**＝だから積で採点する。

    z としか相関しない列・X としか相関しない列は、どちらも順位で下に来なければならない。
    """
    out = B.screen(*_make())
    rank = {r["col"]: i for i, r in enumerate(out["cols"])}
    assert rank[0] < rank[1] and rank[0] < rank[2]     # 交絡が最上位
    by = {r["col"]: r for r in out["cols"]}
    assert by[0]["score"] > by[1]["score"] and by[0]["score"] > by[2]["score"]
    # 片翼だけの列は r_z か r_x のどちらかが 0 付近
    assert abs(by[1]["r_x"]["lt_leader"]) < 0.2
    assert abs(by[2]["r_z"]) < 0.25


def test_a_column_already_in_the_band_collapses_to_zero():
    """**帯に入っている列は 0 に潰れる**——潰れなければ帯の実装がおかしい（器の検算）。"""
    bands, sc, xs, z = _make()
    # 列 0 の値そのもので帯を切る＝帯の中で動かなくなる
    bands = [("b", round(float(v), 1)) for v in sc[:, 0]]
    out = B.screen(bands, sc, xs, z, min_rows=2)
    by = {r["col"]: r for r in out["cols"]}
    assert abs(by[0]["score"]) < 0.02


def test_my_own_board_columns_are_flagged_not_recommended():
    """**自分の場の列は「交絡」ではなく「処置」**＝高い score が出ても帯に入れられない。

    `attackers_left`（まだ攻撃できる自キャラ数）は**実測で `my_field_n` とほぼ同じ動き**を
    するのに、初版は印を付けず交絡候補の 4 位に出してしまった（2026-09-14）。
    """
    own = B.own_board_cols(WIDTH)
    assert 8 in own                                    # my_field_n
    assert all(c in own for c in (63, 64, 65))         # my_field_agg
    off = WIDTH - len(F.EXTRA_COLS)
    assert off + F.EXTRA_COLS.index("attackers_left") in own
    # 相手側・デッキ側は処置ではない
    assert off + F.EXTRA_COLS.index("opp_pool_max_power") not in own
    assert 9 not in own                                # opp_field_n


def test_the_column_names_resolve_for_every_index():
    """全列に名前が付く（付かないと順位表が読めない）。"""
    names = [B.col_name(i, WIDTH) for i in range(WIDTH)]
    assert len(names) == WIDTH and all(names)
    assert B.col_name(0, WIDTH) == "my_life"
    assert B.col_name(8, WIDTH) == "my_field_n"
    off = WIDTH - len(F.EXTRA_COLS)
    assert B.col_name(off, WIDTH) == "extra.%s" % F.EXTRA_COLS[0]


def test_a_constant_column_scores_zero_instead_of_dividing_by_zero():
    bands, sc, xs, z = _make()
    sc[:, 3] = 1.0
    out = B.screen(bands, sc, xs, z)
    by = {r["col"]: r for r in out["cols"]}
    assert by[3]["score"] == 0.0 and by[3]["r_z"] == 0.0


def test_thin_data_returns_none_rather_than_a_number():
    assert B.screen(["b"] * 3, np.zeros((3, WIDTH)), np.zeros((3, 3)), np.zeros(3)) is None
