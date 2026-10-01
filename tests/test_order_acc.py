"""`tests/scripts/order_acc.py` の純関数（順序一致・訪問の下限・帯・集計）。

基盤健全性（`cpu_infra`）: 読み取り専用の計器の算術を固める。順序一致は
**同値のペアを母数から外す**ところが要——外さないと「並べられている」側へ勝手に寄る。
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

import order_acc as O  # noqa: E402


def test_band_of_uses_the_absolute_value():
    assert O.band_of(0.0) == "close"
    assert O.band_of(-0.2) == "close"
    assert O.band_of(0.21) == "mid"
    assert O.band_of(-0.6) == "mid"
    assert O.band_of(0.61) == "decided"
    assert O.band_of(-0.95) == "decided"


def test_pair_agree_perfect_and_reversed():
    x = [1.0, 2.0, 3.0]
    assert O.pair_agree(x, [10.0, 20.0, 30.0]) == (3, 3)
    assert O.pair_agree(x, [30.0, 20.0, 10.0]) == (0, 3)


def test_pair_agree_drops_ties_from_the_denominator():
    """同値のペアは「一致」にも「不一致」にも数えない。"""
    x = [1.0, 1.0, 2.0]                      # (0,1) は x が同値
    y = [5.0, 9.0, 9.0]                      # (1,2) は y が同値
    agree, total = O.pair_agree(x, y)
    assert total == 1                        # 残るのは (0,2) だけ
    assert agree == 1
    # eps を与えると「ほぼ同値」も外れる
    assert O.pair_agree([1.0, 1.001], [0.0, 1.0], x_eps=0.01) == (0, 0)


def test_pair_agree_handles_a_single_candidate():
    assert O.pair_agree([1.0], [1.0]) == (0, 0)


def test_q_floor_takes_the_larger_of_absolute_and_fraction():
    n = [80.0, 40.0, 40.0]                   # 総訪問 160
    assert O.q_floor(n, n_min=5, n_min_frac=0.05) == 8.0     # 割合が勝つ
    assert O.q_floor(n, n_min=20, n_min_frac=0.05) == 20.0   # 絶対数が勝つ
    assert O.q_floor([1.0, 1.0], n_min=5, n_min_frac=0.05) == 5.0


def test_row_stats_reports_the_cost_of_a_wrong_order():
    """方策の 1 位と探索の 1 位が違えば、その Q 差が `q_cost` に出る。"""
    n = np.array([100.0, 60.0])              # 探索は手 0
    q = np.array([0.30, 0.10])
    p = np.array([0.2, 0.8])                 # 方策は手 1
    st = O.row_stats(n, q, p)
    assert st["k"] == 2
    assert st["top1_pn"] is False
    assert st["q_cost"] == pytest.approx(0.20)
    assert st["pn_pairs"] == 1 and st["pn_agree"] == 0
    assert st["pq_pairs"] == 1 and st["pq_agree"] == 0

    # 順序が合っていれば損は 0
    st2 = O.row_stats(n, q, np.array([0.9, 0.1]))
    assert st2["top1_pn"] is True
    assert st2["q_cost"] == pytest.approx(0.0)
    assert st2["pn_agree"] == 1 and st2["pq_agree"] == 1


def test_row_stats_skips_q_when_visits_are_too_thin():
    """訪問が下限に届く候補が 1 本以下なら q のペアは作らない（`q_cost` も出さない）。"""
    n = np.array([158.0, 4.0, 1.0])          # 下限（総訪問の 5%＝9）を超えるのは 1 本だけ
    q = np.array([0.5, -0.9, 0.9])
    p = np.array([0.1, 0.2, 0.7])
    st = O.row_stats(n, q, p)
    assert st["pq_pairs"] == 0
    assert st["q_cost"] is None
    assert st["pn_pairs"] == 3               # 訪問との順序は全ペアで測れる


def test_row_stats_prior_entropy():
    st = O.row_stats(np.array([1.0, 1.0]), np.array([0.0, 0.0]), np.array([0.5, 0.5]))
    assert st["prior_entropy"] == pytest.approx(np.log(2))
    st0 = O.row_stats(np.array([1.0, 1.0]), np.array([0.0, 0.0]), np.array([0.0, 0.0]))
    assert st0["prior_entropy"] is None


def test_block_pools_pairs_not_rows():
    """集計は**ペアの合算**（行ごとの割合の平均ではない）＝候補が多い行が正しく重く出る。"""
    recs = [
        {"k": 10, "pn_agree": 40, "pn_pairs": 45, "pq_agree": 2, "pq_pairs": 4,
         "top1_pn": True, "q_cost": 0.0, "prior_entropy": 1.0, "band": "close"},
        {"k": 2, "pn_agree": 0, "pn_pairs": 1, "pq_agree": 0, "pq_pairs": 1,
         "top1_pn": False, "q_cost": 0.2, "prior_entropy": 0.5, "band": "close"},
    ]
    b = O.block(recs)
    assert b["rows"] == 2
    assert b["pairs_pn"] == 46
    assert b["order_acc_pn"] == pytest.approx(40 / 46, abs=1e-4)
    assert b["order_acc_pq"] == pytest.approx(0.4)
    assert b["top1_pn"] == pytest.approx(0.5)
    assert b["q_cost_mean"] == pytest.approx(0.1)
    assert b["q_cost_n"] == 2
    assert b["pn_edge"] == pytest.approx(40 / 46 - 0.5, abs=1e-4)
    assert O.block([]) is None


def test_block_handles_rows_without_usable_pairs():
    recs = [{"k": 1, "pn_agree": 0, "pn_pairs": 0, "pq_agree": 0, "pq_pairs": 0,
             "top1_pn": True, "q_cost": None, "prior_entropy": None, "band": "mid"}]
    b = O.block(recs)
    assert b["order_acc_pn"] is None and b["order_acc_pq"] is None
    assert b["q_cost_mean"] is None and b["prior_entropy_mean"] is None
    assert b["pn_edge"] is None
