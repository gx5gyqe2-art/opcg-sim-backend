"""`pre_settle_decomp.py`（T145・決着前の較正の「悪化」を構成と予測に分ける）の算術を固める。

押さえるのは 3 つ:

1. **`split_rows`** は `crossing_bridge` の `--pre-settle on` と同じ鍵 `(seed, who, t)` で行を分ける。
2. **恒等式** `LL(全) = (n_残·LL(残) + n_除·LL(除)) / n_全` が数値で成り立つ（`decompose` の `identity`）。
3. **予測は母数に依らない**——同じ行は `全` でも `残` でも同じ確率を受け取る。

**基盤健全性ではない**（鍵の取り違えは「決着前の較正」という判断材料そのものを誤らせる）。必須側。
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bootstrap  # noqa: F401,E402

_SCRIPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import pre_settle_decomp as PD  # noqa: E402


def _row(seed, who, t, tm, to, won):
    return {"seed": seed, "who": who, "t": t, "won": won,
            "tau_me_theory": tm, "tau_opp_theory": to}


def _rows():
    out = []
    for g in range(20):
        # 勝者の決着の行（予測が高く当たる）と、決着前の曖昧な行
        out.append(_row(g, 0, 5, 1.0, 4.0, True))
        out.append(_row(g, 0, 3, 3.0, 3.5, g % 3 != 0))
        out.append(_row(g, 1, 4, 3.0, 2.5, False))
    return out


def test_split_rows_uses_the_same_key_as_the_bridge():
    rows = _rows()
    settled = {(g, 0, 5): True for g in range(20)}
    settled[(0, 1, 4)] = False
    kept, removed = PD.split_rows(rows, settled)
    assert len(removed) == 20 and all(r["t"] == 5 for r in removed)
    assert len(kept) == 40


def test_the_logloss_identity_holds_and_removed_rows_are_the_easy_ones():
    rows = _rows()
    settled = {(g, 0, 5): True for g in range(20)}
    out = PD.decompose(rows, settled)
    assert out["identity"]["abs_err"] < 1e-12
    assert out["removed"]["n"] == 20 and out["removed"]["winner_share"] == 1.0
    # 除いた行は当てやすい＝全体より損失が小さい → 残った行の損失は全体より大きい
    assert out["removed"]["logloss"] < out["all"]["logloss"] < out["kept"]["logloss"]
    assert out["kept"]["n"] == 40


def test_predictions_do_not_depend_on_which_rows_are_scored():
    rows = _rows()
    _s, p_all, _z = PD.score_set(rows)
    kept, _rem = PD.split_rows(rows, {(g, 0, 5): True for g in range(20)})
    _s2, p_kept, _z2 = PD.score_set(kept)
    by_key = {(r["seed"], r["who"], r["t"]): p for r, p in zip(rows, p_all)}
    assert [by_key[(r["seed"], r["who"], r["t"])] for r in kept] == p_kept


def test_mix_logloss_is_the_row_weighted_mean():
    assert PD.mix_logloss([(3, 1.0), (1, 5.0)]) == pytest.approx(2.0)
    assert PD.mix_logloss([]) == 0.0


def test_the_rel_reading_obeys_the_same_identity():
    """**T145**: 出荷の読み（`rel`）でも恒等式は成り立ち、`w_err` が結果に記録される。"""
    rows = _rows()
    settled = {(g, 0, 5): True for g in range(20)}
    out = PD.decompose(rows, settled, w_err="rel", sigma_rel=0.3)
    assert out["w_err"] == "rel"
    assert out["identity"]["abs_err"] < 1e-12
    assert out["removed"]["n"] == 20
