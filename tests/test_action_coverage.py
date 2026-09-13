"""`tests/scripts/action_coverage.py` の純関数（候補の鍵・被覆の突き合わせ・集計）。

基盤健全性（`cpu_infra`）: エンジンは呼ばない。**候補の鍵に DON 配分を入れること**が要
——`move_sig` は don_k を含まないので、鍵を間違えると「同じ手の別配分」が 1 つに潰れて
被覆が 1.0 に見えてしまう（＝枝予算の影響を取り逃がす）。
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

import action_coverage as A  # noqa: E402


def mv(at="ATTACK", uuid="u1", **payload):
    return {"action_type": at, "card_uuid": uuid, "payload": dict(payload)}


def out_of(legal, visits=None, exhausted=0):
    stats = {"legal": legal}
    if visits is not None:
        stats["N"] = visits
    return {"stats": stats, "budget": {"exhausted": exhausted}}


def test_sig_key_separates_don_allocations():
    """同じ攻撃でも DON の枚数が違えば別の候補（鍵が分かれる）。"""
    a = A.sig_key(mv(don_k=0))
    b = A.sig_key(mv(don_k=2))
    assert a != b
    assert A.sig_key(mv(don_k=2)) == b          # 同じものは同じ鍵


def test_sig_key_is_order_insensitive_in_the_payload():
    x = {"action_type": "PLAY", "card_uuid": "u", "payload": {"a": 1, "b": 2}}
    y = {"action_type": "PLAY", "card_uuid": "u", "payload": {"b": 2, "a": 1}}
    assert A.sig_key(x) == A.sig_key(y)


def test_is_don_matches_the_box_action():
    assert A.is_don(mv(at="DON_BOX"))
    assert A.is_don(mv(at="DON_ATTACH"))
    assert not A.is_don(mv(at="ATTACK"))
    assert not A.is_don("ATTACK")


def test_best_index_needs_a_visited_move():
    assert A.best_index(out_of([mv()], visits=[0.0])) is None
    assert A.best_index(out_of([mv(), mv(uuid="u2")], visits=[1.0, 5.0])) == 1
    assert A.best_index(out_of([mv()])) is None


def test_compare_counts_the_moves_the_budget_removed():
    """B にしか無い手が `missing`・B の最善手が A に無ければ `best_in_a` が偽。"""
    m0, m1, m2 = mv(don_k=0), mv(don_k=1), mv(don_k=2)
    a = out_of([m0, m1], visits=[10.0, 1.0], exhausted=3)
    b = out_of([m0, m1, m2], visits=[1.0, 1.0, 50.0])
    row = A.compare(a, b)
    assert row["k_a"] == 2 and row["k_b"] == 3
    assert row["missing"] == 1
    assert row["extra"] == 0
    assert row["coverage"] == pytest.approx(2 / 3)
    assert row["best_in_a"] is False            # B の最善（don_k=2）が A に無い
    assert row["exhausted_a"] == 3


def test_compare_full_coverage():
    m0, m1 = mv(don_k=0), mv(don_k=1)
    row = A.compare(out_of([m0, m1], visits=[5.0, 1.0]),
                    out_of([m0, m1], visits=[5.0, 1.0]))
    assert row["coverage"] == 1.0
    assert row["missing"] == 0 and row["extra"] == 0
    assert row["best_in_a"] is True


def test_compare_tracks_don_candidates_separately():
    d0, d1 = mv(at="DON_BOX", don_k=0), mv(at="DON_BOX", don_k=1)
    plain = mv(at="PLAY", uuid="p")
    row = A.compare(out_of([plain, d0], visits=[1.0, 1.0]),
                    out_of([plain, d0, d1], visits=[1.0, 1.0, 1.0]))
    assert row["don_k_a"] == 1 and row["don_k_b"] == 2
    assert row["don_coverage"] == pytest.approx(0.5)
    assert row["don_missing"] == 1


def test_block_aggregates_and_keeps_errors_visible():
    rows = [
        {"coverage": 1.0, "k_a": 5, "k_b": 5, "missing": 0, "extra": 0, "best_in_a": True,
         "exhausted_a": 0, "exhausted_b": 0, "don_coverage": 1.0, "don_missing": 0},
        {"coverage": 0.5, "k_a": 2, "k_b": 4, "missing": 2, "extra": 0, "best_in_a": False,
         "exhausted_a": 1, "exhausted_b": 0, "don_coverage": 0.5, "don_missing": 1},
        {"error": "RuntimeError: boom", "turn": 3},
    ]
    b = A.block(rows)
    assert b["n"] == 2 and b["errors"] == 1
    assert b["coverage_mean"] == pytest.approx(0.75)
    assert b["coverage_min"] == 0.5
    assert b["missing_any"] == 0.5
    assert b["best_not_in_a"] == 0.5 and b["best_checked"] == 2
    assert b["exhausted_a_any"] == 0.5
    assert b["don_coverage_mean"] == pytest.approx(0.75)
    assert b["don_missing_any"] == 0.5
    assert A.block([]) is None


def test_block_when_every_row_errored():
    b = A.block([{"error": "x"}])
    assert b == {"n": 1, "errors": 1}


def test_turn_band_edges():
    assert A.turn_band(4) == "T<=4"
    assert A.turn_band(5) == "T5-8"
    assert A.turn_band(9) == "T9+"
