"""バッチ式アクター/ラーナーの純粋協調ロジック（鮮度フィルタ・消費追跡・リングバッファ）の検証。

git 非依存の部分だけを固定する（git 入出力を含む end-to-end は pd_*_smoke で別途）。
docs/reports/batched_selfplay_design_20260710.md。
"""
import numpy as np

import conftest  # noqa: F401
import pd_batch_common as C


def _meta(worker, batch_id, against_round, games=100, states=100):
    return {"worker": worker, "batch_id": batch_id, "against_round": against_round,
            "games": games, "states": states}


def test_is_fresh_accept_seen_stale():
    consumed = {"w1": 4}
    # 新規かつ新鮮 → accept
    assert C.is_fresh(_meta("w1", 5, 10), consumed, 10, 3) == "accept"
    # 消費済み（batch_id <= consumed） → seen
    assert C.is_fresh(_meta("w1", 4, 10), consumed, 10, 3) == "seen"
    assert C.is_fresh(_meta("w1", 3, 10), consumed, 10, 3) == "seen"
    # 古すぎる against_round（round-staleness 未満） → stale
    assert C.is_fresh(_meta("w1", 6, 6), consumed, 10, 3) == "stale"   # 6 < 10-3=7
    assert C.is_fresh(_meta("w1", 6, 7), consumed, 10, 3) == "accept"  # 7 == 境界OK
    # 未知workerは consumed=-1 扱い
    assert C.is_fresh(_meta("w9", 0, 10), consumed, 10, 3) == "accept"


def test_plan_consumption_mixed():
    consumed = {"w1": 2, "w2": 0}
    metas = [_meta("w1", 3, 10), _meta("w2", 0, 10), _meta("w3", 5, 4)]
    # w1: 新規新鮮=accept / w2: batch_id0<=consumed0=seen / w3: against 4 < 7 = stale
    accepted, skipped = C.plan_consumption(metas, consumed, 10, 3)
    assert [m["worker"] for m in accepted] == ["w1"]
    assert skipped == {"w2": "seen", "w3": "stale"}


def test_update_consumed_monotonic():
    consumed = {"w1": 2}
    out = C.update_consumed(consumed, [_meta("w1", 5, 10), _meta("w2", 1, 10)])
    assert out == {"w1": 5, "w2": 1}
    # 元の dict は不変
    assert consumed == {"w1": 2}
    # 逆行しない（古いbatchが後から来ても最大を保つ）
    out2 = C.update_consumed(out, [_meta("w1", 3, 10)])
    assert out2["w1"] == 5


def test_ring_append_caps_and_bootstraps():
    a = {"x": np.arange(10), "y": np.arange(10) * 2}
    b = {"x": np.arange(10, 16), "y": np.arange(10, 16) * 2}
    # 空バッファ → new をそのまま（cap内）
    buf = C.ring_append(None, a, cap=100)
    assert np.array_equal(buf["x"], np.arange(10))
    # 連結して cap で末尾切り
    buf = C.ring_append(buf, b, cap=8)
    assert len(buf["x"]) == 8
    assert np.array_equal(buf["x"], np.arange(8, 16))       # 末尾8件
    assert np.array_equal(buf["y"], np.arange(8, 16) * 2)   # キー間で整合
