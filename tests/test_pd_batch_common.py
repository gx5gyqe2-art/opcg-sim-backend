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


def test_is_fresh_referee_label_exempt_from_staleness():
    """レフェリー再ラベル（source="referee_label"・v9）は gen5 固定アンカー由来＝腐らないため
    staleness 免除（against_round=-1 が学習4ラウンド目以降に全棄却される事故の回帰）。
    未消費チェック（seen）は免除しない。"""
    consumed = {"ref": 2}
    ref = dict(_meta("ref", 3, -1), source="referee_label")
    assert C.is_fresh(ref, consumed, 100, 3) == "accept"
    assert C.is_fresh(dict(ref, batch_id=2), consumed, 100, 3) == "seen"
    # source 無し（通常自己対戦バッチ）は従来どおり stale
    assert C.is_fresh(_meta("w1", 9, -1), {}, 100, 3) == "stale"


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


def test_updates_for_scales_with_inflow_and_preserves_ratio():
    """薄まり防止: 学習回数が新規games に比例＝並列でも1局あたりの勾配露出が一定。"""
    gpu = 128
    # K=1（1バッチ128局）→ 1ラウンド（従来と同一・後方互換）
    assert C.updates_for(128, gpu, 16) == 1
    # K=6（6バッチ768局）→ 6ラウンド（直列の games:updates 比を維持）
    assert C.updates_for(768, gpu, 16) == 6
    # 端数は四捨五入（192局→1.5→2）だが最低1
    assert C.updates_for(192, gpu, 16) == 2
    assert C.updates_for(10, gpu, 16) == 1        # 新規僅少でも最低1
    assert C.updates_for(0, gpu, 16) == 1
    # 暴発上限（大量流入でも max で頭打ち）
    assert C.updates_for(100000, gpu, 16) == 16
    # 不正 gpu は 1
    assert C.updates_for(500, 0, 16) == 1


def test_should_generate_backpressure_boundaries():
    """バックプレッシャ: 未消費が depth 本以下なら生成可・超えたら待機（learner停止中の上書き全損防止）。"""
    # learner未稼働（consumed=-1）でも depth 本までは先行生成できる
    assert C.should_generate(0, -1, 2) is True    # 未消費1本目
    assert C.should_generate(1, -1, 2) is True    # 2本目
    assert C.should_generate(2, -1, 2) is False   # 3本目=depth超え→待つ
    # learnerが追いつけば再開
    assert C.should_generate(2, 0, 2) is True
    # depth=0 は「前バッチ消費まで次を作らない」完全同期
    assert C.should_generate(5, 4, 0) is False
    assert C.should_generate(5, 5, 0) is True


def test_pack_unpack_policy_roundtrip():
    """policy教師（可変長L）のnpzパック往復＝直列とのパリティ（policy凍結バグの回帰）。"""
    rng = np.random.default_rng(0)
    pol = [(rng.standard_normal(6).astype(np.float32),
            rng.standard_normal((L, 4)).astype(np.float32),
            (np.ones(L) / L).astype(np.float32)) for L in (3, 1, 5)]
    packed = C.pack_policy(pol)
    # npz 保存/読込を模す（np.savez→np.load 相当のdict渡し）
    out = C.unpack_policy(packed)
    assert len(out) == 3
    for (c0, a0, v0), (c1, a1, v1) in zip(pol, out):
        assert np.allclose(c0, c1) and np.allclose(a0, a1) and np.allclose(v0, v1)
        assert a1.shape[0] == len(v1)
    # 空は空・旧形式（キー無し）は []（後方互換）
    assert C.unpack_policy(C.pack_policy([])) == []
    assert C.unpack_policy({"scalars": np.zeros(3)}) == []


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
