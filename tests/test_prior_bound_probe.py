"""prior/value 分解プローブ（v20・`tests/scripts/prior_bound_probe.py`）の分類則テスト。

実 decide・実復元は回さない（`test_promotion_gate.py` と同じ思想＝純関数のみ）。
この計器の存在理由は「VERIFIED v2 の改善ターゲットが value 較正2周でも 0.00 のまま」の原因を
prior / value / 探索 に切り分けることなので、固定する性質は**分類の優先順が機序の切り分けとして
成立している**こと。2026-07-29 の初回測定（`docs/reports/cpu_v20_prior_value_20260729.md`）で
PRIOR_BOUND が0件・SEARCH_AVERSE が3件だったため、両者の境界を特に固定する。
"""
import conftest  # noqa: F401  (google スタブ注入 & sys.path 設定)
import pytest

import prior_bound_probe as PB

pytestmark = pytest.mark.cpu_infra


def test_ok_when_base_already_hits():
    """基準 sims で accept しているなら対象外（深掘り列が無くても OK）。"""
    assert PB.classify(1.0, None, None, 0.3, -0.5) == "OK"
    assert PB.classify(0.5, None, None, 0.0, -0.5) == "OK"      # 境界は OK 側


def test_explorable_beats_other_causes():
    """深探索で立ち上がるなら、prior が薄くても value が負でも『探索の浅さ』が主因。"""
    assert PB.classify(0.0, 1.0, 1.0, 0.01, -0.3) == "EXPLORABLE"


def test_prior_bound_requires_thin_prior_and_flat_recovery():
    """PRIOR_BOUND は**両方**必要: prior 質量が薄い かつ 一様prior で立ち上がる。
    どちらか欠ければ policy 起因とは言えない（片方だけで断定しない）。"""
    assert PB.classify(0.0, 0.0, 1.0, 0.05, +0.2) == "PRIOR_BOUND"
    # prior は薄いが flat でも解けない → policy のせいではない
    assert PB.classify(0.0, 0.0, 0.0, 0.05, -0.2) == "VALUE_BLIND"
    # flat で解けるが prior は厚い（沈んでいない）→ PRIOR_BOUND にしない
    assert PB.classify(0.0, 0.0, 1.0, 0.60, -0.2) == "VALUE_BLIND"


def test_value_blind_when_value_ranks_correct_move_below():
    """dv ≤ 0＝1手先 value が正着を誤着より下に見ている＝value/表現の問題。境界 0 も含む。"""
    assert PB.classify(0.0, 0.0, 0.0, 0.30, -0.10) == "VALUE_BLIND"
    assert PB.classify(0.0, 0.0, 0.0, 0.30, 0.0) == "VALUE_BLIND"


def test_search_averse_is_the_third_mechanism():
    """prior 1位 かつ 1手先 value も支持（dv>0）なのに深探索が選ばない＝多手先の読みの問題。
    2026-07-29 実測の m2@44 / m4@12 / m5@7 の形（prior でも value でもない第3の機序）。"""
    assert PB.classify(0.0, 0.0, 0.0, 0.74, +0.15, prior_rank=1) == "SEARCH_AVERSE"
    # prior 1位でなければ SEARCH_AVERSE と断定しない（他の原因が混ざる）
    assert PB.classify(0.0, 0.0, 0.0, 0.74, +0.15, prior_rank=3) == "UNRESOLVED"


def test_unrestorable_when_base_missing():
    assert PB.classify(None, None, None, None, None) == "UNRESTORABLE"
