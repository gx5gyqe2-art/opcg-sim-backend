"""コーチゲートの判定則（v9 §4・`coach_gate.py`・mark_gate の後継）。

レフェリー検証済みバンド（band-top プランの初手集合）への所属で候補を判定する:
  1. hit: (action_type, card) 一致・card=None は action_type のみ（PASS/TURN_END 型）
  2. judge: 非退行（base≥0.8 の点で chall≤base−0.4 が無い）＋改善（ヒット計 ≥ base計）
  3. VERIFIED 採録の整合: 全点が (tag, index, 非空 accept 集合) で言及ゲームが実在
純関数のみ（decide は回さない＝高速）。基盤健全性＝cpu_infra。
"""
import os
import sys

import conftest  # noqa: F401  (google スタブ注入 & sys.path 設定)
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "tests", "scripts"))
from coach_gate import (REPLAYS_V2, VERIFIED, VERIFIED_V2, hit, judge,  # noqa: E402
                        min_reliable_delta)

pytestmark = pytest.mark.cpu_infra


def test_hit_card_and_type_only():
    accept = {("ATTACK", "PRB02-008"), ("PASS", None)}
    assert hit({"action_type": "ATTACK", "card": "PRB02-008"}, accept)
    assert not hit({"action_type": "ATTACK", "card": "OP11-041"}, accept)
    assert hit({"action_type": "PASS"}, accept)          # card 省略＝type のみで合格
    assert not hit({"action_type": "TURN_END"}, accept)


def test_judge_regression_and_improvement():
    rows = [("g3", 33, 1.0, 1.0), ("g3", 115, 0.0, 1.0)]
    ok_nr, ok_imp, regs = judge(rows)
    assert ok_nr and ok_imp and not regs                  # 改善のみ＝PASS 側
    rows = [("g3", 33, 1.0, 0.4), ("g3", 115, 0.0, 1.0)]
    ok_nr, ok_imp, regs = judge(rows)
    assert not ok_nr and regs == [("g3", 33, 1.0, 0.4)]   # 確実点の大幅落ち＝退行
    rows = [("g3", 33, 0.4, 0.0)]
    ok_nr, _imp, regs = judge(rows)
    assert ok_nr, "base が不確実（<0.8）な点の揺れは退行扱いしない"


def test_verified_entries_wellformed():
    assert len(VERIFIED) >= 5
    for tag, i, accept in VERIFIED:
        assert tag == "g3" and isinstance(i, int) and i >= 0
        assert accept and all(isinstance(a, tuple) and len(a) == 2 for a in accept)


def test_verified_v2_entries_wellformed():
    """VERIFIED v2（gen7 実対局由来）: tag が REPLAYS_V2 に実在し fixture ファイルもある。
    複数対局・両対面方向（CPU=ナミ/シャンクス）を含む＝g3 の単一対局バイアスの回帰防止。

    下限は 2026-08-04 のユーザレビューで 10→8 へ引き下げ: 局面前提が不自然な点
    （パワー2000 のキャラにドン2枚付与＝m1@42/m1@94/m4@12）、バンドが広すぎて識別力の無い点
    （m4@8）、裁定が誤り/未確定の点（m2@12/m2@64）を取り下げた結果。**点数を保つために
    疑わしい点を残さない**（水増しされたバンドは候補を実力以上に見せる）。"""
    assert len(VERIFIED_V2) >= 8
    tags = {t for t, _i, _a in VERIFIED_V2}
    assert len(tags) >= 3, "複数対局から採録されているはず"
    for tag, i, accept in VERIFIED_V2:
        assert tag in REPLAYS_V2 and isinstance(i, int) and i >= 0
        assert accept and all(isinstance(a, tuple) and len(a) == 2 for a in accept)
    for path in REPLAYS_V2.values():
        assert os.path.exists(path), path


def test_min_reliable_delta_shrinks_with_seeds():
    """点別差の信頼下限（2σ）は √n で縮む。**5seed は 0.63＝ほぼ何も言えない**という
    v22 の実測事実を数値で固定する（5seed の 0.60→0.20 を『退行』と読んだ誤りの再発防止）。"""
    assert min_reliable_delta(5) == pytest.approx(0.632, abs=0.01)
    assert min_reliable_delta(16) == pytest.approx(0.354, abs=0.01)
    assert min_reliable_delta(64) == pytest.approx(0.177, abs=0.01)
    assert min_reliable_delta(16) < min_reliable_delta(5)


def test_min_reliable_delta_guards_zero_seeds():
    assert min_reliable_delta(0) == float("inf")
