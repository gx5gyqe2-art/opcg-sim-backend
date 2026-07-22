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
from coach_gate import VERIFIED, hit, judge  # noqa: E402

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
