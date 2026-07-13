"""ピーク自動アラート（peak_alert.detect_peak・v5 §4-4a）の単体検証。

2指標（mark_improved・arena_wr）の**同時後退**が patience 回連続でアラート＝忘却の早期検知。
単一指標の後退やノイズ的な上下ではアラートしない（誤報抑制）ことと、凍結候補が best 複合スコアの
round になることを固定する。
"""
import pytest

import conftest  # noqa: F401
from peak_alert import detect_peak

pytestmark = pytest.mark.cpu_infra   # 基盤健全性（監視計器）


def _r(rnd, mark, wr):
    return {"round": rnd, "mark_improved": mark, "arena_wr": wr}


def test_no_alert_while_improving():
    recs = [_r(1, 2, 0.50), _r(2, 3, 0.55), _r(3, 4, 0.60)]
    res = detect_peak(recs, patience=2)
    assert not res["alert"]
    assert res["peak_round"] == 3 and res["peak_mark"] == 4


def test_alert_on_simultaneous_regression():
    """ピーク(round3)後、2指標が同時後退して patience=2 回続く＝アラート・凍結候補は round3。"""
    recs = [_r(1, 2, 0.50), _r(2, 4, 0.62), _r(3, 5, 0.66),
            _r(4, 3, 0.60), _r(5, 2, 0.58)]
    res = detect_peak(recs, patience=2, mark_drop=1, wr_drop=0.03)
    assert res["alert"]
    assert res["peak_round"] == 3
    assert res["regressing_streak"] >= 2


def test_single_axis_regression_does_not_alert():
    """mark だけ下がり wr は維持＝同時後退でない＝アラートしない（誤報抑制）。"""
    recs = [_r(1, 3, 0.55), _r(2, 5, 0.66), _r(3, 3, 0.66), _r(4, 3, 0.67)]
    res = detect_peak(recs, patience=2, mark_drop=1, wr_drop=0.03)
    assert not res["alert"]


def test_noise_below_tolerance_does_not_alert():
    """許容幅内の微小な上下（wr −0.01・mark 同値）は後退扱いしない。"""
    recs = [_r(1, 4, 0.60), _r(2, 4, 0.59), _r(3, 4, 0.605)]
    res = detect_peak(recs, patience=2, mark_drop=1, wr_drop=0.03)
    assert not res["alert"]


def test_empty_and_single():
    assert detect_peak([])["alert"] is False
    res = detect_peak([_r(7, 5, 0.7)])
    assert res["peak_round"] == 7 and not res["alert"]
