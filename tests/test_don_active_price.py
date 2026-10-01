"""`don_active_price.py`（T57・アクティブなドン 1 個の価格）の算術を固める。

**説明変数（アクティブ数）は帯に入れず、総在庫は帯に入れる**——ここを取り違えると在庫の価格 `δ` を測り直すだけになる。
相手のターン開始の行は**相手の視点の scalars から自分のドンを読む**（列 4/5・z は自分の側）ことを値で押さえる。
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

import don_active_price as DA  # noqa: E402


def _sc(my_act, my_rest, opp_act, opp_rest, my_life=4, opp_life=4, my_hand=5, opp_hand=5, my_field=2, opp_field=1, turn=6):
    sc = np.zeros(70, np.float32)
    sc[0], sc[1] = my_life, opp_life
    sc[2], sc[3], sc[4], sc[5] = my_act, my_rest, opp_act, opp_rest
    sc[6], sc[7], sc[8], sc[9], sc[10] = my_hand, opp_hand, my_field, opp_field, turn
    return sc


def test_the_band_holds_the_total_stock_and_leaves_the_active_count_free():
    tok = np.zeros((22, 24), np.float32)
    a = DA.row_mid(_sc(5, 1, 3, 3), tok, 1.0)
    b = DA.row_mid(_sc(2, 4, 3, 3), tok, 0.0)
    assert a["band"] == b["band"]                       # 総在庫 6 が同じ＝同じ帯
    assert (a["active"], b["active"]) == (5.0, 2.0)     # アクティブ数だけが違う
    c = DA.row_mid(_sc(5, 2, 3, 3), tok, 1.0)
    assert c["band"] != a["band"]                       # 総在庫 7 は別の帯


def test_the_opponents_turn_row_reads_my_don_from_their_perspective():
    tok = np.zeros((22, 24), np.float32)
    # 相手の視点の scalars: 列 4/5 が「相手（＝自分）のドン」・ライフ・手札・場も入れ替わる
    r = DA.row_opp_turn(_sc(my_act=6, my_rest=0, opp_act=2, opp_rest=3, my_life=2, opp_life=4, my_hand=3, opp_hand=6,
                            my_field=0, opp_field=2), tok, z_me=1.0)
    assert r["active"] == 2.0 and r["z"] == 1.0
    assert r["band"].split("|")[:3] == ["5", "4", "2"]   # 総在庫 5・自ライフ 4・相手ライフ 2（自分の視点に戻す）
    assert r["band"].split("|")[-2:] == ["6", "2"]        # 自分の手札 6・自分の場 2


def test_the_within_slope_is_read_inside_bands_and_bootstrapped_by_game():
    rows = []
    for g in range(40):
        for k, z in ((5, 1.0), (2, 0.0), (5, 1.0), (2, 0.0)):
            rows.append({"active": float(k), "z": z, "band": "6|4|4|T5-8|5|2", "game": g})
    out = DA.summarise({"mid": rows, "opp_turn": []}, reps=20)
    s = out["scenes"]["mid"]
    assert s["delta_active_within"] == pytest.approx(1.0 / 3.0, abs=1e-4)   # 3 個で勝率が 0 → 1
    assert s["ci95_games"][0] == pytest.approx(1.0 / 3.0, abs=1e-4)
    assert s["ratio_to_delta"] == pytest.approx((1.0 / 3.0) / DA.DELTA, abs=1e-3)
    assert "opp_turn" not in out["scenes"]                                # 行が無ければ出さない
