"""`attack_axis_by_result.py`(P8-7(c)以降・C-1・T123の攻撃の行の残差を結果別・軸別に割る)の
算術を固める。

押さえるのは2つ:

1. **`by_result`は`transition_ledger.collect(dump=...)`が積む行(`resp`・`sh`つき)を結果別に束ね、
   `resid_abs`の平均と5軸それぞれの平均(符号つき)を出す**——束ねる鍵は`resp`のみ。
2. **`collect`は`d_mode`を渡すと`kappa_vector.D_MODE`を切り替えてから`transition_ledger.collect`を
   呼ぶ**(渡さなければ既定のまま)。

**基盤健全性ではない**——器の誤りはC-1の「残差はどの軸に乗るか」という診断そのものを誤らせる
(`t139_defender_check.py`と同じ扱い)。
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bootstrap  # noqa: F401,E402

_SCRIPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import attack_axis_by_result as AX  # noqa: E402


# --- 1. by_result ---------------------------------------------------------------------------

def test_by_result_averages_resid_and_axes_per_response_type():
    dump = [
        {"resp": "took", "resid_abs": 0.10,
         "sh": {"th_me": -0.02, "th_opp": 0.03, "a_me": 0.0, "a_opp": 0.0, "j": 0.0}},
        {"resp": "took", "resid_abs": 0.20,
         "sh": {"th_me": -0.04, "th_opp": 0.05, "a_me": 0.0, "a_opp": 0.0, "j": 0.0}},
        {"resp": "nothing", "resid_abs": 0.02,
         "sh": {"th_me": -0.01, "th_opp": 0.01, "a_me": 0.0, "a_opp": 0.0, "j": 0.0}},
    ]
    out = AX.by_result(dump)
    assert set(out.keys()) == {"took", "nothing"}
    assert out["took"]["n"] == 2
    assert out["took"]["resid_abs_mean"] == pytest.approx(0.15)
    assert out["took"]["axis_mean"]["th_me"] == pytest.approx(-0.03)
    assert out["took"]["axis_mean"]["th_opp"] == pytest.approx(0.04)
    assert out["nothing"]["n"] == 1
    assert out["nothing"]["resid_abs_mean"] == pytest.approx(0.02)


def test_by_result_is_empty_for_empty_dump():
    assert AX.by_result([]) == {}


# --- 2. collect ------------------------------------------------------------------------------

def test_collect_passes_d_mode_through_to_kappa_vector(monkeypatch):
    import kappa_vector as KV
    seen = {}

    def fake_tl_collect(dirs, limit_games, dump=None):
        seen["d_mode_at_call"] = KV.D_MODE
        dump.append({"resp": "took", "resid_abs": 0.1,
                    "sh": {"th_me": -0.02, "th_opp": 0.02, "a_me": 0.0, "a_opp": 0.0, "j": 0.0}})
        return {"games": 1, "resid_priority": {"attack_rows": 0.4}}

    monkeypatch.setattr(AX.TL, "collect", fake_tl_collect)
    old = KV.D_MODE
    try:
        out = AX.collect(["dummy"], d_mode="theory")
        assert seen["d_mode_at_call"] == "theory"
        assert out["d_mode"] == "theory"
        assert out["games"] == 1
        assert out["attack_rows_share_of_total_resid"] == 0.4
        assert out["by_result"]["took"]["n"] == 1
    finally:
        KV.set_d_mode(old)


def test_collect_does_not_touch_d_mode_when_omitted(monkeypatch):
    import kappa_vector as KV
    seen = {}

    def fake_tl_collect(dirs, limit_games, dump=None):
        seen["d_mode_at_call"] = KV.D_MODE
        return {"games": 0, "resid_priority": {"attack_rows": None}}

    monkeypatch.setattr(AX.TL, "collect", fake_tl_collect)
    old = KV.D_MODE
    KV.set_d_mode("clock")
    try:
        AX.collect(["dummy"])
        assert seen["d_mode_at_call"] == "clock"          # 渡さなければ既定(現状の値)のまま
    finally:
        KV.set_d_mode(old)


def test_collect_passes_attack_rest_through_to_kappa_vector(monkeypatch):
    """**C-2**: `attack_rest`を渡すと`kappa_vector.ATTACK_REST_MODE`を切り替えてから呼ぶ。"""
    import kappa_vector as KV
    seen = {}

    def fake_tl_collect(dirs, limit_games, dump=None):
        seen["mode_at_call"] = KV.ATTACK_REST_MODE
        return {"games": 0, "resid_priority": {"attack_rows": None}}

    monkeypatch.setattr(AX.TL, "collect", fake_tl_collect)
    old = KV.ATTACK_REST_MODE
    try:
        out = AX.collect(["dummy"], attack_rest="body")
        assert seen["mode_at_call"] == "body"
        assert out["attack_rest_mode"] == "body"
    finally:
        KV.set_attack_rest_mode(old)


def test_collect_does_not_touch_attack_rest_mode_when_omitted(monkeypatch):
    import kappa_vector as KV
    seen = {}

    def fake_tl_collect(dirs, limit_games, dump=None):
        seen["mode_at_call"] = KV.ATTACK_REST_MODE
        return {"games": 0, "resid_priority": {"attack_rows": None}}

    monkeypatch.setattr(AX.TL, "collect", fake_tl_collect)
    old = KV.ATTACK_REST_MODE
    KV.set_attack_rest_mode("body")
    try:
        AX.collect(["dummy"])
        assert seen["mode_at_call"] == "body"             # 渡さなければ既定(現状の値)のまま
    finally:
        KV.set_attack_rest_mode(old)
