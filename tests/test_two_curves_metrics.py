"""`two_curves_metrics.py`（T137b・2 本の曲線の 3 指標）の算術を固める。

**`two_curves.py --dump` が書いた JSON を読むだけ**（記録は読まない）ので、全テストが合成データで
完結する。押さえるのは 4 つ:

1. **`increments`**: `(turns, g, r, g_fam)` から `[(j, ΔG, ΔR, fam)]` を正しく作る（`j` は 0 始まり・
   増分は前の累積との差）。
2. **`phase_of`**: 自席ターン番号 `j` を 3 帯（0-2／3-5／6+）に正しく振る。
3. **`collect` の指標の算術**（増分の相関・傾き・局面別の距離・終点の比・型別の相関）——
   **既知の量を組んだ合成データで検算**（例: `ΔG=ΔR` なら相関 1・傾き 1）。
4. **型が全部ゼロの族（`end`／`attach`）は相関 `None`**（`corr` の定義どおり）。

**基盤健全性ではない**——器の誤りは「曲線が重なっているか」の判定そのものを誤らせる。必須側。
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import _bootstrap  # noqa: F401,E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
import two_curves_metrics as TM  # noqa: E402


# ---- 1. increments -----------------------------------------------------------------------------

def test_increments_returns_deltas_from_zero():
    seat = {"g": [1.0, 3.0, 3.0], "r": [0.5, 0.5, 2.0], "g_fam": [{"attack": 1.0}, {"play": 2.0}, {}]}
    out = TM.increments(seat)
    assert out == [(0, 1.0, 0.5, {"attack": 1.0}), (1, 2.0, 0.0, {"play": 2.0}), (2, 0.0, 1.5, {})]


def test_increments_defaults_missing_g_fam_to_empty_dicts():
    seat = {"g": [1.0], "r": [1.0]}
    out = TM.increments(seat)
    assert out == [(0, 1.0, 1.0, {})]


def test_increments_handles_empty_seat():
    assert TM.increments({"g": [], "r": []}) == []


# ---- 2. phase_of --------------------------------------------------------------------------------

def test_phase_of_buckets_the_own_turn_index():
    assert TM.phase_of(0) == "0-2" and TM.phase_of(2) == "0-2"
    assert TM.phase_of(3) == "3-5" and TM.phase_of(5) == "3-5"
    assert TM.phase_of(6) == "6+" and TM.phase_of(100) == "6+"


# ---- 3. collect の算術 -----------------------------------------------------------------------------

def test_collect_recovers_a_perfect_linear_relationship():
    """`ΔR = ΔG` になるよう組んだ系列 → 相関 1・傾き 1（当てはめではなく検算）。"""
    dump = [{"g": [1.0, 2.0, 4.0], "r": [1.0, 2.0, 4.0], "g_fam": [{"attack": 1.0}] * 3} for _ in range(5)]
    out = TM.collect(dump)
    assert out["increment_corr_raw"] == pytest.approx(1.0)
    assert out["increment_slope"] == pytest.approx(1.0)
    assert out["distance_by_phase"]["0-2"]["mean"] == pytest.approx(0.0)      # G と R が常に等しい


def test_collect_endpoint_ratio_matches_hand_computation():
    dump = [{"g": [2.0], "r": [1.0], "g_fam": [{}]},
            {"g": [3.0], "r": [1.0], "g_fam": [{}]},
            {"g": [5.0], "r": [0.0], "g_fam": [{}]}]           # r=0 は比から除く（0 割り回避）
    out = TM.collect(dump)
    assert out["endpoint_ratio"]["n"] == 2
    assert out["endpoint_ratio"]["mean"] == pytest.approx((2.0 + 3.0) / 2)


def test_collect_distance_by_phase_separates_early_from_late():
    # j=0..5（序盤・中盤）は G=R（距離 0）・j=6（"6+" 帯はここ 1 点だけ）は G が R よりずっと大きい（距離 5）
    dump = [{"g": [1.0] * 6 + [10.0], "r": [1.0] * 6 + [5.0], "g_fam": [{}] * 7}]
    out = TM.collect(dump)
    assert out["distance_by_phase"]["0-2"]["mean"] == pytest.approx(0.0)
    assert out["distance_by_phase"]["3-5"]["mean"] == pytest.approx(0.0)
    assert out["distance_by_phase"]["6+"]["mean"] == pytest.approx(5.0)


def test_collect_family_breakdown_recovers_the_matching_type():
    # attack の増分は R と完全に一致・play の増分は R と無関係（定数）にしておく
    dump = []
    for k in range(1, 6):
        dump.append({"g": [float(k)], "r": [float(k)], "g_fam": [{"attack": float(k), "play": 1.0}]})
    out = TM.collect(dump)
    assert out["by_family"]["attack"]["corr"] == pytest.approx(1.0)
    assert out["by_family"]["play"]["corr"] is None            # play は定数（相関できない）


# ---- 4. 全部ゼロの族は相関 None ------------------------------------------------------------------

def test_zero_families_have_no_correlation():
    dump = [{"g": [float(k)], "r": [float(k)], "g_fam": [{"end": 0.0, "attach": 0.0, "attack": float(k)}]}
            for k in range(1, 5)]
    out = TM.collect(dump)
    assert out["by_family"]["end"]["corr"] is None
    assert out["by_family"]["attach"]["corr"] is None
    assert out["by_family"]["attack"]["corr"] == pytest.approx(1.0)


def test_load_dump_reads_json(tmp_path):
    p = tmp_path / "d.json"
    p.write_text('[{"g": [1.0], "r": [1.0]}]', encoding="utf-8")
    out = TM.load_dump(str(p))
    assert out == [{"g": [1.0], "r": [1.0]}]


def test_cli_reaches_collect(monkeypatch, tmp_path):
    called = {}
    p = tmp_path / "d.json"
    p.write_text("[]", encoding="utf-8")

    def fake_collect(dump):
        called["dump"] = dump
        return {"x": 1}

    monkeypatch.setattr(TM, "collect", fake_collect)
    TM.main(["--dump", str(p)])
    assert called["dump"] == []
