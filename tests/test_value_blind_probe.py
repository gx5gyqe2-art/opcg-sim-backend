"""VALUE_BLIND 原因分析プローブ（v23・`tests/scripts/value_blind_probe.py`）の純関数テスト。

実ネット・実盤面は使わない。固定する性質:
  - **GROUPS は符号化3キーの完全分割**（漏れ・重複があると遮蔽帰属が「測っていない特徴」を
    見逃す／二重計上する）
  - swap_group は対象グループだけを入れ替え、入力を壊さない
  - 線形ネットでは帰属の総和が gap に厳密一致（fwd=rev・分解の健全性）
  - scan_target は展開（自場ID新出）と付与（attached_don 増加）を判別する
  - contrast_stats の echo は「q_root が z の裏付け無く持ち上げる」方向を正で返す
"""
import conftest  # noqa: F401  (google スタブ注入 & sys.path 設定)
import numpy as np
import pytest

import value_blind_probe as VB

pytestmark = pytest.mark.cpu_infra


def _enc(scalar_fill=0.0, field_fill=0.0, idx_fill=0):
    return {"scalars": np.full(55, scalar_fill, dtype=np.float32),
            "field": np.full((10, 8), field_fill, dtype=np.float32),
            "card_idx": np.full(24, idx_fill, dtype=np.int32)}


def test_groups_partition_all_indices_exactly_once():
    seen = {"scalars": [], "field": [], "card_idx": []}
    for key, idxs in VB.GROUPS.values():
        seen[key].extend(idxs)
    assert sorted(seen["scalars"]) == list(range(55))     # SCALARS_V5
    assert sorted(seen["field"]) == list(range(10))       # 2*MAX_FIELD 行
    assert sorted(seen["card_idx"]) == list(range(24))    # 2+10+10+2 枠


def test_swap_group_replaces_only_target_and_preserves_inputs():
    a, b = _enc(0.0), _enc(1.0, 1.0, 7)
    out = VB.swap_group(a, b, "scalars", [2, 3])
    assert out["scalars"][2] == 1.0 and out["scalars"][3] == 1.0
    assert out["scalars"][0] == 0.0 and out["field"].sum() == 0.0
    assert a["scalars"][2] == 0.0 and b["scalars"][0] == 1.0   # 入力不変
    out2 = VB.swap_group(a, b, "field", [5])
    assert out2["field"][5].sum() == 8.0 and out2["field"][4].sum() == 0.0


def test_attribution_sums_to_gap_for_linear_net():
    """線形 vf では遮蔽は加法的＝グループ寄与の総和が gap に厳密一致し fwd=rev。"""
    rng = np.random.default_rng(0)
    w_s, w_f, w_i = rng.standard_normal(55), rng.standard_normal((10, 8)), rng.standard_normal(24)

    def vf(enc):
        return float((enc["scalars"] * w_s).sum() + (enc["field"] * w_f).sum()
                     + (enc["card_idx"] * w_i).sum())

    bad, good = _enc(0.5, 0.2, 3), _enc(-0.1, 0.9, 1)
    gap, rows = VB.attribution(vf, bad, good)
    assert gap == pytest.approx(vf(bad) - vf(good))
    assert sum(r["fwd"] for r in rows) == pytest.approx(gap, abs=1e-6)
    for r in rows:
        assert r["fwd"] == pytest.approx(r["rev"], abs=1e-6)
    assert abs(rows[0]["mean"]) == max(abs(r["mean"]) for r in rows)   # |mean| 降順


def test_scan_target_deploy_and_attach():
    parent, child = _enc(), _enc()
    parent["card_idx"][2] = 5
    child["card_idx"][2] = 5
    child["card_idx"][3] = 9                     # 自場に新出＝展開
    assert VB.scan_target(parent, child) == (9, "deploy")
    child2 = _enc(); child2["card_idx"][2] = 5
    child2["field"][0, 3] = 0.2                  # attached_don 列の増加＝付与
    assert VB.scan_target(parent, child2) == (5, "attach")
    same = _enc(); same["card_idx"][2] = 5
    assert VB.scan_target(parent, same) == (None, None)


def test_contrast_stats_echo_positive_when_q_lifts_without_z():
    """在群: z は平均0のまま q_root だけ +0.5 ＝ echo=+0.5（q が裏付け無く持ち上げ）。"""
    z = [0.0, 0.0, 0.0, 0.0]
    q = [0.5, 0.5, 0.0, 0.0]
    present = [True, True, False, False]
    st = VB.contrast_stats(z, q, present)
    assert st["present"]["n"] == 2 and st["absent"]["n"] == 2
    assert st["dz"] == pytest.approx(0.0)
    assert st["dq"] == pytest.approx(0.5)
    assert st["echo"] == pytest.approx(0.5)


def test_contrast_stats_ignores_nan_q_and_handles_empty_group():
    st = VB.contrast_stats([1.0, -1.0], [np.nan, 0.3], [True, True])
    assert st["present"]["mean_q"] == pytest.approx(0.3)   # NaN は q 平均から除外
    assert st["absent"]["n"] == 0 and st["echo"] is None   # 対照が無ければ差は主張しない
