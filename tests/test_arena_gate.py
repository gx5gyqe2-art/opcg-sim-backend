"""固定N・帯層別アリーナ判定器（v16・`tests/scripts/arena_gate.py`）の純関数テスト。

実対局は回さない（`test_promotion_gate.py` と同じ思想）。この計器の存在理由は
`docs/reports/cpu_v15_ensemble_power_20260726.md` §2＝**24〜120局の判定は検定力不足**で、
「60局で有望→確証で消える」を2回踏んだこと。よって固定する性質は次の2点:
  - 点推定だけでは通さない（**ペア水準95%CI下限 > 0.50** を併せて要求する）
  - 帯ごとに seed 空間を大きく離す（帯内相関で偶然が帯をまたいで共有されないようにする）
"""
import conftest  # noqa: F401  (google スタブ注入 & sys.path 設定)
import pytest

import arena_gate as AG

pytestmark = pytest.mark.cpu_infra


def test_plan_bands_partitions_all_pairs():
    """全ペアが過不足なく帯へ分配される（端数は先頭帯へ寄せる）。"""
    bands = AG.plan_bands(10, 4, 1000)
    assert [len(b) for b in bands] == [3, 3, 2, 2]
    assert sum(len(b) for b in bands) == 10


def test_plan_bands_seeds_are_far_apart_and_unique():
    """帯間の seed 基点が stride ぶん離れており、全 seed が重複しない。"""
    bands = AG.plan_bands(400, 4, 71000)
    assert bands[0][0] == 71000 and bands[1][0] == 171000
    flat = [s for b in bands for s in b]
    assert len(set(flat)) == len(flat)


def test_screen_rejects_below_floor():
    """一次スクリーン: floor 未満で早期棄却＝壊れた候補に本判定の1.5時間を使わない。"""
    assert AG.screen_decision(40, 96, 0.48) == "reject"
    assert AG.screen_decision(48, 96, 0.48) == "continue"   # 境界は継続側


def test_final_requires_both_point_estimate_and_ci():
    """勝率 ≥ frac **かつ** CI下限 > 0.50。片方だけでは通さない。"""
    # 全ペア 1.0（＝候補が両席で勝つ）: 点推定 1.0・分散 0 で明確に通る
    ok, ci = AG.final_decision([1.0] * 40, frac=0.55)
    assert ok and ci["lo"] > 0.5
    # 点推定は 0.55 だが散らばりが大きく CI下限が 0.5 を割る＝通さない
    noisy = [1.0] * 11 + [0.0] * 9      # 11/20 = 0.55 ちょうど・分散最大
    ok2, ci2 = AG.final_decision(noisy, frac=0.55)
    assert ci2["win_rate"] >= 0.55 and ci2["lo"] < 0.5 and not ok2


def test_final_rejects_coin_flip():
    """真に五分（全ペア 0.5）は CI が潰れても点推定で落ちる＝ヌル対照が PASS しない。"""
    ok, ci = AG.final_decision([0.5] * 100, frac=0.55)
    assert not ok and ci["win_rate"] == pytest.approx(0.5)
