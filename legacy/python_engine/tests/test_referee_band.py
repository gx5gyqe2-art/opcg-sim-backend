"""同価値バンド v2（v8 柱B・`counterfactual_referee.same_value`）の対判定則。

CRN（全プランが同じ世界線を共有）を活かした符号検定風の断定則:
  - 同じ世界で勝敗が割れたペアの正味差 (n10−n01) ≥ 3 → 実差＝断定
  - それ未満 かつ 平均残ライフ差 < band → 同価値
  - それ未満 でも ライフ差 ≥ band → 断定はしないが序列（同価値ではない）
較正根拠（@64 実測）: 素朴な勝ち数差は ±1 勝のノイズで断定が往復した
（6世界=同値・12世界=素攻撃+1・16世界=付与+1）。対判定なら3測定とも同価値で一致する。
基盤健全性＝cpu_infra。純関数のみ＝高速。
"""
import os
import sys

import conftest  # noqa: F401  (google スタブ注入 & sys.path 設定)
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "tests", "scripts"))
from counterfactual_referee import same_value  # noqa: E402

pytestmark = pytest.mark.cpu_infra


def _e(outcomes, lifem=0.0):
    return {"outcomes": dict(enumerate(outcomes)), "lifem": lifem}


def test_net_discordance_3_is_decisive():
    """正味不一致3以上＝断定（同価値でない）。ライフ差が小さくても実差。"""
    best = _e([1, 1, 1, 1, 0, 0], lifem=0.1)
    e = _e([0, 0, 0, 1, 0, 0], lifem=0.0)   # n10=3, n01=0
    assert not same_value(best, e)


def test_one_win_flip_is_tie():
    """±1勝の揺れ（@64 実測の形）は同価値。運の共通項が消えた後の差が1世界分しかない。"""
    best = _e([1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1], lifem=0.67)   # 8/12 相当
    e = _e([1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0], lifem=0.33)      # 7/12・n10=1
    assert same_value(best, e)


def test_discordance_cancels():
    """割れが相殺する（n10=2, n01=1→正味1）なら勝ち数差があっても同価値。"""
    best = _e([1, 1, 0, 1], lifem=0.2)
    e = _e([0, 0, 1, 1], lifem=0.0)   # n10=2, n01=1
    assert same_value(best, e)


def test_life_band_separates_saturated():
    """飽和局面（全世界同勝敗＝不一致0）はライフ差 ≥ band で同価値から外れる（序列は付く）。"""
    best = _e([1, 1, 1, 1], lifem=2.0)
    e = _e([1, 1, 1, 1], lifem=0.5)
    assert not same_value(best, e, band=0.5)
    assert same_value(best, e, band=2.0)


def test_missing_worlds_use_common_subset():
    """不成立世界はペアから除外＝共通世界だけで対判定する。"""
    best = _e([1, 1, 1, 1, 1, 1], lifem=0.0)
    e = {"outcomes": {0: False, 1: False, 2: False}, "lifem": 0.0}   # 共通3世界で n10=3
    assert not same_value(best, e)
