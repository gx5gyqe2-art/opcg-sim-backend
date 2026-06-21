"""学習価値関数のローダ＋推論（stdlib-only・PyPy/CPython 両対応）。§2.5.7 残5。

オフライン自己対戦で学習した**線形（ロジスティック回帰）モデル**を `value_model.json` から読み、
`cpu_features.extract_features` の特徴ベクトルから**勝率 [0,1]** を推定する。GBDT へ拡張する場合も
同じ pure-Python 推論で差し替えられる（重みは JSON 同梱＝`build_card_cache` と同方式）。

安全設計:
  - モデル未同梱／読込失敗なら `is_available()=False`＝呼び出し側は現行 `evaluate` にフォールバック。
  - ブレンド率 `OPCG_VALUE_BLEND`（既定 0.0＝完全OFF＝現状と同値）。0 のとき推論は一切走らない。
  - 推論は標準化＋ロジスティックのみ（stdlib `math`）＝µs 級・葉で安全。
"""
import json
import math
import os
from typing import List, Optional

from . import cpu_features

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "value_model.json")

_MODEL = None          # 読込済みモデル（dict）or False（読込失敗/未同梱）
_LOAD_TRIED = False
MODEL_FORMAT = "logreg-standardized-v1"


def _load():
    global _MODEL, _LOAD_TRIED
    if _LOAD_TRIED:
        return _MODEL
    _LOAD_TRIED = True
    try:
        with open(_MODEL_PATH, "r", encoding="utf-8") as f:
            m = json.load(f)
        # 特徴の順序/個数が現行と一致することを検証（不一致＝無効化してフォールバック）。
        if (m.get("format") == MODEL_FORMAT
                and m.get("feature_names") == cpu_features.FEATURE_NAMES
                and len(m.get("weights", [])) == cpu_features.N_FEATURES
                and len(m.get("mean", [])) == cpu_features.N_FEATURES
                and len(m.get("std", [])) == cpu_features.N_FEATURES):
            _MODEL = m
        else:
            _MODEL = False
    except (OSError, ValueError):
        _MODEL = False
    return _MODEL


def is_available() -> bool:
    return bool(_load())


_ALPHA_OVERRIDE = None   # 自己対戦アリーナで「片側だけ α を変える」ための上書き（本番/テストは None）。


def set_alpha_override(a):
    """ブレンド率を一時的に上書き（評価アリーナ用）。None で env 既定に戻す。"""
    global _ALPHA_OVERRIDE
    _ALPHA_OVERRIDE = None if a is None else min(1.0, max(0.0, float(a)))


def blend_alpha() -> float:
    """葉評価へのブレンド率（0=OFF=現状同値）。上書き>env>0 の優先。tests/本番は未設定=0。"""
    if _ALPHA_OVERRIDE is not None:
        return _ALPHA_OVERRIDE
    try:
        a = float(os.environ.get("OPCG_VALUE_BLEND", "0") or "0")
    except ValueError:
        return 0.0
    return min(1.0, max(0.0, a))


def predict_winprob(features: List[float]) -> Optional[float]:
    """標準化＋ロジスティックで勝率 [0,1] を返す。モデル無/長さ不一致は None。"""
    m = _load()
    if not m:
        return None
    w = m["weights"]; mean = m["mean"]; std = m["std"]
    if len(features) != len(w):
        return None
    z = float(m.get("intercept", 0.0))
    for i, x in enumerate(features):
        s = std[i]
        xs = (x - mean[i]) / s if s > 1e-9 else (x - mean[i])
        z += w[i] * xs
    # 数値安定なシグモイド。
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)
