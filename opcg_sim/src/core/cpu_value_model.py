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
MODEL_FORMAT = "logreg-standardized-v1"          # 線形（後方互換）
_FORMATS = (MODEL_FORMAT, "gbdt-v1", "mlp-v1")   # gbdt-v1＝木／mlp-v1＝小型MLP（NNUE路線・pure-Python forward）


def _valid_model(m) -> bool:
    """特徴の順序/個数が現行と一致＋形式ごとの必須フィールドを検証。"""
    if m.get("feature_names") != cpu_features.FEATURE_NAMES:
        return False
    fmt = m.get("format")
    if fmt == MODEL_FORMAT:
        return (len(m.get("weights", [])) == cpu_features.N_FEATURES
                and len(m.get("mean", [])) == cpu_features.N_FEATURES
                and len(m.get("std", [])) == cpu_features.N_FEATURES)
    if fmt == "gbdt-v1":
        return isinstance(m.get("trees"), list) and m.get("n_features") == cpu_features.N_FEATURES
    if fmt == "mlp-v1":
        layers = m.get("layers")
        if not (isinstance(layers, list) and layers
                and m.get("n_features") == cpu_features.N_FEATURES
                and len(m.get("mean", [])) == cpu_features.N_FEATURES
                and len(m.get("std", [])) == cpu_features.N_FEATURES):
            return False
        # 第1層の入力次元＝特徴数・最終層は1ユニット（勝率ロジット）。
        return (len(layers[0]["W"][0]) == cpu_features.N_FEATURES and len(layers[-1]["b"]) == 1)
    return False


def _load():
    global _MODEL, _LOAD_TRIED
    if _LOAD_TRIED:
        return _MODEL
    _LOAD_TRIED = True
    try:
        with open(_MODEL_PATH, "r", encoding="utf-8") as f:
            m = json.load(f)
        _MODEL = m if _valid_model(m) else False
    except (OSError, ValueError):
        _MODEL = False
    return _MODEL


def load_model_file(path: str):
    """外部モデル JSON を読みスキーマ検証して dict を返す（読込失敗/検証NG は None）。

    本番経路（`predict_winprob` の既定）には影響しない。オフライン検証ハーネスが**同梱外の候補モデル**を
    外部セットで採点するための入口（`predict_winprob(features, model=...)` に渡す）。
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            m = json.load(f)
    except (OSError, ValueError):
        return None
    return m if _valid_model(m) else None


def _tree_predict(node, x: List[float]) -> float:
    """GBDT 木 1 本の予測（`v`=葉値／`f,t,l,r`=内部ノード・`train_gbdt.tree_predict` と同一規約）。"""
    while "v" not in node:
        node = node["l"] if x[node["f"]] <= node["t"] else node["r"]
    return node["v"]


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


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _mlp_predict(m, features: List[float]) -> float:
    """mlp-v1 の forward pass（標準化→各層 W·x+b＋活性→最終1ユニットを sigmoid）。pure-Python＝PyPy 可。

    `layers[k]` = {"W": [out][in], "b": [out], "act": "relu"|"tanh"|"linear"}。最終層は linear・1ユニット。
    """
    mean, std = m["mean"], m["std"]
    v = [((features[i] - mean[i]) / std[i] if std[i] > 1e-9 else features[i] - mean[i])
         for i in range(len(features))]
    for layer in m["layers"]:
        W, b, act = layer["W"], layer["b"], layer["act"]
        out = []
        for j in range(len(b)):
            Wj = W[j]
            s = b[j]
            for i in range(len(v)):
                s += Wj[i] * v[i]
            out.append(s)
        if act == "relu":
            out = [o if o > 0.0 else 0.0 for o in out]
        elif act == "tanh":
            out = [math.tanh(o) for o in out]
        v = out
    return _sigmoid(v[0])


def predict_winprob(features: List[float], model=None) -> Optional[float]:
    """勝率 [0,1] を返す。線形（標準化＋ロジスティック）/ GBDT（木の走査）両対応。モデル無/長さ不一致は None。

    `model` 省略時は同梱 `value_model.json`（本番経路・既定）。明示の dict を渡すと**そのモデル**で推論する
    （オフライン検証ハーネスが候補モデルを外部セットで採点する用途・推論の単一情報源を保つ）。
    """
    m = model if model is not None else _load()
    if not m or len(features) != cpu_features.N_FEATURES:
        return None
    if m.get("format") == "gbdt-v1":
        raw = float(m.get("base_score", 0.0))
        lr = float(m.get("learning_rate", 1.0))
        for t in m["trees"]:
            raw += lr * _tree_predict(t, features)
        return _sigmoid(raw)
    if m.get("format") == "mlp-v1":
        return _mlp_predict(m, features)
    # 線形（logreg-standardized-v1）。
    w = m["weights"]; mean = m["mean"]; std = m["std"]
    z = float(m.get("intercept", 0.0))
    for i, x in enumerate(features):
        s = std[i]
        xs = (x - mean[i]) / s if s > 1e-9 else (x - mean[i])
        z += w[i] * xs
    return _sigmoid(z)
