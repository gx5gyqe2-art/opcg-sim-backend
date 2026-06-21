"""学習価値関数（§2.5.7 残5）の **学習＋エクスポート**（オフライン・dev専用・stdlib-only）。

`collect_value_data.py` が出した JSONL を読み、標準化＋ロジスティック回帰（pure-Python 勾配降下・L2）で
勝率モデルを学習し、`opcg_sim/src/core/value_model.json`（`cpu_value_model` が読む形）へ書き出す。
外部ライブラリ不要＝ローカルでも Cloud Run Jobs でも動く。重みは小さく、リポジトリ同梱で本番反映。

実行例:
    OPCG_LOG_SILENT=1 python tests/train_value.py --data /tmp/value_data.jsonl --epochs 300
"""
import argparse
import json
import math
import os
import random
import sys
from typing import List

import conftest  # noqa: F401
from opcg_sim.src.core import cpu_features

_OUT = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                    "opcg_sim", "src", "core", "value_model.json")


def _load_rows(path: str):
    X, Y = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if len(r["f"]) == cpu_features.N_FEATURES:
                X.append([float(v) for v in r["f"]])
                Y.append(int(r["y"]))
    return X, Y


def _standardize_params(X: List[List[float]]):
    n = len(X); d = len(X[0])
    mean = [0.0] * d
    for row in X:
        for j in range(d):
            mean[j] += row[j]
    mean = [m / n for m in mean]
    var = [0.0] * d
    for row in X:
        for j in range(d):
            var[j] += (row[j] - mean[j]) ** 2
    std = [math.sqrt(v / n) for v in var]
    std = [s if s > 1e-9 else 1.0 for s in std]   # 分散ゼロ（bias等）は素通し
    return mean, std


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def train(X, Y, epochs=300, lr=0.1, l2=1e-4, seed=0):
    n = len(X); d = len(X[0])
    mean, std = _standardize_params(X)
    Xs = [[(row[j] - mean[j]) / std[j] for j in range(d)] for row in X]
    w = [0.0] * d
    b = 0.0
    idx = list(range(n))
    rng = random.Random(seed)
    for _ in range(epochs):
        rng.shuffle(idx)
        gw = [0.0] * d
        gb = 0.0
        for i in idx:
            row = Xs[i]
            z = b + sum(w[j] * row[j] for j in range(d))
            err = _sigmoid(z) - Y[i]
            for j in range(d):
                gw[j] += err * row[j]
            gb += err
        for j in range(d):
            w[j] -= lr * (gw[j] / n + l2 * w[j])
        b -= lr * (gb / n)
    return w, b, mean, std


def _metrics(X, Y, w, b, mean, std):
    d = len(w); n = len(X)
    ll = 0.0; correct = 0
    for i in range(n):
        z = b + sum(w[j] * ((X[i][j] - mean[j]) / std[j]) for j in range(d))
        p = _sigmoid(z)
        p = min(1 - 1e-12, max(1e-12, p))
        ll += -(Y[i] * math.log(p) + (1 - Y[i]) * math.log(1 - p))
        if (p >= 0.5) == bool(Y[i]):
            correct += 1
    return ll / n, correct / n


def main(argv=None):
    ap = argparse.ArgumentParser(description="価値関数の学習＋エクスポート")
    ap.add_argument("--data", default="/tmp/value_data.jsonl")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--out", default=_OUT)
    args = ap.parse_args(argv)

    X, Y = _load_rows(args.data)
    if len(X) < 50:
        print(f"データ不足: {len(X)} 行（>=50 必要）"); return 1
    # 8:2 ホールドアウトで汎化を見る。
    cut = int(len(X) * 0.8)
    w, b, mean, std = train(X[:cut], Y[:cut], epochs=args.epochs, lr=args.lr, l2=args.l2)
    tr_ll, tr_acc = _metrics(X[:cut], Y[:cut], w, b, mean, std)
    va_ll, va_acc = _metrics(X[cut:], Y[cut:], w, b, mean, std) if len(X) - cut > 0 else (0, 0)
    print(f"rows={len(X)} pos_rate={sum(Y)/len(Y):.3f} | train logloss={tr_ll:.4f} acc={tr_acc:.3f}"
          f" | val logloss={va_ll:.4f} acc={va_acc:.3f}")

    model = {
        "format": "logreg-standardized-v1",
        "feature_names": cpu_features.FEATURE_NAMES,
        "weights": w, "intercept": b, "mean": mean, "std": std,
        "trained_rows": len(X), "val_acc": va_acc, "val_logloss": va_ll,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(model, f, ensure_ascii=False)
    print(f"wrote model → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
