"""価値関数の**小型MLP**をオフライン学習し `mlp-v1` JSON へ書き出す（NNUE路線・dev専用）。

線形(logreg)/GBDT が val_acc ~0.65 で頭打ちだったのに対し、小容量の MLP で非線形を入れて天井を破れるか試す。
学習は numpy（dev のみ・pip 導入可）。**推論は重みを pure-Python/JSON 化**＝`cpu_value_model._mlp_predict`
（stdlib のみ）で読む＝PyPy 同梱可・デプロイ制約クリア。`train_value.py`/`train_gbdt.py` と同じ JSONL を読む。

実行例:
    python tests/train_mlp.py --data tests/value_hard.jsonl --hidden 48 24 --epochs 300 --out tests/cand_mlp.json
"""
import argparse
import json
import sys

import numpy as np

import conftest  # noqa: F401
from opcg_sim.src.core import cpu_features


def _load(path):
    X, Y = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if len(r["f"]) != cpu_features.N_FEATURES:
                continue
            X.append(r["f"]); Y.append(float(r["y"]))
    return np.asarray(X, dtype=np.float64), np.asarray(Y, dtype=np.float64)


def _init_layers(dims, rng):
    """He 初期化（relu 前提）。各層 (W:[in,out], b:[out])。"""
    layers = []
    for i in range(len(dims) - 1):
        nin, nout = dims[i], dims[i + 1]
        W = rng.standard_normal((nin, nout)) * np.sqrt(2.0 / nin)
        b = np.zeros(nout)
        layers.append([W, b])
    return layers


def _forward(layers, X):
    """relu 隠れ層→線形出力ロジット。activations を返す（逆伝播用）。"""
    acts = [X]
    h = X
    for k, (W, b) in enumerate(layers):
        z = h @ W + b
        h = np.maximum(z, 0.0) if k < len(layers) - 1 else z   # 最終層は線形
        acts.append(h)
    return acts


def _train(X, Y, hidden, epochs, lr, l2, batch, patience, seed=0):
    rng = np.random.default_rng(seed)
    n = len(X)
    idx = rng.permutation(n)
    X, Y = X[idx], Y[idx]
    nval = max(1, int(0.15 * n))
    Xtr, Ytr, Xva, Yva = X[nval:], Y[nval:], X[:nval], Y[:nval]
    mean = Xtr.mean(0); std = Xtr.std(0); std[std < 1e-9] = 1.0
    Xtr = (Xtr - mean) / std; Xva = (Xva - mean) / std

    dims = [cpu_features.N_FEATURES] + list(hidden) + [1]
    layers = _init_layers(dims, rng)
    # Adam
    mW = [[np.zeros_like(W), np.zeros_like(b)] for W, b in layers]
    vW = [[np.zeros_like(W), np.zeros_like(b)] for W, b in layers]
    b1, b2, eps = 0.9, 0.999, 1e-8
    t = 0
    best = {"loss": 1e9, "layers": None, "ep": 0}

    def _logits(L, A):
        return _forward(L, A)[-1][:, 0]

    def _val():
        zl = _logits(layers, Xva)
        p = 1.0 / (1.0 + np.exp(-zl))
        p = np.clip(p, 1e-7, 1 - 1e-7)
        ll = -np.mean(Yva * np.log(p) + (1 - Yva) * np.log(1 - p))
        acc = np.mean((p >= 0.5) == (Yva >= 0.5))
        return ll, acc

    for ep in range(epochs):
        perm = rng.permutation(len(Xtr))
        for s in range(0, len(Xtr), batch):
            bi = perm[s:s + batch]
            xb, yb = Xtr[bi], Ytr[bi]
            acts = _forward(layers, xb)
            z = acts[-1][:, 0]
            p = 1.0 / (1.0 + np.exp(-z))
            g = ((p - yb) / len(yb))[:, None]          # dL/dlogit
            t += 1
            for k in range(len(layers) - 1, -1, -1):
                W, b = layers[k]
                a_prev = acts[k]
                gW = a_prev.T @ g + l2 * W
                gb = g.sum(0)
                # Adam 更新
                for arr, grad, mv, vv in ((W, gW, mW[k][0], vW[k][0]), (b, gb, mW[k][1], vW[k][1])):
                    mv *= b1; mv += (1 - b1) * grad
                    vv *= b2; vv += (1 - b2) * (grad * grad)
                    mhat = mv / (1 - b1 ** t); vhat = vv / (1 - b2 ** t)
                    arr -= lr * mhat / (np.sqrt(vhat) + eps)
                if k > 0:
                    g = (g @ W.T) * (acts[k] > 0.0)    # relu 勾配
        ll, acc = _val()
        if ll < best["loss"] - 1e-5:
            best = {"loss": ll, "acc": acc, "layers": [[W.copy(), b.copy()] for W, b in layers], "ep": ep}
        elif ep - best["ep"] >= patience:
            break
    return best, mean, std


def _export(best, mean, std, path):
    layers_json = []
    L = best["layers"]
    for k, (W, b) in enumerate(L):
        act = "relu" if k < len(L) - 1 else "linear"
        layers_json.append({"W": W.T.tolist(), "b": b.tolist(), "act": act})   # [out][in]
    model = {
        "format": "mlp-v1",
        "feature_names": cpu_features.FEATURE_NAMES,
        "n_features": cpu_features.N_FEATURES,
        "mean": mean.tolist(), "std": std.tolist(),
        "layers": layers_json,
        "val_acc": float(best["acc"]), "val_logloss": float(best["loss"]),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(model, f, ensure_ascii=False)
    return model


def main(argv=None):
    ap = argparse.ArgumentParser(description="価値関数の小型MLP学習→mlp-v1 JSON")
    ap.add_argument("--data", default="tests/value_hard.jsonl")
    ap.add_argument("--hidden", type=int, nargs="+", default=[48, 24])
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--out", default="tests/cand_mlp.json")
    args = ap.parse_args(argv)

    X, Y = _load(args.data)
    print(f"data: {len(X)} rows, {cpu_features.N_FEATURES} features, pos_rate={Y.mean():.3f}")
    best, mean, std = _train(X, Y, args.hidden, args.epochs, args.lr, args.l2, args.batch, args.patience)
    _export(best, mean, std, args.out)
    print(f"MLP {args.hidden}: val_acc={best['acc']:.4f} val_logloss={best['loss']:.4f} "
          f"(best ep {best['ep']}) → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
