"""学習価値の**非線形化**（GBDT・天井上げ Phase）= pure-Python 勾配ブースティング木の学習＋エクスポート。

線形ロジ回帰（`train_value.py`・val_acc 0.725）では「特徴の組み合わせの妙」を表現できない。GBDT は
浅い回帰木を勾配方向に積んで非線形な相互作用/閾値を学習する。stdlib-only（外部ライブラリ無し）・推論は
木の走査だけ＝µs 級で葉で安全。`collect_value_data.py` の JSONL を読み、
`opcg_sim/src/core/value_model.json` 互換の **gbdt-v1** 形式（木の入れ子 dict）を書き出す。

設計（XGBoost 風・二値ロジ損失）:
  - 各反復で p=sigmoid(raw)・勾配 g=y-p・ヘシアン h=p(1-p) を計算し、g を目的に回帰木を 1 本学習。
  - 木の分割は gain = GL²/(HL+λ)+GR²/(HR+λ)-G²/(H+λ) を最大化（Newton）。葉値 = G/(H+λ)。
  - 候補閾値は特徴ごとの分位点（ヒストグラム・既定 24）に限定して pure-Python でも現実的な速度に。
  - raw += lr·tree_pred（shrinkage）。

実行例:
    OPCG_LOG_SILENT=1 python tests/train_gbdt.py --data /tmp/value_data.jsonl --trees 120 --depth 3
"""
import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional

import conftest  # noqa: F401
from opcg_sim.src.core import cpu_features

_OUT = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                    "opcg_sim", "src", "core", "value_model.json")


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


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


def _candidate_thresholds(X: List[List[float]], d: int, n_bins: int) -> List[List[float]]:
    """特徴ごとの分位点しきい値（ヒストグラム候補）。定数列は空＝分割不可。"""
    n = len(X)
    cands: List[List[float]] = []
    for j in range(d):
        vals = sorted(X[i][j] for i in range(n))
        uniq = sorted(set(vals))
        if len(uniq) < 2:
            cands.append([])
            continue
        if len(uniq) <= n_bins:
            ts = [(uniq[k] + uniq[k + 1]) / 2.0 for k in range(len(uniq) - 1)]
        else:
            ts = []
            for b in range(1, n_bins):
                q = vals[min(n - 1, int(b * n / n_bins))]
                ts.append(q)
            ts = sorted(set(ts))
        cands.append(ts)
    return cands


def _build_tree(idx, X, g, h, cands, depth, max_depth, min_leaf, lam):
    G = sum(g[i] for i in idx); H = sum(h[i] for i in idx)
    leaf_val = G / (H + lam)
    if depth >= max_depth or len(idx) < 2 * min_leaf:
        return {"v": leaf_val}
    base = G * G / (H + lam)
    best = None  # (gain, f, thr, left, right)
    for f in range(len(cands)):
        for thr in cands[f]:
            GL = HL = 0.0; nl = 0
            left = []
            for i in idx:
                if X[i][f] <= thr:
                    left.append(i); GL += g[i]; HL += h[i]; nl += 1
            nr = len(idx) - nl
            if nl < min_leaf or nr < min_leaf:
                continue
            GR = G - GL; HR = H - HL
            gain = GL * GL / (HL + lam) + GR * GR / (HR + lam) - base
            if best is None or gain > best[0]:
                best = (gain, f, thr, left, set(left))
    if best is None or best[0] <= 1e-9:
        return {"v": leaf_val}
    _, f, thr, left, lset = best
    right = [i for i in idx if i not in lset]
    return {"f": f, "t": thr,
            "l": _build_tree(left, X, g, h, cands, depth + 1, max_depth, min_leaf, lam),
            "r": _build_tree(right, X, g, h, cands, depth + 1, max_depth, min_leaf, lam)}


def tree_predict(node: Dict[str, Any], x: List[float]) -> float:
    """木 1 本の予測（cpu_value_model の推論と同一規約：`v`=葉／`f,t,l,r`=内部）。"""
    while "v" not in node:
        node = node["l"] if x[node["f"]] <= node["t"] else node["r"]
    return node["v"]


def train(X, Y, trees=120, depth=3, lr=0.1, lam=1.0, min_leaf=20, n_bins=24):
    n = len(X); d = len(X[0])
    base_rate = sum(Y) / n
    base_rate = min(1 - 1e-6, max(1e-6, base_rate))
    base_score = math.log(base_rate / (1 - base_rate))   # 初期 raw=ベース率の log-odds
    cands = _candidate_thresholds(X, d, n_bins)
    raw = [base_score] * n
    forest: List[Dict[str, Any]] = []
    idx_all = list(range(n))
    for _ in range(trees):
        p = [_sigmoid(raw[i]) for i in range(n)]
        g = [Y[i] - p[i] for i in range(n)]
        h = [p[i] * (1 - p[i]) for i in range(n)]
        tree = _build_tree(idx_all, X, g, h, cands, 0, depth, min_leaf, lam)
        for i in range(n):
            raw[i] += lr * tree_predict(tree, X[i])
        forest.append(tree)
    return {"base_score": base_score, "learning_rate": lr, "trees": forest}


def _raw(model, x):
    return model["base_score"] + model["learning_rate"] * sum(tree_predict(t, x) for t in model["trees"])


def _metrics(X, Y, model):
    n = len(X); ll = 0.0; correct = 0
    for i in range(n):
        p = _sigmoid(_raw(model, X[i]))
        p = min(1 - 1e-12, max(1e-12, p))
        ll += -(Y[i] * math.log(p) + (1 - Y[i]) * math.log(1 - p))
        if (p >= 0.5) == bool(Y[i]):
            correct += 1
    return ll / n, correct / n


def main(argv=None):
    ap = argparse.ArgumentParser(description="価値関数の GBDT 学習＋エクスポート（非線形）")
    ap.add_argument("--data", default="/tmp/value_data.jsonl")
    ap.add_argument("--trees", type=int, default=120)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--min-leaf", type=int, default=20)
    ap.add_argument("--bins", type=int, default=24)
    ap.add_argument("--out", default=_OUT)
    args = ap.parse_args(argv)

    from value_dataset import load_rows, split
    rows = load_rows(args.data)
    if len(rows) < 50:
        print(f"データ不足: {len(rows)} 行（>=50 必要）"); return 1
    # **試合単位 split**（リーク防止・全トレーナ共通の seed/val_frac＝同一 val 試合集合）。
    Xtr, Ytr, Xva, Yva, meta = split(rows, val_frac=0.15, seed=0)
    Ytr = [int(y) for y in Ytr]; Yva = [int(y) for y in Yva]
    m = train(Xtr, Ytr, trees=args.trees, depth=args.depth, lr=args.lr,
              lam=args.lam, min_leaf=args.min_leaf, n_bins=args.bins)
    tr_ll, tr_acc = _metrics(Xtr, Ytr, m)
    va_ll, va_acc = _metrics(Xva, Yva, m) if Xva else (0.0, 0.0)
    print(f"[gbdt {args.trees}t/d{args.depth}] split={meta['mode']} games={meta['n_games']}(val {meta['val_games']}) "
          f"train={meta['n_train']} val={meta['n_val']} | train acc={tr_acc:.3f} | "
          f"val logloss={va_ll:.4f} acc={va_acc:.4f}")

    model = {
        "format": "gbdt-v1",
        "feature_names": cpu_features.FEATURE_NAMES,
        "n_features": cpu_features.N_FEATURES,
        "base_score": m["base_score"], "learning_rate": m["learning_rate"], "trees": m["trees"],
        "trained_rows": len(Xtr), "val_acc": va_acc, "val_logloss": va_ll,
        "params": {"trees": args.trees, "depth": args.depth, "lr": args.lr, "lam": args.lam},
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(model, f, ensure_ascii=False)
    print(f"wrote model → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
