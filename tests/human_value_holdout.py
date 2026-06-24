"""人間ログの『正直な汎化』検証 — Leave-One-Game-Out 交差検証（オフライン・dev専用・stdlib-only）。

`eval_value_on_set.py` は学習に使っていない外部セットで採点するが、**1 対局＝多数の相関した局面**のため、
同じ対局の行が学習側と採点側に分かれて混ざると（行単位分割）汎化が楽観化する。本ツールは

  - 采取 JSON 1 ファイル = 1 対局 = **1 グループ**として、
  - 「1 対局を抜いて残りで学習 → 抜いた 1 対局で採点」を全対局で回し、
  - **out-of-fold 予測をプール**して採点する（＝どの採点行も、その対局を学習に使っていない）。

これが「カンニングなし」の汎化指標。比較のため、同じプール行に対する **同梱モデル**と、楽観バイアスの
大きさを示す **in-sample（全行で学習し全行で採点）** も併記する。

安全: 学習は in-memory の候補モデルのみ。**同梱 `value_model.json` は一切書き換えない**（読み取り専用）。

実行例:
    OPCG_LOG_SILENT=1 python tests/human_value_holdout.py
    OPCG_LOG_SILENT=1 python tests/human_value_holdout.py --in tests/human_captures/ --epochs 300
"""
import argparse
import glob
import os
import sys
from typing import Callable, List, Optional, Tuple

import conftest  # noqa: F401
import eval_value_on_set as E
import human_log_ingest as ingest
import train_gbdt
import train_value
from opcg_sim.src.core import cpu_features, cpu_value_model

Group = Tuple[str, List[List[float]], List[int]]


# --- データ: 1 ファイル = 1 対局 = 1 グループ ---------------------------------------

def load_groups(paths: List[str]) -> List[Group]:
    """采取ファイル群を読み、有効サンプルを持つものを (名前, X, Y) のグループとして返す。"""
    groups: List[Group] = []
    for p in sorted(paths):
        try:
            rows = ingest.rows_from_file(p)
        except (OSError, ValueError):
            continue
        if not rows:
            continue
        X = [r["f"] for r in rows]
        Y = [r["y"] for r in rows]
        groups.append((os.path.basename(p), X, Y))
    return groups


def expand_dir(inp: str) -> List[str]:
    if os.path.isdir(inp):
        return sorted(glob.glob(os.path.join(inp, "*.json")))
    return [inp]


# --- 候補モデル学習（本番と同じ logreg-standardized-v1 形式で in-memory 出力） ------

def train_candidate(X: List[List[float]], Y: List[int], epochs=300, lr=0.1, l2=1e-4) -> Optional[dict]:
    """学習し本番推論経路が読める dict を返す。単一クラスしか無い等で学習不能なら None。"""
    if len(set(Y)) < 2:
        return None
    w, b, mean, std = train_value.train(X, Y, epochs=epochs, lr=lr, l2=l2)
    return {
        "format": cpu_value_model.MODEL_FORMAT,
        "feature_names": cpu_features.FEATURE_NAMES,
        "weights": w, "intercept": b, "mean": mean, "std": std,
    }


def train_gbdt_candidate(X: List[List[float]], Y: List[int],
                         trees=120, depth=3, lr=0.1) -> Optional[dict]:
    """非線形（GBDT）候補を本番推論経路が読む gbdt-v1 dict で返す。単一クラスは None。"""
    if len(set(Y)) < 2:
        return None
    m = train_gbdt.train(X, Y, trees=trees, depth=depth, lr=lr)
    return {
        "format": "gbdt-v1",
        "feature_names": cpu_features.FEATURE_NAMES,
        "n_features": cpu_features.N_FEATURES,
        "base_score": m["base_score"], "learning_rate": m["learning_rate"], "trees": m["trees"],
    }


def _predict_with(f: List[float], model) -> float:
    """本番経路で勝率を出す（単一情報源）。None は 0.5（=情報なし）に倒す。"""
    p = cpu_value_model.predict_winprob(f, model=model)
    return 0.5 if p is None else p


# --- Leave-One-Game-Out（純粋・関数注入でテスト可能） ------------------------------

def logo_oof(groups: List[Group],
             train_fn: Callable[[List[List[float]], List[int]], object],
             predict_fn: Callable[[List[float], object], float],
             ) -> Tuple[List[float], List[int], List[dict]]:
    """各グループを 1 度ずつ held-out にし、残りで学習したモデルで held-out 行を予測。

    返り値: (プールした out-of-fold 予測, 同順の正解 y, フォールド別メタ)。
    学習不能なフォールド（train_fn が None を返す）はスキップしメタに記録する。
    どの予測行も、その行が属する対局を学習に使っていない（グループ混入ゼロ）。
    """
    oof_probs: List[float] = []
    oof_ys: List[int] = []
    folds: List[dict] = []
    for i, (name, Xte, Yte) in enumerate(groups):
        Xtr: List[List[float]] = []
        Ytr: List[int] = []
        for j, (_, Xj, Yj) in enumerate(groups):
            if j == i:
                continue
            Xtr.extend(Xj)
            Ytr.extend(Yj)
        model = train_fn(Xtr, Ytr) if Xtr else None
        if model is None:
            folds.append({"game": name, "n": len(Yte), "trained": False, "acc": None})
            continue
        probs = [predict_fn(f, model) for f in Xte]
        oof_probs.extend(probs)
        oof_ys.extend(Yte)
        folds.append({"game": name, "n": len(Yte), "trained": True,
                      "acc": E.accuracy(probs, Yte)})
    return oof_probs, oof_ys, folds


# --- 指標ブロック（既存 eval_value_on_set の純粋指標を再利用） ----------------------

def metrics(probs: List[float], ys: List[int]) -> dict:
    pos = sum(ys) / len(ys)
    const = [pos] * len(ys)
    return {
        "n": len(ys), "pos_rate": pos,
        "acc": E.accuracy(probs, ys), "logloss": E.logloss(probs, ys),
        "brier": E.brier(probs, ys),
        "base_acc": E.accuracy(const, ys), "base_logloss": E.logloss(const, ys),
    }


def _fmt(label: str, m: dict) -> str:
    lift = m["base_logloss"] - m["logloss"]
    verdict = "当てている" if lift > 0 and m["acc"] > m["base_acc"] else "定数予測を超えていない（汎化弱）"
    return (f"  --- {label} ---\n"
            f"    rows={m['n']}  acc={m['acc']:.4f} (定数 {m['base_acc']:.4f})  "
            f"logloss={m['logloss']:.4f} (定数 {m['base_logloss']:.4f})  brier={m['brier']:.4f}\n"
            f"    → logloss 改善 {lift:+.4f}・判定: {verdict}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="人間ログ Leave-One-Game-Out 汎化検証（読み取り専用）")
    ap.add_argument("--in", dest="inp", default=os.path.join(os.path.dirname(__file__), "human_captures"),
                    help="采取 JSON ディレクトリ or ファイル（既定: tests/human_captures）")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--model", choices=("linear", "gbdt"), default="linear",
                    help="候補モデルの種類（linear=ロジ回帰 / gbdt=非線形）")
    ap.add_argument("--trees", type=int, default=120)
    ap.add_argument("--depth", type=int, default=3)
    args = ap.parse_args(argv)

    groups = load_groups(expand_dir(args.inp))
    if len(groups) < 2:
        print(f"対局が不足: {len(groups)} 件（Leave-One-Game-Out には 2 対局以上が必要）")
        return 1

    total_rows = sum(len(Y) for _, _, Y in groups)
    pos = sum(sum(Y) for _, _, Y in groups) / total_rows
    print("=== 人間ログ Leave-One-Game-Out 検証（正直な汎化） ===")
    print(f"  games={len(groups)}  rows={total_rows}  pos_rate={pos:.3f}  候補={args.model}")

    if args.model == "gbdt":
        def _train(X, Y):
            return train_gbdt_candidate(X, Y, trees=args.trees, depth=args.depth, lr=args.lr)
    else:
        def _train(X, Y):
            return train_candidate(X, Y, epochs=args.epochs, lr=args.lr, l2=args.l2)

    # ① 候補モデル: LOGO out-of-fold（カンニングなし＝本指標）
    oof_probs, oof_ys, folds = logo_oof(groups, _train, _predict_with)
    if not oof_ys:
        print("  全フォールドが学習不能（各対局のラベルが偏りすぎ）。対局数を増やしてください。")
        return 1
    logo_m = metrics(oof_probs, oof_ys)
    print(_fmt("候補モデル（LOGO out-of-fold＝カンニングなし・本指標）", logo_m))

    # ② 同梱モデル: 同じ採点行を参照線として採点
    bundled_probs = [_predict_with(f, None) for (_, X, _) in groups for f in X]
    bundled_ys = [y for (_, _, Y) in groups for y in Y]
    print(_fmt("同梱モデル（同じ行・参照線）", metrics(bundled_probs, bundled_ys)))

    # ③ in-sample: 全行で学習し全行で採点（楽観値・参考）
    allX = [f for (_, X, _) in groups for f in X]
    allY = [y for (_, _, Y) in groups for y in Y]
    insample = _train(allX, allY)
    if insample is not None:
        in_probs = [_predict_with(f, insample) for f in allX]
        in_m = metrics(in_probs, allY)
        print(_fmt("候補モデル（in-sample・楽観値・参考）", in_m))
        print(f"  楽観バイアス（in-sample acc − LOGO acc）= {in_m['acc'] - logo_m['acc']:+.4f}")

    # フォールド別 OOF 正答率（どの対局が読めて/読めないか）
    print("  --- 対局別 out-of-fold acc ---")
    for fo in folds:
        a = f"{fo['acc']:.3f}" if fo["acc"] is not None else "skip(単一クラス)"
        print(f"    {fo['game']:<28} n={fo['n']:>3}  acc={a}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
