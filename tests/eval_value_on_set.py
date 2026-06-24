"""学習価値モデルの『外部セット汎化』検証ハーネス（オフライン・dev専用・stdlib-only）。検証セット(a)。

人間 vs CPU 対戦から採取したラベル付き特徴セット（JSONL 1行 `{"f":[...],"y":0/1}`）に対し、
**学習済みモデル**（同梱 `value_model.json` または候補ファイル）が勝敗を当てられるかを採点する。
天井上げの本丸＝人間ログ活用の **(a) 検証セット**：自己対戦の内部ホールドアウトより『実戦汎化』の本物指標。

学習（`train_value`）はデータ内の 8:2 ホールドアウトで val を測るが、本ハーネスは**学習に使っていない外部セット**
でのみ採点する＝実戦分布での汎化を独立に測る。読み取り専用：モデルも `value_model.json` も一切書き換えない。

指標:
  - acc（>=0.5 を勝ち予測とした正答率）/ logloss / Brier（二乗誤差）/ ECE（キャリブレーション誤差）
  - 参照線＝定数予測（p=pos_rate）の logloss/acc。モデルがこれを上回って初めて『当てている』。

実行例:
    OPCG_LOG_SILENT=1 python tests/eval_value_on_set.py --data /tmp/human_value.jsonl
    OPCG_LOG_SILENT=1 python tests/eval_value_on_set.py --data /tmp/human_value.jsonl --model /tmp/cand.json
"""
import argparse
import json
import math
import sys
from typing import List, Optional, Tuple

import conftest  # noqa: F401
from opcg_sim.src.core import cpu_features, cpu_value_model

_EPS = 1e-12


def load_rows(path: str) -> Tuple[List[List[float]], List[int]]:
    """JSONL（`{"f":[...],"y":0/1}`）を読み、特徴長が現行スキーマと一致する行だけ返す。"""
    X, Y = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if len(r.get("f", [])) == cpu_features.N_FEATURES and r.get("y") in (0, 1):
                X.append([float(v) for v in r["f"]])
                Y.append(int(r["y"]))
    return X, Y


# --- 純粋指標（テスト可能・モデル非依存） ----------------------------------------

def accuracy(probs: List[float], ys: List[int]) -> float:
    return sum(1 for p, y in zip(probs, ys) if (p >= 0.5) == bool(y)) / len(ys)


def logloss(probs: List[float], ys: List[int]) -> float:
    s = 0.0
    for p, y in zip(probs, ys):
        p = min(1 - _EPS, max(_EPS, p))
        s += -(y * math.log(p) + (1 - y) * math.log(1 - p))
    return s / len(ys)


def brier(probs: List[float], ys: List[int]) -> float:
    return sum((p - y) ** 2 for p, y in zip(probs, ys)) / len(ys)


def calibration(probs: List[float], ys: List[int], n_bins: int = 10):
    """予測確率を n_bins 等幅ビンに分け、(範囲, 件数, 平均予測, 実勝率) と ECE を返す。"""
    buckets = [[] for _ in range(n_bins)]
    for p, y in zip(probs, ys):
        b = min(n_bins - 1, int(p * n_bins))
        buckets[b].append((p, y))
    rows = []
    ece = 0.0
    n = len(ys)
    for i, bk in enumerate(buckets):
        lo, hi = i / n_bins, (i + 1) / n_bins
        if not bk:
            rows.append((lo, hi, 0, None, None))
            continue
        mp = sum(p for p, _ in bk) / len(bk)
        my = sum(y for _, y in bk) / len(bk)
        rows.append((lo, hi, len(bk), mp, my))
        ece += (len(bk) / n) * abs(mp - my)
    return rows, ece


def predict_all(X: List[List[float]], model=None) -> List[float]:
    """各特徴ベクトルの勝率予測。model=None なら同梱 value_model.json（本番モデル）。"""
    out = []
    for f in X:
        p = cpu_value_model.predict_winprob(f, model=model)
        out.append(0.5 if p is None else p)
    return out


def evaluate(X: List[List[float]], Y: List[int], model=None) -> dict:
    """セット全体の指標一式を返す（読み取り専用・純粋）。"""
    probs = predict_all(X, model=model)
    pos_rate = sum(Y) / len(Y)
    const = [pos_rate] * len(Y)   # 定数予測の参照線
    cal_rows, ece = calibration(probs, Y)
    return {
        "n": len(Y), "pos_rate": pos_rate,
        "acc": accuracy(probs, Y), "logloss": logloss(probs, Y),
        "brier": brier(probs, Y), "ece": ece,
        "base_acc": accuracy(const, Y), "base_logloss": logloss(const, Y),
        "calibration": cal_rows,
    }


def _fmt_report(r: dict, label: str) -> str:
    lines = [
        f"=== value-model 外部セット検証: {label} ===",
        f"  rows={r['n']}  pos_rate={r['pos_rate']:.3f}",
        f"  acc      = {r['acc']:.4f}   (定数予測 {r['base_acc']:.4f})",
        f"  logloss  = {r['logloss']:.4f}   (定数予測 {r['base_logloss']:.4f})",
        f"  brier    = {r['brier']:.4f}",
        f"  ECE      = {r['ece']:.4f}",
        "  --- キャリブレーション（予測ビン: 件数 平均予測→実勝率） ---",
    ]
    for lo, hi, cnt, mp, my in r["calibration"]:
        if cnt == 0:
            continue
        lines.append(f"    [{lo:.1f},{hi:.1f}) n={cnt:>4}  {mp:.3f} → {my:.3f}")
    lift_ll = r["base_logloss"] - r["logloss"]
    verdict = "当てている" if lift_ll > 0 and r["acc"] > r["base_acc"] else "定数予測を超えていない（汎化弱）"
    lines.append(f"  → logloss 改善 {lift_ll:+.4f} vs 定数予測・判定: {verdict}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="価値モデルの外部セット汎化検証（読み取り専用）")
    ap.add_argument("--data", required=True, help="ラベル付き特徴 JSONL（{'f':[...],'y':0/1}）")
    ap.add_argument("--model", default=None,
                    help="候補モデル JSON（省略時は同梱 value_model.json）")
    args = ap.parse_args(argv)

    X, Y = load_rows(args.data)
    if not X:
        print(f"有効行なし: {args.data}（特徴長 {cpu_features.N_FEATURES} 一致・y∈{{0,1}} の行が0）")
        return 1

    model: Optional[dict] = None
    label = "同梱 value_model.json"
    if args.model:
        model = cpu_value_model.load_model_file(args.model)
        if model is None:
            print(f"モデル読込/検証失敗: {args.model}")
            return 1
        label = args.model
    elif not cpu_value_model.is_available():
        print("同梱 value_model.json が読めない/スキーマ不一致")
        return 1

    print(_fmt_report(evaluate(X, Y, model=model), label))
    return 0


if __name__ == "__main__":
    sys.exit(main())
