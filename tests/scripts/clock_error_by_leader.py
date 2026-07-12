"""残りターン補助ヘッドの対面別誤差分解（v4 監視 diagnostics・docs/cpu_v4_plan.md §4-3）。

平均誤差（learner ログの aux±T）はリーダー対面ごとの系統偏りを隠しうる（ユーザ指摘 2026-07-12:
残りターンは対面・デッキ依存）。batch.npz（スキーマ v2）の局面を「自リーダー」「対面ペア」で
グループ化し、補助ヘッドの MAE（ターン換算）を分解する。読み取り専用＝学習に影響しない。
偏りが大きい対面が見つかったら §5.5-2（自デッキ残の集約特徴）の切り分け材料にする。

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/clock_error_by_leader.py \
    --net /tmp/ckpt/value.npz --batch /tmp/w1/batch.npz --batch /tmp/w2/batch.npz --top 8
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
from collections import defaultdict

import numpy as np

import os as _os, sys as _sys  # noqa: E402  test bootstrap
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap  # noqa: E402,F401
import rl_net as RN
import rl_encoder as E
from cpu_selfplay import _load_db
from opcg_sim.src.learned.config import V4_TURNS_SCALE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True, help="value.npz（補助ヘッド入り）")
    ap.add_argument("--batch", action="append", required=True, help="batch.npz（複数可）")
    ap.add_argument("--turns-scale", type=float, default=V4_TURNS_SCALE)
    ap.add_argument("--top", type=int, default=8, help="ワースト/ベスト表示件数")
    ap.add_argument("--min-n", type=int, default=100, help="表示する最低局面数（少数サンプルのノイズ除外）")
    args = ap.parse_args()

    db = _load_db()
    vocab = E.build_vocab(db)
    idx2id = {v: k for k, v in vocab.items()}
    net = RN.ValueNet.load(args.net)

    parts = {k: [] for k in ("scalars", "field", "card_idx", "turns_left")}
    for p in args.batch:
        z = np.load(p)
        if "turns_left" not in z.files:
            print(f"  [skip] {p}: スキーマ v1（turns_left 無し）", flush=True)
            continue
        for k in parts:
            parts[k].append(z[k])
    if not parts["scalars"]:
        print("対象データなし"); return 1
    data = {k: np.concatenate(v) for k, v in parts.items()}
    fin = np.isfinite(data["turns_left"])
    data = {k: v[fin] for k, v in data.items()}
    n = len(data["turns_left"])

    # chunk 推論（メモリ節約）
    pred = np.empty(n)
    for s in range(0, n, 8192):
        mb = {k: data[k][s:s + 8192] for k in ("scalars", "field", "card_idx")}
        pred[s:s + 8192] = net.predict_aux(mb)
    pred_t = pred * args.turns_scale
    true_t = np.clip(data["turns_left"], 0, args.turns_scale)
    err = np.abs(pred_t - true_t)
    bias = pred_t - true_t

    print(f"=== 時計誤差の対面別分解: {n}局面 / 全体 MAE ±{err.mean():.2f}T "
          f"(bias {bias.mean():+.2f}T) ===", flush=True)

    def group(keys, label):
        g_err, g_bias = defaultdict(list), defaultdict(list)
        for i in range(n):
            g_err[keys[i]].append(err[i]); g_bias[keys[i]].append(bias[i])
        rows = [(k, len(v), float(np.mean(v)), float(np.mean(g_bias[k])))
                for k, v in g_err.items() if len(v) >= args.min_n]
        rows.sort(key=lambda r: -r[2])
        print(f"\n--- {label}（n≥{args.min_n}・MAE降順・ワースト{args.top}/ベスト{args.top}）---")
        shown = rows[:args.top] + ([("...", 0, 0.0, 0.0)] if len(rows) > 2 * args.top else []) \
            + rows[-args.top:] if len(rows) > args.top else rows
        for k, cnt, m, b in shown:
            if k == "...":
                print("   ...")
            else:
                print(f"  {k:<24} n={cnt:<6} MAE ±{m:.2f}T  bias {b:+.2f}T")
        return rows

    my = [idx2id.get(int(i), "?") for i in data["card_idx"][:, 0]]
    group(my, "自リーダー別")
    pair = [f"{idx2id.get(int(a), '?')} vs {idx2id.get(int(b), '?')}"
            for a, b in zip(data["card_idx"][:, 0], data["card_idx"][:, 1])]
    group(pair, "対面ペア別")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
