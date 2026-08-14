"""g15: G系 v11 スパイクの訓練（2026-08-14・`g15_gen` の対）。

bb_train と同型の MSE 訓練だが2点が違う:
  - **card_idx は実 ID のまま**（G系の本領＝埋め込みを使う。bb の PAD 固定と正反対）
  - `--scalar-cols` で scalars を接頭辞切り出しできる＝**同一コーパスから A/B 両腕**
    （v11 腕=97列そのまま・v10 腕=73列切り出し。対局・行・分割は完全同一）

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/g15_train.py \\
    --dirs g15_corpus/part1,... --scalar-cols 97 --epochs 12 --out /tmp/g15_v11
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import glob
import json

import numpy as np

import os as _os, sys as _sys  # noqa: E402
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: E402,F401

import rl_encoder as E  # noqa: E402
import rl_net as RN  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", required=True, help="g15_gen シャードのディレクトリ（カンマ区切り）")
    ap.add_argument("--scalar-cols", type=int, required=True,
                    help="scalars の接頭辞切り出し列数（97=v11腕・73=v10腕）")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--d-emb", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    parts = {"scalars": [], "field": [], "card_idx": [], "value": []}
    for d in args.dirs.split(","):
        for f in sorted(glob.glob(os.path.join(d.strip(), "g15_*.npz"))):
            z = np.load(f)
            for k in parts:
                parts[k].append(z[k])
    X = {k: np.concatenate(v) for k, v in parts.items()}
    n = len(X["value"])
    assert n > 0, "コーパスが空"
    X["scalars"] = X["scalars"][:, :args.scalar_cols]
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(n)
    n_val = max(1, int(n * args.val_frac))
    va, tr = order[:n_val], order[n_val:]
    print(f"コーパス {n} 行（train {len(tr)} / val {len(va)}）"
          f"・scalars {args.scalar_cols}列・勝敗均衡 {X['value'].mean():+.3f}")

    from opcg_sim.src.core.cpu_learned import LearnedEngine
    vocab_size = len(LearnedEngine().vocab)
    ver = {E.scalars_dim(v): v for v in E.known_versions()}[args.scalar_cols]
    net = RN.ValueNet(vocab_size=vocab_size, d_emb=args.d_emb, hidden=args.hidden,
                      feat_dim=E.feature_dim(ver), seed=args.seed)

    def mse(idx):
        s = 0.0
        for k in range(0, len(idx), 4096):
            sel = idx[k:k + 4096]
            b = {kk: X[kk][sel] for kk in ("scalars", "field", "card_idx")}
            p = net.predict(b)
            s += float(((p - X["value"][sel]) ** 2).sum())
        return s / len(idx)

    print(f"学習前: train {mse(tr):.4f} / val {mse(va):.4f}")
    for ep in range(args.epochs):
        rng.shuffle(tr)
        for k in range(0, len(tr), args.batch):
            sel = tr[k:k + args.batch]
            b = {kk: X[kk][sel] for kk in ("scalars", "field", "card_idx")}
            _pred, cache = net.forward(b)
            net.step(net.backward(cache, X["value"][sel]), lr=args.lr)
        print(f"  ep{ep + 1}: train {mse(tr):.4f} / val {mse(va):.4f}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    net.save(os.path.join(args.out, "value.npz"))
    meta = {"rows": int(n), "epochs": args.epochs, "scalar_cols": args.scalar_cols,
            "d_emb": args.d_emb, "hidden": args.hidden,
            "val_mse": round(mse(va), 4), "card_idx": "実ID（G系）"}
    with open(os.path.join(args.out, "meta_g15_train.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print("G15_TRAIN_DONE " + json.dumps(meta, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    _sys.exit(main())
