"""bb1: 骨組みネットの訓練（B系 Phase 1・2026-08-12・`bb_gen` の対）。

ランダム合成世界のコーパス（bb_gen 出力・card_idx なし）で、**ID埋め込みを使わない**
value ネットを MSE 訓練する。ID排除は入力の card_idx を全 PAD(0) 固定で実現＝
ValueNet 本体は G系と同一実装（分離規約: G系モジュール無変更）。

出力: `--out/value.npz`（B系成果物・G系の genN とは別置き規約だが Phase 1 は /tmp で可）。

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/bb_train.py \\
    --dirs /tmp/bb1_corpus --epochs 12 --out /tmp/bb1_net
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

N_IDX_V9 = 24        # v9 の card_idx 枠数（2 リーダー + 場10 + 手札10 + ステージ2）＝全 PAD で埋める


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", required=True, help="bb_gen シャードのディレクトリ（カンマ区切り）")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--d-emb", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--enc-version", type=int, default=9,
                    help="符号化世代（コーパスと一致させる。bb2=10）")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    parts = {"scalars": [], "field": [], "value": []}
    for d in args.dirs.split(","):
        for f in sorted(glob.glob(os.path.join(d.strip(), "bb1_*.npz"))):
            z = np.load(f)
            for k in parts:
                parts[k].append(z[k])
    X = {k: np.concatenate(v) for k, v in parts.items()}
    n = len(X["value"])
    assert n > 0, "コーパスが空"
    X["card_idx"] = np.zeros((n, N_IDX_V9), np.int64)      # 骨組み規約: 全 PAD＝ID情報ゼロ
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(n)
    n_val = max(1, int(n * args.val_frac))
    va, tr = order[:n_val], order[n_val:]
    print(f"コーパス {n} 行（train {len(tr)} / val {len(va)}）・勝敗均衡 {X['value'].mean():+.3f}")

    from cpu_selfplay import _load_db
    from opcg_sim.src.core.cpu_learned import LearnedEngine
    vocab_size = len(LearnedEngine().vocab)                # G系と同じ語彙次元（埋め込みは死重）
    net = RN.ValueNet(vocab_size=vocab_size, d_emb=args.d_emb, hidden=args.hidden,
                      feat_dim=E.feature_dim(args.enc_version), seed=args.seed)

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
    meta = {"rows": int(n), "epochs": args.epochs, "lr": args.lr,
            "d_emb": args.d_emb, "hidden": args.hidden,
            "val_mse": round(mse(va), 4), "card_idx": "PAD固定"}
    with open(os.path.join(args.out, "meta_bb_train.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print("BB_TRAIN_DONE " + json.dumps(meta, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    _sys.exit(main())
