"""大容量 value のフルスクラッチ訓練（v13・微調整族の局所最適からの脱出）。

背景: gen6 への微調整は12回の判定で例外なく対gen6 アリーナ 0.41-0.49（policy 学習・disagree 重み・
乖離教師・エコー破り・順位ペアの全族）。一方 **ネット容量は gen2 から据え置き**（hidden=256・
パラメータ 23万）で、教師は root 16k＋子盤面 34k＝約5万サンプルまで育っている。温スタート微調整
では初期値の谷から出られないため、**容量を上げて初期値から訓練し直す**のが未検証の構造レバー。

- policy は gen6 のまま凍結（v12 の封印: policy 微調整は1エポックでも実戦を壊す）
- 符号化は現行 v5（feat_dim=135）・**vocab と効果表は gen6 から引き継ぐ**（index ズレ防止）
- 子盤面 value の併合は `--use-child`（v11 で微調整には有害だったが、初期値から学ぶ場合は
  データ量2倍の効果が上回る可能性＝A/B で判定する）

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/value_scratch_train.py \
    --hidden 512 --epochs 20 --out /tmp/scratch512
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import shutil
import time

import numpy as np

import os as _os, sys as _sys  # noqa: E402  test bootstrap
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap  # noqa: E402,F401
import rl_net as RN
import rl_encoder as E
from ref_finetune_smoke import (collect_ref_batches, split_idx, build_rank_pairs, pair_acc,
                                rank_finetune)

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS = os.path.join(REPO, "opcg_sim", "data", "learned")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=512, help="隠れ層（gen6=256）")
    ap.add_argument("--d-emb", type=int, default=24, help="カード埋め込み次元（gen6=24）")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3, help="スクラッチ訓練の学習率")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--use-child", action="store_true",
                    help="子盤面 value も訓練に併合（データ約3倍・A/B 判定用）")
    ap.add_argument("--rank-epochs", type=int, default=0, help="順位ヒンジの後段微調整（v12.1）")
    ap.add_argument("--rank-lr", type=float, default=5e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    vdata, _pol = collect_ref_batches()
    if vdata is None:
        print("教師バッチが見つからない"); return 1
    child = vdata.get("_child")
    keys = ("scalars", "field", "card_idx")
    n_root = len(vdata["value"])
    tr, va = split_idx(n_root, 0.15)
    tr_data = {k: vdata[k][tr] for k in keys}
    tr_data["value"] = vdata["value"][tr]
    va_data = {k: vdata[k][va] for k in keys}
    va_y = vdata["value"][va]
    if args.use_child and child is not None:
        tr_data = {k: np.concatenate([tr_data[k], child[k]]) for k in keys + ("value",)}
    extra = (f"＋子盤面 {len(child['value'])}"
             if args.use_child and child is not None else "")
    print(f"訓練 {len(tr_data['value'])} 行（root {len(tr)}{extra}）"
          f" / val {len(va)} 行（root のみ＝比較互換）")

    base = RN.ValueNet.load(os.path.join(MODELS, "gen6_value.npz"))
    fd = E.feature_dim(5)
    assert base.feat_dim == fd, f"符号化不一致 base={base.feat_dim} 現行v5={fd}"
    net = RN.ValueNet(vocab_size=base.Emb.shape[0] - 1, d_emb=args.d_emb, hidden=args.hidden,
                      feat_dim=fd, seed=args.seed, lead_slots=base.lead_slots,
                      eff_table=base.EffF, eff_proj=base.eff_proj or 16)
    net.vocab_ids = list(base.vocab_ids) if base.vocab_ids else None   # index ズレ防止（必須）
    n_par = sum(a.size for a in (net.Emb, net.W1, net.b1, net.W2, net.b2))
    print(f"新ネット: hidden={args.hidden} d_emb={args.d_emb} パラメータ {n_par}"
          f"（gen6=229817）")

    t0 = time.time()
    tm, vm = RN.train(net, tr_data, epochs=args.epochs, lr=args.lr, batch=args.batch,
                      val_frac=0.1, seed=args.seed)
    pred = net.predict(va_data)
    mae = float(np.abs(pred - va_y).mean())
    corr = float(np.corrcoef(pred, va_y)[0, 1])
    b_pred = base.predict(va_data)
    b_mae = float(np.abs(b_pred - va_y).mean())
    b_corr = float(np.corrcoef(b_pred, va_y)[0, 1])
    print(f"train mse {tm:.3f}→val {vm:.3f}（{time.time() - t0:.0f}s）")
    print(f"held-out root: scratch MAE={mae:.3f} corr={corr:.3f} / gen6 MAE={b_mae:.3f} "
          f"corr={b_corr:.3f}")

    if args.rank_epochs > 0 and child is not None:
        pairs = build_rank_pairs(child)
        p_tr = [p for p in pairs if p[2] % 7 != 0]
        p_va = [p for p in pairs if p[2] % 7 == 0]
        a0 = pair_acc(net, child, p_va); ab = pair_acc(base, child, p_va)
        rank_finetune(net, child, p_tr, epochs=args.rank_epochs, lr=args.rank_lr)
        a1 = pair_acc(net, child, p_va)
        print(f"順位正答(val): scratch {a0:.3f}→{a1:.3f} / gen6 {ab:.3f}")

    os.makedirs(args.out, exist_ok=True)
    net.save(os.path.join(args.out, "value.npz"))
    shutil.copy(os.path.join(MODELS, "gen6_policy.npz"),
                os.path.join(args.out, "policy.npz"))   # policy は gen6 凍結（封印）
    print(f"saved → {args.out}/value.npz（policy は gen6 コピー）")
    return 0


if __name__ == "__main__":
    _sys.exit(main())
