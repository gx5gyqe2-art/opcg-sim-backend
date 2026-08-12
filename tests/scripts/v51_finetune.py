"""v51: 乖離盤面教師による G14 の較正微調整（G系・2026-08-12・`lethal_teacher_gen` の対）。

50点の実現証明つき教師（確信して外した盤面・value=実測EV）へ **MSE で水準を寄せ**、
蒸留アンカー（一般盤面で G14 の予測へ引き戻す・v33 と同じ交互バッチ）で他を固定する。
順位ヒンジでなく MSE なのは、v50/v51 の欠陥が**順位でなく較正（水準）**だから。

検証（学習器内で前後測定・判定は外部）:
  - **転移**: v50 リーサル45点（学習に不使用）・特に |誤差|>0.85 の乖離8点
  - 一般60点ホールドアウト（bb1 整備・学習に不使用）
  - 一般盤面の摂動 std（アンカー633盤面）
外部ゲート: コーチゲート9点（非退行）→アリーナ（採否はユーザ判断）。

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/v51_finetune.py \\
    --teachers tests/fixtures/candidates/v51_teacher --epochs 120 --out /tmp/cand_v51
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

import rl_net as RN  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS = os.path.join(REPO, "opcg_sim", "data", "learned")
FIXT = os.path.join(REPO, "tests", "fixtures", "candidates", "v51_teacher")


def _load(globs):
    parts = {k: [] for k in ("scalars", "field", "card_idx", "value")}
    for g in globs:
        for f in sorted(glob.glob(g)):
            z = np.load(f)
            for k in parts:
                parts[k].append(z[k])
    return {k: np.concatenate(v) for k, v in parts.items()}


def _metrics(net, X):
    p = net.predict({k: X[k] for k in ("scalars", "field", "card_idx")})
    e = p - X["value"]
    return p, {"MAE": round(float(np.mean(np.abs(e))), 3),
               "r": round(float(np.corrcoef(p, X["value"])[0, 1]), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teachers", default=FIXT)
    ap.add_argument("--base", default="gen14")
    ap.add_argument("--anchor-dirs", default="/tmp/anchor_replays")
    ap.add_argument("--anchor-rows", type=int, default=600)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--anchor-scale", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    T = _load([os.path.join(args.teachers, "v51t_*.npz")])
    print(f"教師 {len(T['value'])} 行（value=実測EV）")
    L45 = _load([os.path.join(args.teachers, "lethal45_v50.npz")])
    H60 = _load([os.path.join(args.teachers, "holdout60_generic.npz")])

    net = RN.ValueNet.load(os.path.join(MODELS, f"{args.base}_value.npz"))
    base = RN.ValueNet.load(os.path.join(MODELS, f"{args.base}_value.npz"))

    # アンカー（v33 規約: 一般盤面を base 予測へ MSE で引き戻す・新特徴列はゼロのまま）
    aparts = {k: [] for k in ("scalars", "field", "card_idx")}
    for d in args.anchor_dirs.split(","):
        for f in sorted(glob.glob(os.path.join(d, "dense_*.npz"))):
            z = np.load(f)
            for k in aparts:
                aparts[k].append(z[k])
    A = {k: np.concatenate(v) for k, v in aparts.items()}
    rng = np.random.default_rng(17)
    sel = rng.permutation(len(A["scalars"]))[:args.anchor_rows]
    A = {k: A[k][sel] for k in A}
    yA = base.predict(A)
    print(f"アンカー {len(yA)} 盤面")

    # 学習前の基準
    p0_45, m0_45 = _metrics(net, L45)
    _p0_60, m0_60 = _metrics(net, H60)
    dec_mask = np.abs(p0_45 - L45["value"]) >= 0.85     # v50 の乖離族（学習外・転移の的）
    print(f"学習前: 教師MAE {_metrics(net, T)[1]['MAE']} / L45 {m0_45} / 乖離族{int(dec_mask.sum())}点"
          f" MAE {np.mean(np.abs((p0_45 - L45['value'])[dec_mask])):.3f} / H60 {m0_60}")

    nT = len(T["value"])
    for ep in range(args.epochs):
        order = rng.permutation(nT)
        for k in range(0, nT, 32):
            selb = order[k:k + 32]
            b = {kk: T[kk][selb] for kk in ("scalars", "field", "card_idx")}
            _p, cache = net.forward(b)
            net.step(net.backward(cache, T["value"][selb]), lr=args.lr)
            asel = rng.integers(0, len(yA), 192)         # 交互にアンカーで引き戻す
            ab = {kk: A[kk][asel] for kk in ("scalars", "field", "card_idx")}
            _pa, cachea = net.forward(ab)
            net.step(net.backward(cachea, yA[asel]), lr=args.lr * args.anchor_scale)
        if (ep + 1) % 30 == 0:
            print(f"  ep{ep+1}: 教師MAE {_metrics(net, T)[1]['MAE']}", flush=True)

    p1_45, m1_45 = _metrics(net, L45)
    _p1_60, m1_60 = _metrics(net, H60)
    drift = net.predict(A) - yA
    res = {"teacher_mae": _metrics(net, T)[1]["MAE"],
           "L45": {"before": m0_45, "after": m1_45},
           "deceptive8": {"n": int(dec_mask.sum()),
                          "before_mae": round(float(np.mean(np.abs((p0_45 - L45["value"])[dec_mask]))), 3),
                          "after_mae": round(float(np.mean(np.abs((p1_45 - L45["value"])[dec_mask]))), 3)},
           "H60": {"before": m0_60, "after": m1_60},
           "anchor_drift_std": round(float(drift.std()), 4)}
    os.makedirs(args.out, exist_ok=True)
    net.save(os.path.join(args.out, "value.npz"))
    import shutil
    shutil.copyfile(os.path.join(MODELS, f"{args.base}_policy.npz"),
                    os.path.join(args.out, "policy.npz"))
    with open(os.path.join(args.out, "meta_v51.json"), "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print("V51_FINETUNE_RESULT " + json.dumps(res, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    _sys.exit(main())
