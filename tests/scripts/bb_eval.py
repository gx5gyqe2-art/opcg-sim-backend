"""bb1: 骨組みネットの実盤面ホールドアウト評価（B系 Phase 1 の Go/No-Go・2026-08-12）。

**問い**（docs/cpu_backbone_plan.md Phase 1）: ランダム合成世界だけで訓練した
**ID埋め込みなし**の骨組みネットは、**実カードの実盤面**の価値をどこまで説明できるか。

正解基準＝レフェリー勝率ラベル（`lethal_calibration_probe --out` の npz・教師正本
sims48×6世界）。**gen14 との一致は判定に使わない**（固有性監査 #2＝偏った物差しの輸入）
——gen14 の同ラベルへの較正は**参考線**として並記する。

指標: MAE / RMSE / Pearson r / 符号一致率。骨組みは card_idx を PAD に潰して評価
（訓練時と同じ規約）。参考線 gen14 は通常符号化（実ID込み）で評価＝それぞれの土俵。

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/bb_eval.py \\
    --net /tmp/bb1_net/value.npz --holdout /tmp/bb1_holdout
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


def metrics(pred, y):
    pred, y = np.asarray(pred, float), np.asarray(y, float)
    mae = float(np.mean(np.abs(pred - y)))
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    r = float(np.corrcoef(pred, y)[0, 1]) if len(y) > 2 else float("nan")
    sign = float(np.mean(np.sign(pred) == np.sign(y)))
    return {"MAE": round(mae, 3), "RMSE": round(rmse, 3),
            "r": round(r, 3), "符号一致": round(sign, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True, help="骨組み value.npz（bb_train 出力）")
    ap.add_argument("--holdout", required=True,
                    help="実盤面ラベル npz のディレクトリ（lethal_calibration_probe --out）")
    args = ap.parse_args()

    parts = {"scalars": [], "field": [], "card_idx": [], "value": []}
    for f in sorted(glob.glob(os.path.join(args.holdout, "*.npz"))):
        z = np.load(f)
        for k in parts:
            parts[k].append(z[k])
    X = {k: np.concatenate(v) for k, v in parts.items()}
    y = X["value"]
    n = len(y)
    print(f"ホールドアウト {n} 実盤面（レフェリー勝率ラベル）")

    bb = RN.ValueNet.load(args.net)
    pad_idx = np.zeros_like(X["card_idx"])
    p_bb = bb.predict({"scalars": X["scalars"], "field": X["field"], "card_idx": pad_idx})

    from opcg_sim.src.core.cpu_learned import LearnedEngine
    import rl_encoder as E
    g = LearnedEngine().vnet                       # 参考線＝出荷既定（G14）・実ID込み
    # ホールドアウトが v10 行（73列）でも G14 は v9 接頭辞（append-only 契約）で評価できる
    Xg = dict(X, scalars=X["scalars"][:, :E.scalars_dim(9)])
    p_g = g.predict({k: Xg[k] for k in ("scalars", "field", "card_idx")})

    base = np.zeros(n)                             # 無情報基準（常に0）
    print(f"\n  {'指標':<8} {'骨組み(ID無)':>14} {'G14参考線':>12} {'常に0':>8}")
    m_bb, m_g, m_0 = metrics(p_bb, y), metrics(p_g, y), metrics(base, y)
    for k in ("MAE", "RMSE", "r", "符号一致"):
        print(f"  {k:<8} {m_bb[k]:>14} {m_g[k]:>12} {m_0[k]:>8}")
    print("\n  判定の読み方: 骨組みが G14 参考線に迫る/超えるなら「物理言語で実盤面の価値は"
          "説明できる」＝Phase 1 GO。常に0 と同等なら情報を学べていない＝No-Go。")
    print("BB_EVAL " + json.dumps({"n": n, "bb": m_bb, "g14": m_g, "zero": m_0},
                                  ensure_ascii=False))
    return 0


if __name__ == "__main__":
    _sys.exit(main())
