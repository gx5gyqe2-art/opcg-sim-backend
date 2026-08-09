"""本体 value の較正監査（v47 手順0・2026-08-09・読み取り専用）。

**問い**: 本体 value（盤面→勝率）は分布全体で正しいか。7点の検証点でなく、
ラベル済みコーパス全体で predict() とレフェリー実測を突き合わせ、
(1) 較正曲線（予測ビン→実測平均）、(2) 層別バイアス（ライフ差/手札差/自ライフ/ターン）を出す。

- ラベルは `--label auto`: `win_w`（v44+ の per-world 生結果）があれば純粋勝率 z を再計算、
  無ければ `value` 列（margin_blend 済み＝ライフ差タイブレーク w=0.25 込み）。
  **blend はライフ差バイアスを減衰させる向きに働く**（劣勢盤面のラベルを下げ優勢盤面を上げる）
  ため、blend 込みで出た単調バイアスの真の値は表示より大きい（保守側の測定）。
- 群内相関（同一決定点の兄弟盤面）があるため、バイアスの SE は group 単位のクラスタで計算。

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/value_calibration_audit.py \
    --name "戦闘出口w4" --dirs /tmp/defcf_v35_all
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import glob
import numpy as np
import os as _os, sys as _sys  # noqa: E402
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: E402,F401
from opcg_sim.src.learned import value_net as RN  # noqa: E402


def load(dirs, glob_pat="*.npz", max_shards=None):
    cols = {}
    fs = []
    for d in dirs:
        fs += sorted(glob.glob(os.path.join(d, glob_pat)))
    if max_shards:
        fs = fs[:max_shards]
    for f in fs:
        z = np.load(f)
        for k in z.files:
            cols.setdefault(k, []).append(z[k])
    if not cols:
        return None
    n0 = {k: len(v) for k, v in cols.items()}
    keep = {k for k, c in n0.items() if c == len(fs)}   # 全シャード共通の列のみ
    return {k: np.concatenate(cols[k]) for k in keep}

def audit(name, col, label, note=""):
    X = {k: col[k] for k in ("scalars", "field", "card_idx")}
    pred = np.asarray(net.predict(X)).ravel()
    lab = np.asarray(label).ravel()
    ok = np.isfinite(lab)
    pred, lab = pred[ok], lab[ok]
    grp = col["group"][ok] if "group" in col else np.arange(len(pred))
    sc = col["scalars"][ok]
    diff = pred - lab
    def cse(mask):  # クラスタ（group）単位の bias と SE
        if not mask.any():
            return None
        gs = {}
        for g, d in zip(grp[mask], diff[mask]):
            gs.setdefault(int(g), []).append(d)
        m = np.array([np.mean(v) for v in gs.values()])
        return float(m.mean()), float(m.std(ddof=1) / max(np.sqrt(len(m)), 1)) if len(m) > 1 else 0.0, int(mask.sum()), len(m)
    print(f"\n=== {name}  n={len(pred)} 群={len(set(grp.tolist()))} {note}")
    r = float(np.corrcoef(pred, lab)[0, 1])
    b, se, _, ng = cse(np.ones(len(pred), bool))
    print(f"  相関 r={r:.3f}  全体バイアス {b:+.3f} ±{2*se:.3f}(2SE)  MAE={np.abs(diff).mean():.3f}")
    print("  較正曲線（予測ビン → 実測平均）:")
    edges = [-1.5, -0.6, -0.3, -0.1, 0.1, 0.3, 0.6, 1.5]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (pred >= lo) & (pred < hi)
        if m.sum() < 8:
            continue
        st = cse(m)
        print(f"    pred[{lo:+.1f},{hi:+.1f}) n={m.sum():4d}: 予測平均 {pred[m].mean():+.3f} 実測平均 {lab[m].mean():+.3f}  バイアス {st[0]:+.3f} ±{2*st[1]:.3f}")
    strata = [
        ("ライフ差(自-相)", sc[:, 0] - sc[:, 1], [(-9, -1.5), (-1.5, -0.5), (-0.5, 0.5), (0.5, 1.5), (1.5, 9)]),
        ("手札差(自-相)",   sc[:, 6] - sc[:, 7], [(-9, -1.5), (-1.5, -0.5), (-0.5, 0.5), (0.5, 1.5), (1.5, 9)]),
        ("自ライフ",        sc[:, 0],            [(-1, 1.5), (1.5, 2.5), (2.5, 3.5), (3.5, 9)]),
        ("ターン数",        sc[:, 10],           [(0, 6.5), (6.5, 10.5), (10.5, 40)]),
    ]
    for sname, vals, bins in strata:
        print(f"  層別バイアス — {sname}:")
        for lo, hi in bins:
            m = (vals >= lo) & (vals < hi)
            if m.sum() < 10:
                continue
            st = cse(m)
            flag = " ←" if abs(st[0]) > 2 * st[1] and abs(st[0]) > 0.05 else ""
            print(f"    [{lo:+.1f},{hi:+.1f}) n={st[2]:4d} 群={st[3]:3d}: バイアス {st[0]:+.3f} ±{2*st[1]:.3f}{flag}")



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="表示名（分布の説明）")
    ap.add_argument("--dirs", required=True, help="コーパスのディレクトリ（カンマ区切り）")
    ap.add_argument("--glob", default="*.npz")
    ap.add_argument("--net", default="opcg_sim/data/learned/gen13_value.npz")
    ap.add_argument("--label", default="auto", choices=("auto", "winw", "value"))
    args = ap.parse_args()
    global net
    net = RN.ValueNet.load(args.net)
    col = load([d for d in args.dirs.split(",") if d], args.glob)
    if col is None:
        print("コーパスが空"); return 1
    if args.label != "value" and "win_w" in col:
        lab = 2.0 * np.nanmean(col["win_w"], axis=1) - 1.0
        note = f"（win_w 純粋勝率・worlds={col['win_w'].shape[1]}）"
    elif args.label == "winw":
        print("win_w 列が無い"); return 1
    else:
        lab = col["value"]
        note = "（value=margin_blend 混合ラベル＝ライフ差バイアスは保守側に出る）"
    audit(args.name, col, lab, note)
    print("VALUE_CALIB_AUDIT_DONE")
    return 0


if __name__ == "__main__":
    _sys.exit(main())
