"""単調再較正のフィットと検証（v47 手順2・2026-08-09）。

`value_calibration_audit.py` が見つけた水準のずれを、**順位を一切壊さない**単調写像
（`ValueNet.set_calib`）として当てる。フィットは等調回帰（PAVA）で、ビン端の恣意性を持たない。

**なぜ単調変換から試すか**: 本体の再学習は v40（ゲート満点なのにアリーナ 0.447）を
繰り返す危険があるが、単調変換は**あらゆる直接比較（箱の出口選択・枝の順位）を bit 保存**
するので、壊しうるのは「探索が値を平均する経路」だけ＝影響範囲が構造的に限定される。

**検証（--holdout）**: 対局単位で分割して当てはめの汎化を見る。同一対局の盤面は相関する
ので、盤面単位で分割すると楽観的に出る（v47 で SE のクラスタ単位を対局へ直したのと同型）。

**単調変換で消えない誤りも出す**: ライフ差など**特徴依存**のバイアスは出力の変換では消せない。
変換前後の層別バイアスを併記し、「残った分＝本体の再学習でしか直らない分」を明示する。

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/value_calib_fit.py \\
    --dirs /tmp/v47/w1,/tmp/v47/w5 --glob "vlabel_*.npz" --knots 9 --holdout 0.3 \\
    --out /tmp/gen13_calib.npz
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
from opcg_sim.src.learned import value_net as RN  # noqa: E402


def load(dirs, pat):
    cols, fs = {}, [f for d in dirs for f in sorted(glob.glob(os.path.join(d, pat)))]
    for f in fs:
        z = np.load(f)
        for k in z.files:
            cols.setdefault(k, []).append(z[k])
    if not fs:
        return None, 0
    keep = {k for k, v in cols.items() if len(v) == len(fs)}
    return {k: np.concatenate(cols[k]) for k in keep}, len(fs)


def pava(y, w=None):
    """重み付き等調回帰（Pool Adjacent Violators・pure）: y を単調増加へ射影する。"""
    y = [float(v) for v in y]
    w = [1.0] * len(y) if w is None else [float(v) for v in w]
    ys, ws, ns = [], [], []
    for yi, wi in zip(y, w):
        ys.append(yi); ws.append(wi); ns.append(1)
        while len(ys) > 1 and ys[-2] > ys[-1]:
            tw = ws[-2] + ws[-1]
            ys[-2:] = [(ys[-2] * ws[-2] + ys[-1] * ws[-1]) / tw]
            ws[-2:] = [tw]; ns[-2:] = [ns[-2] + ns[-1]]
    out = []
    for v, n in zip(ys, ns):
        out.extend([v] * n)
    return np.array(out)


def fit_knots(pred, lab, n_knots):
    """(pred, lab) から単調ノット (xs, ys) を作る。

    等調回帰した値を、pred の分位点で n_knots 本に間引く＝`np.interp` 用の区分線形。
    ノットは狭義単調増加でなければならない（`set_calib` の契約）ので、x が重複したら
    1e-6 ずつずらす。y は等調なので単調性が保証される。"""
    o = np.argsort(pred, kind="stable")
    p_s, l_s = pred[o], lab[o]
    g = pava(l_s)
    qs = np.linspace(0.0, 1.0, n_knots)
    idx = np.clip((qs * (len(p_s) - 1)).astype(int), 0, len(p_s) - 1)
    xs, ys = p_s[idx].astype(float), g[idx].astype(float)
    for i in range(1, len(xs)):                        # x の狭義単調化
        if xs[i] <= xs[i - 1]:
            xs[i] = xs[i - 1] + 1e-6
    ys = pava(ys)                                      # 間引きで崩れた単調性を復元
    return xs, ys


def report(tag, pred, lab, games):
    """バイアスと RMSE を対局クラスタ SE つきで出す。"""
    d = pred - lab
    by = {}
    for g, v in zip(games, d):
        by.setdefault(int(g), []).append(v)
    m = np.array([np.mean(v) for v in by.values()])
    se = float(m.std(ddof=1) / np.sqrt(len(m))) if len(m) > 1 else 0.0
    print(f"  {tag:<10} バイアス {m.mean():+.3f} ±{2*se:.3f}(2SE)  "
          f"RMSE {np.sqrt((d**2).mean()):.3f}  MAE {np.abs(d).mean():.3f}")
    return m.mean(), np.sqrt((d ** 2).mean())


def strata_table(tag, pred, lab, sc, games):
    """ライフ差の層別バイアス（単調変換で消えるか消えないかを見る唯一の表）。"""
    ld = sc[:, 0] - sc[:, 1]
    print(f"  {tag} — ライフ差(自-相) 層別バイアス:")
    for lo, hi in ((-9, -1.5), (-1.5, -0.5), (-0.5, 0.5), (0.5, 1.5), (1.5, 9)):
        m = (ld >= lo) & (ld < hi)
        if m.sum() < 10:
            continue
        d = pred[m] - lab[m]
        by = {}
        for g, v in zip(games[m], d):
            by.setdefault(int(g), []).append(v)
        gm = np.array([np.mean(v) for v in by.values()])
        se = float(gm.std(ddof=1) / np.sqrt(len(gm))) if len(gm) > 1 else 0.0
        print(f"    [{lo:+.1f},{hi:+.1f}) n={m.sum():4d} 対局={len(gm):3d}: "
              f"{gm.mean():+.3f} ±{2*se:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", required=True)
    ap.add_argument("--glob", default="vlabel_*.npz")
    ap.add_argument("--net", default="opcg_sim/data/learned/gen13_value.npz")
    ap.add_argument("--knots", type=int, default=9)
    ap.add_argument("--holdout", type=float, default=0.3, help="検証に回す**対局**の割合")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default="", help="較正を焼いたネットの保存先（空=保存しない）")
    args = ap.parse_args()

    col, nf = load([d for d in args.dirs.split(",") if d], args.glob)
    if col is None:
        print("コーパスが空"); return 1
    net = RN.ValueNet.load(args.net)
    net.set_calib(None, None)                          # 生の出力でフィットする
    X = {k: col[k] for k in ("scalars", "field", "card_idx")}
    pred = np.asarray(net.predict(X)).ravel()
    lab = (2.0 * np.nanmean(col["win_w"], axis=1) - 1.0 if "win_w" in col
           else np.asarray(col["value"]).ravel())
    games = col["group"] // 100
    ok = np.isfinite(lab) & np.isfinite(pred)
    pred, lab, games, sc = pred[ok], lab[ok], games[ok], col["scalars"][ok]

    ug = np.array(sorted(set(games.tolist())))
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(ug))
    n_ho = max(1, int(round(args.holdout * len(ug)))) if args.holdout > 0 else 0
    ho = set(ug[perm[:n_ho]].tolist())
    is_ho = np.array([int(g) in ho for g in games])
    print(f"{nf}シャード {len(pred)}盤面 {len(ug)}対局（学習{len(ug)-n_ho} / 検証{n_ho}対局）")

    xs, ys = fit_knots(pred[~is_ho], lab[~is_ho], args.knots)
    net.set_calib(xs, ys)                              # 単調性はここで検査される
    cal = net.apply_calib(pred)

    print("\n【学習側】")
    report("変換前", pred[~is_ho], lab[~is_ho], games[~is_ho])
    report("変換後", cal[~is_ho], lab[~is_ho], games[~is_ho])
    if n_ho:
        print("【検証側（未使用の対局）】")
        report("変換前", pred[is_ho], lab[is_ho], games[is_ho])
        report("変換後", cal[is_ho], lab[is_ho], games[is_ho])

    print("\n推定された写像 g:")
    for x, y in zip(xs, ys):
        print(f"    {x:+.3f} → {y:+.3f}  （{y - x:+.3f}）")

    print("\n単調変換で消える誤り / 消えない誤り:")
    strata_table("変換前", pred, lab, sc, games)
    strata_table("変換後", cal, lab, sc, games)

    if args.out:
        net.save(args.out)
        print(f"\n保存: {args.out}（calib を焼いたネット・胴体と全ヘッドは無改変）")
    print("VALUE_CALIB_FIT_RESULT " + json.dumps(
        {"boards": len(pred), "games": len(ug), "knots": args.knots,
         "calib_x": [round(v, 4) for v in xs], "calib_y": [round(v, 4) for v in ys]},
        ensure_ascii=False))
    return 0


if __name__ == "__main__":
    _sys.exit(main())
