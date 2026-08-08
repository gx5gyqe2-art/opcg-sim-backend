"""教師ラベルの信頼度測定（v44・2026-08-08）: そのラベルは信号か、引きの当たり外れか。

`plan_cf_gen` のシャードは**世界ごとの生の結果**（`win_w` / `life_w`）を持つ。本スクリプトは
そこから任意の予算 K（≤ 生成時の worlds）の z を再計算し、**信頼度を K の関数として**出す。
コーパスを作り直さずに「worlds をいくつにすべきか」に答えるのが目的。

なぜ要るか（v41/v42 の教訓）: コーパスを作る→学習する→ゲート→アリーナ、は1周に1〜2日かかる。
v41 は「学習しても直らない」で1周、v42 は「アリーナで有意退行」で568局ぶんを溶かした。どちらも
**教師の信頼度を先に測っていれば着手前に分かった**（v41 実測: 因果 z の粒度は worlds=4 で 0.5、
対して教えたい枝間マージンは 0.02〜0.03）。生成コストのほぼ全部は「worlds × プラン数 ×
終局までのロールアウト」（実測 245秒/窓 @worlds8）なので、**世界を使い回して予算を変える**のが
測定の唯一の安価な方法になる。

測り方（各 K について）: 世界を無作為に2つの互いに素な K/2 組へ分け、それぞれ独立にラベルを作る。
  - **半々一致率**: 全ペアで2組が同じ向きの順位を出す割合。0.5＝コインフリップ（ラベルはノイズのみ）。
  - **δ選抜後の一致率**: 「片方で |Δz|>δ と判定したペア」をもう片方が追認する割合。学習は δ で
    選抜したペアだけを使うので、**これが実際に教わる順位の正答率**。真の差が小さい決定点では
    |Δz|>δ が立つのは引きが偏ったときだけなので、平均への回帰でここが 0.5 を割ることがある
    （＝逆向きの教師が過半。v44 実測で確認）。
  - **推定ラベル1σ**: 2組の z 差の標準偏差 /√2。枝間マージンとの比が本質。
分割は `--repeats` 回繰り返して平均する（分け方の当たり外れを均す）。

**K は「半分ずつ」で測る点に注意**: K=16 の行は 8世界ラベル同士の比較なので、本番（16世界を
全部使う）より**保守側**の数字になる。予算の当たりを見るには十分。

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/label_reliability.py \\
    --dirs /tmp/plancf_w32 --budgets 4,8,16,32 --delta 0.25
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

from plan_cf_gen import causal_z, margin_blend  # ラベル式は生成器と同一の正を使う  # noqa: E402


def load(dirs, pattern="plancf_*.npz"):
    """win_w / life_w / group を連結する（per-world 列の無い旧シャードは除外して報告）。"""
    cols = {k: [] for k in ("win_w", "life_w", "group")}
    n_files = n_skipped = 0
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, pattern))):
            z = np.load(f)
            if "win_w" not in z.files:
                n_skipped += 1
                continue
            for k in cols:
                cols[k].append(z[k])
            n_files += 1
    if not n_files:
        return None, 0, n_skipped
    return {k: np.concatenate(v) for k, v in cols.items()}, n_files, n_skipped


def z_from(win_w, life_w, sel):
    """世界の部分集合 `sel` からラベル z を作る（生成器と同一式・欠測は除外）。"""
    w, l = win_w[sel], life_w[sel]
    ok = np.isfinite(w)
    if not ok.any():
        return np.nan
    ld = l[ok & np.isfinite(l)]
    return margin_blend(causal_z(float(w[ok].sum()), int(ok.sum())),
                        float(ld.mean()) if ld.size else None)


def pairs_of(group):
    """同一 group 内の全 2 組み合わせ（pure）。"""
    import collections
    by = collections.defaultdict(list)
    for i, g in enumerate(group):
        by[int(g)].append(i)
    return [(idxs[a], idxs[b]) for idxs in by.values()
            for a in range(len(idxs)) for b in range(a + 1, len(idxs))]


def measure(col, budget, delta, repeats, rng):
    """予算 `budget` での (半々一致率, δ選抜後一致率, 選抜数, 1σ) を repeats 回平均で返す。"""
    win, life, ps = col["win_w"], col["life_w"], pairs_of(col["group"])
    n_worlds = win.shape[1]
    half = budget // 2
    ag, ag_sel, n_sel, sig = [], [], [], []
    for _ in range(repeats):
        perm = rng.permutation(n_worlds)
        sa, sb = perm[:half], perm[half:2 * half]
        za = np.array([z_from(win[i], life[i], sa) for i in range(len(win))])
        zb = np.array([z_from(win[i], life[i], sb) for i in range(len(win))])
        da = np.array([za[a] - za[b] for a, b in ps])
        db = np.array([zb[a] - zb[b] for a, b in ps])
        good = np.isfinite(da) & np.isfinite(db)
        live = good & (da != 0) & (db != 0)       # 同値ペアは順位を教えられないので除外
        if live.any():
            ag.append(float((np.sign(da[live]) == np.sign(db[live])).mean()))
        sel = live & (np.abs(da) > delta)
        if sel.any():
            ag_sel.append(float((np.sign(da[sel]) == np.sign(db[sel])).mean()))
        n_sel.append(int(sel.sum()))
        if good.any():
            sig.append(float(np.std(da[good] - db[good]) / np.sqrt(2.0)))
    m = lambda v: float(np.mean(v)) if v else float("nan")   # noqa: E731
    return m(ag), m(ag_sel), m(n_sel), m(sig), len(ps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", required=True, help="コーパスのディレクトリ（カンマ区切り）")
    ap.add_argument("--glob", default="plancf_*.npz")
    ap.add_argument("--budgets", default="", help="測る worlds 予算（カンマ区切り・既定＝2の冪で全域）")
    ap.add_argument("--delta", type=float, default=0.25, help="順位ペアに採る z 差の下限")
    ap.add_argument("--margin", type=float, default=0.03,
                    help="比較対象の枝間マージン（gen12 実測 0.02〜0.03）")
    ap.add_argument("--repeats", type=int, default=8, help="世界の分け方を変えて平均する回数")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    col, n_files, n_skipped = load([d for d in args.dirs.split(",") if d], args.glob)
    if col is None:
        print(f"per-world 列（win_w）を持つシャードが無い（v44 以降の生成器で作り直す・"
              f"skip {n_skipped}）")
        return 1
    n_worlds = col["win_w"].shape[1]
    groups = len(set(col["group"].tolist()))
    budgets = ([int(b) for b in args.budgets.split(",") if b] if args.budgets
               else [b for b in (4, 8, 16, 32, 64) if b <= n_worlds])
    budgets = [b for b in budgets if 2 <= b <= n_worlds]
    print(f"{n_files}シャード {len(col['win_w'])}盤面 {groups}群 worlds={n_worlds}"
          f"（skip {n_skipped}）・δ={args.delta}・分け方{args.repeats}回平均", flush=True)
    print(f"{'予算K':>6} {'半々一致率':>10} {'δ選抜後':>10} {'選抜数':>7} "
          f"{'ラベル1σ':>10} {'σ/マージン':>10}", flush=True)
    res = []
    rng = np.random.default_rng(args.seed)
    for b in budgets:
        ag, ag_sel, n_sel, sig, n_pairs = measure(col, b, args.delta, args.repeats, rng)
        print(f"{b:>6} {ag:>10.3f} {ag_sel:>10.3f} {n_sel:>7.1f} {sig:>10.3f} "
              f"{sig / args.margin:>10.1f}", flush=True)
        res.append({"budget": b, "agree": round(ag, 4), "agree_selected": round(ag_sel, 4),
                    "n_selected": round(n_sel, 1), "sigma": round(sig, 4),
                    "sigma_over_margin": round(sig / args.margin, 2), "pairs": n_pairs})
    print(f"\nLABEL_RELIABILITY_RESULT {json.dumps({'groups': groups, 'worlds': n_worlds, 'rows': res}, ensure_ascii=False)}",
          flush=True)
    return 0


if __name__ == "__main__":
    _sys.exit(main())
