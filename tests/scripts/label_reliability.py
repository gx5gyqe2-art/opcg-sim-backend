"""教師ラベルの信頼度測定（v44・2026-08-08）: そのラベルは信号か、引きの当たり外れか。

`plan_cf_gen.py --split-halves` 系のシャードは、**同じ決定点のラベルを独立2組**
（world を偶奇で分割）で持つ（`value_a` / `value_b`）。本スクリプトはその2組が
**順位について一致するか**を測る。

なぜ要るか（v41/v43 の教訓）: コーパスを作る→学習する→ゲート→アリーナ、は1周に
1〜2日かかる。v41 は「学習しても直らない」で1周、v42 は「アリーナで有意退行」で
568局ぶんを溶かした。どちらも**教師の信頼度を先に測っていれば着手前に分かった**
（v41 実測: 因果 z の粒度は worlds=4 で 0.5、対して教えたい枝間マージンは 0.02〜0.03）。
本スクリプトは**本生成に着手する前の足切り**であり、worlds を増やす投資判断の根拠。

出す数字:
  - **半々一致率**: 全ペアについて、A半分とB半分が同じ向きの順位を出す割合。
    0.5 = 完全なコインフリップ（ラベルはノイズのみ）／1.0 = 完全に再現。
  - **δ選抜後の一致率**: 「A半分で |Δz|>δ と判定したペア」をB半分が追認する割合。
    学習は δ で選抜したペアだけを使うので、**これが実際に教わる順位の正答率**。
    勝者の呪い（引きが偏ったペアほど選ばれる）があると、選抜後の方が下がる。
  - **推定ノイズ幅**: 半分同士の z 差の標準偏差から、そのラベルの1σを推定する。
    これが枝間マージン（gen12 実測 0.02〜0.03）と比べてどれくらいかが本質。

分割半は **worlds/2 ぶんの予算**で測っている点に注意（worlds=32 の走行なら、
出てくる一致率は「worlds=16 のラベルの信頼度」）。本番ラベルは全世界を使うので
実際の信頼度はこれより少し高い＝**本スクリプトの数字は保守側**。

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/label_reliability.py \\
    --dirs /tmp/plancf_w8,/tmp/plancf_w32 --labels w8,w32 --delta 0.25
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
import _bootstrap  # noqa: E402,F401


def load(dirs, pattern="plancf_*.npz"):
    """value/value_a/value_b/group を連結する（分割半の無いシャードは除外して報告）。"""
    cols = {k: [] for k in ("value", "value_a", "value_b", "group")}
    n_files = n_skipped = 0
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, pattern))):
            z = np.load(f)
            if "value_a" not in z.files:
                n_skipped += 1
                continue
            for k in cols:
                cols[k].append(z[k])
            n_files += 1
    if not n_files:
        return None, 0, n_skipped
    return {k: np.concatenate(v) for k, v in cols.items()}, n_files, n_skipped


def pairs_of(group):
    """同一 group 内の全 2 組み合わせ（index ペア）を列挙する（pure）。"""
    import collections
    by = collections.defaultdict(list)
    for i, g in enumerate(group):
        by[int(g)].append(i)
    out = []
    for idxs in by.values():
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                out.append((idxs[a], idxs[b]))
    return out


def reliability(col, delta):
    """(半々一致率, δ選抜後の一致率, 選抜数, 推定ノイズ1σ) を返す（pure）。"""
    va, vb, z = col["value_a"], col["value_b"], col["value"]
    ps = [(a, b) for a, b in pairs_of(col["group"])
          if np.isfinite(va[a]) and np.isfinite(va[b])
          and np.isfinite(vb[a]) and np.isfinite(vb[b])]
    if not ps:
        return float("nan"), float("nan"), 0, float("nan"), 0
    da = np.array([va[a] - va[b] for a, b in ps])
    db = np.array([vb[a] - vb[b] for a, b in ps])
    # 同値ペア（片方が引き分け）は一致判定から外す＝順位を教えられないペアなので
    live = (da != 0) & (db != 0)
    agree = float((np.sign(da[live]) == np.sign(db[live])).mean()) if live.any() else float("nan")
    sel = live & (np.abs(da) > delta)             # A半分で δ 選抜（学習が実際に使う条件）
    agree_sel = float((np.sign(da[sel]) == np.sign(db[sel])).mean()) if sel.any() else float("nan")
    # 独立2回の差 (da-db) の分散は 2σ²（σ＝半分ラベルの1σ）
    sigma = float(np.std(da - db) / np.sqrt(2.0))
    return agree, agree_sel, int(sel.sum()), sigma, len(ps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", required=True, help="コーパスのディレクトリ（カンマ区切り・腕ごと）")
    ap.add_argument("--labels", default="", help="表示名（カンマ区切り・--dirs と同数）")
    ap.add_argument("--glob", default="plancf_*.npz")
    ap.add_argument("--delta", type=float, default=0.25, help="順位ペアに採る z 差の下限")
    ap.add_argument("--margin", type=float, default=0.03,
                    help="比較対象の枝間マージン（gen12 実測 0.02〜0.03）")
    args = ap.parse_args()

    dirs = [d for d in args.dirs.split(",") if d]
    labels = (args.labels.split(",") + [""] * len(dirs))[:len(dirs)]
    res = []
    for d, lab in zip(dirs, labels):
        col, n_files, n_skipped = load([d], args.glob)
        name = lab or d
        if col is None:
            print(f"{name}: 分割半つきシャードが無い（--split-halves で生成したものが要る・"
                  f"skip {n_skipped}）", flush=True)
            continue
        agree, agree_sel, n_sel, sigma, n_pairs = reliability(col, args.delta)
        groups = len(set(col["group"].tolist()))
        print(f"\n=== {name} ===", flush=True)
        print(f"  {n_files}シャード {len(col['value'])}盤面 {groups}群 ペア{n_pairs}", flush=True)
        print(f"  半々一致率            {agree:.3f}   （0.5=コインフリップ / 1.0=完全再現）",
              flush=True)
        print(f"  δ={args.delta} 選抜後の一致率  {agree_sel:.3f}   （選抜 {n_sel}ペア＝"
              f"学習が実際に教わる順位の正答率）", flush=True)
        print(f"  推定ラベル1σ          {sigma:.3f}   （枝間マージン {args.margin} の "
              f"{sigma / args.margin:.1f}倍）", flush=True)
        res.append({"name": name, "files": n_files, "boards": int(len(col["value"])),
                    "groups": groups, "pairs": n_pairs, "agree": round(agree, 4),
                    "agree_selected": round(agree_sel, 4), "n_selected": n_sel,
                    "sigma": round(sigma, 4), "sigma_over_margin": round(sigma / args.margin, 2)})
    if res:
        print(f"\nLABEL_RELIABILITY_RESULT {json.dumps(res, ensure_ascii=False)}", flush=True)
    return 0


if __name__ == "__main__":
    _sys.exit(main())
