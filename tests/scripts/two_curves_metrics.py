#!/usr/bin/env python3
"""**T137b**（2026-09-23）: `two_curves.py`（T137a）が作る `G(t)`／`R(t)` の系列から、
**3 つの指標**を測る——**ターン増分の一致**・**局面別の距離**・**終点の比**。**型別**の内訳も添える。

## 問い

T137a は器（`G(t)`・`R(t)` の系列）だけを作った。**曲線が重なるか**は測っていない。本 T が測る。

## 式（新定数ゼロ・既存の系列を集計するだけ）

* **増分の一致**: `ΔG_j = G_j − G_{j−1}`・`ΔR_j = R_j − R_{j−1}`（`j` は自席ターン番号・0 始まり）。
  **相関**（生・`j` の影を抜いた〔`rate_tracking.demean_by`〕もの）と**回帰の傾き**（`ΔR` を `ΔG` で
  説明する最小二乗の傾き）。**予告**（測る前）: 1 手ごとの価格 対 実現の相関は既に 0.535／0.554
  （T84）と分かっている。**積んだ系列の増分はそれと同じ量**（1 ターンぶんへ粒度を変えただけ）なので、
  **相関はおおむね同じ水準になるはず**（ターンにまとめても新しい情報は増えない・下がりもしない）。
* **局面別の距離**: `|G_j − R_j|`（ズレの絶対値）を自席ターン番号 `j` で 3 帯（0-2／3-5／6+・
  `own_turn_index` の規約に合わせ 0 始まり）に分けて集計。**予告**: 終盤ほど開く
  （T107 の「`A` の後半の過大・生存者バイアス」と整合するはず）。
* **終点の比**: `ΣG / ΣR`（局×席ごと・0 割りは除く）の分布（平均・中央値・四分位）。
* **型別**（T137a で追加した `g_fam`・T81 と同じ族＝attack/play/effect/attach/other）:
  各族の **1 ターンぶんの `ΔG_fam`** と、その턴の `ΔR` との相関——**どの型の理論値が実現の損害と
  一番よく動くか**（T81 の「相関 0.36 の中身は型」の追試を、積んだ系列でやり直す）。

## 測るもの

両記録（実／合成）で上記 4 つを出す。`--dump`（T137a の出力）を読むだけで、記録を読み直さない
（`two_curves.py --dump` を先に実行しておくこと）。

使い方:

    python tests/scripts/two_curves.py --in <dir> --dump rows.json   # 先に T137a の系列を作る
    python tests/scripts/two_curves_metrics.py --dump rows.json [--json out.json]
"""

import argparse
import json
import os
import sys

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from rate_tracking import corr, demean_by  # noqa: E402

#: 自席ターン番号 `j`（0 始まり）の帯——`own_turn_index` の規約に合わせる
PHASE_BANDS = ((0, 2, "0-2"), (3, 5, "3-5"), (6, 10 ** 9, "6+"))


def phase_of(j):
    for lo, hi, name in PHASE_BANDS:
        if lo <= j <= hi:
            return name
    return PHASE_BANDS[-1][2]


def increments(seat):
    """`(turns, g, r, g_fam)` → `[(j, dG, dR, fam_dict)]`（`j` は 0 始まりの自席ターン番号）。"""
    g, r = seat["g"], seat["r"]
    fam = seat.get("g_fam") or [{}] * len(g)
    out = []
    prev_g = prev_r = 0.0
    for j, (gv, rv, fv) in enumerate(zip(g, r, fam)):
        out.append((j, gv - prev_g, rv - prev_r, fv))
        prev_g, prev_r = gv, rv
    return out


def load_dump(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def endpoint_ratios(dump):
    ratios = []
    for row in dump:
        g_end = row["g"][-1] if row["g"] else 0.0
        r_end = row["r"][-1] if row["r"] else 0.0
        if abs(r_end) > 1e-9:
            ratios.append(g_end / r_end)
    return ratios


def collect(dump):
    dg_all, dr_all, j_all = [], [], []
    dist_by_phase = {name: [] for _lo, _hi, name in PHASE_BANDS}
    fam_dg = {}          # 型 -> [(dG_fam, dR)]
    for row in dump:
        seat = {"g": row["g"], "r": row["r"], "g_fam": row.get("g_fam")}
        for j, dg, dr, fv in increments(seat):
            dg_all.append(dg); dr_all.append(dr); j_all.append(j)
            g_j = row["g"][j]; r_j = row["r"][j]
            dist_by_phase[phase_of(j)].append(abs(g_j - r_j))
            for fam, v in fv.items():
                fam_dg.setdefault(fam, []).append((v, dr))
    dg_all = np.asarray(dg_all); dr_all = np.asarray(dr_all)
    raw_corr = corr(dg_all, dr_all)
    dg_dm = demean_by(dg_all, j_all); dr_dm = demean_by(dr_all, j_all)
    dm_corr = corr(dg_dm, dr_dm)
    slope = None
    if dg_all.std() > 1e-12:
        slope = float(np.polyfit(dg_all, dr_all, 1)[0])
    ratios = np.asarray(endpoint_ratios(dump))
    fam_out = {}
    for fam, pairs in fam_dg.items():
        # **その型の行が 0 の局・席も母数に入れる**（他の型しか使わなかったターンは `ΔG_fam=0` として並べる）
        vs = np.asarray([p[0] for p in pairs]); ds = np.asarray([p[1] for p in pairs])
        fam_out[fam] = {"n": int(vs.size), "corr": corr(vs, ds)}
    return {"n_rows": len(dg_all), "n_seats": len(dump),
            "increment_corr_raw": round(raw_corr, 4) if raw_corr is not None else None,
            "increment_corr_demeaned": round(dm_corr, 4) if dm_corr is not None else None,
            "increment_slope": round(slope, 4) if slope is not None else None,
            "distance_by_phase": {name: {"n": len(vs), "mean": round(float(np.mean(vs)), 4) if vs else None}
                                  for name, vs in dist_by_phase.items()},
            "endpoint_ratio": {"n": int(ratios.size),
                               "mean": round(float(ratios.mean()), 4) if ratios.size else None,
                               "median": round(float(np.median(ratios)), 4) if ratios.size else None,
                               "p25": round(float(np.percentile(ratios, 25)), 4) if ratios.size else None,
                               "p75": round(float(np.percentile(ratios, 75)), 4) if ratios.size else None},
            "by_family": fam_out}


def build_parser():
    ap = argparse.ArgumentParser(description="2 本の曲線の 3 指標を測る（T137b）")
    ap.add_argument("--dump", required=True, help="two_curves.py --dump が書いた JSON")
    ap.add_argument("--json", default="")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    dump = load_dump(a.dump)
    out = collect(dump)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
