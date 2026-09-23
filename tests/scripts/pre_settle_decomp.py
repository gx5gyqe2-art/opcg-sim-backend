#!/usr/bin/env python3
"""**決着前の較正が「悪化した」のは何のせいか**（T145・2026-09-23・ユーザ指示「その 3 つを順に閉じてください」の 3 つ目）。

## 問い

T138b は `win_calib --pre-settle on` で決着後の行を除くと、対数損失が 0.68→0.72（実）／0.78→0.82（合成）に
**悪化**し、4 条件中 3 条件でコインに負けると報告した。だが `win_calib` の予測 `W(D)` は**行ごとに決まり、
母数に依らない**（`σ_D` は固定・当てはめない）。だから `on` の数字は「**同じ予測を、減らした行の上で採点した**」
だけのはずで、予測そのものは 1 ビットも変わっていない。

## 式（新定数ゼロ）

行の集合 `全 = 残 ∪ 除`（`除` = `lethal_rule.settled_map` が `True` の行）。対数損失は行の平均なので

    LL(全) = ( n_残 · LL(残) + n_除 · LL(除) ) / n_全          （恒等式）

**`LL(残) > LL(全)` は `LL(除) < LL(全)`（除いた行が当てやすかった）と同じこと**——予測の劣化ではなく**構成の変化**。

## 検算の予告

1. `残` の行数と予測は `--pre-settle on` の `win_calib` と**完全に一致**する（同じ行・同じ予測）。
2. 上の恒等式が数値で成り立つ（丸め誤差まで）。
3. `除` の行は**ほとんど勝者の行**で、予測勝率が高く、対数損失は `全` より**ずっと小さい**。
4. 上位分位の過信の「拡大」も同じ構成の効果——`除` を抜くと最上位分位に**まだ決まっていない行**が繰り上がる。
5. **出荷の読み（`rel`）でも同じ構成の効果が出る**が、`rel` は `abs` より残った行でもコインに近いか勝つ
   （T118 が `rel` を既定にした理由＝長い時計の行の自信過剰を比で平らにする）。T138b は `abs` だけを引いていた。

使い方:

    python tests/scripts/pre_settle_decomp.py --in <records_dir> [<records_dir> ...] [--games N] [--json out.json]
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

import crossing_bridge as CB  # noqa: E402
import theory_order as TO  # noqa: E402
import win_calib as WC  # noqa: E402
from theory_order import MU, THETA  # noqa: E402


def split_rows(rows_out, settled):
    """`rows_out`（`crossing_bridge.collect` の行・`seed`／`who`／`t` を持つ）を `(残, 除)` に分ける。
    **`crossing_bridge` の `--pre-settle on` と同じ判定**（`settled.get((seed, w, t))`）。"""
    kept, removed = [], []
    for r in rows_out:
        (removed if settled.get((int(r["seed"]), int(r["who"]), int(r["t"]))) else kept).append(r)
    return kept, removed


def mix_logloss(parts):
    """**恒等式の右辺**: `[(n, LL), ...]` の行数重みの平均。"""
    n = sum(p[0] for p in parts)
    return sum(p[0] * p[1] for p in parts) / max(1, n)


def logloss_raw(p, z):
    """丸めない対数損失（恒等式の検算用・`win_calib.score` と同じ刻み）。"""
    p = np.clip(np.asarray(p, float), WC.EPS, 1.0 - WC.EPS); z = np.asarray(z, float)
    if not len(p):
        return 0.0
    return float(-(z * np.log(p) + (1 - z) * np.log(1 - p)).mean())


def score_set(rows, slope="theory", nbin=10, w_err="abs", sigma_rel=None):
    """行の集合を `win_calib` と同じ読みで採点する（`w_err`＝`abs`｜`rel`・`rel` は `sigma_rel` が要る）。
    空なら `None`。"""
    if not rows:
        return None, [], []
    rs = WC.rows_of(rows, slope)
    z = [r[3] for r in rs]
    old = TO.W_ERR_MODE
    try:
        TO.set_w_err_mode(w_err)
        p = WC.probs_of(rs, sigma_rel if w_err == "rel" else None)
    finally:
        TO.set_w_err_mode(old)
    sc = WC.score(p, z, nbin) if len(set(z)) > 1 else {"n": len(z), "base_win_rate": float(np.mean(z))}
    return sc, p, z


def decompose(rows_out, settled, slope="theory", nbin=10, w_err="abs", sigma_rel=None):
    """**3 つの集合（全・残・除）を同じ予測で採点**し、恒等式と除いた行の中身を並べる。"""
    kept, removed = split_rows(rows_out, settled)
    kw = {"w_err": w_err, "sigma_rel": sigma_rel}
    s_all, p_all, z_all = score_set(rows_out, slope, nbin, **kw)
    s_kept, p_kept, z_kept = score_set(kept, slope, nbin, **kw)
    s_rem, p_rem, z_rem = score_set(removed, slope, nbin, **kw)
    ll_all = logloss_raw(p_all, z_all)
    ll_kept = logloss_raw(p_kept, z_kept)
    ll_rem = logloss_raw(p_rem, z_rem)
    mix = mix_logloss([(len(p_kept), ll_kept), (len(p_rem), ll_rem)])
    rem_info = {"n": len(removed),
                "winner_share": round(float(np.mean(z_rem)), 4) if z_rem else None,
                "p_mean": round(float(np.mean(p_rem)), 4) if p_rem else None,
                "logloss": round(ll_rem, 4)}
    # **最上位分位に誰が居たか**: `全` の最上位分位のうち、除かれた行の割合
    top = {}
    if p_all:
        idx = np.argsort(np.asarray(p_all, float), kind="mergesort")
        cut = int(len(idx) * (nbin - 1) / nbin)
        top_idx = set(int(i) for i in idx[cut:])
        keys_rem = set((int(r["seed"]), int(r["who"]), int(r["t"])) for r in removed)
        n_top_rem = sum(1 for i in top_idx
                        if (int(rows_out[i]["seed"]), int(rows_out[i]["who"]), int(rows_out[i]["t"])) in keys_rem)
        top = {"top_bin_n": len(top_idx), "top_bin_removed": n_top_rem,
               "top_bin_removed_share": round(n_top_rem / max(1, len(top_idx)), 4)}
    return {"w_err": w_err, "all": s_all, "kept": s_kept, "removed": rem_info,
            "identity": {"logloss_all": round(ll_all, 6), "logloss_mix": round(mix, 6),
                         "abs_err": float(abs(ll_all - mix))},
            "top_bin": top}


def collect(dirs, limit_games=0, slope="theory", nbin=10, sigma_rel=None):
    """記録を 1 度だけ読み（`--pre-settle off`）、決着の旗で分けて採点する。

    **読みは 2 つとも出す**——`abs`（T138b の報告が引いた読み）と `rel`（**出荷の既定**
    `theory_order.W_ERR_MODE`・T118）。`sigma_rel` が引けなければ `rel` は出さない。"""
    import lethal_rule as LR
    old = CB.PRE_SETTLE_MODE
    try:
        CB.set_pre_settle_mode("off")
        rows_out, _ledger, stats, _th, _tc = CB.collect(dirs, limit_games, THETA, MU, "const")
    finally:
        CB.set_pre_settle_mode(old)
    settled = LR.settled_map(dirs, limit_games)
    if sigma_rel is None:
        sigma_rel = CB.sigma_rel_for(dirs)
    out = {"games": stats.get("games"), "w_err_default": TO.W_ERR_MODE,
           "abs": decompose(rows_out, settled, slope, nbin, "abs")}
    if sigma_rel is not None:
        out["rel"] = decompose(rows_out, settled, slope, nbin, "rel", sigma_rel)
        out["sigma_rel"] = round(float(sigma_rel), 4)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="決着前の較正の悪化を構成と予測に分ける（T145）")
    ap.add_argument("--in", dest="src", nargs="+", required=True)
    ap.add_argument("--games", type=int, default=0)
    ap.add_argument("--slope", default="theory")
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)
    out = collect(a.src, a.games, a.slope, a.bins)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
