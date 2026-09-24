#!/usr/bin/env python3
"""**勝率の読みの較正を測る**（T118・2026-09-19・ユーザ指示「それらの全課題についての修正、検証をお願いします」）。

## 何を測るか

交点の橋は `D = τ_opp − τ_me` の**符号**で勝者を当てる（的中 0.663／0.636）。その `D` を**確率**に写すのが
`theory_order.prob_of_d`（`W(D) = Φ(D/σ_D)`）で、**識別力は符号と 1 ビットも変わらない**（単調変換）。
変わるのは**較正**だけ——そして既定の読みは**決着帯を言い過ぎる**:

| 10 分位 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 予測 − 実勝率（実） | +0.06 | +0.16 | +0.10 | +0.08 | −0.04 | −0.06 | −0.09 | −0.04 | +0.11 | **+0.26** |

最上位帯は**予測 0.91 対 実勝率 0.65**。**対数損失の超過はほぼ全部この 1 帯**（それを除くと合成でもコインを下回る）。

## 機構（測って分かったこと）

**上限（打ち切り）でも `Θ` でもない**——最上位帯の τ_opp が打ち切りになる割合は 0.9%／4.1% で、
その帯の `Θ`/その後の損害は 0.99〜1.07（`Θ` は正しい大きさ）。**膨らんでいるのは `A` 側**
（相手の τ の残差 +10.7／+11.3 ターン 対 自分の +1.4／+2.3）で、**その帯の行は序盤**（j_me 1.8／1.5）＝
**相手の速さが後から育つのを読めていない**（A_opp 0.054 → その後の最大 0.388）。

**残差は「ばらつき」ではなく「伸び」**: 予測 τ の五分位ごとに **予測 ÷ 実際 が 0.85 → 2.84**（合成 0.93 → 2.75）。
だから**同じ `σ` で割る**（`abs`）と長い時計の行で自信過剰になる。`rel` は**比で読む**ことでこれを平坦化する。

**飽和は理論の欠陥であってゲームの性質ではない**——最上位帯の**中だけ**で見ても
`τ_opp/τ_me`（比）の AUC は 0.748／0.614（`D` 自身は 0.589／0.468＝無情報）＝**同じ 2 本の時計を比で読めば当たる**。

使い方:

    python tests/scripts/win_calib.py --in <records_dir> [--games N] [--w-err abs|rel] [--json out.json]
"""

import argparse
import json
import math
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
from theory_order import MU, THETA  # noqa: E402

#: 対数損失を潰さないための刻み（**確率 0/1 を返す読みは存在しない**が、`erf` の飽和で 0 になりうる）
EPS = 1e-6


def rows_of(rows_out, slope="theory", cap=None):
    """判断行を `(D, τ_me, τ_opp, 勝敗)` に落とす（`summarise` の `bias` と同じ母数・同じ打ち切り）。"""
    cap = CB.RACE_CAP if cap is None else float(cap)
    out = []
    for r in rows_out:
        tm = min(float(r["tau_me_" + slope]), cap)
        to = min(float(r["tau_opp_" + slope]), cap)
        out.append((to - tm, tm, to, 1.0 if r["won"] else 0.0))
    return out


def probs_of(rs, sigma_rel=None, scale_mode="hyp"):
    """行ごとの予測勝率。`W_ERR_MODE` は呼ぶ側が立てる（`sigma_rel` は `rel` のときだけ使う）。"""
    old = TO.SIGMA_REL
    try:
        if sigma_rel is not None:
            TO.set_sigma_rel(sigma_rel)
        # **T151-2**: `rows_out` は**ターン開始の 2 本の時計**の行＝手番の半ターンが正確に掛かる瞬間（`mover=True`）。
        return [TO.prob_of_d(d, t_me=tm, t_opp=to, scale_mode=scale_mode, mover=True) for d, tm, to, _z in rs]
    finally:
        TO.set_sigma_rel(old)


def auc_of(p, z):
    """順位で書いた AUC（同値は 0.5 で数える）。"""
    p = np.asarray(p, float); z = np.asarray(z, float)
    pos, neg = p[z > 0.5], p[z <= 0.5]
    if not len(pos) or not len(neg):
        return None
    # 同値を 0.5 で数える＝Mann-Whitney U / (n_pos·n_neg)
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), float)
    srt = np.concatenate([pos, neg])[order]
    i = 0
    while i < len(srt):
        j = i
        while j + 1 < len(srt) and srt[j + 1] == srt[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    u = ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0
    return float(u / (len(pos) * len(neg)))


def calib_bins(p, z, nbin=10):
    """等頻度 `nbin` 箱の較正表（予測の平均・実勝率・差）。"""
    p = np.asarray(p, float); z = np.asarray(z, float)
    idx = np.argsort(p, kind="mergesort")
    out = []
    edges = np.linspace(0, len(p), nbin + 1).astype(int)
    for b in range(nbin):
        sl = idx[edges[b]:edges[b + 1]]
        if not len(sl):
            continue
        out.append({"bin": b + 1, "n": int(len(sl)),
                    "p_mean": round(float(p[sl].mean()), 4),
                    "win_rate": round(float(z[sl].mean()), 4),
                    "gap": round(float(p[sl].mean() - z[sl].mean()), 4)})
    return out


def tau_resid_by_bin(rows_out, p, nbin=10, cap=None):
    """**T151-4（P4）**: `p` の等頻度 `nbin` 箱ごとの **τ の残差**（理論 − 実際）を 2 本の時計で分けて出す。
    `resid_opp = min(τ_opp, cap) − t_opp_act`（相手が実際にあと何自席ターン打ったか）・
    `resid_me = min(τ_me, cap) − t_me_act`。T118 が最上位帯で見つけた「相手の τ の残差 +10.7 対 自分の +1.4」を
    切替（`--opp-clock`）の前後で同じ物差しで測るための器。`won`／`lost` の全体も出す（`summarise` の `bias` と同じ
    ——勝った行は `resid_me`・負けた行は `resid_opp`）。`calib_bins` と同じ切り方（`argsort` の等頻度）。"""
    cap = CB.RACE_CAP if cap is None else float(cap)
    p = np.asarray(p, float)
    ro = np.array([min(float(r["tau_opp_theory"]), cap) - float(r["t_opp_act"]) for r in rows_out], float)
    rm = np.array([min(float(r["tau_me_theory"]), cap) - float(r["t_me_act"]) for r in rows_out], float)
    won = np.array([bool(r["won"]) for r in rows_out])
    idx = np.argsort(p, kind="mergesort")
    edges = np.linspace(0, len(p), nbin + 1).astype(int)
    bins = []
    for b in range(nbin):
        sl = idx[edges[b]:edges[b + 1]]
        if not len(sl):
            continue
        bins.append({"bin": b + 1, "n": int(len(sl)), "p_mean": round(float(p[sl].mean()), 4),
                     "resid_opp": round(float(ro[sl].mean()), 3), "resid_me": round(float(rm[sl].mean()), 3)})
    return {"bins": bins,
            "won_resid_me": round(float(rm[won].mean()), 3) if won.any() else None,
            "lost_resid_opp": round(float(ro[~won].mean()), 3) if (~won).any() else None,
            "all_resid_opp": round(float(ro.mean()), 3) if len(ro) else None,
            "all_resid_me": round(float(rm.mean()), 3) if len(rm) else None}


def score(p, z, nbin=10):
    """対数損失・Brier・AUC・ECE と、**コイン（定数予測）との比較**。"""
    p = np.clip(np.asarray(p, float), EPS, 1.0 - EPS); z = np.asarray(z, float)
    base = float(z.mean())
    ll = float(-(z * np.log(p) + (1 - z) * np.log(1 - p)).mean())
    ll0 = float(-(z * math.log(max(EPS, base)) + (1 - z) * math.log(max(EPS, 1 - base))).mean())
    bins = calib_bins(p, z, nbin)
    ece = float(sum(b["n"] * abs(b["gap"]) for b in bins) / max(1, len(p)))
    return {"n": int(len(p)), "base_win_rate": round(base, 4),
            "logloss": round(ll, 4), "logloss_const": round(ll0, 4),
            "beats_coin": bool(ll < ll0),
            "brier": round(float(((p - z) ** 2).mean()), 4),
            "auc": (round(auc_of(p, z), 4) if auc_of(p, z) is not None else None),
            "ece": round(ece, 4), "calibration": bins}


def stretch(rs, nq=5):
    """**予測 τ の五分位ごとに「予測 ÷ 実際」**（`rel` の患部そのもの）。

    ここでの「実際」は**その席が実際に使った残りターン**ではなく、**同じ行の勝敗と両方の τ から作れない**ので
    呼ぶ側が `t_act` を渡す形にはしない——代わりに**予測 τ の大きさごとの残差の散らばり**を出し、
    「ばらつきが一定か（`abs` の前提）」「比が一定か（`rel` の前提）」のどちらに近いかを見る。"""
    d = np.array([r[0] for r in rs], float)
    tm = np.array([r[1] for r in rs], float)
    to = np.array([r[2] for r in rs], float)
    s = np.array([TO.clock_scale(a, b) for a, b in zip(tm, to)], float)
    q = np.quantile(s, np.linspace(0, 1, nq + 1))
    out = []
    for i in range(nq):
        m = (s >= q[i]) & (s <= q[i + 1]) if i == nq - 1 else (s >= q[i]) & (s < q[i + 1])
        if m.sum() < 5:
            continue
        out.append({"q": i + 1, "n": int(m.sum()),
                    "scale_mean": round(float(s[m].mean()), 3),
                    "d_mean": round(float(d[m].mean()), 3),
                    "d_sd": round(float(d[m].std()), 3),
                    "d_over_scale_sd": round(float((d[m] / np.maximum(1e-9, s[m])).std()), 4)})
    return out


def collect_calib(dirs, limit_games=0, slope="theory", sigma_rel=None, scale_mode="hyp", nbin=10):
    """記録を 1 度読んで **`abs` と `rel` を並べる**（`sigma_rel` が無ければ `rel` は出さない）。"""
    rows_out, ledger, stats, turn_harm, theta_check = CB.collect(dirs, limit_games, THETA, MU, "const")
    rs = rows_of(rows_out, slope)
    z = [r[3] for r in rs]
    old = TO.W_ERR_MODE
    out = {"slope": slope, "games": stats.get("games"), "rows": len(rs),
           "pre_settle": CB.PRE_SETTLE_MODE,
           "sigma_d": round(float(TO.SIGMA_D), 4), "scale_mode": scale_mode,
           "stretch": stretch(rs)}
    try:
        TO.set_w_err_mode("abs")
        out["abs"] = score(probs_of(rs), z, nbin)
        if sigma_rel is not None:
            TO.set_w_err_mode("rel")
            out["rel"] = score(probs_of(rs, sigma_rel, scale_mode), z, nbin)
            out["sigma_rel"] = round(float(sigma_rel), 4)
            # **1 次同次な尺度はどれでも識別力が同じ**ことの検算（AUC が一致する）
            out["scale_auc"] = {m: score(probs_of(rs, sigma_rel, m), z, nbin)["auc"]
                                for m in ("hyp", "sum", "mean", "max", "geo")}
    finally:
        TO.set_w_err_mode(old)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="勝率の読みの較正（T118）")
    ap.add_argument("--in", dest="src", nargs="+", required=True)
    ap.add_argument("--games", type=int, default=0)
    ap.add_argument("--slope", default="theory")
    ap.add_argument("--sigma-rel", type=float, default=None,
                    help="`rel` の σ（省略時は輪郭の表から別のセットの値を引く）")
    ap.add_argument("--scale", default="hyp", choices=("hyp", "sum", "mean", "max", "geo"))
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--pre-settle", default=CB.PRE_SETTLE_MODE, choices=CB.PRE_SETTLE_MODES,
                    help="**T138b** 決着後（`lethal_rule.settled_map`）の行を除いて較正を測るか")
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)
    CB.set_pre_settle_mode(a.pre_settle)            # **T138b**
    sr = a.sigma_rel if a.sigma_rel is not None else CB.sigma_rel_for(a.src)
    out = collect_calib(a.src, a.games, a.slope, sr, a.scale, a.bins)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
