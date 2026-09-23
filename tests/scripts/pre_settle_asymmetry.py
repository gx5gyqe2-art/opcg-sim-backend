#!/usr/bin/env python3
"""**決着前の較正の非対称を測る**（T146b／T146c・2026-09-23・ユーザ指示「上２つのうまく行っていない
ことは先にやりたい」の 1 つ目）。

## 問い

T146a（`2026-09-23_presettle_sigma.md`）は「σ_rel が決着後の行を混ぜて狭く出ている」という仮説を
測って外れた——σ の幅を決着前の行だけから作り直しても対数損失はほぼ動かなかった。だが**較正表を
見ると対称な自信過剰ではなかった**——最下位分位（自分が劣勢と読む行）は予測が実勝率より**低い**
（過小評価）のに、中〜上位分位（自分が優勢と読む行）は予測が実勝率よりずっと**高い**（過大評価）。
σ を対称に広げても直らない形である。

本器はこの非対称を 2 段で測る:

* **T146b**: 優勢側（`p > 0.5`）と劣勢側（`p < 0.5`）の較正を分けて出す。非対称の大きさを数字にする。
* **T146c**: 同じ分け方を自席ターン数（`j_me`＝経過ターン・序盤／中盤／終盤）でさらに割る。
  T118 が全行の最上位分位で見つけた機構（「相手の速さ `A_opp` が後から育つのを読めていない」＝
  序盤ほど自分が優勢だと読みすぎる）が、決着前の行全体でも同じ形かを確かめる。

## 式（新定数ゼロ・当てはめない）

新しい価格式は書かない。既存の `crossing_bridge.collect(pre_settle=True)` の行
（`p`＝`theory_order.prob_of_d`・`z`＝勝敗・`j_me`＝経過ターン）を、**既に持っている値の符号や大きさで
分けるだけ**（優勢／劣勢の分け方も新しい閾値ではなく `p` の定義そのものである 0.5）。

## 検算の予告

1. 優勢側の較正の差（予測−実勝率）は劣勢側より大きい（符号は正・劣勢側は負かゼロに近い）。
2. 優勢側だけで対数損失を測るとコインに負ける幅が広がり、劣勢側だけならコインに近いか勝つ。
3. 優勢側の過大評価は序盤（`j_me` が小さい）ほど大きい（T118 の「相手の速さが後から育つ」機構どおり）。

使い方:

    python tests/scripts/pre_settle_asymmetry.py --in <records_dir> [<records_dir> ...] \
        [--games N] [--sigma-rel VALUE] [--json out.json]
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

#: 序盤／中盤／終盤の境目（自席ターン番号 `j_me`・0 始まり）。**新しい閾値ではなく境目を宣言するだけ**
#: ——`j_me <= J_EARLY_MAX` を序盤・`> J_LATE_MIN` を終盤・その間を中盤とする（T146c）。
J_EARLY_MAX = 1
J_LATE_MIN = 3
STAGES = ("early", "mid", "late")


def stage_of(j_me):
    j = int(j_me)
    if j <= J_EARLY_MAX:
        return "early"
    if j > J_LATE_MIN:
        return "late"
    return "mid"


def rows_with_p(rows_out, slope="theory", sigma_rel=None, w_err="rel"):
    """`rows_out`（`crossing_bridge.collect` の行）に予測確率 `p` を付けて返す
    （`win_calib.rows_of`／`probs_of` をそのまま呼ぶ・新しい式は書かない）。"""
    rs = WC.rows_of(rows_out, slope)
    old = TO.W_ERR_MODE
    try:
        TO.set_w_err_mode(w_err)
        p = WC.probs_of(rs, sigma_rel if w_err == "rel" else None)
    finally:
        TO.set_w_err_mode(old)
    out = []
    for r, (d, tm, to, z), pi in zip(rows_out, rs, p):
        out.append({"p": float(pi), "z": float(z), "d": float(d), "won": bool(r["won"]),
                    "j_me": int(r["j_me"]), "stage": stage_of(r["j_me"])})
    return out


def score_group(rows):
    """1 群の較正（`win_calib.score`・10 分位）。空なら `None`。"""
    if len(rows) < 10:
        return {"n": len(rows)}
    p = [r["p"] for r in rows]
    z = [r["z"] for r in rows]
    if len(set(z)) < 2:
        return {"n": len(rows), "base_win_rate": float(np.mean(z))}
    return WC.score(p, z, 10)


def favorite_split(rows):
    """**T146b**: `p > 0.5`（優勢側）と `p < 0.5`（劣勢側）に分けて採点する。`p == 0.5` はどちらにも数えない
    （境目そのものは較正が定義できない）。"""
    fav = [r for r in rows if r["p"] > 0.5]
    dog = [r for r in rows if r["p"] < 0.5]
    return {"favorite": {"n": len(fav), "score": score_group(fav)},
            "underdog": {"n": len(dog), "score": score_group(dog)}}


def stage_split(rows):
    """**T146c**: 優勢／劣勢のそれぞれを序盤／中盤／終盤でさらに割る。"""
    out = {}
    for fav_name, pred in (("favorite", lambda r: r["p"] > 0.5), ("underdog", lambda r: r["p"] < 0.5)):
        out[fav_name] = {}
        for st in STAGES:
            sub = [r for r in rows if pred(r) and r["stage"] == st]
            out[fav_name][st] = {"n": len(sub), "score": score_group(sub)}
    return out


def gap_of(score):
    """較正の差（予測平均 − 実勝率）の絶対値の重み付き平均（`calibration` が無ければ `None`）。"""
    if not score or "calibration" not in score:
        return None
    bins = score["calibration"]
    n = sum(b["n"] for b in bins)
    if not n:
        return None
    return float(sum(b["n"] * b["gap"] for b in bins) / n)


def collect(dirs, limit_games=0, slope="theory", sigma_rel=None, w_err="rel"):
    """記録を 1 度読み（決着前だけ）、優勢／劣勢・序盤〜終盤で割った較正を返す。"""
    old = CB.PRE_SETTLE_MODE
    try:
        CB.set_pre_settle_mode("on")
        rows_out, _ledger, stats, _th, _tc = CB.collect(dirs, limit_games, THETA, MU, "const")
    finally:
        CB.set_pre_settle_mode(old)
    if sigma_rel is None:
        sigma_rel = CB.sigma_rel_for(dirs)
    rows = rows_with_p(rows_out, slope, sigma_rel, w_err)
    fs = favorite_split(rows)
    ss = stage_split(rows)
    return {"games": stats.get("games"), "n": len(rows), "sigma_rel": sigma_rel, "w_err": w_err,
           "favorite_split": fs,
           "favorite_signed_gap": gap_of(fs["favorite"]["score"]),
           "underdog_signed_gap": gap_of(fs["underdog"]["score"]),
           "stage_split": ss,
           "stage_gaps": {fav: {st: gap_of(ss[fav][st]["score"]) for st in STAGES} for fav in ("favorite", "underdog")}}


def main(argv=None):
    ap = argparse.ArgumentParser(description="決着前の較正の非対称（優勢側の過大評価）を測る（T146b/T146c）")
    ap.add_argument("--in", dest="src", nargs="+", required=True)
    ap.add_argument("--games", type=int, default=0)
    ap.add_argument("--slope", default="theory")
    ap.add_argument("--sigma-rel", type=float, default=None)
    ap.add_argument("--w-err", default="rel", choices=("abs", "rel"))
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)
    out = collect(a.src, a.games, a.slope, a.sigma_rel, a.w_err)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
