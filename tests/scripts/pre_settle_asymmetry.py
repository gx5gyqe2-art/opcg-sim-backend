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

**実測後（測った後に足した検算・T149a）**: 予告 3 は外れ、終盤ほど大きかった。**行は独立ではない**——
1 局の中で同じ席が優勢のまま続けば、その局の終盤の行がまとめて 1 つの外れを反映する。**局を単位に
束ねても「終盤 > 序盤」が残るか**を、`per_game_gap`／`bootstrap_stage_diff` で確かめる（局の復元抽出・
新しい式は書かない・既存の `p`／`z` の平均を局単位でまとめるだけ）——**残った**（局の束ねの見かけ
ではない・`2026-09-23_presettle_late_robust.md`）。

**T149b**: 経過（段）と残り時間（`s`＝`clock_scale`）の 2 軸に分けたところ、**段の中で `s` を追っても
一貫した傾向は無いが、`s` をそろえて段を比べると経過の効果が残った**（`2026-09-23_presettle_s_axis.md`）
——**σ の伸縮不足より、経過そのもの（生存・まだ決め切れていない選択効果）を支持する形**。

**T149c**: 終盤の優勢側行を「序盤から同じ席が優勢のまま（`persistent`）」と「終盤で新しく優勢に
なった（`flipped`）」に分ける（`survivorship_split`）——`persistent` が多く・`gap` も大きいなら、
「局がまだ終わっていないこと自体」が勝率の読みに無い情報だという読みを支持する。**結果は記録間で
向きが割れた**（実は `flipped`（0.386）≫`persistent`（0.112）・合成は `persistent`（0.264）＞
`flipped`（0.233））——T149c の報告は「2 値化ではなく連続変数（継続ターン数）で見る」を未着手の
候補として挙げた。

**T149f**（本節・T149c の未着手候補）: `persistent`／`flipped` の 2 値化を、**その席が何ターン連続で
優勢（`p>0.5`）だったか**という連続変数（`leader_streak_of`）に置き換え、終盤の優勢側行を継続ターン数の
分位で割って `gap` を見る（`streak_axis_table`）。**予告（測る前）**:

1. **実は単調減少**を予告する——`flipped`（継続の短い側）の `gap`（0.386）が `persistent`（継続の
   長い側）の `gap`（0.112）よりずっと大きいので、継続が短いほど `gap` が大きいはず。
2. **合成は実よりずっと平坦**を予告する——`persistent`（0.264）と `flipped`（0.233）の差が小さいので、
   継続ターン数を分位で割っても傾向は弱いか一貫しないはず。
3. **2 値化の境目（1 ターン）をまたいだだけの見かけの逆転**である可能性——連続変数で見ると、実・合成
   どちらも同じ形（例: 継続 1〜2 ターンだけ高い・その先は平坦）で、2 値の境目の取り方が記録間の
   「向きの逆転」を作っていただけ、という可能性も観察する（当てはめない・予告はしない）。

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
                    "j_me": int(r["j_me"]), "stage": stage_of(r["j_me"]),
                    "seed": r.get("seed"), "who": r.get("who"),
                    # **T149b**: `s`＝`theory_order.clock_scale`（`W(D)` の物差しがそのまま使う残り時間の尺度）
                    "s": float(TO.clock_scale(tm, to))})
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


def mean_gap(rows):
    """**符号つきの較正の差**＝`mean(p) − mean(z)`。`gap_of`（`calibration` の n 重み平均）と**恒等式で
    一致する**（分位で束ねても崩れない・T149a で確かめる不変量。`calib_bins` が値を 4 桁に丸めるので
    完全なビット一致ではなく丸め誤差 1e-3 程度）。空なら `None`。"""
    if not rows:
        return None
    return float(np.mean([r["p"] for r in rows]) - np.mean([r["z"] for r in rows]))


def per_game_gap(rows):
    """行を局（`seed`）ごとにまとめ、**局ごとの** `mean(p − z)` を返す（`{seed: gap}`）。
    **T149a**: 1 局が多くの行を出すほど大きく数える行単位の平均と違い、**局 1 つ＝1 票**にする土台。"""
    acc = {}
    for r in rows:
        acc.setdefault(r["seed"], []).append(r["p"] - r["z"])
    return {s: float(np.mean(vs)) for s, vs in acc.items()}


def game_weighted_gap(rows):
    """**局を 1 票ずつ数えた較正の差**（`per_game_gap` の値の平均）。空／局が無ければ `None`。"""
    pg = per_game_gap(rows)
    return float(np.mean(list(pg.values()))) if pg else None


def bootstrap_stage_diff(rows, stage_a="late", stage_b="early", n_boot=2000, seed=0):
    """**T149a**: 優勢（または劣勢）側の行を局×段に束ね、**局を単位にした復元抽出**で
    `mean(stage_a) − mean(stage_b)` の分布を作る（既定 late−early・新しい式は書かない）。
    局が 5 未満なら測らない（母数不足）。"""
    by_seed_stage = {}
    for r in rows:
        by_seed_stage.setdefault(r["seed"], {}).setdefault(r["stage"], []).append(r["p"] - r["z"])
    seeds = list(by_seed_stage.keys())
    if len(seeds) < 5:
        return {"n_games": len(seeds), "diff": None, "ci95": None}

    def stage_mean(sample_seeds, stage):
        vals = [float(np.mean(by_seed_stage[s][stage])) for s in sample_seeds if stage in by_seed_stage[s]]
        return float(np.mean(vals)) if vals else None

    point_a = stage_mean(seeds, stage_a)
    point_b = stage_mean(seeds, stage_b)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        samp = rng.choice(seeds, size=len(seeds), replace=True)
        a, b = stage_mean(list(samp), stage_a), stage_mean(list(samp), stage_b)
        if a is not None and b is not None:
            diffs.append(a - b)
    diffs = np.asarray(diffs, float)
    ci = ([float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]
          if len(diffs) else None)
    return {"n_games": len(seeds), "point_" + stage_a: point_a, "point_" + stage_b: point_b,
            "diff": (point_a - point_b) if (point_a is not None and point_b is not None) else None,
            "ci95": ci, "n_boot_valid": int(len(diffs)),
            "excludes_zero": bool(ci and (ci[0] > 0 or ci[1] < 0))}


#: **T149b**: 「残り時間」の軸を割る分位数（既定 4 分位・各段の中で別々に切る）。
S_QUANTILES = 4


def s_axis_table(rows, n_q=S_QUANTILES):
    """**T149b**: 優勢側の行を、**経過**（`stage`）×**残り時間**（`s`＝`clock_scale` の分位・
    段ごとに別々に切る）の 2 次元で割り、`mean_gap` を出す。

    `σ_rel·s` の伸縮が残り時間を正しく織り込んでいるなら、**同じ段の中で `s` の分位を追っても
    `gap` は平らなはず**（`s` は既に物差しに入っている軸）。段をまたいで `gap` が変わるのに、
    段の中で `s` を追っても変わらなければ、**残り時間の伸縮ではなく経過そのもの**（生存・選択・
    T149c）が疑わしくなる。逆に段の中でも `s` の分位を追うほど `gap` が増えれば、**伸縮の不足**
    （T146a の候補 1）が疑わしい。当てはめではなく観察——新しい価格式は書かない。"""
    out = {}
    for st in STAGES:
        sub = [r for r in rows if r["stage"] == st]
        if len(sub) < n_q * 5:
            out[st] = {"n": len(sub), "quantiles": []}
            continue
        ss = np.asarray([r["s"] for r in sub], float)
        edges = np.quantile(ss, np.linspace(0, 1, n_q + 1))
        cells = []
        for i in range(n_q):
            lo, hi = edges[i], edges[i + 1]
            m = (ss >= lo) & (ss <= hi) if i == n_q - 1 else (ss >= lo) & (ss < hi)
            cell_rows = [r for r, keep in zip(sub, m) if keep]
            cells.append({"q": i + 1, "n": len(cell_rows), "s_lo": round(float(lo), 3),
                         "s_hi": round(float(hi), 3), "gap": mean_gap(cell_rows)})
        out[st] = {"n": len(sub), "quantiles": cells}
    return out


def early_leader_of(rows):
    """**T149c**: 局（`seed`）ごとに、**序盤で最初に優勢だった席**を返す（`{seed: 優勢の who}`）。
    行の `p` はその行自身の席（`who`）から見た予測なので、`p<0.5` の行は**もう片方の席**が優勢
    （`1 - who`）。序盤の行が無い・最初の序盤行が `p==0.5` なら入れない（判定不能は「無い」で扱う）。"""
    first = {}
    for r in rows:
        if r["stage"] != "early" or r["seed"] in first:
            continue
        if r["p"] == 0.5:
            continue
        first[r["seed"]] = r["who"] if r["p"] > 0.5 else (1 - r["who"])
    return first


def survivorship_split(rows):
    """**T149c**: 終盤（`stage=="late"`）の優勢側の行を、**序盤から同じ席が優勢のまま続いている
    （persistent）**か、**終盤になって新しく優勢になった（flipped）**か、**序盤のデータが無い
    （no_early_data）**かに分け、`mean_gap` を比べる。

    `persistent` が多く・`gap` も大きいなら、**「局がまだ終わっていないこと自体」が勝率の読みに
    無い情報**という読みを支持する（序盤に優勢と読まれたのに終盤まで決め切れていない局は、
    その時点で理論の見立てが外れていた可能性が高い局の集まりだから）。当てはめではなく観察。"""
    leader = early_leader_of(rows)
    late_fav = [r for r in rows if r["stage"] == "late" and r["p"] > 0.5]
    groups = {"persistent": [], "flipped": [], "no_early_data": []}
    for r in late_fav:
        el = leader.get(r["seed"])
        if el is None:
            groups["no_early_data"].append(r)
        elif el == r["who"]:
            groups["persistent"].append(r)
        else:
            groups["flipped"].append(r)
    return {name: {"n": len(rs), "gap": mean_gap(rs), "n_games": len(per_game_gap(rs))}
           for name, rs in groups.items()}


def leader_streak_of(rows):
    """**T149f**: 各行について、**その席（`who`）がその行まで何ターン連続で優勢（`p>0.5`）だったか**
    を数える（同じ `seed`・`who` の行を `j_me` 昇順に並べ、この行を含めて `p>0.5` が続く長さ）。
    `survivorship_split` の `persistent`／`flipped` の 2 値化を連続変数に置き換える（T149c の報告が
    挙げた未着手の候補）。`p<=0.5` の行は 0（優勢でない行に継続は無い）。戻り値は
    `{(seed, who, j_me): streak}`。"""
    by_key = {}
    for r in rows:
        by_key.setdefault((r["seed"], r["who"]), []).append(r)
    out = {}
    for rs in by_key.values():
        streak = 0
        for r in sorted(rs, key=lambda r: r["j_me"]):
            streak = streak + 1 if r["p"] > 0.5 else 0
            out[(r["seed"], r["who"], r["j_me"])] = streak
    return out


#: **T149f**: 継続ターン数を割る分位数（既定 4 分位・`S_QUANTILES` と揃える）。
STREAK_QUANTILES = S_QUANTILES


def streak_axis_table(rows, n_q=STREAK_QUANTILES):
    """**T149f**: `survivorship_split` と同じ集合（終盤・優勢側の行）を、連続変数の継続ターン数
    （`leader_streak_of`）の分位で割り、`mean_gap` を出す。`n` が `n_q*5` 未満なら分位を作らず `n` だけ
    返す（`s_axis_table` と同じ規約）。当てはめではなく観察。"""
    streaks = leader_streak_of(rows)
    late_fav = [r for r in rows if r["stage"] == "late" and r["p"] > 0.5]
    if len(late_fav) < n_q * 5:
        return {"n": len(late_fav), "quantiles": []}
    vals = np.asarray([streaks.get((r["seed"], r["who"], r["j_me"]), 1) for r in late_fav], float)
    edges = np.quantile(vals, np.linspace(0, 1, n_q + 1))
    cells = []
    for i in range(n_q):
        lo, hi = edges[i], edges[i + 1]
        m = (vals >= lo) & (vals <= hi) if i == n_q - 1 else (vals >= lo) & (vals < hi)
        cell_rows = [r for r, keep in zip(late_fav, m) if keep]
        cells.append({"q": i + 1, "n": len(cell_rows), "streak_lo": float(lo), "streak_hi": float(hi),
                     "gap": mean_gap(cell_rows)})
    return {"n": len(late_fav), "quantiles": cells}


def robustness_check(rows):
    """**T149a のまとめ**: 優勢側・劣勢側それぞれで、行単位／局単位の較正の差と、
    late−early の局単位ブートストラップを並べる。"""
    out = {}
    for name, pred in (("favorite", lambda r: r["p"] > 0.5), ("underdog", lambda r: r["p"] < 0.5)):
        sub = [r for r in rows if pred(r)]
        out[name] = {"row_weighted_gap": mean_gap(sub), "game_weighted_gap": game_weighted_gap(sub),
                     "n_games": len(per_game_gap(sub)),
                     "late_minus_early": bootstrap_stage_diff(sub, "late", "early")}
    return out


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
           "stage_gaps": {fav: {st: gap_of(ss[fav][st]["score"]) for st in STAGES} for fav in ("favorite", "underdog")},
           # **T149a**: 行単位の恒等式検算（`mean_gap` == `gap_of`）＋局単位の頑健性
           "mean_gap_check": {"favorite": mean_gap([r for r in rows if r["p"] > 0.5]),
                              "underdog": mean_gap([r for r in rows if r["p"] < 0.5])},
           "robustness": robustness_check(rows),
           # **T149b**: 優勢側の gap を経過（段）×残り時間（s の分位）で割る
           "s_axis": {"favorite": s_axis_table([r for r in rows if r["p"] > 0.5]),
                      "underdog": s_axis_table([r for r in rows if r["p"] < 0.5])},
           # **T149c**: 終盤の優勢側行を「序盤から同じ席が優勢のまま（persistent）」か
           # 「終盤で新しく優勢になった（flipped）」かに分ける
           "survivorship": survivorship_split(rows),
           # **T149f**: persistent/flipped の 2 値化を継続ターン数の連続変数に置き換える
           "streak_axis": streak_axis_table(rows)}


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
