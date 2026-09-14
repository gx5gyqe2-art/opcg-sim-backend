"""**4 つの価格は 1 つの `w` を共有するか**（転換の検定・読み取り専用）。

`docs/cpu_theory_gap.md` の **T24**・`game_theory.md` **§17.1.6** の事前登録した予測。

## 何を検定するか

§17.1 は「価格は基本量ではなく**時計の従属量**」と言う:

```
price(X) = w · [ ∂T_opp/∂X − ∂T_me/∂X ]        w = テンポ 1 ターンの価値
```

**`w` は 1 つ**で、4 つの価格はそれを共有する。時計の差分（`1/A_opp`・`1/(c̄·A_opp)` など）は
**デッキと盤面の量**なので、競走の接近度では動かない。したがって:

> **競走の接近度で帯を切ると、価格の「水準」は揃って動き、「比」は動かない。**

- **水準**（`λ`・`μ`・`δ`・`ν`）は `w(状態)` と一緒に上下する。
  接戦ほど 1 ターンの価値が高いので、**接戦帯で全部が高い**はず。
- **比**（`λ/μ` ＝ `c̄`・`δ/μ` ＝ `Δc`・`ν/μ`）は**帯で動かない**はず。

**比が動けばこの枠は誤り。** ——★の式が成り立たないことの直接の証拠になる。

## 測り方

`hand_value_slope` の within 推定をそのまま使う（**軸ごとに説明変数を帯から外す**
＝`band_key(axis)`）。本器が足すのは**競走の接近度による層別**だけ:

| 軸 | 価格 | 帯（説明変数を除く 4 つ） |
|---|---|---|
| `hand` | **`μ`** | 自ライフ・相手ライフ・ターン帯・自場 |
| `life` | **`λ`** | **手札**・相手ライフ・ターン帯・自場 |
| `don` | **`δ`** | 自ライフ・相手ライフ・ターン帯・**手札** |
| `field` | **`ν`** | 自ライフ・相手ライフ・ターン帯・**手札** |

接近度は **`|v0|`**（ネットが見た盤面の傾きの絶対値・`order_acc.band_of` と同じ切り方）。

> **`|v0|` を使う限界**: これは**ネットの意見**であって状態そのものではない。
> 同じ自己対戦から出ているので独立ではない。**状態から時計を推定して切る**のが本来だが、
> `T_me` は自分の場（`ν` の説明変数）を含むので、**4 軸で共通の帯にできない**。
> だから**4 軸に共通で使える 1 つのスカラ**として `|v0|` を採る。

**比には必ず CI を付ける**（`measurement.md` §14-15＝比の統計量は驚くほど動く）。
対局を復元抽出して 4 本の傾きを引き直し、比の分布から取る。

## 読み方（事前登録）

- **水準が揃って動く**（接戦帯で 4 つとも高い）＝★の式と整合。
- **比の CI が帯をまたいで重なる**＝**比は動かない**＝★の式を支持。
- **比の CI が重ならない**＝**★の式は誤り**（価格は 1 つの `w` を共有していない）。
- **水準が揃わない**（ある価格だけ逆に動く）＝その価格の時計への翻訳（§17.1.2）が誤り。

**限界**: 観察された傾きであって因果ではない・`|v0|` はネット由来・自席ターンのみ。

実行例:
  OPCG_LOG_SILENT=1 python tests/scripts/price_scaling.py --in ~/w39 --out ~/price_scaling.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from opcg_sim.learned.train import plan_labels as PL  # noqa: E402
from hand_value_slope import (AXES, AXIS_PRICE, SC_MY_DON, SC_MY_FIELD,  # noqa: E402
                              SC_MY_HAND, SC_MY_LIFE, band_key, within_slope)
from order_acc import band_of  # noqa: E402
from theory_order import PWR_EPS, incoming_x  # noqa: E402

ROW_COLS = ("who", "turn", "seed", "z", "kind", "step", "pol_len", "pol_chosen", "pol_v0")
#: 軸 → 説明変数にする scalars の列
AXIS_COL = {"hand": SC_MY_HAND, "life": SC_MY_LIFE, "don": SC_MY_DON, "field": SC_MY_FIELD}
RACES = ("close", "mid", "decided")
#: 比の分子・分母（**`μ` を基準にする**——手札は 4 通貨で唯一「攻めにも守りにも使える」）
RATIOS = (("lambda_over_mu", "life", "hand"), ("delta_over_mu", "don", "hand"),
          ("nu_over_mu", "field", "hand"))


#: 接近度の切り方（**説明変数の下流で切ってはいけない**・2026-09-14 に `v0` が壊れて判った）
SPLITS = ("a_opp", "v0")


def _extra(dd, n):
    return {"sc": np.asarray(dd["scalars"])[:n].astype(np.float32),
            "tok": np.asarray(dd["tokens"])[:n].astype(np.float32)}


def race_of(v0, tok_row, split="a_opp"):
    """競走の接近度の型。

    | `split` | 量 | 説明変数の下流か |
    |---|---|---|
    | **`a_opp`** | **相手の通る攻撃数**（`T_opp` の分母・§17.1） | **いいえ**——相手側の量だけ |
    | `v0` | ネットが見た盤面の傾き（絶対値） | **はい**——**使ってはいけない**（下記） |

    > **`v0` で切ると器が壊れる**（2026-09-14 に実測）——`|v0|` は**ネットが手札・ライフ・
    > ドン・場を入力にして出した値**なので、**4 つの説明変数すべての下流**にある。
    > そこで帯を切ると **`X → v0 → z` の経路を塞ぐ**＝**測りたい効果そのものを消す**。
    > 実際に接戦帯で **`μ` が −0.0254**（手札 1 枚の価値が負）になった。
    > **`ν` を測るとき自分の場で帯を切れないのと同じ誤り**である。
    > 残してあるのは**壊れ方を再現できるようにするため**で、既定にはしない。

    `a_opp` は「相手の時計がどれだけ速いか」＝**速いほど残りターンが少ない**＝
    ★の式では **`w` が高い**＝**4 つの価格が揃って上がる**はず。
    """
    if split == "v0":
        return band_of(float(v0))
    a = sum(1 for x in incoming_x(tok_row) if x >= -PWR_EPS)
    return "a%d" % min(3, a)


def collect(dirs, limit_games=0, split="a_opp"):
    """自席ターン最初の main 行 → (seed, 接近度, scalars, 勝敗)。"""
    recs = []
    games = 0
    for rows, pol, ex, L, ptr, idx in PL.iter_games(dirs, row_cols=ROW_COLS,
                                                    extra_fn=_extra):
        games += 1
        if limit_games and games > limit_games:
            break
        seed = int(rows["seed"][idx[0]])
        seen = set()
        for i in idx:
            w, t = int(rows["who"][i]), int(rows["turn"][i])
            if t < 1 or int(rows["kind"][i]) != 0 or not PL.is_own_turn(w, t):
                continue
            if (w, t) in seen:
                continue
            seen.add((w, t))
            z = float(rows["z"][i])
            if z == 0.0:
                continue
            recs.append({"seed": seed,
                         "race": race_of(rows["pol_v0"][i], ex["tok"][i], split),
                         "sc": np.asarray(ex["sc"][i], np.float64),
                         "z": 1.0 if z > 0 else 0.0})
    return recs, games


def price(recs, axis):
    """1 つの軸の価格（`hand_value_slope` の within 推定・軸ごとに帯を入れ替える）。"""
    col = AXIS_COL[axis]
    rows = [{"band": band_key(r["sc"], axis), "h": float(r["sc"][col]), "z": r["z"]}
            for r in recs]
    return within_slope(rows, "z", h_key="h")


def prices(recs):
    """4 軸まとめて。キーは価格名（`mu`／`lambda`／`delta`／`nu`）。"""
    out = {}
    for ax in AXES:
        s = price(recs, ax)
        out[AXIS_PRICE[ax]] = {"value": s["within"], "se": s["within_se"],
                               "rows": s["rows_used"], "bands": s["bands_used"]}
    return out


def _ratios(p):
    out = {}
    for name, num_ax, den_ax in RATIOS:
        a = p[AXIS_PRICE[num_ax]]["value"]
        b = p[AXIS_PRICE[den_ax]]["value"]
        out[name] = (float(a / b) if (a is not None and b not in (None, 0.0)) else None)
    return out


def _boot_ratio_ci(recs, reps=200, seed=0):
    """**対局を復元抽出して比の CI**（比の統計量には必ず CI・`measurement.md` §14-15）。"""
    if reps <= 0:
        return {name: [None, None] for name, _a, _b in RATIOS}
    by_game = {}
    for r in recs:
        by_game.setdefault(r["seed"], []).append(r)
    gids = list(by_game)
    if len(gids) < 3:
        return {name: [None, None] for name, _a, _b in RATIOS}
    rng = np.random.default_rng(seed)
    acc = {name: [] for name, _a, _b in RATIOS}
    for _ in range(int(reps)):
        pick = rng.integers(0, len(gids), len(gids))
        sub = [r for k in pick for r in by_game[gids[k]]]
        rr = _ratios(prices(sub))
        for name, v in rr.items():
            if v is not None and np.isfinite(v):
                acc[name].append(v)
    out = {}
    for name, vals in acc.items():
        out[name] = ([round(float(np.percentile(vals, 2.5)), 4),
                      round(float(np.percentile(vals, 97.5)), 4)] if len(vals) >= 10
                     else [None, None])
    return out


def _overlap(a, b):
    if None in a or None in b:
        return None
    return not (a[1] < b[0] or b[1] < a[0])


#: 比の CI がこれより広ければ**情報を持っていない**＝重なっても「動かない」の証拠にならない。
#: 幅は点推定の絶対値との比で見る（比の統計量なので絶対値では測れない）。
CI_WIDTH_MAX_REL = 2.0


def informative(ci, point):
    """**CI が狭くて初めて「重なる＝動かない」が言える**（2026-09-14 に判定を直した）。

    初版は幅 [−9, 16]・[−40, 42] のような CI どうしが重なるのを見て
    **「比は動かない＝★の式を支持」と判定していた**——**雑音を支持と読んでいた**。
    これは罠 17（A/B の前に分解能を数える）を CI に当てはめたもの。
    """
    if ci is None or None in ci or point is None or point == 0.0:
        return False
    return bool((ci[1] - ci[0]) / abs(point) <= CI_WIDTH_MAX_REL)


#: 価格として**あり得ない符号**（手札・ライフ・ドン・体はどれも正の価値を持つ）。
#: 負が出たら**帯の取り方が説明変数の下流を塞いでいる**疑いが濃い。
def price_signs_ok(p):
    vals = [v["value"] for v in p.values() if v["value"] is not None]
    return bool(vals and all(v > 0.0 for v in vals))


#: 帯として採る最小の行数。**薄い帯は判定に入れない**——2026-09-14 に `a0`（52 行）の
#: 負の価格だけで「器が壊れている」と判定が出た（値そのものは単なる雑音だった）。
MIN_BAND_ROWS = 200


def summarise(recs, reps=200, seed=0, races=None, min_rows=MIN_BAND_ROWS):
    """接近度の帯ごとに 4 価格と比を出し、**事前登録した読み方で判定する**。"""
    races = tuple(races) if races else tuple(sorted({r["race"] for r in recs}))
    out = {"by_race": {}, "dropped_thin": []}
    for race in races + ("all",):
        sub = recs if race == "all" else [r for r in recs if r["race"] == race]
        if race != "all" and len(sub) < min_rows:
            out["dropped_thin"].append({"race": race, "n": len(sub)})
            continue
        if len(sub) < 50:
            continue
        p = prices(sub)
        out["by_race"][race] = {"n": len(sub), "prices": p, "ratios": _ratios(p),
                                "ratio_ci95": _boot_ratio_ci(sub, reps, seed)}
    got = [r for r in races if r in out["by_race"]]
    # **価格が負に出た帯は器が壊れている**（説明変数の下流で帯を切ると起きる）
    out["price_signs_ok"] = {r: price_signs_ok(out["by_race"][r]["prices"])
                             for r in out["by_race"]}
    # **水準は揃って動くか**——接戦帯で 4 つとも高いのが★の式の含み
    if len(got) >= 2:
        # **最も速い帯と最も遅い帯**を比べる（`a_opp` なら攻撃が多い方が「速い」）
        fast, slow = got[-1], got[0]
        out["compared"] = {"fast": fast, "slow": slow}
        c = out["by_race"][fast]["prices"]
        d = out["by_race"][slow]["prices"]
        hi = {k: (c[k]["value"] is not None and d[k]["value"] is not None
                  and c[k]["value"] > d[k]["value"]) for k in c}
        out["levels_higher_in_fast"] = hi
        out["levels_move_together"] = bool(len(set(hi.values())) == 1)
    # **比は動かないか**——CI が帯をまたいで重なれば支持
    ov = {}
    for name, _a, _b in RATIOS:
        pairs = [(x, y) for i, x in enumerate(got) for y in got[i + 1:]]
        ov[name] = {("%s_vs_%s" % (x, y)):
                    _overlap(out["by_race"][x]["ratio_ci95"][name],
                             out["by_race"][y]["ratio_ci95"][name]) for x, y in pairs}
    out["ratio_ci_overlap"] = ov
    out["ratio_ci_overlap"] = ov
    # **情報を持つ CI の組だけで判定する**（幅が広すぎる帯は棄権）
    info = {r: all(informative(out["by_race"][r]["ratio_ci95"][n],
                               out["by_race"][r]["ratios"][n]) for n, _a, _b in RATIOS)
            for r in got}
    out["ci_informative"] = info
    usable = [r for r in got if info[r]]
    pairs = [(x, y) for i, x in enumerate(usable) for y in usable[i + 1:]]
    flat = [ov[n]["%s_vs_%s" % (x, y)] for n, _a, _b in RATIOS for x, y in pairs
            if ov[n].get("%s_vs_%s" % (x, y)) is not None]
    out["ratios_are_flat"] = bool(flat and all(flat))
    bad_sign = [r for r in got if not out["price_signs_ok"].get(r, True)]
    if bad_sign:
        out["verdict"] = "instrument_broken_negative_price"
        out["broken_bands"] = bad_sign
    elif not flat:
        out["verdict"] = "undecided_ci_too_wide"
    elif all(flat):
        out["verdict"] = "supports_one_w"
    else:
        out["verdict"] = "ratios_move_framework_wrong"
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="src", nargs="+", required=True, help="n_records のディレクトリ")
    ap.add_argument("--limit-games", type=int, default=0)
    ap.add_argument("--split", default="a_opp", choices=SPLITS,
                    help="接近度の切り方。既定 `a_opp`＝相手の通る攻撃数（`T_opp` の分母）。"
                         "**`v0` は説明変数の下流なので壊れる**（再現用に残してある）")
    ap.add_argument("--min-band-rows", type=int, default=MIN_BAND_ROWS,
                    help="帯として採る最小の行数（薄い帯は判定に入れない）")
    ap.add_argument("--boot-reps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)

    t0 = time.time()
    recs, games = collect(a.src, a.limit_games, a.split)
    res = {"games": games, "rows": len(recs), "split": a.split,
           "summary": summarise(recs, a.boot_reps, a.seed, min_rows=a.min_band_rows),
           "args": {k: v for k, v in vars(a).items() if k != "out"},
           "seconds": round(time.time() - t0, 1)}
    txt = json.dumps(res, ensure_ascii=False, indent=2)
    print(txt)
    if a.out:
        with open(os.path.expanduser(a.out), "w", encoding="utf-8") as fh:
            fh.write(txt + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
