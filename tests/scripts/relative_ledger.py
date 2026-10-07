#!/usr/bin/env python3
"""**紐付けの法則を測る**（T122・2026-09-20・`game_theory.md` §17.9・ユーザ指示「進めてみてください」）。

## 法則（§17.9・測る前に書いてある）

出荷の勝率は `W = Φ(D/(σ_rel·s))` で `D` も `s` も時計について **1 次同次**。したがって
**`z` は 0 次同次＝`W` は 2 本の時計の「比」 `r = T_me/T_opp` だけの関数**。そこから 1 行で出る:

    ΔW = K(r) · Δ log( T_opp / T_me )          K(r) = φ(z)·r(1+r) / ( σ_rel·(1+r²)^{3/2} )

**4 つの軸はすべて同じスカラー `K` を共有し、違うのは分母だけ**（攻撃 `p/Θ_opp`・守り `p/Θ_me`・
体 `ΔA/A_me`・遅く `ΔA/A_opp`）。つまり**価格は絶対額ではなく「相対変化」で入る**。

## 2×2 ＋ 3（腕の組み方）

**「相対化」と「重み」のどちらが効くのかを分けて測る**ため、2×2 に組む:

|  | 重み無し（`κ=1`） | 重み `K(r)` |
|---|---|---|
| **絶対額の価格** | `abs_flat`（T121 の勝者） | `plac_K`（法則の重みだけ・相対化しない） |
| **相対変化** | `rel_flat`（相対化だけ） | **`rel_K`（法則そのもの）** |

* `abs_kappa` … **出荷の帳簿**（`κ = w(D)/w̄`）
* `rel_exact` … **一次近似をやめる**（`Φ(z') − Φ(z)` を直に引く・P3）
* `plac_shift` … **プラセボ**: `K` を**前の局の同じ行番号**から取る（重みと局面の結びつきだけ壊す）

## **判定に AUC を単独で使ってはいけない**（本器で気づいたこと）

**局ごとの帳簿の和を AUC で読むと、決着帯を増幅した腕が機械的に勝つ**——決着帯では勝敗が
**もう決まっている**ので、そこを大きく数えるほど「当たる」。`K(r)` は決着帯で 0 に落ちる
（`r → 0` で `K ∝ r`）＝**`ΔW` として正しい振る舞い**なのに、AUC では罰される。

**だから本器は 3 つを並べて出す**:

1. **`auc` / `corr`（全帯）**——参考。決着帯の寄与が入る。
2. **`before`（最後の自席ターンを外した帯）**——**勝敗がまだ決まっていない帯での判別**＝こちらを主に読む。
3. **`sep` と `bias`（当てはめゼロの較正）**——`ΔW` の帳簿なら
   `Σ ≈ z − W₀` なので **`sep = 平均(Σ|勝) − 平均(Σ|負)` は 1 に、`bias = 平均(Σ) − (平均 z − 平均 W₀)` は 0** に
   なるべき。**係数を 1 つも当てはめずに較正を読める**（T119 の「判別と較正は別の 2 条件」の較正側）。

**`Δ log(T_opp/T_me)` は近道の式（`p/Θ`）ではなく、時計を作り直して引く**——
`curve` の読みでは `T = Θ/A` ではない（輪郭を歩く）ので、近道は成り立たない。
**法則の形（対数比の差）は読みに依らないので、そちらを実装する**。

## 検算の予告（§17.9.8・ここで測るもの）

* **P1**: `rel_flat` が `abs_flat` と `abs_kappa` の両方に勝つ（AUC・相関・両記録）
* **P2**: `rel_K` が P1 の判別力を落とさずに較正（`slope_fwd` が 1 へ）を改善する＝**判別と較正の両立**
* **P3**: `rel_exact` が `rel_K` に勝つのは**最後の自席ターンだけ**（他の帯では差が出ない）
* **P5**: **通貨の一律の付け替えで帳簿は 1 ビットも動いてはならない**（`Θ` と価格を同時に c 倍）
* **P7**: **両席の `A` に共通の掛け算誤差**を入れても相対の腕は動かない（`abs` の腕は動く）

**P4（守りと攻めが同じ `K` を共有するか）と P6（4 軸の重みの比）はこの母数では測れない**
——守りの窓の行が記録に無く（P8 待ち）、`a_opp` の軸もほぼ空（`card_effect_harm` が
【起動メイン】のほぼ全部に 0 を返す）。**測れないものは測れないと書く**。

使い方:

    python tests/scripts/relative_ledger.py --in <records_dir> [--games N] [--json out.json]
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

from opcg_sim.learned.train import plan_labels as PL  # noqa: E402
import crossing_bridge as CB  # noqa: E402
import kappa_vector as KV  # noqa: E402
import theory_rs as TR  # noqa: E402
from theory_bridge import POL_COLS, ROW_COLS, _extra  # noqa: E402
from theory_rs import MU, THETA  # noqa: E402

#: 腕の名前（出力の順序もこれ）
ARMS = ("abs_flat", "abs_kappa", "rel_flat", "rel_K", "rel_exact", "plac_K", "plac_shift")
#: `σ` の合わせ方（定数・Rust の `core::drv_rl`）
KAPPA_SIGMA_MODE = "match"


def calib_of(xs, zs, w0s):
    """**当てはめゼロの較正**（`ΔW` の帳簿なら `Σ ≈ z − W₀`）。

    `sep` = 平均(Σ|勝) − 平均(Σ|負)（**理想 1.0**）・`bias` = 平均(Σ) − (平均 z − 平均 W₀)（**理想 0**）。
    **係数を 1 つも当てはめない**ので、傾き（回帰）と違って尺度の取り替えに騙されない（T119）。

    **注意**: `sep = 1` が厳密に成り立つのは **`W₀` が勝ち負けで釣り合っているとき**
    （`sep = 1 − 平均(W₀|勝) + 平均(W₀|負)`）。**釣り合いも一緒に出す**（`w0_gap`）ので、
    `sep` を読むときは必ずそちらを見る。両席を同じ母数で積んでいるので構造的にはほぼ 0 になる。"""
    x = np.asarray(xs, float); z = np.asarray(zs, float); w0 = np.asarray(w0s, float)
    pos, neg = z > 0.5, z <= 0.5
    if not pos.any() or not neg.any():
        return {"sep": None, "bias": None}
    return {"sep": round(float(x[pos].mean() - x[neg].mean()), 4),
            "bias": round(float(x.mean() - (z.mean() - w0.mean())), 4),
            # `sep` の理想が 1 になる条件（勝ち負けで `W₀` が釣り合っているか）
            "w0_gap": round(float(w0[pos].mean() - w0[neg].mean()), 4)}


#: **不変量の検算に使う倍率**（値は何でもよい・法則が厳密に不変なので）。`P5`＝通貨／`P7`＝両席の速さ。
INV_C5, INV_C7 = 3.0, 1.3



def collect(dirs, limit_games=0, theta=THETA, mu=MU, scale_a=1.0, scale_currency=1.0, pre_settle=False):
    """記録を 1 度読んで **7 つの腕**を並べる（席×局ごとに積む）。

    `scale_a`（**P7**）は**両席の `A` に共通の掛け算誤差**を入れる／`scale_currency`（**P5**）は
    **耐久も価格も同時に c 倍**する＝どちらも**相対の腕は 1 ビットも動いてはならない**。

    **T138b**: `pre_settle=True` なら決着後（`lethal_rule.settled_map` が `True`）の行を読まない——
    **`before`（最後の自席ターンを外すだけの目分量）を、規則の決着点に差し替える**。
    1 局ぶんの計算は Rust の局の駆動（`core::drv_rl`・プラセボの前の局の `K` は局をまたいで渡す）。"""
    settled = None
    if pre_settle:
        import lethal_rule as LR
        settled = LR.settled_map(dirs, limit_games)
    prof = CB.profile_for(dirs)
    if not prof:
        raise ValueError("D_MODE=KV.D_MODE なのに損害の輪郭が引けない（%s）" % (dirs,))
    sr = CB.sigma_rel_for(dirs, slope="curve")
    if sr is None:
        raise ValueError("σ_rel が引けない＝黙って別の物差しに落とさない（T118 の規約）")
    TR.set_sigma_rel(sr)
    seat_decks = KV._seat_decks(dirs)     # **T128**: `A` の流入・効果はデッキの中身から出る
    arms = {k: [] for k in ARMS}
    # **決着帯を外した帯**（最後の自席ターンを落とす）＝勝敗がまだ決まっていない所での判別
    before = {k: [] for k in ARMS}
    # **P3**: 最後の自席ターンだけ（とどめの帯・厳密形が効くならここだけで効くはず）
    last = {k: [] for k in ARMS}
    zs, w0s = [], []
    rs = []
    stats = {"games": 0, "rows": 0, "priced": 0, "dead": 0, "last_turn_rows": 0, "capped": 0,
             "k_sum": 0.0, "dlog_abs_sum": 0.0, "by_family": {},
             # **P5／P7 を行ごとに厳密に検算する**（AUC の比較ではなく一致の検算）
             "inv": {"p5_n": 0, "p5_bad": 0, "p5_max": 0.0, "p5_bad_offcap": 0,
                     "p7_n": 0, "p7_bad": 0, "p7_max": 0.0, "p7_bad_offcap": 0}}
    part_rows = []                        # **T145**: 局ごとの `rel_K` の内訳
    prev_ks = []                          # プラセボ用: **前の局**の同じ行番号の `K`
    games = 0
    for game in PL.iter_games(dirs, row_cols=ROW_COLS, pol_cols=POL_COLS, extra_fn=_extra):
        games += 1
        if limit_games and games > limit_games:
            break
        seed = TR.seed_of(game)
        pin = {"decks": TR.deck_list(KV._deck_pair(seat_decks, seed)), "settled": TR.settled_in(settled, seed)}
        c = TR.cfg(theta, mu, prof=prof, sr=float(sr), scale_a=float(scale_a), scale_currency=float(scale_currency),
                   MIRROR_ME=bool(TR.RUN["MIRROR_ME"]), parts=bool(pre_settle))
        res = TR.game_call("relative_ledger", game, {"cfg": c, "in": pin, "stats": stats, "carry": {"prev_ks": prev_ks}})
        prev_ks = res["carry"]["prev_ks"]
        rs.extend(res["rs"])
        o = res["out"]
        if o is not None:
            zs.append(o["z0"])
            w0s.append(o["w0"])
            for nm, vals in ((arms, o["acc"]), (before, o["before"]), (last, o["last"])):
                for k, v in zip(nm, vals):
                    nm[k].append(v)
            if pre_settle:                # **T145**: `rel_K` の内訳（席 0 視点・宣言した行の旗だけ読む）
                part_rows.append((o["z0"], o["acc"][ARMS.index("rel_K")],
                                  {(int(w), bool(lt), bool(d)): v for w, lt, d, v in o["part"]}))
        new = res["stats"]
        stats.clear()
        stats.update(new)
    n = max(1, stats["priced"])
    out = {"games": stats["games"], "rows": stats["rows"], "priced": stats["priced"],
           "d_mode": KV.D_MODE, "sigma_rel": round(sr, 4),
           "kappa_sigma_mode": KAPPA_SIGMA_MODE, "attack_rest_mode": KV.ATTACK_REST_MODE,
           "pre_settle": bool(pre_settle),
           "scale_a": scale_a, "scale_currency": scale_currency,
           "dead_rows": stats["dead"], "dead_share": round(stats["dead"] / n, 4),
           "last_turn_rows": stats["last_turn_rows"],
           "capped_share": round(stats["capped"] / n, 4),
           "K_mean": round(stats["k_sum"] / n, 4),
           # `r` は片側の時計が 0 に行く行で発散するので**中央値**で読む（平均は尾に支配される）
           "r_median": round(float(np.median(rs)) if rs else 0.0, 4),
           "dlog_abs_mean": round(stats["dlog_abs_sum"] / n, 5),
           "w0_mean": round(float(np.mean(w0s)) if w0s else 0.5, 4),
           "by_family": stats["by_family"],
           # **P5／P7**: 行ごとの厳密な検算（`bad_offcap` が 0 なら「破れは打ち切りだけ」）
           "invariance": {k: (round(v, 12) if isinstance(v, float) else v)
                          for k, v in stats["inv"].items()},
           # 全帯（参考・決着帯の寄与が入る）＋**当てはめゼロの較正**
           "arms": {kk: dict(KV._score(arms[kk], zs), **calib_of(arms[kk], zs, w0s)) for kk in ARMS},
           # **主に読む帯**: 最後の自席ターンを外した＝勝敗がまだ決まっていない所
           "before": {kk: KV._score(before[kk], zs) for kk in ARMS},
           # **P3**: とどめの帯だけ（厳密形が効くならここだけで効くはず）
           "last_turn": {kk: KV._score(last[kk], zs) for kk in ARMS}}
    if pre_settle:
        out["parts"] = parts_of(part_rows)
    # **N-3**（値段の窓は `joint` だけ）
    out["cut_price"] = {"mode": TR.SW["CUT_PRICE_MODE"],
                        **{k: v for k, v in stats.items() if str(k).startswith("cut_")}}
    return out


def parts_of(part_rows):
    """**T145**: `rel_K` の局ごとの和を割った表。`part_rows` は `(z0, 全体の和, {(席, 最後か, 宣言か): 和})`。

    * `mean_winner_view` … 各部分の和を**勝者の視点**に直した平均（勝者の最後のターン・敗者の最後のターン・
      宣言した行〔勝者／敗者〕・それ以外）。**正なら勝者の側に積んでいる**。
    * `auc` … 全体の和から部分を引いた「腕」の AUC（`z` は席 0 の勝敗）:
      `all`（全行）／`minus_declared`（宣言した行を引く＝T138b の `--pre-settle on` と同じ行）／
      `minus_both_last`（両席の最後の自席ターンを引く＝`before` と同じ行）／
      `minus_winner_last`・`minus_loser_last`（片方の席の最後のターンだけ引く）。"""
    names = ("winner_last", "loser_last", "winner_declared", "loser_declared", "rest")
    sums = {k: [] for k in names}
    arms = {k: [] for k in ("all", "minus_declared", "minus_both_last", "minus_winner_last",
                            "minus_loser_last")}
    zs = []
    for z0, tot, part in part_rows:
        win = 0 if z0 > 0.5 else 1
        sg = 1.0 if win == 0 else -1.0                   # 席 0 視点 → 勝者視点
        wl = sum(v for (w, lt, _d), v in part.items() if w == win and lt)
        ll = sum(v for (w, lt, _d), v in part.items() if w != win and lt)
        wd = sum(v for (w, _lt, d), v in part.items() if w == win and d)
        ld = sum(v for (w, _lt, d), v in part.items() if w != win and d)
        rest = sum(v for (_w, lt, d), v in part.items() if not lt and not d)
        for k, v in zip(names, (wl, ll, wd, ld, rest)):
            sums[k].append(v * sg)
        zs.append(z0)
        arms["all"].append(tot)
        arms["minus_declared"].append(tot - wd - ld)
        arms["minus_both_last"].append(tot - wl - ll)
        arms["minus_winner_last"].append(tot - wl)
        arms["minus_loser_last"].append(tot - ll)
    return {"games": len(zs),
            "mean_winner_view": {k: round(float(np.mean(v)) if v else 0.0, 5) for k, v in sums.items()},
            "auc": {k: KV._score(v, zs).get("auc") for k, v in arms.items()}}



def build_parser():
    """**T134**: 引数の組み立てを分ける——**切替が器に無いと「動かなかった」を誤って読む**ので、
    テストから**同じ parser を見て**居ることを確かめられるようにする（本 T で実際に踏んだ）。"""
    ap = argparse.ArgumentParser(description="紐付けの法則を測る（T122）")
    ap.add_argument("--in", dest="src", nargs="+", required=True)
    ap.add_argument("--games", type=int, default=0)
    ap.add_argument("--theta-side", dest="theta_side", choices=TR.SWITCH_VALUES["THETA_SIDE_MODE"], default=None,
                    help="**T133**: `Θ` を両席で同じ式にするか（既定は現状の `legacy`）")
    ap.add_argument("--mirror", choices=("on", "off"), default=None,
                    help="**H-4g** 自分の耐久も相手と同じ守る側の計算で読む（既定 on）")
    ap.add_argument("--scale-a", type=float, default=1.0, help="**P7**: 両席の A に共通の掛け算誤差")
    ap.add_argument("--scale-currency", type=float, default=1.0, help="**P5**: 耐久と価格を同時に c 倍")
    ap.add_argument("--pre-settle", dest="pre_settle", default="off", choices=("off", "on"),
                    help="**T138b** 決着後（`lethal_rule.settled_map`）の行を除いて測るか")
    ap.add_argument("--json", default="")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    if a.theta_side:
        CB.set_theta_side_mode(a.theta_side)          # **T133**
    if a.mirror:
        TR.set_switch("MIRROR_ME", a.mirror == "on")  # **H-4g**
    out = collect(a.src, a.games, scale_a=a.scale_a, scale_currency=a.scale_currency,
                  pre_settle=(a.pre_settle == "on"))
    out["rule_stats"] = dict(CB.RULE_STATS)               # **H-4g**: 使った計画ごとの地平の縮み
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
