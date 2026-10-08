#!/usr/bin/env python3
"""**局面の傾き `κ` をスカラーからベクトル（勾配）にする**（T121・2026-09-20・ユーザ指示「Gをお願いします」）。

## 問い

帳簿（線形の橋）は **手の値段 × κ(局面)** を積んで勝敗を順序づけようとしている。
その `κ` は `w(D)/w̄`＝**勝率曲線の導関数**なので、**形は連鎖律**である:

    ΔW = W(s') − W(s) ≈ F'(x) · Δx = κ · v(a)          （x は「有利さ」のスカラー）

**だからこの形は 3 つの仮定を置いている**:

1. **勝率が 1 本のスカラー `x` の関数である**（十分統計量が 1 次元）
2. **1 手の変化が小さい**（一次近似が効く）
3. **すべての通貨が同じ 1 本の軸を、同じ換算率で動かす**

**このうち 1 と 3 は既に反証されている**（測る前に言える）:

* **1 の反証（T118）**: `D` が十分統計量なら**`D` の単調変換はどれも識別力を変えられない**。
  ところが `D/(σ_rel·s)` にすると **AUC 0.6924 → 0.7625**。**`D` の外に情報がある**＝十分統計量は 1 次元でない。
* **3 の反証（T111）**: **速さ `A` と耐久 `Θ` の交換レートは `1/τ`＝局面の量**。
  1 つのスカラー `κ` では**2 つの違う換算率を表せない**。

## 式（新定数ゼロ・連鎖律を正しく書くだけ）

`D` は **2 つの到達時刻の差**（T90・T75）＝ `D = τ(私が死ぬまで) − τ(相手が死ぬまで)`。
`D` を軸ごとに偏微分して、手の量をその軸に載せる:

    ΔW ≈ w(D) · Σ_i (∂D/∂x_i) · Δx_i

**スカラー `κ` は「成分すべてが同じ値」と置いた特別な場合**にすぎない。

## **`D` の読み方が 2 つあり、生きている軸の数が違う**（本器の骨）

| 読み | `τ` の出し方 | 生きている軸 |
|---|---|---|
| **`curve`（帳簿の正本・T75）** | **実測の損害の輪郭**（`tests/fixtures/harm_profile.json`）を `Θ` に届くまで積む | **`Θ_me` と `Θ_opp` の 2 本だけ** |
| `theory`（積み上がる歩き・T127） | 状態から出す加速で `Θ` に届くまで歩く（表を使わない） | **4 本**（`Θ` 2 本 ＋ `A` 2 本） |

（波C・2026-10-05: `clock`〔`τ = min(CAP, Θ/A)`・T90〕と `curve_scaled`〔輪郭をその席の `A` で伸縮・T126〕は削除——
凍結ブランチ `claude/theory-switches-final` で再現する。）

**これが本器の一番大事な観測**——**帳簿の `D` には速さの軸が存在しない**。輪郭は
**両席共通の固定の表**なので、体を出しても付与しても `τ` は 1 ターンも動かない。
だから帳簿は「自分の速さを上げる手」を**原理的に値付けできない**（`share_probe` が
`RATE_DON_MODE` を切り替えても帳簿が 1 桁も動かないことを示したのと同じ事実の裏側）。
本器は**その割合を数える**（`dead_share`）。

**`curve` の中でも `κ` はスカラーではない**: `∂D/∂Θ_me` と `∂D/∂Θ_opp` は
**別々の τ における輪郭の高さの逆数**（`1/prof[j+τ_me]` と `1/prof[j+τ_opp]`）なので**値が違う**。
**スカラー `κ` はこの 2 つを 1 つに潰している**。

## 手をどの軸に載せるか（型ごと・規則から）

| 型 | 動かす軸 | Δx の単位 |
|---|---|---|
| 攻撃 | **`Θ_opp` を削る**（＋`ATTACK_REST_MODE=return`なら**ブロッカーが攻めてレストになる分を`Θ_me`から戻る側へ移す**・C-5c） | 価格（そのまま） |
| 出す | **`A_me` を上げる** | **1 ターンあたり**（`attack_value`＝体の毎ターンの攻撃の価値） |
| 付与 | **`A_me` を上げる** | 1 ターンあたり（`attack_value(p+1000k) − attack_value(p)`） |
| 効果・除去 | **`Θ_opp` を削る ＋ `A_opp` を下げる** | 価格 ＋ 1 ターンあたり |
| 効果・その他 | **`A_me` を上げる** | 1 ターンあたり（価格 ÷ 残りターン） |
| 守り | **`Θ_me` を守る** | 価格（そのまま） |

**単位の変換がここの本体**——今の価格は「速さの軸の量」を **`ko_p` の幾何和（一定率）**で価格へ換算して
1 本にまとめている。**正しい率は `1/τ`（局面依存・T111）**。だから本器は**換算をやめ、軸ごとに勾配を掛ける**。

## 仮定 2（一次近似）も測る——**線形化をやめた腕**を並べる

勾配は「1 手の変化が小さい」ことを前提にする。**序盤はそれが成り立たない**（体を 1 つ出すと
`A_me` が倍になる＝`τ` が半分になる）。**そこで同じ式のまま差分を正確に取る腕も並べる**（新定数ゼロ）:

* `exact` … `D(x+Δx) − D(x)`（**勾配を使わず `D` を 2 回計算する**）
* `exactw` … `W(D(x+Δx)) − W(D(x))`（**一次近似をどこにも使わない**・物差しは T118 の `σ_rel·s`）

**この 3 つは同じ式の近似の深さだけが違う**（`vector` ⊂ `exact` ⊂ `exactw`）ので、
差が出たらそれは**「どの仮定が壊れているか」の切り分けになる**。

## 5 つの腕と 2 つのプラセボ

* `flat` … `κ = 1`（**重みを知らないなら 1**＝最小仮定）
* `scalar` … 今の形 `w(D)/w̄`（**帳簿の正本**）
* **`vector`** … 上の勾配（一次）
* **`exact`** … 正確な `ΔD`
* **`exactw`** … 正確な `ΔW`
* プラセボ A **軸の入れ替え**（同じ大きさ・**軸だけ決定的に置換**）＝勝ちが「軸の意味」から来ているかの対照
* プラセボ B **勾配の大きさだけ**（成分の平均を全軸に使う＝スカラーと同じだが正規化だけ変える）

**判定は尺度に依らない量（AUC・相関）で行う**——傾き（較正）は正規化で動くので主にしない（T119）。

使い方:

    python tests/scripts/kappa_vector.py --in <records_dir> [--games N] [--json out.json]
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
import theory_rs as TR  # noqa: E402
from theory_bridge import POL_COLS, ROW_COLS, _extra  # noqa: E402
from theory_rs import MU, THETA  # noqa: E402

#: **C-5c**: 攻めたブロッカーは消さずに戻る側へ移す（定数・Rust の `core::drv_kv`）
ATTACK_REST_MODE = "return"
#: `D` の読み（`curve`＝帳簿の正本・損害の輪郭）。`theory`（積み上がる歩き・T127）は Rust に移していない
#: （段 7・凍結ブランチ `claude/theory-switches-final` で再現する）
D_MODE = "curve"
LIVE_AXES = {"curve": ("th_me", "th_opp")}


def _seat_decks(dirs):
    """記録から `{seed: (デッキ p1, デッキ p2)}` を引く（**T128**・`A` の流入と効果の材料）。

    **橋と同じ解決**（`crossing_bridge.collect`）。引けなければ `None` を返さず**落とす**
    ——黙って `A` が盤面だけに落ちると、伸びが丸ごと消えたまま数字が出てしまう（2026-09-20 の実害）。"""
    import deck_refill as DR
    sd = DR.decks_by_seed(dirs)
    if not sd:
        raise ValueError("デッキが引けない＝`A` の流入・効果の項が作れない（%s）" % (dirs,))
    return sd


def _deck_of(seat_decks, seed, w):
    """`(seed, 席)` → その席のデッキの card id 列（無ければ `None`）。"""
    if not seat_decks:
        return None
    return (seat_decks.get(int(seed)) or (None, None))[int(w)]


def _deck_pair(seat_decks, seed):
    """**N-3**: `(席 0 のデッキ, 席 1 のデッキ)`（無ければ `None`）。"""
    if not seat_decks:
        return None
    return (_deck_of(seat_decks, seed, 0), _deck_of(seat_decks, seed, 1))


def auc_of(scores, labels):
    s = np.asarray(scores, float); y = np.asarray(labels, float)
    pos, neg = (y > 0.5), (y <= 0.5)
    if not pos.any() or not neg.any():
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float)
    ss = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    n_p, n_n = int(pos.sum()), int(neg.sum())
    return float((ranks[pos].sum() - n_p * (n_p + 1) / 2.0) / (n_p * n_n))


def _score(xs, zs):
    x = np.asarray(xs, float); z = np.asarray(zs, float)
    if len(x) < 10 or x.var() <= 0.0:
        return {"n": len(x), "auc": None, "corr": None, "slope": None}
    xd = x - x.mean()
    a = auc_of(x, z)
    return {"n": int(len(x)),
            "auc": (round(a, 4) if a is not None else None),
            "corr": round(float(np.corrcoef(x, z)[0, 1]), 4),
            "slope": round(float((xd * (z - z.mean())).sum() / (xd * xd).sum()), 5)}


def collect(dirs, limit_games=0, theta=THETA, mu=MU):
    """記録を 1 度読んで **5 つの腕 ＋ 2 つのプラセボ**を並べる（席×局ごとに積む）。
    1 局ぶんの計算（`A`・`Θ`・勾配・腕の積み上げ）は Rust の局の駆動（`core::drv_kv`）。"""
    prof = CB.profile_for(dirs)
    if not prof:
        raise ValueError("D_MODE=D_MODE なのに損害の輪郭が引けない（%s）" % (dirs,))
    # **物差し（T118）**: `W_ERR_MODE=rel` なら `σ_rel × s(τ_me, τ_opp)`。**別のセットの値を使う**（§0.1 条件 1）。
    sr = None
    if TR.CLOCK["W_ERR_MODE"] == "rel":
        sr = CB.sigma_rel_for(dirs, slope="curve")
        if sr is None:
            raise ValueError("W_ERR_MODE=rel なのに σ_rel が引けない＝黙って abs に落とさない")
        TR.set_sigma_rel(sr)
    # **`κ = w(D)/w̄` の分子の `σ_T` と分母 `w̄`**（`scalar` の腕・`w(D)` を使う `vector`／`exact`／`exactw` の腕も同じ `σ_T`）を
    # `theory_bridge`（`w̄` を測った器）と同じ 1 つの読み込みで表から引く（2026-10-08・引けない／出所が合わなければ落ちる）
    kc = CB.kappa_clock(dirs)
    w_bar = kc["w_bar"]
    seat_decks = _seat_decks(dirs)
    arms = {k: [] for k in ("flat", "scalar", "vector", "exact", "exactw",
                            "plac_axis", "plac_mag")}
    zs = []
    stats = {"games": 0, "rows": 0, "priced": 0, "dead": 0, "dead_v": 0.0, "live_v": 0.0,
             "tau_me_sum": 0.0, "tau_opp_sum": 0.0, "g_me_sum": 0.0, "g_opp_sum": 0.0,
             "by_family": {}, "by_axis": {}}
    games = 0
    for game in PL.iter_games(dirs, row_cols=ROW_COLS, pol_cols=POL_COLS, extra_fn=_extra):
        games += 1
        if limit_games and games > limit_games:
            break
        pin = {"decks": TR.deck_list(_deck_pair(seat_decks, TR.seed_of(game)))}
        res = TR.game_call("kappa_vector", game, {"cfg": TR.cfg(theta, mu, prof=prof, kappa_stats=True), "in": pin, "stats": stats,
                                                  "carry": {}})
        new = res["stats"]
        stats.clear()
        stats.update(new)
        out = res["out"]
        if out is None:
            continue
        zs.append(out["z0"])
        for k, v in zip(arms, out["acc"]):
            arms[k].append(v)
    n = max(1, stats["priced"])
    out = {"games": stats["games"], "rows": stats["rows"], "priced": stats["priced"],
           "d_mode": D_MODE, "live_axes": list(LIVE_AXES[D_MODE]), "attack_rest_mode": ATTACK_REST_MODE,
           # **その読みに軸が無い手の割合**（件数と価格の絶対値・帳簿が値付けできない分）
           "dead_rows": stats["dead"], "dead_share": round(stats["dead"] / n, 4),
           "dead_value_share": round(stats["dead_v"] / max(1e-12, stats["dead_v"] + stats["live_v"]), 4),
           # **2 つの耐久の軸の重みが違うこと**（スカラー κ はこれを 1 つに潰している）
           "grad_th_me_mean": round(stats["g_me_sum"] / n, 4),
           "grad_th_opp_mean": round(stats["g_opp_sum"] / n, 4),
           "tau_me_mean": round(stats["tau_me_sum"] / n, 4),
           "tau_opp_mean": round(stats["tau_opp_sum"] / n, 4),
           "sigma_rel": (round(sr, 4) if sr is not None else None),
           "w_err_mode": TR.CLOCK["W_ERR_MODE"],
           # `κ` の分母と、`κ` の行の平均（定義上の値は 1）
           "w_bar": round(w_bar, 4), "sigma_t": round(kc["sigma_t"], 4),
           "kappa_mean": round(stats.get("kappa_sum", 0.0) / n, 4),
           "by_family": stats["by_family"], "by_axis": stats["by_axis"],
           "arms": {kk: _score(arms[kk], zs) for kk in arms}}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="κ をベクトル（勾配）にする（T121）")
    ap.add_argument("--in", dest="src", nargs="+", required=True)
    ap.add_argument("--games", type=int, default=0)
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)
    out = collect(a.src, a.games)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
