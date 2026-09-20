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
| `clock`（交点の橋の静的な形・T90） | `τ = min(CAP, Θ/A)` | **4 本**（`Θ` 2 本 ＋ `A` 2 本） |

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
| 攻撃 | **`Θ_opp` を削る** | 価格（そのまま） |
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

    python tests/scripts/kappa_vector.py --in <records_dir> [--games N] [--d-mode curve|clock] [--json out.json]
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

from opcg_sim.learned.train import plan_labels as PL  # noqa: E402
import crossing_bridge as CB  # noqa: E402
import guard_afford as GA  # noqa: E402
import theory_order as TO  # noqa: E402
from theory_bridge import POL_COLS, ROW_COLS, _extra, _state_of, move_family  # noqa: E402
from theory_order import (MU, SC_MY_DON, SC_MY_LEADER_POWER, SC_MY_LIFE, SC_OPP_LEADER_POWER,  # noqa: E402
                          SC_OPP_LIFE, THETA, opp_bodies_of, own_attackers_of, score_candidate,
                          slot_power, theta_of)

#: 速さの床（`crossing_bridge.SLOPE_FLOOR` と同じ意味・0 割りを避ける）
A_FLOOR = CB.SLOPE_FLOOR
#: **歩きの打ち切り**（`crossing_bridge.RACE_CAP`・既存の定数）。`A → 0` の行で `τ` が数百ターンに飛ぶのを止める。
TAU_CAP = CB.RACE_CAP
#: 軸の名前（順序は固定・プラセボの置換もこの順で回す）
AXES = ("th_me", "th_opp", "a_me", "a_opp")

#: **`D` の読み方**（上の表）。`curve`＝**帳簿の正本**（輪郭・軸は 2 本）／`clock`＝2 本の時計（軸は 4 本）。
D_MODES = ("curve", "curve_scaled", "clock")
D_MODE = "curve"
#: 読みごとに**生きている軸**。`curve` に速さの軸は**存在しない**（輪郭は固定の表）。
#: **`curve_scaled` は 4 本とも生きている**——輪郭をその席の `A` で伸縮するので速さが時計に入る。
LIVE_AXES = {"curve": ("th_me", "th_opp"), "curve_scaled": AXES, "clock": AXES}

#: **`curve_scaled` の分母**（`crossing_bridge.profile_th_for` が返す理論の速さの輪郭）。
#: 読む側が `set_profile_th` で入れる。**入っていないのに `curve_scaled` を頼んだら落ちる**
#: （黙って `curve` に落ちない＝T118 の規約と同じ）。
PROFILE_TH = None


def set_profile_th(prof_th):
    global PROFILE_TH
    PROFILE_TH = list(prof_th) if prof_th else None
    return PROFILE_TH


def profile_scale(rate, j, prof_th=None):
    """**輪郭をその席の速さで伸縮する倍率**＝`A / prof_th[j]`（橋の `curve_scaled` と同じ式・T126）。

    **これが入ると輪郭の読みが尺度不変になる**——輪郭は比 `prof/prof_th`（無次元）としてしか入らず、
    **単位は `A` が持つ**ので、通貨を c 倍すれば 1 ターンの損害も `Θ` も同じだけ c 倍になる
    （T122 の P5 が `curve` で 96.2% 破れていた患部）。
    **`prof_th[0] = 0` を分母にしてはいけない**（T126 で踏んだ）——`RATE_T1_MODE=on` は
    「最初の自席ターンは 1 本も打てない」という**規則**なので、そこに典型の速さは**存在しない**。
    床 `SLOPE_FLOOR` で割ると倍率が 55 倍まで飛び、**5.0% の行が打ち切りに貼り付いた**（実測）。
    **規則どおりに「速さが定義される最初のターン」まで進めて割る**（新定数ゼロ）。"""
    th = prof_th if prof_th is not None else PROFILE_TH
    if not th:
        raise ValueError("curve_scaled には理論の速さの輪郭が要る（profile_th_for）")
    i = min(max(0, int(j)), len(th) - 1)
    while i < len(th) - 1 and float(th[i]) <= CB.SLOPE_FLOOR:
        i += 1                                  # 速さが定義される最初のターンまで進める
    return float(rate) / max(CB.SLOPE_FLOOR, float(th[i]))


def set_d_mode(name):
    global D_MODE
    if name not in D_MODES:
        raise ValueError("D_MODE は %s のいずれか（%r）" % (D_MODES, name))
    D_MODE = name


def tau_of(theta, rate):
    """**片側の時計** `τ = min(CAP, Θ/A)`（`clock` の読み・T90 の `static`）。"""
    return min(float(TAU_CAP), float(theta) / max(A_FLOOR, float(rate)))


def d_of(st, prof=None):
    """**2 つの到達時刻の差** `D = τ(私が死ぬまで) − τ(相手が死ぬまで)`（正なら自分が先に届く）。

    `st` は `(Θ_me, Θ_opp, A_me, A_opp, j)`。`j` は自席ターン番号（輪郭の読み出し位置）。
    `curve` は `crossing_bridge.tau_from_profile`（**帳簿が使っているのと同じ関数**）。"""
    th_me, th_opp, a_me, a_opp, j = st
    if D_MODE in ("curve", "curve_scaled"):
        if prof is None:
            raise ValueError("D_MODE=%s には損害の輪郭が要る（profile_for）" % D_MODE)
        # **T126**: `curve_scaled` は**その席の速さ**で輪郭を伸縮する（相手の時計は相手の `A` で）
        s_me = s_opp = 1.0
        if D_MODE == "curve_scaled":
            s_me = profile_scale(a_opp, j)      # 自分が死ぬまで＝**相手**が削る速さ
            s_opp = profile_scale(a_me, j)      # 相手が死ぬまで＝**自分**が削る速さ
        return float(CB.tau_from_profile(max(0.0, float(th_me)), int(j), prof, s_me)
                     - CB.tau_from_profile(max(0.0, float(th_opp)), int(j), prof, s_opp))
    return tau_of(th_me, a_opp) - tau_of(th_opp, a_me)


def apply_dx(st, dx):
    """`(Θ_me, Θ_opp, A_me, A_opp, j)` に `Δx` を足す（**耐久は 0 未満に、速さは床未満にしない**＝規則）。"""
    th_me, th_opp, a_me, a_opp, j = st
    return (max(0.0, float(th_me) + float(dx.get("th_me", 0.0))),
            max(0.0, float(th_opp) + float(dx.get("th_opp", 0.0))),
            max(A_FLOOR, float(a_me) + float(dx.get("a_me", 0.0))),
            max(A_FLOOR, float(a_opp) + float(dx.get("a_opp", 0.0))), j)


def grad_clock(th_me, th_opp, a_me, a_opp):
    """**`clock` の読みの勾配**（4 成分・解析・新定数ゼロ）。`D = Θ_me/A_opp − Θ_opp/A_me` の偏微分そのまま。

        ∂D/∂Θ_me  = +1/A_opp          自分の耐久を 1 守る
        ∂D/∂Θ_opp = −1/A_me           相手の耐久を 1 削る（削る＝ Δ<0 なので寄与は +）
        ∂D/∂A_me  = +Θ_opp/A_me²      自分の速さを 1 上げる
        ∂D/∂A_opp = −Θ_me/A_opp²      相手の速さを 1 下げる

    **符号は「その軸の量が増えたとき `D` がどう動くか」**なので、使う側は「削る」なら
    `Δx < 0` を渡す（符号を 2 回付けない）。**打ち切りに当たっている側は 0**（`τ` がもう動かない）。"""
    am = max(A_FLOOR, float(a_me)); ao = max(A_FLOOR, float(a_opp))
    live_me = float(th_me) / ao < float(TAU_CAP)
    live_opp = float(th_opp) / am < float(TAU_CAP)
    return {"th_me": (1.0 / ao) if live_me else 0.0,
            "th_opp": (-1.0 / am) if live_opp else 0.0,
            "a_me": (float(th_opp) / (am * am)) if live_opp else 0.0,
            "a_opp": (-float(th_me) / (ao * ao)) if live_me else 0.0}


def grad_of(st, prof=None):
    """**`D` の勾配**（読みに応じて）。`clock` は解析（`grad_clock`）、`curve` は**中心差分**。

    `curve` の `τ` は輪郭に沿った区分線形なので、中心差分は**その `τ` における輪郭の高さの逆数**
    `1/prof[j+τ]` をそのまま返す（解析と同じ値・刻みは数値の都合で式の定数ではない）。
    **生きていない軸は 0**（`curve` に速さの軸は存在しない）。"""
    th_me, th_opp, a_me, a_opp, j = st
    if D_MODE == "clock":
        return grad_clock(th_me, th_opp, a_me, a_opp)
    g = {k: 0.0 for k in AXES}
    axes = (("th_me", 0), ("th_opp", 1)) if D_MODE == "curve" else (
        ("th_me", 0), ("th_opp", 1), ("a_me", 2), ("a_opp", 3))
    for k, i in axes:
        x = float(st[i])
        h = max(1e-6, 1e-4 * max(1.0, abs(x)))
        up = list(st); up[i] = x + h
        dn = list(st); dn[i] = max(0.0, x - h)
        g[k] = (d_of(tuple(up), prof) - d_of(tuple(dn), prof)) / (up[i] - dn[i])
    return g


def rate_of_row(sc, tok, ci_row, idx2cid, cards, theta=THETA, mu=MU):
    """その席の **A**（1 自席ターンに積む損害）。**橋と同じ `seat_slope`**（`SLOPE_*` の既定に従う）。

    **必ずその席のターンの「最初の行」で呼ぶ**——ターン途中の行では殴り終わった体の
    `CAN_ATTACK` が落ちていて `A` がほぼ 0 になり、`τ = Θ/A` が数百ターンに飛ぶ（実測）。
    橋も `turn_start[(w, t)]` で読んでいる（`crossing_bridge.collect`）。"""
    sc = np.asarray(sc); tok = np.asarray(tok)
    olp = float(sc[SC_OPP_LEADER_POWER]) * 1e4 or 5000.0
    return float(CB.seat_slope(sc, tok, ci_row, idx2cid, cards, olp, theta, mu))


def g_of_row(sc, tok, ci_row, idx2cid, cards):
    """**その席の手札 1 枚あたりの価格**（T76／T79・帳簿の `_g_of_row` と同じ式）。
    `THETA_HAND_MODE` が `count` なら `None`（＝`μ`）。"""
    part = CB.THETA_HAND_PART[CB.THETA_HAND_MODE]
    if part is None:
        return None
    return CB.hand_price_mean(sc, tok, ci_row, idx2cid, cards, part=part)


def state_of_row(sc, tok, a_me, a_opp, j, g_me=None, g_opp=None):
    """行から **(Θ_me, Θ_opp, A_me, A_opp, j)** を組む（両席・完全情報・§0.05）。

    `Θ` は**その行**から両席分読める（`threshold` と `threshold_of_me` が対の式）。
    **手札 1 枚あたりの価格は両席それぞれの手札から**（T79・`curve_d_of_row` と同じ渡し方）
    ——`scalar` の腕を**帳簿の `κ` そのもの**にするために要る（`μ` で代用すると別物になる）。
    `A` は**両席ぶんをターンの最初の行から**渡してもらう（`rate_of_row` の注意書き）。"""
    sc = np.asarray(sc); tok = np.asarray(tok)
    return (float(CB.threshold_of_me(sc, tok, g_hand=g_me)),
            float(CB.threshold(sc, tok, g_hand=g_opp)),
            float(a_me), float(a_opp), int(j))


def axis_of_move(fam, v, sig, cid, cards, sc, tok, olp, r_turns, don_k=0):
    """**その手が動かす軸と量**を返す（`{軸: Δx}`・規則と型から）。

    **単位に注意**——`th_*` は価格そのまま、`a_*` は**1 ターンあたり**。
    価格 `v` は「速さの軸の量」を一定率で価格へ換算済みなので、`a_*` の軸では**換算前の量を作り直す**
    （`attack_value`＝毎ターンの攻撃の価値）。"""
    out = {}
    v = float(v)
    if fam == "attack":
        out["th_opp"] = -v                      # 相手の耐久を削る（価格の単位のまま）
    elif fam == "play":
        info = cards.info(cid) if (cards is not None and cid) else None
        p = float((info or {}).get("power") or 0.0)
        if p > 0.0:
            # 出した体が**毎ターン**出す攻撃の価値（＝`theory_slope_parts` がこの体に足す分）
            out["a_me"] = float(TO.attack_value(p, olp, True, THETA, MU))
        else:
            out["a_me"] = v / max(1.0, float(r_turns))     # 体でない札は残りターンで割る（過小側）
    elif fam == "attach":
        # 付与 k 枚＝一番強い攻め手のパワーが +1000k（規則）
        xs = own_attackers_of(tok, olp)
        p = (max(xs) + olp) if xs else olp
        k = max(0, int(don_k))
        out["a_me"] = float(TO.attack_value(p + 1000.0 * k, olp, True, THETA, MU)
                            - TO.attack_value(p, olp, True, THETA, MU))
    elif fam == "effect":
        import deck_refill as DR
        mlp = float(np.asarray(sc)[SC_MY_LEADER_POWER]) * 1e4 or 5000.0
        harm = float(DR.card_effect_harm(cid, mlp, r_turns) or 0.0) if cid else 0.0
        if harm > 0.0:
            out["th_opp"] = -harm               # 除去は相手の耐久を削り
            out["a_opp"] = -harm / max(1.0, float(r_turns))   # 相手の速さも下げる（体が減る）
        else:
            out["a_me"] = v / max(1.0, float(r_turns))        # 引く・探す・付与は自分の速さ
    elif fam == "guard":
        # 自分の耐久を守る（`g` は既に正味）。**本器の母数からは来ない**——`move_family` は `guard` を
        # 返さない（守りは相手ターンの窓で、記録に「実際の攻撃」が無い＝P8 待ち）。表を完成させて
        # 置き、テストが直接呼んで軸を固定する。
        out["th_me"] = v
    else:
        out["th_opp"] = -v                      # 型が読めない手は耐久の軸へ（従来と同じ扱い）
    return out


def _perm_axes(d, shift=1):
    """**プラセボ A**: 大きさはそのまま・**軸だけ決定的に回す**（意味を壊す対照）。"""
    keys = list(AXES)
    return {keys[(keys.index(k) + shift) % len(keys)]: val for k, val in d.items()}


def dot(grad, dx):
    return float(sum(float(grad.get(k, 0.0)) * float(v) for k, v in dx.items()))


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
    """記録を 1 度読んで **5 つの腕 ＋ 2 つのプラセボ**を並べる（席×局ごとに積む）。"""
    cards = PL.Cards()
    idx2cid = {i: c for c, i in GA._vocab().items()}
    prof = CB.profile_for(dirs)
    if D_MODE in ("curve", "curve_scaled") and not prof:
        raise ValueError("D_MODE=D_MODE なのに損害の輪郭が引けない（%s）" % (dirs,))
    if D_MODE == "curve_scaled":
        # **T126**: 輪郭をその席の `A` で伸縮する読み＝分母が要る（引けなければ落ちる）
        _th = CB.profile_th_for(dirs)
        if not _th:
            raise ValueError("curve_scaled なのに理論の速さの輪郭が引けない（%s）" % (dirs,))
        set_profile_th(_th)
    # **物差し（T118）**: `W_ERR_MODE=rel` なら `σ_rel × s(τ_me, τ_opp)`。**別のセットの値を使う**（§0.1 条件 1）。
    sr = None
    if TO.W_ERR_MODE == "rel":
        sr = CB.sigma_rel_for(dirs, slope="curve")
        if sr is None:
            raise ValueError("W_ERR_MODE=rel なのに σ_rel が引けない＝黙って abs に落とさない")
        TO.set_sigma_rel(sr)
    arms = {k: [] for k in ("flat", "scalar", "vector", "exact", "exactw",
                            "plac_axis", "plac_mag")}
    zs = []
    stats = {"games": 0, "rows": 0, "priced": 0, "dead": 0, "dead_v": 0.0, "live_v": 0.0,
             "tau_me_sum": 0.0, "tau_opp_sum": 0.0, "g_me_sum": 0.0, "g_opp_sum": 0.0,
             "by_family": {}, "by_axis": {}}
    games = 0
    for r, pol, ex, L, ptr, idx in PL.iter_games(dirs, row_cols=ROW_COLS, pol_cols=POL_COLS,
                                                 extra_fn=_extra):
        games += 1
        if limit_games and games > limit_games:
            break
        stats["games"] += 1
        acc = {k: 0.0 for k in arms}
        z_of = {}
        # **1 周目**: 席ごとに「その自席ターンの**最初の**決定行」から `A` を作る（橋と同じ `turn_start`）。
        rate_at_turn, g_at_turn = {}, {}
        for i in idx:
            if int(r["kind"][i]) != 0:
                continue
            w, t = int(r["who"][i]), int(r["turn"][i])
            if PL.is_own_turn(w, t) and (w, t) not in rate_at_turn:
                rate_at_turn[(w, t)] = rate_of_row(ex["sc"][i], ex["tok"][i], ex["ci"][i],
                                                   idx2cid, cards, theta, mu)
                g_at_turn[(w, t)] = g_of_row(ex["sc"][i], ex["tok"][i], ex["ci"][i], idx2cid, cards)

        def _opp_at(w, t):
            """相手の **(A, g)**＝相手の**直近の自席ターンの最初の行**から読んだもの。無ければ `None`。"""
            ts = [tt for (ww, tt) in rate_at_turn if ww == 1 - w and tt < t]
            if not ts:
                return None
            key = (1 - w, max(ts))
            return rate_at_turn[key], g_at_turn[key]
        for i in idx:
            z = float(r["z"][i])
            if z != 0.0:
                z_of[int(r["who"][i])] = 1.0 if z > 0 else 0.0
            if int(r["kind"][i]) != 0:
                continue
            w, t = int(r["who"][i]), int(r["turn"][i])
            if not PL.is_own_turn(w, t):
                continue
            k = int(L[i]); ch = int(r["pol_chosen"][i])
            if k < 1 or ch < 0 or ch >= k:
                continue
            stats["rows"] += 1
            pair = _opp_at(w, t)
            if pair is None:
                continue                       # 相手がまだ 1 ターンも打っていない（橋と同じ扱い）
            ao, g_opp = pair
            sc, tok, ci = ex["sc"][i], ex["tok"][i], ex["ci"][i]
            b = int(ptr[i]) + ch
            sig = json.loads(pol["pol_sig"][b])
            fam = move_family(sig)
            st0 = state_of_row(sc, tok, rate_at_turn[(w, t)], ao, CB.own_turn_index(t),
                               g_me=g_at_turn[(w, t)], g_opp=g_opp)
            rt = max(1.0, min(5.0, float(np.asarray(sc)[SC_OPP_LIFE])))
            th = theta_of(tok, float(np.asarray(sc)[SC_MY_LIFE]), float(np.asarray(sc)[SC_MY_DON]),
                          mode="const", theta=theta)
            olp = float(np.asarray(sc)[SC_OPP_LEADER_POWER]) * 1e4 or 5000.0
            mlp = float(np.asarray(sc)[SC_MY_LEADER_POWER]) * 1e4 or 5000.0
            ctx = {"theta": th, "mu": mu, "opp_leader_power": olp, "my_leader_power": mlp,
                   "r_turns": rt, "don_k": 1, "attackers": own_attackers_of(tok, olp),
                   "don_active": float(np.asarray(sc)[SC_MY_DON]),
                   "st": _state_of(sc, ci, idx2cid),
                   "opp_bodies": opp_bodies_of(tok, mlp, rt, th, mu, ci_row=ci, idx2cid=idx2cid)}
            tl = sig[2] if len(sig) > 2 else None
            v = score_candidate(sig, str(pol["pol_cid"][b]) or None,
                                (str(pol["pol_tcid"][b]) or None) if tl else None, ctx, cards,
                                src_power=slot_power(tok, int(pol["pol_si"][b])),
                                tgt_power=slot_power(tok, int(pol["pol_ti"][b])),
                                don_k=int(pol["pol_k"][b]))
            if v is None:
                continue
            stats["priced"] += 1
            stats["by_family"][fam] = stats["by_family"].get(fam, 0) + 1
            d = d_of(st0, prof)
            wd = TO.w_of_d(d)
            grad = grad_of(st0, prof)
            dx = axis_of_move(fam, v, sig, str(pol["pol_cid"][b]) or None, cards, sc, tok, olp, rt,
                              don_k=int(pol["pol_k"][b]))
            live = LIVE_AXES[D_MODE]
            if not any(ax in live for ax in dx):
                # **その読みに軸が無い手**＝帳簿が原理的に値付けできない手（数えて報告する）
                stats["dead"] += 1
                stats["dead_v"] += abs(float(v))
            else:
                stats["live_v"] += abs(float(v))
            for ax in dx:
                stats["by_axis"][ax] = stats["by_axis"].get(ax, 0) + 1
            stats["tau_me_sum"] += tau_of(st0[0], st0[3])
            stats["tau_opp_sum"] += tau_of(st0[1], st0[2])
            stats["g_me_sum"] += abs(grad["th_me"]); stats["g_opp_sum"] += abs(grad["th_opp"])
            sgn = 1.0 if w == 0 else -1.0                    # 席 0 の視点で積む（帳簿と同じ）
            acc["flat"] += float(v) * sgn
            acc["scalar"] += float(v) * sgn * float(TO.state_factor(d, "curve"))
            acc["vector"] += wd * dot(grad, dx) * sgn
            # **線形化をやめた 2 本**（同じ式・近似の深さだけが違う）
            st1 = apply_dx(st0, dx)
            d1 = d_of(st1, prof)
            acc["exact"] += wd * (d1 - d) * sgn
            t0 = (tau_of(st0[0], st0[3]), tau_of(st0[1], st0[2]))
            t1 = (tau_of(st1[0], st1[3]), tau_of(st1[1], st1[2]))
            acc["exactw"] += (TO.prob_of_d(d1, t_me=t1[0], t_opp=t1[1])
                              - TO.prob_of_d(d, t_me=t0[0], t_opp=t0[1])) * sgn
            acc["plac_axis"] += wd * dot(grad, _perm_axes(dx)) * sgn
            # プラセボ B: 勾配の成分を**平均で置き換える**（＝スカラーと同じ形・正規化だけ違う）
            gbar = sum(abs(x) for x in grad.values()) / max(1, len(grad))
            acc["plac_mag"] += wd * gbar * sum(abs(x) for x in dx.values()) * sgn
        if len(z_of) < 2:
            continue
        zs.append(z_of.get(0, 0.0))
        for kk in arms:
            arms[kk].append(acc[kk])
    n = max(1, stats["priced"])
    out = {"games": stats["games"], "rows": stats["rows"], "priced": stats["priced"],
           "d_mode": D_MODE, "live_axes": list(LIVE_AXES[D_MODE]),
           # **その読みに軸が無い手の割合**（件数と価格の絶対値・帳簿が値付けできない分）
           "dead_rows": stats["dead"], "dead_share": round(stats["dead"] / n, 4),
           "dead_value_share": round(stats["dead_v"] / max(1e-12, stats["dead_v"] + stats["live_v"]), 4),
           # **2 つの耐久の軸の重みが違うこと**（スカラー κ はこれを 1 つに潰している）
           "grad_th_me_mean": round(stats["g_me_sum"] / n, 4),
           "grad_th_opp_mean": round(stats["g_opp_sum"] / n, 4),
           "tau_me_mean": round(stats["tau_me_sum"] / n, 4),
           "tau_opp_mean": round(stats["tau_opp_sum"] / n, 4),
           "sigma_rel": (round(sr, 4) if sr is not None else None),
           "w_err_mode": TO.W_ERR_MODE,
           "by_family": stats["by_family"], "by_axis": stats["by_axis"],
           "arms": {kk: _score(arms[kk], zs) for kk in arms}}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="κ をベクトル（勾配）にする（T121）")
    ap.add_argument("--in", dest="src", nargs="+", required=True)
    ap.add_argument("--games", type=int, default=0)
    ap.add_argument("--d-mode", dest="d_mode", choices=D_MODES, default=None)
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)
    if a.d_mode:
        set_d_mode(a.d_mode)
    out = collect(a.src, a.games)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
