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

    python tests/scripts/relative_ledger.py --in <records_dir> [--games N] [--d-mode curve|clock] [--json out.json]
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
import kappa_vector as KV  # noqa: E402
import theory_order as TO  # noqa: E402
from theory_bridge import POL_COLS, ROW_COLS, _extra, _state_of, move_family  # noqa: E402
from theory_order import (MU, SC_MY_DON, SC_MY_LEADER_POWER, SC_MY_LIFE, SC_OPP_LEADER_POWER,  # noqa: E402
                          SC_OPP_LIFE, THETA, opp_bodies_of, own_attackers_of, score_candidate,
                          slot_power, theta_of)

#: 時計の床（0 割りと `log 0` を避ける・`crossing_bridge.SLOPE_FLOOR` と同じ意味）
T_FLOOR = 1e-6
#: 腕の名前（出力の順序もこれ）
ARMS = ("abs_flat", "abs_kappa", "rel_flat", "rel_K", "rel_exact", "plac_K", "plac_shift")


def clocks_of(st, prof=None):
    """状態 `(Θ_me, Θ_opp, A_me, A_opp, j)` → **2 本の時計 `(T_me, T_opp)`**（§17.9 の記号）。

    **`T_me` は「自分が相手を倒しきるまで」**＝相手の耐久 `Θ_opp` を自分の速さで削る時間、
    **`T_opp` は「相手が自分を倒しきるまで」**。`KV.D_MODE` が読み方を決める
    （`curve`＝輪郭を歩く〔帳簿の正本〕／**`curve_scaled`＝輪郭をその席の `A` で伸縮**〔T126〕／`clock`＝`min(CAP, Θ/A)`）。

    **`D = T_opp − T_me`** は `KV.d_of` と同じ値になる（同じ関数を呼んでいる）。"""
    # **C-5c**: 7 つ組なら末尾の**戻る分**（レスト中のブロッカー）が各歩きの的に 2 段目から足される
    th_me, th_opp, a_me, a_opp, j, b_me, b_opp = KV.split_state(st)
    if KV.D_MODE in ("curve", "curve_scaled"):
        if prof is None:
            raise ValueError("D_MODE=%s には損害の輪郭が要る（profile_for）" % KV.D_MODE)
        # **T126**: `curve_scaled` は**削る側の速さ**で輪郭を伸縮する（`kappa_vector.d_of` と同じ式）
        s_me = s_opp = 1.0
        if KV.D_MODE == "curve_scaled":
            s_me = KV.profile_scale(a_me, j)      # 自分が相手を倒すまで＝**自分**の速さ
            s_opp = KV.profile_scale(a_opp, j)    # 相手が自分を倒すまで＝**相手**の速さ
        t_me = float(CB.tau_from_profile(max(0.0, float(th_opp)), int(j), prof, s_me, step=b_opp))
        t_opp = float(CB.tau_from_profile(max(0.0, float(th_me)), int(j), prof, s_opp, step=b_me))
    elif KV.D_MODE == "theory":
        # **T127**: 加速を状態から出す（表を使わない）。削る側の速さでそれぞれ歩く。
        t_me = KV.tau_theory(th_opp, a_me, KV.RATE_SHAPE["me"], j, step=b_opp)
        t_opp = KV.tau_theory(th_me, a_opp, KV.RATE_SHAPE["opp"], j, step=b_me)
    else:
        t_me = KV.tau_of(th_opp, a_me, step=b_opp)
        t_opp = KV.tau_of(th_me, a_opp, step=b_me)
    return max(T_FLOOR, t_me), max(T_FLOOR, t_opp)


def z_of(t_me, t_opp, sigma_rel):
    """`z = D / (σ_rel·s)`＝**0 次同次**（2 本を同時に c 倍しても動かない）。"""
    s = TO.clock_scale(t_me, t_opp)
    if s <= 0.0 or sigma_rel <= 0.0:
        return 0.0
    return (float(t_opp) - float(t_me)) / (float(sigma_rel) * s)


def w_of(t_me, t_opp, sigma_rel):
    """`W = Φ(z)`（出荷の `prob_of_d` と同じ式・引数を時計で受ける形）。"""
    return 0.5 * (1.0 + math.erf(z_of(t_me, t_opp, sigma_rel) / math.sqrt(2.0)))


def k_of(t_me, t_opp, sigma_rel):
    """**法則のスカラー `K(r)`**（§17.9.3）＝`φ(z)·r(1+r) / (σ_rel·(1+r²)^{3/2})`・`r = T_me/T_opp`。

    **これは `dW/dlog(T_opp/T_me)` そのもの**（`game_theory.md` §17.9.9 の検算が 8 桁で一致）。
    `s` に `hyp` 以外を選ぶと `K` の閉じた形は変わるが、**判別は変わらない**（どれも `r` の単調関数・T118）。"""
    r = float(t_me) / max(T_FLOOR, float(t_opp))
    z = z_of(t_me, t_opp, sigma_rel)
    phi = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    return phi * r * (1.0 + r) / (max(1e-12, float(sigma_rel)) * (1.0 + r * r) ** 1.5)


def dlog_of(st0, st1, prof=None):
    """**`Δ log(T_opp/T_me)`**——**時計を作り直して引く**（近道の `p/Θ` は `T=Θ/A` のときしか成り立たない）。"""
    a0, b0 = clocks_of(st0, prof)
    a1, b1 = clocks_of(st1, prof)
    return math.log(b1 / a1) - math.log(b0 / a0)


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


def _invariance(inv, st0, dx, prof, sigma_rel, dl, kk, capped):
    """**行ごとに P5／P7 を厳密に検算する**（腕の AUC を run ごとに比べるのではなく一致を見る）。

    **P5**: 耐久も価格も同時に c 倍（＝通貨の付け替え）／**P7**: 両席の速さを同時に c 倍。
    どちらも **`r` が不変なので `Δlog` と `K` は 1 ビットも動いてはならない**。
    **打ち切り（`TAU_CAP`）と床は絶対量**なので、そこに当たった行だけは動く＝**別に数える**。
    **倍率を掛けた側で初めて打ち切りを跨ぐ行も在る**ので、**両方の状態を見て**「打ち切り無しでも破れたか」を数える。"""
    for tag, st0b, dxb in (
            # **P5 は通貨の付け替え**なので **`A` も同じだけ倍にする**（`A` は「損害/ターン」）。
            # **T127 で直した**——`curve` の読みでは `A` が時計に入らないので無害だったが、
            # `A` が時計に入る読み（`curve_scaled`／`theory`）では**耐久だけ 3 倍にするのは
            # 単位の変更ではなく物理の変更**（同じ速さで 3 倍の耐久＝3 倍の時間）になっていた。
            # **T122／T126 の P5 の数字はこの誤った定義で測ったもの**（`curve` については結論は変わらない）。
            # **C-5c**: 7 つ組なら**戻る分も同じ通貨**（P5 では c 倍・P7 ではそのまま）
            ("p5", (st0[0] * INV_C5, st0[1] * INV_C5, st0[2] * INV_C5, st0[3] * INV_C5, st0[4])
                   + tuple(x * INV_C5 for x in st0[5:]),
             {k: v * INV_C5 for k, v in dx.items()}),
            ("p7", (st0[0], st0[1], st0[2] * INV_C7, st0[3] * INV_C7, st0[4]) + tuple(st0[5:]),
             {k: v for k, v in dx.items() if k in ("th_me", "th_opp", "th_me_back", "th_opp_back")}
             | {k: v * INV_C7 for k, v in dx.items() if k in ("a_me", "a_opp")})):
        st1b = KV.apply_dx(st0b, dxb)
        a, b = clocks_of(st0b, prof)
        a1, b1 = clocks_of(st1b, prof)
        dlb = dlog_of(st0b, st1b, prof)
        kb = k_of(a, b, sigma_rel)
        err = max(abs(dlb - dl), abs(kb - kk))
        lim = KV.TAU_CAP - 1e-9
        hit = capped or a >= lim or b >= lim or a1 >= lim or b1 >= lim
        inv[tag + "_n"] += 1
        inv[tag + "_max"] = max(inv[tag + "_max"], err)
        if err > 1e-9:
            inv[tag + "_bad"] += 1
            if not hit:
                inv[tag + "_bad_offcap"] += 1


def collect(dirs, limit_games=0, theta=THETA, mu=MU, scale_a=1.0, scale_currency=1.0, pre_settle=False,
            parts=False):
    """記録を 1 度読んで **7 つの腕**を並べる（席×局ごとに積む）。

    `scale_a`（**P7**）は**両席の `A` に共通の掛け算誤差**を入れる／`scale_currency`（**P5**）は
    **耐久も価格も同時に c 倍**する＝どちらも**相対の腕は 1 ビットも動いてはならない**。

    **T138b**: `pre_settle=True` なら決着後（`lethal_rule.settled_map` が `True`）の行を読まない——
    **`before`（最後の自席ターンを外すだけの目分量）を、規則の決着点に差し替える**。

    **T145**: `parts=True` なら `rel_K` の局ごとの和を**勝者／敗者の最後の自席ターン**と**宣言した行**に
    割って出す（`out["parts"]`）——T138b の `arms` の符号反転が「決着後を除いた」ことではなく
    「**片方の席の最後のターンだけ**を除いた」ことから来るかを、同じ行の上で確かめる。"""
    cards = PL.Cards()
    idx2cid = {i: c for c, i in GA._vocab().items()}
    settled = None
    flags = None                          # **T145**: 落とさずに旗だけ読む（`parts`）
    if pre_settle or parts:
        import lethal_rule as LR
        flags = LR.settled_map(dirs, limit_games)
        if pre_settle:
            settled = flags
    part_rows = []                        # **T145**: 局ごとの `rel_K` の内訳（席 0 視点）
    prof = CB.profile_for(dirs)
    if KV.D_MODE in ("curve", "curve_scaled") and not prof:
        raise ValueError("D_MODE=KV.D_MODE なのに損害の輪郭が引けない（%s）" % (dirs,))
    if KV.D_MODE == "curve_scaled":
        # **T126**: 輪郭をその席の `A` で伸縮する読み＝分母が要る（引けなければ落ちる）
        _th = CB.profile_th_for(dirs)
        if not _th:
            raise ValueError("curve_scaled なのに理論の速さの輪郭が引けない（%s）" % (dirs,))
        KV.set_profile_th(_th)
    sr = CB.sigma_rel_for(dirs, slope="curve")
    if sr is None:
        raise ValueError("σ_rel が引けない＝黙って別の物差しに落とさない（T118 の規約）")
    TO.set_sigma_rel(sr)
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
    prev_ks = []                          # プラセボ用: **前の局**の同じ行番号の `K`
    games = 0
    for r, pol, ex, L, ptr, idx in PL.iter_games(dirs, row_cols=ROW_COLS, pol_cols=POL_COLS,
                                                 extra_fn=_extra):
        games += 1
        if limit_games and games > limit_games:
            break
        stats["games"] += 1
        acc = {k: 0.0 for k in arms}
        acc_before = {k: 0.0 for k in arms}
        acc_last = {k: 0.0 for k in arms}
        # **T145**: `rel_K` を (席, 最後の自席ターンか, 宣言した行か) で割る（席 0 視点のまま積む）
        acc_part = {}
        cur_ks = []
        w0 = None
        z_of_seat = {}
        rate_at_turn, g_at_turn = {}, {}
        shape_at = {}
        seed_g = int(r["seed"][idx[0]]) if len(idx) else -1
        for i in idx:
            if int(r["kind"][i]) != 0:
                continue
            w, t = int(r["who"][i]), int(r["turn"][i])
            if PL.is_own_turn(w, t) and (w, t) not in rate_at_turn:
                dk = KV._deck_of(seat_decks, seed_g, w)            # **T128**
                rate_at_turn[(w, t)] = KV.rate_of_row(ex["sc"][i], ex["tok"][i], ex["ci"][i],
                                                      idx2cid, cards, theta, mu, deck_ids=dk,
                                                      j=CB.own_turn_index(t)) * float(scale_a)
                g_at_turn[(w, t)] = KV.g_of_row(ex["sc"][i], ex["tok"][i], ex["ci"][i], idx2cid, cards)
                shape_at[(w, t)] = (KV.rate_terms_of_row(ex["sc"][i], ex["tok"][i], ex["ci"][i],
                                                       idx2cid, cards, theta, mu, deck_ids=dk)
                                       if KV.D_MODE == "theory" else None)
        last_turn_of = {}
        for (w, t) in rate_at_turn:
            last_turn_of[w] = max(t, last_turn_of.get(w, -1))

        def _opp_at(w, t):
            ts = [tt for (ww, tt) in rate_at_turn if ww == 1 - w and tt < t]
            if not ts:
                return None
            key = (1 - w, max(ts))
            return rate_at_turn[key], g_at_turn[key]
        for i in idx:
            z = float(r["z"][i])
            if z != 0.0:
                z_of_seat[int(r["who"][i])] = 1.0 if z > 0 else 0.0
            if int(r["kind"][i]) != 0:
                continue
            w, t = int(r["who"][i]), int(r["turn"][i])
            if not PL.is_own_turn(w, t):
                continue
            if settled is not None and settled.get((seed_g, w, t)):
                continue                                  # **T138b**: 決着後の行は除く（pre_settle）
            k = int(L[i]); ch = int(r["pol_chosen"][i])
            if k < 1 or ch < 0 or ch >= k:
                continue
            stats["rows"] += 1
            pair = _opp_at(w, t)
            if pair is None:
                continue
            if KV.D_MODE == "theory":
                # **T127**: 席ごとの速さの形（相手は直近の自席ターンの形）を行ごとに入れる
                _ts = [tt for (ww, tt) in rate_at_turn if ww == 1 - w and tt < t]
                KV.set_rate_shape(shape_at.get((w, t)),
                             shape_at.get((1 - w, max(_ts))) if _ts else None)
            ao, g_opp = pair
            sc, tok, ci = ex["sc"][i], ex["tok"][i], ex["ci"][i]
            b = int(ptr[i]) + ch
            sig = json.loads(pol["pol_sig"][b])
            fam = move_family(sig)
            st0 = KV.state_of_row(sc, tok, rate_at_turn[(w, t)], ao, CB.own_turn_index(t),
                                  g_me=g_at_turn[(w, t)], g_opp=g_opp, ci_row=ci, idx2cid=idx2cid, cards=cards)
            # **P5**: 通貨の付け替え＝耐久も価格も同じ c 倍（速さはそのまま＝時計は c 倍される）
            st0 = ((st0[0] * scale_currency, st0[1] * scale_currency, st0[2], st0[3], st0[4])
                   + tuple(x * scale_currency for x in st0[5:]))     # 戻る分も同じ通貨（C-5c）
            rt = max(1.0, min(5.0, float(np.asarray(sc)[SC_OPP_LIFE])))
            th = theta_of(tok, float(np.asarray(sc)[SC_MY_LIFE]), float(np.asarray(sc)[SC_MY_DON]),
                          mode="const", theta=theta)
            olp = float(np.asarray(sc)[SC_OPP_LEADER_POWER]) * 1e4 or 5000.0
            mlp = float(np.asarray(sc)[SC_MY_LEADER_POWER]) * 1e4 or 5000.0
            ctx = {"theta": th, "mu": mu, "opp_leader_power": olp, "my_leader_power": mlp,
                   "r_turns": rt, "don_k": 1, "attackers": own_attackers_of(tok, olp),
                   "don_active": float(np.asarray(sc)[SC_MY_DON]),
                   "st": _state_of(sc, ci, idx2cid),
                   # **T150f-2**: 見送った登場の価値（`misalloc_play`）に要る手札の card_id 列
                   "hand": TO.hand_ids_of(ci, idx2cid),
                   "opp_bodies": opp_bodies_of(tok, mlp, rt, th, mu, ci_row=ci, idx2cid=idx2cid)}
            tl = sig[2] if len(sig) > 2 else None
            v = score_candidate(sig, str(pol["pol_cid"][b]) or None,
                                (str(pol["pol_tcid"][b]) or None) if tl else None, ctx, cards,
                                src_power=slot_power(tok, int(pol["pol_si"][b])),
                                tgt_power=slot_power(tok, int(pol["pol_ti"][b])),
                                don_k=int(pol["pol_k"][b]))
            if v is None:
                continue
            v = float(v) * float(scale_currency)
            stats["priced"] += 1
            stats["by_family"][fam] = stats["by_family"].get(fam, 0) + 1
            dx = KV.axis_of_move(fam, v, sig, str(pol["pol_cid"][b]) or None, cards, sc, tok, olp, rt,
                                 don_k=int(pol["pol_k"][b]))
            st1 = KV.apply_dx(st0, dx)
            t_me, t_opp = clocks_of(st0, prof)
            d = t_opp - t_me
            kk = k_of(t_me, t_opp, sr)
            dl = dlog_of(st0, st1, prof)
            if dl == 0.0:
                stats["dead"] += 1
            stats["k_sum"] += kk
            rs.append(t_me / max(T_FLOOR, t_opp))
            stats["dlog_abs_sum"] += abs(dl)
            # **打ち切りに当たっているか**——`tau_of`／輪郭の打ち切りは**絶対量**なので、
            # そこに当たった行では下の不変量（P5・P7）が成り立たない（破れの出どころ）。
            capped = (t_me >= KV.TAU_CAP - 1e-9) or (t_opp >= KV.TAU_CAP - 1e-9)
            stats["capped"] += int(capped)
            _invariance(stats["inv"], st0, dx, prof, sr, dl, kk, capped)
            sgn = 1.0 if w == 0 else -1.0                # 席 0 の視点で積む（帳簿と同じ）
            if w0 is None:                               # **較正用**: 席 0 視点の開始時の勝率
                w0 = w_of(t_me, t_opp, sr) if w == 0 else 1.0 - w_of(t_me, t_opp, sr)
            ex_w = w_of(*clocks_of(st1, prof), sigma_rel=sr) - w_of(t_me, t_opp, sr)
            plac_k = prev_ks[len(cur_ks)] if len(cur_ks) < len(prev_ks) else kk
            cur_ks.append(kk)
            one = {"abs_flat": v * sgn,
                   "abs_kappa": v * sgn * float(TO.state_factor(d, "curve", t_me=t_me, t_opp=t_opp)),
                   "rel_flat": dl * sgn,
                   "rel_K": kk * dl * sgn,
                   "rel_exact": ex_w * sgn,
                   "plac_K": kk * v * sgn,
                   "plac_shift": plac_k * dl * sgn}
            is_last = (t == last_turn_of.get(w))         # **P3**: 最後の自席ターンか
            stats["last_turn_rows"] += int(is_last)
            for kk2, val in one.items():
                acc[kk2] += val
                (acc_last if is_last else acc_before)[kk2] += val
            if flags is not None:
                pk = (w, bool(is_last), bool(flags.get((seed_g, w, t))))
                acc_part[pk] = acc_part.get(pk, 0.0) + one["rel_K"]
        prev_ks = cur_ks
        if len(z_of_seat) < 2:
            continue
        zs.append(z_of_seat.get(0, 0.0))
        w0s.append(0.5 if w0 is None else w0)
        for kk2 in arms:
            arms[kk2].append(acc[kk2]); before[kk2].append(acc_before[kk2])
            last[kk2].append(acc_last[kk2])
        if flags is not None:
            part_rows.append((z_of_seat.get(0, 0.0), acc["rel_K"], acc_part))
    n = max(1, stats["priced"])
    out = {"games": stats["games"], "rows": stats["rows"], "priced": stats["priced"],
           "d_mode": KV.D_MODE, "sigma_rel": round(sr, 4),
           "kappa_sigma_mode": TO.KAPPA_SIGMA_MODE, "attack_rest_mode": KV.ATTACK_REST_MODE,
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
    if flags is not None:
        out["parts"] = parts_of(part_rows)
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
    ap.add_argument("--d-mode", dest="d_mode", choices=KV.D_MODES, default=None)
    ap.add_argument("--kappa-sigma", dest="kappa_sigma", choices=TO.KAPPA_SIGMA_MODES, default=None,
                    help="**T122/§17.9.6-1**: κ の物差しを W に合わせるか（既定は現状の `abs`）")
    ap.add_argument("--theta-side", dest="theta_side", choices=CB.THETA_SIDE_MODES, default=None,
                    help="**T133**: `Θ` を両席で同じ式にするか（既定は現状の `legacy`）")
    ap.add_argument("--slope-take", dest="slope_take", choices=CB.SLOPE_TAKE_MODES, default=None,
                    help="**T134**: `A` の「受ける費用」を自分のライフで決めるか（既定は現状の `const`）")
    ap.add_argument("--attack-rest", dest="attack_rest", choices=KV.ATTACK_REST_MODES, default=None,
                    help="**C-2**: 攻撃した体のレスト費用をΘ_meへ足すか（既定 `off`・C-5c は `return`）")
    ap.add_argument("--theta-return", dest="theta_return", choices=CB.THETA_RETURN_MODES, default=None,
                    help="**C-5c**: レスト中のブロッカーを次の自席ターンから戻る耐久として持つか（既定 `off`）")
    ap.add_argument("--scale-a", type=float, default=1.0, help="**P7**: 両席の A に共通の掛け算誤差")
    ap.add_argument("--scale-currency", type=float, default=1.0, help="**P5**: 耐久と価格を同時に c 倍")
    ap.add_argument("--scale-clamp", dest="clamp", default="",
                    help="**診断用**（例 0.5,2）: `curve_scaled` の倍率を締める。モデルの提案ではない")
    ap.add_argument("--pre-settle", dest="pre_settle", default="off", choices=("off", "on"),
                    help="**T138b** 決着後（`lethal_rule.settled_map`）の行を除いて測るか")
    ap.add_argument("--parts", action="store_true",
                    help="**T145** `rel_K` の和を勝者／敗者の最後のターンと宣言した行に割って出す")
    ap.add_argument("--json", default="")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    if a.d_mode:
        KV.set_d_mode(a.d_mode)
    if a.kappa_sigma:
        TO.set_kappa_sigma_mode(a.kappa_sigma)
    if a.theta_side:
        CB.set_theta_side_mode(a.theta_side)          # **T133**
    if a.slope_take:
        CB.set_slope_take_mode(a.slope_take)          # **T134**
    if a.attack_rest:
        KV.set_attack_rest_mode(a.attack_rest)        # **C-2**
    if a.theta_return:
        CB.set_theta_return_mode(a.theta_return)      # **C-5c**
    if a.clamp:
        KV.set_scale_clamp([float(x) for x in a.clamp.split(",")])
    out = collect(a.src, a.games, scale_a=a.scale_a, scale_currency=a.scale_currency,
                  pre_settle=(a.pre_settle == "on"), parts=a.parts)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
