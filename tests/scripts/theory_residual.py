"""**理論の量のうち、ネットが持っていないのはどれか**（補助ヘッド／入力列の選別・読み取り専用）。

`docs/cpu_theory_gap.md` §8.1 の **T1c**。ユーザ指示 2026-09-13
「どの計算結果が勝敗に関連しているかで、どれが一番効くかを分析する」への回答だが、
**素の勝敗との相関では選別できない**——`Θ` も `G` も `S` も「勝っている側の相貌」と相関し、
それは **`V` が既に捉えている分**である。選別に使うのは**ネットの残差**:

```
残差 = z − P̂          （z＝勝敗 0/1・P̂ = (V+1)/2）
その量が残差を説明する ⇔ ネットが持っていない情報
```

## 入力列と補助ヘッドで基準が違う

```
入力列（今の状態から計算できる）      期待される効き = 残差の説明力
補助ヘッド（未来からラベルを作る）    期待される効き = 残差の説明力 × 盤面から予測できる度合い
```

補助ヘッドが 2 因子なのは、**残差をよく説明しても盤面から予測できない量はヘッドにしても
学べない**から。本器は量を `now`（状態から決まる＝入力列の候補）と
`future`（記録の未来から作る＝ヘッドの候補）に分けて出し、`future` には
**帯だけでどれだけ説明できるか**（`band_r2`＝盤面の代理）を併記する。

## 推定（今日の教訓を 2 つ入れている）

- **帯の中だけで傾きを取る**（within 推定）。帯 `= (自ライフ, 相手ライフ, ターン帯, 手札)`。
  盤面と相関する量を素に回帰すると、交絡がそのまま乗る。
- **SE は対局でクラスタする**（同じ局の行は独立でない）。帯で中心化した後、
  **局ごとに分子と分母を積んで** `SE² = Σ_g (num_g − β·den_g)² / (Σ den)²`（クラスタ頑健）。
- **順位は標準化した効き `|β|·sd(q)`** で付ける（単位が違う量を並べるため）
  ＝「その量が 1 標準偏差動くと残差がどれだけ動くか」。

## 読み方（事前登録）

- `|β|·sd(q)` が **0.01（勝率 1 ポイント）以上**で CI が 0 を含まない量は、
  **ネットが持っていない情報**＝足す価値がある。
- **`now` の量はそのまま入力列の候補**（決定的に計算できるので第 2 因子は 1）。
- **`future` の量は `band_r2` が高いものだけヘッドにする**（低ければ盤面から予測できない）。
- すべての量が 0.01 未満なら、**理論の量はネットに既に入っている**＝
  入力を足す線は捨てて、方策の側（T1）へ寄せる。

## T11: **`future` の量の β は勝敗の写しかもしれない**（`--split`・`cpu_theory_gap.md` §7.2）

上の読み方には穴が 1 つ在る。**帯の中で、負ければ残りライフを全部失う**ので `life_spent` は
大きくなり、`resid = z − P̂` も `z` で動く＝**その量が `z` 以外に何の情報も持っていなくても
β < 0 は出る**。2026-09-13 にこれが主たる説明に昇格した（1 手先の流れは `aux[0]`／`aux[3]` で
全世代が持ち、決着までの流れは r10 が 0.8 枚の精度で当てているのに、残差も強さも動かない）。

`--split` は量を**状態から読める部分と読めない残り**に分けて、効きを別々に測る:

```
f = f̂(s) + u        f̂(s) = E[f | 状態]（**z を使わずに**推定）
resid ~ f̂           … 状態の関数なので**勝敗の写しではありえない**＝本物の情報
resid ~ u           … 勝敗との結合が住む場所＝artefact の通り道
```

**判定（事前登録）**: `β(f̂)` が有意で効きが基準以上 ⇒ `real_information`
（V は状態から読める量に対して較正がずれている＝ヘッドで直せる）。
`f̂` が効かず `u` だけが効く ⇒ `outcome_artefact`（**ヘッドの線は閉じる**）。

`f̂` の推定器は 2 つ:

1. **主 `--pred-net`**: **r10 の時間軸ヘッド**の出力（`my_life_end`／`opp_life_end` の予測）。
   **23 量の張る空間の外**にある本物の状態予測器なので、これが主。`--net` と別に指定できる＝
   **r3 の残差を r10 の予測で割る**（V に伝わっていないことの検査）と
   **r10 自身の残差を測る**（ヘッドが自分の V に伝わったかの検査）を撃ち分けられる。
2. **副（`--pred-net` 無し）**: 全 `q_` 列から**局で out-of-fold に交差適合**（ネット不要の下限）。
   局で割るのが要点——同じ局の行を訓練と予測に跨がせると `f̂` が `u` を吸って `z` と結合する。

実行例:
  OPCG_LOG_SILENT=1 python tests/scripts/theory_residual.py --net ~/nrel_r3.npz \\
    --in ~/w32/*/n_records --holdout-mod 7 --out ~/theory_residual_w32.json

  # T11（主）: r3 の残差を r10 のヘッドの予測で割る
  OPCG_LOG_SILENT=1 python tests/scripts/theory_residual.py --net ~/nrel_r3.npz \\
    --pred-net ~/nrel_r10.npz --split f_life_spent f_life_taken \\
    --in ~/w32/*/n_records --holdout-mod 7 --out ~/split_r3_by_r10.json
"""
import argparse
import json
import math
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

from plan_value_map import _pad  # noqa: E402
from theory_order import add_nu_mode_arg, apply_nu_mode, c_of, nu_of, saturation_x, THETA, MU  # noqa: E402
from opcg_sim.learned import n_rel as NL  # noqa: E402
from opcg_sim.learned.train import n_rel_train as NT  # noqa: E402
from opcg_sim.learned.train import plan_labels as PL  # noqa: E402
from opcg_sim.learned.train.n_eff_feat import build_eff_tables  # noqa: E402

ROW_COLS = ("who", "turn", "seed", "z", "kind", "step", "pol_len", "pol_v0")

#: scalars の列（`rust/opcg_engine/src/encode/scalars.rs`）
SC_MY_LIFE, SC_OPP_LIFE = 0, 1
SC_MY_DON = 2
SC_MY_HAND, SC_OPP_HAND = 6, 7
SC_MY_FIELD, SC_OPP_FIELD = 8, 9
SC_TURN = 10
SC_MY_LEADER_POWER, SC_OPP_LEADER_POWER = 12, 13
SC_MY_DECK, SC_OPP_DECK = 16, 17
#: トークンの枠（自L,相L,自場5,相場5,手札10）と S 列（`n_rel_feat.S_COLS`）
SLOT_OWN_FIELD = slice(2, 7)
SLOT_OPP_FIELD = slice(7, 12)
SLOT_HAND = slice(12, 22)
S_POWER, S_REST, S_BLOCKER, S_COUNTER = 0, 3, 6, 7
#: 量の分類（`now`＝入力列の候補／`future`＝補助ヘッドの候補）
KIND_NOW, KIND_FUTURE = "now", "future"


def turn_band(t):
    return "T<=4" if t <= 4 else ("T5-8" if t <= 8 else "T9+")


def band_key(sc):
    """帯＝(自ライフ, 相手ライフ, ターン帯, 手札)。

    **帯に入れた量は測れない**（帯の中で動かないので傾きが定義できない）＝
    `q_L`（自ライフ）と `q_H`（手札）は必ず `n=0`・`beta=null` になる。**これは仕様**で、
    データの欠落ではない（2026-09-13 の回収で作業セッションが欠落と報告した）。
    それらを測りたいときは帯の定義を変える（`hand_value_slope --axis life|hand` が担当）。
    """
    return "|".join((str(int(round(float(sc[SC_MY_LIFE])))),
                     str(int(round(float(sc[SC_OPP_LIFE])))),
                     turn_band(int(round(float(sc[SC_TURN])))),
                     str(int(round(float(sc[SC_MY_HAND]))))))


def _round10(x):
    """パワーの差を 10 の桁で丸める（f32 の丸めで 1000 の倍数が崩れるのを吸収）。"""
    return float(round(float(x) / 10.0) * 10.0)


def _occupied(tok, sl):
    return [j for j in range(sl.start, sl.stop) if float(np.abs(tok[j]).sum()) > 0.0]


def now_quantities(sc, tok, theta=THETA, mu=MU):
    """**状態から決まる**理論の量（入力列の候補）。`docs/game_theory.md` の記号に合わせる。"""
    L = float(sc[SC_MY_LIFE]); oppL = float(sc[SC_OPP_LIFE])
    H = float(sc[SC_MY_HAND]); don = float(sc[SC_MY_DON])
    F = float(sc[SC_MY_FIELD]); oppF = float(sc[SC_OPP_FIELD])
    my_lp = float(sc[SC_MY_LEADER_POWER]) * 1e4
    opp_lp = float(sc[SC_OPP_LEADER_POWER]) * 1e4
    own = _occupied(tok, SLOT_OWN_FIELD)
    opp = _occupied(tok, SLOT_OPP_FIELD)
    hand = _occupied(tok, SLOT_HAND)

    A = oppF + 1.0                                  # 相手の攻撃回数（リーダー＋場）
    A_me = F + 1.0
    t_hat = max(1.0, math.ceil(oppL / max(A_me, 1.0)))   # 規則で置いた時計（§7 の T）
    N = A * t_hat
    B = float(sum(1 for j in own if float(tok[j, S_BLOCKER]) > 0.5))
    G = max(0.0, N - L - B)
    S = H + (N - G) + t_hat - G * theta             # §7 の余裕（`c̄ ≈ Θ` で近似）
    theta_life = A * t_hat - B                      # §11 の `θ`
    # 攻め: いちばん強い自分の枠の超過パワーと、飽和点からの距離
    # **10 の桁で丸める**——パワーは 1000 の倍数だが、符号化を戻すと f32 の丸めで
    # 6999.999… になる（今日 2 回踏んだ罠・`PWR_EPS` と同じ扱い）。
    x_max = _round10(max([float(tok[j, S_POWER]) * 1e4 - opp_lp for j in own]
                         + [my_lp - opp_lp]))
    xs = saturation_x(theta)
    # 守り: 飛んでくる最大パワーと、それを止める費用
    x_in = _round10(max([float(tok[j, S_POWER]) * 1e4 - my_lp for j in opp] + [0.0]))
    c_in = c_of(x_in)
    guard_total = float(sum(float(tok[j, S_COUNTER]) for j in hand))
    nu_sum = float(sum(nu_of(float(tok[j, S_POWER]) * 1e4, opp_lp, t_hat, theta, mu)
                       for j in own))
    return {
        "L": L, "H": H, "don": don, "F": F, "deck_left": float(sc[SC_MY_DECK]) * 50.0,
        "A": A, "B": B, "t_hat": float(t_hat), "N": N,
        "G": G, "S": S, "theta_life": theta_life, "L_minus_theta": L - theta_life,
        "x_max": x_max, "x_over_sat": x_max - xs,        # >0 なら飽和点を超えて積んでいる
        "n_dead_attackers": float(sum(1 for j in own
                                      if float(tok[j, S_POWER]) * 1e4 - opp_lp < 0)),
        "n_rested": float(sum(1 for j in own if float(tok[j, S_REST]) > 0.5)),
        "c_in": c_in, "c_in_minus_theta": c_in - theta,
        "guard_slack": guard_total / 1000.0 - c_in,
        "nu_sum": nu_sum,
    }


def future_quantities(rows, sc_all, idx, who, pos):
    """**記録の未来から作る**量（補助ヘッドの候補・`pos` は `idx` の中の位置）。

    | 量 | 意味 | 対応するヘッド |
    |---|---|---|
    | `t_real` | この行から先の**自席ターン数**（実現した決着までの時間） | 時間軸ヘッド（r10 で実装済み） |
    | `life_spent` | この先で**自分が失うライフ**の枚数 | `λ` の実現値（`game_theory.md` §9） |
    | `life_taken` | この先で**相手から取るライフ**の枚数 | 攻めの側の実現値 |
    | `own_rows_left` | この先の自席 main 行の数 | 手数の実現値（`ν` の代理） |

    **未来を見ているので入力にはできない**（ヘッドのラベルとしてだけ使える）。
    """
    turns = set()
    own_rows = 0
    my_life_end = float(sc_all[idx[pos]][SC_MY_LIFE])
    opp_life_end = float(sc_all[idx[pos]][SC_OPP_LIFE])
    for k in range(pos, len(idx)):
        i = idx[k]
        if int(rows["who"][i]) != who:
            continue
        turns.add(int(rows["turn"][i]))
        if int(rows["kind"][i]) == 0:
            own_rows += 1
        my_life_end = float(sc_all[i][SC_MY_LIFE])
        opp_life_end = float(sc_all[i][SC_OPP_LIFE])
    my_now = float(sc_all[idx[pos]][SC_MY_LIFE])
    opp_now = float(sc_all[idx[pos]][SC_OPP_LIFE])
    return {"t_real": float(len(turns)), "own_rows_left": float(own_rows),
            "life_spent": max(0.0, my_now - my_life_end),
            "life_taken": max(0.0, opp_now - opp_life_end)}


def within_cluster_slope(rows, q_key, y_key="resid", band="band", seed="seed", min_n=2):
    """帯の中の傾きと、**対局でクラスタした**SE（`SE² = Σ_g (num_g − β·den_g)² / (Σ den)²`）。

    帯で中心化してから局ごとに分子・分母を積む＝within 推定のクラスタ頑健分散。
    `sd` は帯で中心化した後の標準偏差（**順位は `|β|·sd` で付ける**）。
    """
    by_band = {}
    for r in rows:
        by_band.setdefault(r[band], []).append(r)
    num = den = 0.0
    per_game = {}
    qd_all = []
    used_rows = used_bands = 0
    for _b, sub in by_band.items():
        if len(sub) < min_n:
            continue
        q = np.array([float(r[q_key]) for r in sub], np.float64)
        y = np.array([float(r[y_key]) for r in sub], np.float64)
        if float(q.var()) <= 0.0:
            continue
        qd = q - q.mean(); yd = y - y.mean()
        qd_all.append(qd)
        used_rows += len(sub); used_bands += 1
        for r, a, c in zip(sub, qd, yd):
            g = r[seed]
            p = per_game.setdefault(g, [0.0, 0.0])
            p[0] += float(a * c); p[1] += float(a * a)
        num += float((qd * yd).sum()); den += float((qd * qd).sum())
    if den <= 0.0 or len(per_game) < 2:
        return {"n": used_rows, "bands": used_bands, "beta": None, "se": None,
                "sd": None, "effect": None, "ci95": None, "games": len(per_game)}
    beta = num / den
    resid = np.array([p[0] - beta * p[1] for p in per_game.values()], np.float64)
    se = float(np.sqrt(float((resid ** 2).sum())) / den)
    sd = float(np.concatenate(qd_all).std())
    eff = abs(beta) * sd
    return {"n": used_rows, "bands": used_bands, "games": len(per_game),
            "beta": round(beta, 6), "se": round(se, 6), "sd": round(sd, 4),
            "effect": round(eff, 5),
            "ci95": [round(beta - 1.96 * se, 6), round(beta + 1.96 * se, 6)]}


def band_r2(rows, q_key, band="band"):
    """**帯だけでその量をどれだけ説明できるか**（`future` の量が盤面から読めるかの代理）。

    盤面の一部（ライフ・ターン帯・手札）でしか無いので**下限の代理**。
    r10 の時間軸ヘッドは「盤面から `T` が読める」ことを実際に示している（MAE 0.73 対 定数 1.79）。
    """
    by = {}
    for r in rows:
        by.setdefault(r[band], []).append(float(r[q_key]))
    allv = np.array([v for vs in by.values() for v in vs], np.float64)
    if len(allv) < 2 or float(allv.var()) <= 0.0:
        return None
    ss_tot = float(((allv - allv.mean()) ** 2).sum())
    ss_res = 0.0
    for vs in by.values():
        a = np.array(vs, np.float64)
        ss_res += float(((a - a.mean()) ** 2).sum())
    return round(1.0 - ss_res / ss_tot, 4)


HAT_PREFIX, UNP_PREFIX = "hat_", "unp_"
#: 交差適合の fold 数（局で割る）と ridge（列を標準化してあるので小さくてよい）
FOLDS, RIDGE = 5, 1e-6
SPLIT_REAL, SPLIT_ARTEFACT, SPLIT_NONE = "real_information", "outcome_artefact", "no_signal"
#: **`f̂` が弱いと `outcome_artefact` は名乗れない**（下の `split_verdict` に理由）
SPLIT_WEAK = "predictor_too_weak"
MIN_HAT_R2 = 0.5


def _band_index(recs, band="band"):
    by = {}
    for i, r in enumerate(recs):
        by.setdefault(r[band], []).append(i)
    return by


def center_in_band(recs, key, band="band"):
    """帯で中心化した列（within 推定が見ているのと同じ変動）。"""
    y = np.array([float(r[key]) for r in recs], np.float64)
    for ix in _band_index(recs, band).values():
        y[ix] -= y[ix].mean()
    return y


def _design(recs, feats, band="band"):
    """帯で中心化して標準化した説明行列（列の尺度が桁で違うので標準化は必須）。"""
    X = np.array([[float(r[f]) for f in feats] for r in recs], np.float64)
    for ix in _band_index(recs, band).values():
        X[ix] -= X[ix].mean(0)
    sd = X.std(0)
    sd[sd <= 0.0] = 1.0
    return X / sd


def cross_fit(recs, target, feats, folds=FOLDS, ridge=RIDGE, seed="seed", band="band"):
    """**局で out-of-fold** に `f̂ = E[target | 状態]` を作る（戻り値は行と同じ長さ）。

    局で割るのが要点——同じ局の行を訓練と予測に跨がせると `f̂` が `u`（読めない残り）を
    吸い、そこに住む `z` との結合を `f̂` 側へ持ち込む＝artefact を「本物の情報」に見せる。
    """
    X = _design(recs, feats, band)
    y = center_in_band(recs, target, band)
    games = sorted({r[seed] for r in recs}, key=str)
    fold_of = {g: i % folds for i, g in enumerate(games)}
    hat = np.zeros(len(recs), np.float64)
    fitted = 0
    for k in range(folds):
        te = [i for i, r in enumerate(recs) if fold_of[r[seed]] == k]
        tr = [i for i, r in enumerate(recs) if fold_of[r[seed]] != k]
        if not te or len(tr) <= X.shape[1]:
            continue
        A = X[tr].T @ X[tr] + ridge * len(tr) * np.eye(X.shape[1])
        w = np.linalg.solve(A, X[tr].T @ y[tr])
        hat[te] = X[te] @ w
        fitted += len(te)
    return hat, y, fitted


def split_future(recs, name, feats=None, pred_key=None, folds=FOLDS, min_effect=0.01):
    """`f = f̂(s) + u` に分けて残差への効きを別々に測る（T11・`cpu_theory_gap.md` §7.2）。

    `pred_key` が在ればそれを `f̂` として使う（**23 量の外**の予測器＝r10 のヘッドの出力）。
    無ければ `feats`（既定は全 `q_` 列）から局で交差適合する。

    判定は **`β(f̂)`**——`f̂` は状態の関数なので**勝敗の写しではありえない**。
    """
    if not recs or name not in recs[0]:
        return None
    if pred_key is not None:
        hat = center_in_band(recs, pred_key)
        y = center_in_band(recs, name)
        source, fitted = pred_key, len(recs)
    else:
        feats = feats or sorted(k for k in recs[0] if k.startswith("q_"))
        hat, y, fitted = cross_fit(recs, name, feats, folds)
        source = f"cross_fit({len(feats)} q_ cols, {folds} folds by game)"
    sub = [dict(r) for r in recs]
    for r, h, v in zip(sub, hat, y):
        r[HAT_PREFIX + name] = float(h)
        r[UNP_PREFIX + name] = float(v - h)
    var = float(y.var())
    out = {"name": name, "source": source, "rows_fitted": fitted,
           # `f̂` が帯の中の変動をどれだけ説明できたか（推定器の質そのもの）
           "hat_r2": round(1.0 - float(((y - hat) ** 2).mean()) / var, 4) if var > 0 else None,
           "hat": within_cluster_slope(sub, HAT_PREFIX + name),
           "unpredictable": within_cluster_slope(sub, UNP_PREFIX + name),
           "whole": within_cluster_slope(sub, name)}
    out["verdict"] = split_verdict(out, min_effect)
    return out


def _carries(s, min_effect):
    """その部分が残差を「持っている」か＝効きが基準以上で CI が 0 を跨がない。"""
    if not s or s.get("effect") is None or s.get("se") is None:
        return False
    lo, hi = s["ci95"]
    return bool((lo > 0 or hi < 0) and s["effect"] >= min_effect)


def split_verdict(sp, min_effect=0.01, min_hat_r2=MIN_HAT_R2):
    """**`β(f̂)` が持っていれば本物の情報**（ヘッドで直せる）・`u` だけなら勝敗の写し。

    **ただし `β(f̂)` ≈ 0 を artefact と読めるのは `f̂` が強いときだけ**。推定器が弱いと
    （縮んだ予測＝ネットもridgeも平均へ寄せる）読める分が `u` に残り、`β(u)` がそれを拾う
    ＝**弱い `f̂` は artefact の偽陽性を作る**。だから `hat_r2 < min_hat_r2` で `f̂` が
    何も持たないときは判定を出さず **`predictor_too_weak`** を返す（`β(f̂)` は
    「本物の情報」の**下限**であって、`hat_r2` がその下限の被覆率）。
    """
    if not sp:
        return None
    if _carries(sp.get("hat"), min_effect):
        return SPLIT_REAL
    r2 = sp.get("hat_r2")
    if r2 is None or r2 < min_hat_r2:
        return SPLIT_WEAK
    return SPLIT_ARTEFACT if _carries(sp.get("unpredictable"), min_effect) else SPLIT_NONE


#: `time_labels.TIME_COLS` の列（予測器として使うのはこの 2 つだけ）
T_MY_LIFE_END, T_OPP_LIFE_END = 5, 6
#: `--pred-net` が書く列（`f_life_spent`／`f_life_taken` の**状態からの予測**）
PRED_KEY = {"f_life_spent": "p_life_spent", "f_life_taken": "p_life_taken"}


def time_predictions(pred_net, pred_rt, sc, ci, tok):
    """r10 の時間軸ヘッド → `(pred_my_life_end, pred_opp_life_end)`。

    **`net.time` が立っていないネットを渡してはいけない**——ヘッドの重みが初期値のままで、
    未学習のヘッドは定数予測より悪い（`2026-09-13_time_head.md` 結果 (b) で 1.698 対 0.613）。
    """
    if not getattr(pred_net, "time", False):
        raise SystemExit("--pred-net のネットは時間軸ヘッドを持っていない（--time-weight で訓練したもの）")
    rel = NT.relations_or_zeros(pred_net, ci, tok, pred_rt)
    _v, tm = pred_net.value_with_time(sc, ci, tok, *rel)
    tm = np.asarray(tm, np.float32)
    return tm[:, T_MY_LIFE_END], tm[:, T_OPP_LIFE_END]


def collect(net, rt, dirs, holdout_mod=7, limit_games=0, bs=512, theta=THETA, mu=MU,
            pred_net=None, pred_rt=None):
    """holdout の行 → 残差と理論の量（自席ターンの main 行だけ・`kind==0`）。

    `pred_net` を渡すと、そのネットの時間軸ヘッドから**状態だけで作った流れの予測**
    （`p_life_spent`／`p_life_taken`）も同じ行に書く（T11 の主の推定器）。
    """
    recs = []
    stats = {"games": 0, "rows": 0, "skipped_draw": 0}
    games = 0
    pend = []

    def flush():
        if not pend:
            return
        sc = np.stack([p[1] for p in pend]); ci = np.stack([p[2] for p in pend])
        tok = np.stack([p[3] for p in pend])
        rel = NT.relations_or_zeros(net, ci, tok, rt)
        v = np.asarray(net.value(sc, ci, tok, *rel), np.float32).reshape(-1)
        if pred_net is not None:
            my_end, opp_end = time_predictions(pred_net, pred_rt, sc, ci, tok)
        for k, (rec, _s, _c, _t) in enumerate(pend):
            p_hat = (float(v[k]) + 1.0) / 2.0
            rec["p_hat"] = round(p_hat, 5)
            rec["resid"] = rec["z"] - p_hat
            if pred_net is not None:
                # 教師と同じ作り方（`future_quantities`）で予測側も作る＝0 で切る
                rec["p_life_spent"] = max(0.0, float(_s[SC_MY_LIFE]) - float(my_end[k]))
                rec["p_life_taken"] = max(0.0, float(_s[SC_OPP_LIFE]) - float(opp_end[k]))
            recs.append(rec)
        pend.clear()

    for rows, pol, ex, L, ptr, idx in PL.iter_games(dirs, row_cols=ROW_COLS, pol_cols=(),
                                                   extra_fn=_extra):
        seed = int(rows["seed"][idx[0]])
        if holdout_mod and seed % holdout_mod != 0:
            continue
        games += 1
        if limit_games and games > limit_games:
            break
        stats["games"] += 1
        zs = {int(rows["who"][i]): float(rows["z"][i]) for i in idx}
        for pos, i in enumerate(idx):
            if int(rows["kind"][i]) != 0:
                continue
            who = int(rows["who"][i]); t = int(rows["turn"][i])
            if t < 1 or not PL.is_own_turn(who, t):
                continue
            z = zs.get(who, 0.0)
            if z == 0.0:
                stats["skipped_draw"] += 1
                continue
            sc_row = ex["sc"][i]; tok_row = ex["tok"][i]
            rec = {"seed": seed, "turn": t, "z": 1.0 if z > 0 else 0.0,
                   "band": band_key(sc_row), "v0": float(rows["pol_v0"][i])}
            rec.update({("q_" + k): v for k, v in
                        now_quantities(sc_row, tok_row, theta, mu).items()})
            rec.update({("f_" + k): v for k, v in
                        future_quantities(rows, ex["sc"], idx, who, pos).items()})
            stats["rows"] += 1
            pend.append((rec, sc_row, ex["ci"][i], tok_row))
            if len(pend) >= bs:
                flush()
    flush()
    return recs, stats


def _extra(dd, n):
    sc, tok = _pad(np.asarray(dd["scalars"])[:n].astype(np.float32),
                   np.asarray(dd["tokens"])[:n].astype(np.float32))
    return {"sc": sc, "tok": tok,
            "ci": np.asarray(dd["card_idx"])[:n, :NL.N_TOK].astype(np.int64)}


def rank(recs, min_effect=0.01):
    """量を**標準化した効き `|β|·sd`** で並べる（`now` と `future` を分けて）。"""
    if not recs:
        return {"n": 0}
    keys = [k for k in recs[0] if k.startswith(("q_", "f_"))]
    rows = []
    for k in keys:
        out = within_cluster_slope(recs, k)
        out["name"] = k
        out["kind"] = KIND_NOW if k.startswith("q_") else KIND_FUTURE
        if out["kind"] == KIND_FUTURE:
            out["band_r2"] = band_r2(recs, k)
        # **`se` は 0 になりうる**（雑音の無い量）。`and out["se"]` と書くと 0 が偽で
        # その量が黙って落ちる（テストで検出・2026-09-13）。
        if out["effect"] is not None and out["se"] is not None:
            ci = out["ci95"]
            out["significant"] = bool(ci[0] > 0 or ci[1] < 0)
            out["worth_adding"] = bool(out["significant"] and out["effect"] >= min_effect)
        rows.append(out)
    rows.sort(key=lambda r: -(r["effect"] or 0.0))
    return {"n": len(recs), "min_effect": min_effect, "ranked": rows}


def verdict(r):
    """事前登録: **どの量も 0.01 未満なら理論の量はネットに既に入っている**。"""
    if not r or not r.get("ranked"):
        return None
    worth = [x for x in r["ranked"] if x.get("worth_adding")]
    if not worth:
        return "net_already_has_it"
    if any(x["kind"] == KIND_NOW for x in worth):
        return "add_inputs"
    return "add_heads_only"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--net", required=True, help="NRel の npz（value だけ使う）")
    ap.add_argument("--in", dest="src", nargs="+", required=True)
    ap.add_argument("--holdout-mod", type=int, default=7)
    ap.add_argument("--limit-games", type=int, default=0)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--theta", type=float, default=THETA)
    ap.add_argument("--mu", type=float, default=MU)
    ap.add_argument("--min-effect", type=float, default=0.01,
                    help="足す価値の下限（勝率・既定 0.01＝1 ポイント）")
    ap.add_argument("--pred-net", default="",
                    help="T11 の主の推定器＝時間軸ヘッドを持つネット（r10）。"
                         "`--net` と別で良い＝別のネットの残差をこの予測で割れる")
    ap.add_argument("--split", nargs="*", default=[],
                    help="T11: 状態から読める部分と読めない残りに分ける量"
                         f"（既定の候補は {' '.join(PRED_KEY)}）")
    ap.add_argument("--folds", type=int, default=FOLDS, help="交差適合の fold 数（局で割る）")
    add_nu_mode_arg(ap)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    apply_nu_mode(a)

    t0 = time.time()
    stats_t, ab, abm, pwr, isl, _vocab = build_eff_tables()
    tables = (stats_t, ab, abm, pwr, isl)
    net = NL.NRelNet.load(a.net, tables)
    rt = None if "rel" in (net.ablate or ()) else tables
    pred_net = pred_rt = None
    if a.pred_net:
        pred_net = net if os.path.abspath(a.pred_net) == os.path.abspath(a.net) else \
            NL.NRelNet.load(a.pred_net, tables)
        pred_rt = None if "rel" in (pred_net.ablate or ()) else tables
    recs, stats = collect(net, rt, a.src, a.holdout_mod, a.limit_games, a.batch, a.theta, a.mu,
                          pred_net, pred_rt)
    rk = rank(recs, a.min_effect)
    splits = [s for s in (split_future(recs, nm,
                                       pred_key=PRED_KEY.get(nm) if pred_net is not None else None,
                                       folds=a.folds, min_effect=a.min_effect)
                          for nm in a.split) if s]
    res = {"nu_mode": a.nu_mode, "net": os.path.basename(a.net), "stats": stats,
           "pred_net": os.path.basename(a.pred_net) if a.pred_net else None,
           "resid_mean": round(float(np.mean([r["resid"] for r in recs])), 5) if recs else None,
           "rank": rk, "verdict": verdict(rk),
           "splits": splits,
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
