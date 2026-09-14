"""無差別点 `Θ` を**盤面から直接**出す（記録だけ・ネット不要・`λ` を通らない）。

**なぜ要るか**（2026-09-14・`cpu_theory_gap.md` §8.1 の T2・ユーザ指示「机上で確認できる
理論値を先に」）: `Θ` の既定 **1.15** は `Θ = λ/μ − 1 − τ_value` から出ているが、
**その `λ`（相手ターンの 0.1362）は恒等式を 32% 超過する**（`2026-09-14_life_price.md`）。
割り戻すと `Θ ≈ 0.67` になり、**飽和点 `x*` が 1000 未満**に落ちて
「安い攻撃を受けたのは誤り」という判定まで反転しうる。**連鎖が 3 段あるので独立に出す。**

### `Θ` は順序統計量である（§8 の定義に戻る）

> `Θ` は「**この先まだ守る、最も高い攻撃の費用**」＝「`G` 回は守らねばならない」という
> 制約に紐づく**シャドー価格**（`game_theory.md` §8）。

したがって**盤面から直接計算できる**——`λ`・`μ`・`τ_value` を一切通らない:

```
N = これから来る攻撃数（相手のリーダー＋相手のキャラ）
B = 自分のアクティブなブロッカー数
L = 自分のライフ（0 になった次の被弾で負け＝受けられるのは残り L 回）
G = max(0, N − L − B)                     必ず守る回数
Θ = 「来る攻撃の c(x) を安い順に並べて G 番目」   ＝ 守る中で最も高いものの費用
```

`G = 0` なら**領域 1**（全部受けても死なない）＝**制約が無い＝シャドー価格も無い**
（守るのは純粋な無駄・§8）。この窓は `Θ` の集計から外す。

### 混同してはいけない 3 つの量（§8 の表）

| 量 | 定義 | 本器の欄 |
|---|---|---|
| **`Θ`** | **G 番目に安い攻撃の費用**（シャドー価格・**これが無差別点**） | `theta` |
| `c_mean_blocked` | 守る G 本の**平均**費用 | 定義上 `≤ Θ` |
| `c_mean_all` | 来る攻撃**全部**の平均費用 | 「x の分布で重み付けした `c`」 |

**`Θ` は平均ではなく順序統計量**なので、平均で代用すると必ず過小に出る。

### 相手のドンの扱い（上下で挟む）

自席の main 行で読むと、**相手のキャラにはまだドンが乗っていない**（付与は相手のターンに起きる）
＝`x` は**過小**、したがって `c(x)` と `Θ` も**過小**。そこで 2 本出す:

| 腕 | 相手のドン | 読み |
|---|---|---|
| `no_don` | 乗せない | **`Θ` の下限** |
| **`real_don`** | **実測の付与率（ドンの 47%）** | **本命**（下記） |
| `all_don` | 持っているドン**全部**を乗せる | **`Θ` の上限**（現実にはここまで付与しない） |

> **実測の付与は持っているドンの 47% である**（2026-09-14・`2026-09-14_don_accounting.md`）
> ——1 ターンのドン 6.29 個のうち**付与 2.94・カードを出すのに 2.93**・余り 0.51。
> **全部を付与に回す `all_don` は現実より強い想定**なので、判定は **`real_don`** で行う。

> **相手のドンの枚数は `scalars` から直接読めない**。自席の main 行では相手の `don_active` は
> **0.40**・`don_rested` は 2.76 しか無く、**付与済みの分（2.94・`2026-09-14_don_accounting.md`）は
> どちらにも入らない**（キャラに乗っている）。リフレッシュで全部戻るので、
> **自分のアクティブなドン（自席で 6.29）を「相手が次のターンに持つ枚数」の代理に使う**
> ——ドンは両者 1 ターン 1 個ずつ増えるので差は 1 以内である。
>
> 配り方は**高い攻撃から順に 1 個ずつ**（round-robin）＝**攻撃側の最適ではない**
> （§10 は「段を跨ぐように配れ」と言う）。中立な配り方なので `Θ` の上限としては緩い。

### 恒等式との照合

`Θ` が出れば §9 の恒等式は**残差で `τ_value` を与える**（今まで逆算の幅しか無かった量）:

```
τ_value = λ/μ − 1 − Θ
```

**`τ_value` が [0, 0.7] に入れば体系は自己整合**。負に出れば `Θ` か `λ` のどちらかが誤り。

### 限界（必ず添える）

- **`N` は「相手のリーダー＋相手のキャラ数」**（`race_state._state` と同じ規約）。
  相手のターンには全部アクティブに戻るので `can_attack_now` では絞らない。
  **相手がキャラを出して攻撃を増やす分は入らない**＝`N` は過小側。
- **`B` は今アクティブなブロッカー**（相手のターンまでに増えうる）＝過小側。
- **`c(x)` は合成 800 デッキの平均曲線**（リーダーごとの差 2.09〜3.00 は入らない・§18）。
- **ブロッカーで止めた分は手札を使わない**ので `G` から引いているが、
  ブロッカー自身が KO される費用は数えていない。

実行例:
  OPCG_LOG_SILENT=1 python tests/scripts/theta_price.py --in ~/tvs_rec --out ~/theta.json
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
from race_state import _extra as _race_extra  # noqa: E402
from race_state import (_PWR_SCALE, _SLOT_OPP, _SLOT_OWN, _T_BLK, _T_CHAR,  # noqa: E402
                        _T_PWR)
from theory_order import c_of  # noqa: E402

ROW_COLS = ("who", "turn", "seed", "z", "kind", "step", "pol_len")
SC_MY_LIFE, SC_OPP_LIFE = 0, 1
SC_MY_DON, SC_OPP_DON = 2, 4
SC_MY_HAND = 6
SC_TURN = 10
#: `λ/μ − 1`（§18 の実測・自席と相手ターン）。`τ_value = λ/μ − 1 − Θ` の照合に使う
LAM_OVER_MU_MINUS_1 = {"own": 1.674, "opp": 1.473}
#: 恒等式で割り戻した相手ターンの `λ/μ − 1`（`2026-09-14_life_price.md` §4）
LAM_OVER_MU_MINUS_1_FIXED = 0.860
#: 飽和点を探す刻み
X_GRID = tuple(range(0, 11000, 1000))
#: **実測の付与率**＝1 ターンのドンのうち付与に回る割合（2.94 / 6.29・`don_accounting`）
DON_SHARE = 0.467


def turn_band(t):
    return "T<=4" if t <= 4 else ("T5-8" if t <= 8 else "T9+")


def incoming(tk_row, with_don=0):
    """**これから来る攻撃の超過パワー `x`**（自席の main 行から相手の枠を読む）。

    攻撃側は**相手のリーダー（枠 1）＋相手のキャラ（枠 7〜11）**、守るのは自分のリーダー（枠 0）。
    `with_don` を渡すと**いちばん高い攻撃から順に 1 個 +1000 ずつ**乗せる
    （相手は最も通る攻撃を伸ばすので、これが `x` の上限側）。
    """
    tk = np.asarray(tk_row)
    mine = float(tk[0, _T_PWR]) * _PWR_SCALE
    opp = tk[_SLOT_OPP]
    pwr = [float(tk[1, _T_PWR]) * _PWR_SCALE]                      # 相手リーダー
    ch = opp[:, _T_CHAR] > 0.5
    pwr += [float(v) * _PWR_SCALE for v in opp[ch, _T_PWR]]
    pwr.sort(reverse=True)                                          # 高い順
    for k in range(int(max(with_don, 0))):
        if not pwr:
            break
        pwr[k % len(pwr)] += 1000.0                                 # 高い側から順に配る
    return [p - mine for p in pwr]


def my_blockers(tk_row):
    tk = np.asarray(tk_row)
    own = tk[_SLOT_OWN]
    ch = own[:, _T_CHAR] > 0.5
    return int(((own[:, _T_BLK] > 0.5) & ch).sum())


def x_star(theta):
    """**飽和点**＝`c(x) ≥ Θ` になる最小の x（これを超えて積むのは無駄）。"""
    for x in X_GRID:
        if c_of(float(x)) >= theta:
            return x
    return X_GRID[-1]


def window(sc_row, tk_row, with_don=0):
    """1 つの自席 main 行 → シャドー価格 `Θ` とその材料。`G = 0` なら `theta` は None。"""
    xs = incoming(tk_row, with_don)
    n = len(xs)
    life = int(round(float(sc_row[SC_MY_LIFE])))
    blk = my_blockers(tk_row)
    g = max(0, n - life - blk)
    cs = sorted(c_of(x) for x in xs)
    out = {"n_attacks": n, "life": life, "blockers": blk, "G": g,
           "turn": int(round(float(sc_row[SC_TURN]))),
           "c_mean_all": (round(float(np.mean(cs)), 4) if cs else None)}
    if g <= 0 or g > len(cs):
        out["theta"] = None                     # 領域 1＝制約が無い＝シャドー価格が無い
        return out
    out["theta"] = round(cs[g - 1], 4)          # **G 番目に安い**＝守る中で最も高い
    out["c_mean_blocked"] = round(float(np.mean(cs[:g])), 4)
    out["x_star"] = x_star(cs[g - 1])
    return out


def collect(dirs, limit_games=0, don_share=0.0):
    recs = []
    games = 0
    for rows, pol, ex, L, ptr, idx in PL.iter_games(dirs, row_cols=ROW_COLS, pol_cols=(),
                                                    extra_fn=_extra):
        games += 1
        if limit_games and games > limit_games:
            break
        seed = int(rows["seed"][idx[0]])
        seen = set()
        for i in idx:
            w = int(rows["who"][i]); t = int(rows["turn"][i])
            if t < 1 or not PL.is_own_turn(w, t) or int(rows["kind"][i]) != 0:
                continue
            if (w, t) in seen:
                continue
            seen.add((w, t))
            sc = ex["sc"][i]
            # **相手が次のターンに持つドンの代理は「自分のいまのアクティブなドン」**。
            # 相手の `don_active` は自席では 0.40 しか無い（使い切っている＋付与分は
            # キャラに乗っていてどちらの列にも出ない）ので、そのまま使うと上限にならない。
            don = int(round(float(sc[SC_MY_DON]) * don_share))
            r = window(sc, ex["tk"][i], don)
            r["seed"] = seed
            recs.append(r)
    return recs, games


def _extra(dd, n):
    _sc, tk = _race_extra(dd, n)
    return {"sc": np.asarray(dd["scalars"])[:n, :14].astype(np.float32), "tk": tk}


def _mean_ci(vals, seeds):
    """対局クラスタの SE（1 局から複数の窓を採るので行で割ると過小・§14-7）。"""
    if not vals:
        return None
    by = {}
    for v, s in zip(vals, seeds):
        by.setdefault(s, []).append(v)
    per = np.array([float(np.mean(v)) for v in by.values()], float)
    m = float(np.mean(np.array(vals, float)))
    se = float(per.std(ddof=1) / np.sqrt(len(per))) if len(per) > 1 else None
    out = {"mean": round(m, 4), "n": len(vals), "games": len(per),
           "median": round(float(np.median(np.array(vals, float))), 4)}
    if se is not None:
        out["ci95"] = [round(m - 1.96 * se, 4), round(m + 1.96 * se, 4)]
    return out


def summarise(recs, side="opp"):
    if not recs:
        return None
    constrained = [r for r in recs if r["theta"] is not None]
    out = {"windows": len(recs), "constrained": len(constrained),
           "region1_share": round(1.0 - len(constrained) / len(recs), 4),
           "n_attacks": round(float(np.mean([r["n_attacks"] for r in recs])), 3),
           "blockers": round(float(np.mean([r["blockers"] for r in recs])), 3),
           "G": round(float(np.mean([r["G"] for r in recs])), 3)}
    if not constrained:
        return out
    sd = [r["seed"] for r in constrained]
    out["theta"] = _mean_ci([r["theta"] for r in constrained], sd)
    out["c_mean_blocked"] = _mean_ci([r["c_mean_blocked"] for r in constrained], sd)
    out["c_mean_all"] = _mean_ci([r["c_mean_all"] for r in constrained], sd)
    out["x_star"] = _mean_ci([float(r["x_star"]) for r in constrained], sd)
    # **飽和点の分布**（1000 未満が多いなら「超過パワーを積む価値が無い」）
    xs = [r["x_star"] for r in constrained]
    out["x_star_dist"] = {str(x): round(float(np.mean([v == x for v in xs])), 4)
                          for x in sorted(set(xs))}
    # 恒等式の残差＝`τ_value`
    th = out["theta"]["mean"]
    out["tau_implied"] = {
        "from_measured_lam": round(LAM_OVER_MU_MINUS_1[side] - th, 4),
        "from_identity_fixed_lam": round(LAM_OVER_MU_MINUS_1_FIXED - th, 4),
        "note": "τ_value = λ/μ − 1 − Θ（[0, 0.7] に入れば自己整合・負なら Θ か λ が誤り）",
    }
    # ターン帯ごと（`Θ` は盤面の関数なので動くはず・§11）
    out["by_turn"] = {}
    for tb in ("T<=4", "T5-8", "T9+"):
        sub = [r for r in constrained if turn_band(r["turn"]) == tb]
        if len(sub) >= 30:
            out["by_turn"][tb] = {"n": len(sub),
                                  "theta": _mean_ci([r["theta"] for r in sub],
                                                    [r["seed"] for r in sub]),
                                  "G": round(float(np.mean([r["G"] for r in sub])), 3)}
    # ライフごと（`Θ` はライフが減るほど上がるはず＝守る本数が増える）
    out["by_life"] = {}
    for lv in range(0, 7):
        sub = [r for r in constrained if r["life"] == lv]
        if len(sub) >= 30:
            out["by_life"][str(lv)] = {"n": len(sub), "G": round(float(np.mean(
                [r["G"] for r in sub])), 3), "theta": round(float(np.mean(
                    [r["theta"] for r in sub])), 4)}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="src", nargs="+", required=True, help="n_records のディレクトリ")
    ap.add_argument("--limit-games", type=int, default=0)
    ap.add_argument("--don-share", type=float, default=DON_SHARE,
                    help="相手が付与に回すドンの割合（既定は実測の 0.467）")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)

    t0 = time.time()
    res = {"theta_default_now": 1.15, "lam_over_mu_minus_1": LAM_OVER_MU_MINUS_1,
           "lam_over_mu_minus_1_fixed": LAM_OVER_MU_MINUS_1_FIXED}
    res["don_share_measured"] = DON_SHARE
    for arm, share in (("no_don", 0.0), ("real_don", a.don_share), ("all_don", 1.0)):
        recs, games = collect(a.src, a.limit_games, share)
        res["games"] = games
        res[arm] = summarise(recs)
        res[arm]["don_share"] = share
    res["seconds"] = round(time.time() - t0, 1)
    txt = json.dumps(res, ensure_ascii=False, indent=2)
    print(txt)
    if a.out:
        with open(os.path.expanduser(a.out), "w", encoding="utf-8") as fh:
            fh.write(txt + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
