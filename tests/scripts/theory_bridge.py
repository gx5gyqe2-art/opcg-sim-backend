"""**理論の点数は勝敗に換算できるか**（T28・橋・読み取り専用）。

ユーザ提案 2026-09-14「**打った手によるなにかしらの累積スコアを定義して、それを勝敗に紐づける**」。
`docs/cpu_theory_gap.md` **§0.3**（設計）・**§0.4**（暫定値の台帳）。

## なぜ要るか——**勝率は 1 局 1 ビットで、分解能が足りない**

いまの検証は最終的にアリーナの勝率に依るが、**384 局で ±35 Elo**・作っている差は **5 Elo 級**
（`measurement.md` §4）＝**原理的に見えない**。**1 手ごとに値が付けば標本が 2 桁増える。**

## 何を足すか

**各判断点で「その場の最善からいくら落としたか」**（`≤ 0`）を席ごとに足す:

```
S  = Σ_t s_t            ΔS = S_自席 − S_相手席        →  ΔS は勝敗を予測するか
```

**要点は `s_t` が「同じ局面の中の差」であること**——**局面の良し悪しはその場で差し引かれる**ので、
`S` は「どれだけ良い局面に居たか」ではなく**「各決断でいくら取りこぼしたか」**の累計になる。

### 攻め側（自席ターンの本体）

```
s_t = θ(実際に打った手) − max_a θ(a)          候補の中の差
```

### 守り側（相手ターンの窓）——**候補リストは使わない**（ユーザ指摘 2026-09-14）

記録の守りの窓には**候補リストが無い**（実測 0%）。しかし**理論は規則を持っている**
（`c(x) ≤ Θ` なら守る）ので、**状態から助言が出て、記録から実際の行動が出る**:

```
守る費用 = c(x)·μ        受ける費用 = Θ·μ
s_t     = −( 実際に払った費用 − min(2 つのうち払えた方) )
```

> **払えなかった行は誤りと数えない**（`measurement.md` §1「〜しないではなく〜できない」）。
> カウンターを持っていてもドンが足りなければ**守れない**ので、
> **`guard_afford` の予算計算（無料カウンター＋ナップサック）をそのまま使う**。
> ブロッカーが居れば無料で守れる。

## 暫定値（§0.4 の台帳・**感度を付けて回す**）

| | 穴 | この器での扱い |
|---|---|---|
| **P2** | イベント・効果の値付けが無い | `--silent {zero,exclude}` ——**両方出して挟む** |
| **P3** | `Θ` が 2 経路で食い違う | `--theta` ——1.15／1.325／1.88 で回す |
| **P1** | `w(状態)` の式が無い | **帯ごとに傾きを出す**（揃えば実害なし） |

## 読み方（事前登録）

- **`ΔS` が勝敗を予測する**＝理論の点数に**勝率への為替レート**が付く
  （`dP(win)/dΔS`）＝**以後は棋譜を数え直すだけで理論の変更を評価できる**。
- **予測しない**＝理論の順序は勝敗と結びついていない＝**橋は架からない**。
- **攻め側だけ効く／守り側だけ効く**なら、**どちらの半分が正しいか**が分かれる。

**限界**: 観察であって因果ではない。**「勝っている席は良い手を打つ余裕がある」**という
選択交絡は行内の差では消えない（帯で層別して見る）。守りの窓は
**そのターン最初の窓だけ**・来る攻撃は**最大のもの**で近似する。

実行例:
  OPCG_LOG_SILENT=1 python tests/scripts/theory_bridge.py --in ~/w41 --out ~/bridge.json
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
import guard_afford as GA  # noqa: E402
from order_acc import band_of  # noqa: E402
from theory_order import (MU, PWR_EPS, POL_COLS, SC_MY_DON, SC_MY_LEADER_POWER,  # noqa: E402
                          SC_MY_LIFE, SC_OPP_LEADER_POWER, SC_OPP_LIFE, THETA,
                          c_of, incoming_x, opp_chars_of, score_candidate, slot_power,
                          theta_of)

ROW_COLS = ("who", "turn", "seed", "z", "kind", "step", "pol_len", "pol_chosen", "pol_v0",
            "sig")   # `sig` は `PL.label_game`（守り側の take/guard の判定）が要る
#: **P3** の感度（§0.4）——盤面経路 1.325・恒等式 1.88・出荷既定 1.15
THETA_SWEEP = (1.15, 1.325, 1.88)
#: **P2** の扱い（理論が値を付けられなかった行）
SILENT_MODES = ("zero", "exclude")
#: **T28-c**——「払えた」の判定が粗いせいで反転しているのではないかを分ける閾値。
#: 守る力が来る攻撃をこれだけ上回っていれば**余裕で払えた**と見なす（暫定値・感度を取る）。
#: **ブロッカーが居る行は無条件で余裕**（レストするだけでドンを使わない）。
MARGIN_COMFORT = 2000.0


def _extra(dd, n):
    return {"sc": np.asarray(dd["scalars"])[:n].astype(np.float32),
            "tok": np.asarray(dd["tokens"])[:n].astype(np.float32),
            "ci": np.asarray(dd["card_idx"])[:n]}


def guard_step(tok, sc, played, free, paid, theta=THETA, mu=MU, margin_comfort=None):
    """**守りの窓 1 つ**の取りこぼし（`≤ 0`）と、判定に使った内訳。

    **払えなかった行は誤りと数えない**——`measurement.md` §1。
    """
    xs = [x for x in incoming_x(tok) if x >= -PWR_EPS]
    if not xs:
        return None
    x = max(xs)                                   # そのターン最大の攻撃で近似
    blocker = bool((np.asarray(tok)[GA.SLOT_OWN_FIELD][:, GA.S_BLOCKER] > 0.5).any())
    budget = int(round(float(sc[SC_MY_DON])))
    afford_pw = free + GA.knapsack(paid, budget)
    can_guard = bool(blocker or afford_pw >= x)
    cost_take = float(theta) * float(mu)
    cost_guard = float(c_of(x)) * float(mu)
    best = min(cost_take, cost_guard) if can_guard else cost_take
    actual = cost_guard if played == "guard" else cost_take
    if played == "guard" and not can_guard:
        # 守れないはずの行で守っている＝予算の見積りが渋い。**誤りにしない**
        actual = best
    # **T28-c**: 余裕の大きさ。**貧しい席を減点していないか**を分けるために出す。
    margin = (float("inf") if blocker else float(afford_pw) - float(x))
    return {"s": -max(0.0, actual - best), "x": x, "can_guard": can_guard, "margin": margin,
            "comfortable": bool(can_guard and margin >= (MARGIN_COMFORT
                                                        if margin_comfort is None
                                                        else float(margin_comfort))),
            "played": played, "theory_says": ("guard" if (can_guard and cost_guard < cost_take)
                                              else "take")}


def _state_of(sc, ci, idx2cid):
    """判断点の状態（条件の判定用）。**記録だけで作れる**——リーダーは `card_idx` の
    0/1（vocab index）・ステージの有無は 22/23。"""
    try:
        import condition_value as CV
    except Exception:
        return None
    ci = np.asarray(ci)
    return CV.state_from_scalars(sc, idx2cid.get(int(ci[0])), idx2cid.get(int(ci[1])),
                                 my_stage=int(ci[22]) > 0, opp_stage=int(ci[23]) > 0)


def _add(rec, band, s, side):
    """**行ごとに帯へ足す**（T28-b）——決着後の雑さが接戦帯に混ざらないようにする。

    `side="grdc"` は**余裕で払えた守りの行だけ**の別勘定（T28-c）＝`grd` と二重に足す。
    """
    if side != "grdc":
        rec["s_%s" % side] += float(s)
        rec["n_%s" % side] += 1
    b = rec["band"].setdefault(band, {"s": 0.0, "n": 0, "s_atk": 0.0, "n_atk": 0,
                                      "s_grd": 0.0, "n_grd": 0,
                                      # **T28-c**: 余裕で払えた守りの行だけの集計
                                      "s_grdc": 0.0, "n_grdc": 0})
    b["s"] += float(s); b["n"] += 1
    b["s_%s" % side] += float(s); b["n_%s" % side] += 1


def collect(dirs, limit_games=0, theta=THETA, mu=MU, theta_mode="const", nu_targets="leader",
            silent="zero", margin_comfort=None):
    """(局, 席) ごとに攻め側と守り側の取りこぼしを足す。"""
    cards = PL.Cards()
    idx2cid = {i: c for c, i in GA._vocab().items()}
    per = {}
    stats = {"games": 0, "atk_rows": 0, "atk_silent": 0, "grd_rows": 0, "grd_no_attack": 0,
             # **T28-c**: 余裕で払えた守りの行の数
             "grd_comfortable": 0}
    games = 0
    for rows, pol, ex, L, ptr, idx in PL.iter_games(dirs, row_cols=ROW_COLS,
                                                    pol_cols=POL_COLS, extra_fn=_extra):
        games += 1
        if limit_games and games > limit_games:
            break
        stats["games"] += 1
        seed = int(rows["seed"][idx[0]])
        life0 = ex["sc"][:, 0]
        labels, _unk = PL.label_game(rows, pol, life0, L, ptr, idx, cards)
        seen = set()
        for n, i in enumerate(idx):
            w, t = int(rows["who"][i]), int(rows["turn"][i])
            if t < 1:
                continue
            z = float(rows["z"][i])
            key = (seed, w)
            rec = per.setdefault(key, {"seed": seed, "who": w, "z": None,
                                       "s_atk": 0.0, "s_grd": 0.0, "n_atk": 0, "n_grd": 0,
                                       "n_silent": 0, "v0": [],
                                       # **T28-b: 行ごとに帯を決めてから足す**
                                       "band": {}})
            if z != 0.0:
                rec["z"] = 1.0 if z > 0 else 0.0
            sc, tok = ex["sc"][i], ex["tok"][i]
            if PL.is_own_turn(w, t):
                if int(rows["kind"][i]) != 0:
                    continue
                k = int(L[i])
                ch = int(rows["pol_chosen"][i])
                if k < 2 or ch < 0 or ch >= k:
                    continue
                stats["atk_rows"] += 1
                th = theta_of(tok, float(sc[SC_MY_LIFE]), float(sc[SC_MY_DON]),
                              mode=theta_mode, theta=theta)
                ctx = {"theta": th, "mu": mu,
                       "opp_leader_power": float(sc[SC_OPP_LEADER_POWER]) * 1e4,
                       "my_leader_power": float(sc[SC_MY_LEADER_POWER]) * 1e4,
                       "r_turns": max(1.0, min(5.0, float(sc[SC_OPP_LIFE]))), "don_k": 1,
                       # **条件の判定に使う状態**（`condition_value.py`・2026-09-14）。
                       # リーダーとステージは `card_idx` の 0/1 と 22/23 に在る。
                       "st": _state_of(sc, ex["ci"][i], idx2cid)}
                if nu_targets == "board":
                    ctx["opp_chars"] = opp_chars_of(tok)
                b = int(ptr[i])
                vals = []
                for j in range(b, b + k):
                    sig = json.loads(pol["pol_sig"][j])
                    tl = sig[2] if len(sig) > 2 else None
                    v = score_candidate(sig, str(pol["pol_cid"][j]) or None,
                                        (str(pol["pol_tcid"][j]) or None) if tl else None,
                                        ctx, cards,
                                        src_power=slot_power(tok, pol["pol_si"][j]),
                                        tgt_power=slot_power(tok, pol["pol_ti"][j]),
                                        don_k=pol["pol_k"][j])
                    vals.append(v)
                scored = [v for v in vals if v is not None]
                played_v = vals[ch]
                bnd = band_of(abs(float(rows["pol_v0"][i])))
                if played_v is None or len(scored) < 2:
                    stats["atk_silent"] += 1
                    rec["n_silent"] += 1
                    if silent == "zero":
                        # **無言の行を「取りこぼし 0」として母数に入れる**
                        _add(rec, bnd, 0.0, "atk")
                    continue                     # `exclude` は母数にも入れない
                _add(rec, bnd, float(played_v) - max(scored), "atk")
                rec["v0"].append(abs(float(rows["pol_v0"][i])))
            else:
                if (w, t) in seen or int(labels[n]) < 0:
                    continue
                seen.add((w, t))
                played = PL.PLAN_CLASSES[int(labels[n])]
                if played not in ("take", "guard"):
                    continue
                free, paid, _slots = GA.hand_counters(tok, ex["ci"][i], idx2cid, cards)
                got = guard_step(tok, sc, played, free, paid, theta, mu, margin_comfort)
                if got is None:
                    stats["grd_no_attack"] += 1
                    continue
                stats["grd_rows"] += 1
                bnd = band_of(abs(float(rows["pol_v0"][i])))
                _add(rec, bnd, got["s"], "grd")
                if got["comfortable"]:
                    stats["grd_comfortable"] += 1
                    _add(rec, bnd, got["s"], "grdc")   # **余裕で払えた行だけの別勘定**
    return per, stats


def pair_by_band(per, band):
    """**その帯の行だけ**で 2 席を突き合わせる（T28-b の本体）。"""
    by = {}
    for (seed, w), r in per.items():
        by.setdefault(seed, {})[w] = r
    out = []
    for seed, seats in by.items():
        a, b = seats.get(0), seats.get(1)
        if a is None or b is None or a["z"] is None or b["z"] is None:
            continue
        ba, bb = a["band"].get(band), b["band"].get(band)
        if not ba or not bb or ba["n"] < 1 or bb["n"] < 1:
            continue
        out.append({"seed": seed, "z": a["z"],
                    "dS": ba["s"] - bb["s"],
                    "dS_per_row": ba["s"] / ba["n"] - bb["s"] / bb["n"],
                    "dS_atk": (ba["s_atk"] / ba["n_atk"] if ba["n_atk"] else 0.0)
                              - (bb["s_atk"] / bb["n_atk"] if bb["n_atk"] else 0.0),
                    "dS_grd": (ba["s_grd"] / ba["n_grd"] if ba["n_grd"] else 0.0)
                              - (bb["s_grd"] / bb["n_grd"] if bb["n_grd"] else 0.0),
                    # **T28-c**: 両席が「余裕で払えた」行を持つ局だけで意味を持つ
                    "dS_grdc": ((ba["s_grdc"] / ba["n_grdc"] if ba["n_grdc"] else None)
                                if (ba["n_grdc"] and bb["n_grdc"]) else None),
                    "n_grdc": (ba["n_grdc"], bb["n_grdc"]),
                    "n": ba["n"] + bb["n"], "dn": ba["n"] - bb["n"],
                    "silent": a["n_silent"] + b["n_silent"], "v0": None})
    return out


def pair_games(per, silent="zero"):
    """(局) ごとに 2 席を突き合わせて `ΔS` を作る（全帯まとめ・T28 の形）。"""
    by = {}
    for (seed, w), r in per.items():
        by.setdefault(seed, {})[w] = r
    out = []
    for seed, seats in by.items():
        if len(seats) != 2:
            continue
        a, b = seats.get(0), seats.get(1)
        if a is None or b is None or a["z"] is None or b["z"] is None:
            continue
        if silent == "exclude" and (a["n_silent"] or b["n_silent"]):
            # **無言の行があった局を丸ごと外す**のは厳しすぎるので、
            # `exclude` は「無言の行を母数から外す」＝既に加算していない＝ここでは何もしない。
            pass
        na, nb = a["n_atk"] + a["n_grd"], b["n_atk"] + b["n_grd"]
        if na < 1 or nb < 1:
            continue
        out.append({"seed": seed, "z": a["z"],
                    "dS": (a["s_atk"] + a["s_grd"]) - (b["s_atk"] + b["s_grd"]),
                    "dS_atk": a["s_atk"] - b["s_atk"],
                    "dS_grd": a["s_grd"] - b["s_grd"],
                    # **1 手あたりに正規化した版**——手数の差が `ΔS` を動かすため。
                    # **先手は手番が 1 つ多い**ので、生の和は席順を拾いうる。
                    "dS_per_row": (a["s_atk"] + a["s_grd"]) / na
                                  - (b["s_atk"] + b["s_grd"]) / nb,
                    "n": na + nb, "dn": na - nb,
                    "silent": a["n_silent"] + b["n_silent"],
                    "v0": float(np.mean(a["v0"])) if a["v0"] else None})
    return out


def auc(scores, labels):
    """順位の AUC（**当てはめない**）。同値は 0.5 として数える。"""
    s = np.asarray(scores, np.float64); y = np.asarray(labels, np.float64)
    pos, neg = (y > 0.5), (y <= 0.5)
    n_p, n_n = int(pos.sum()), int(neg.sum())
    if n_p == 0 or n_n == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), np.float64)
    ss = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[pos].sum() - n_p * (n_p + 1) / 2.0) / (n_p * n_n))


def slope(pairs, key="dS"):
    """**為替レート**——`ΔS` 1 単位あたり勝率がどれだけ動くか（最小二乗）。"""
    x = np.array([p[key] for p in pairs], np.float64)
    y = np.array([p["z"] for p in pairs], np.float64)
    if len(x) < 10 or float(x.var()) <= 0.0:
        return None
    xd = x - x.mean()
    return float((xd * (y - y.mean())).sum() / (xd * xd).sum())


def _boot(pairs, reps=200, seed=0, key="dS"):
    if reps <= 0 or len(pairs) < 10:
        return [None, None]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(int(reps)):
        sub = [pairs[k] for k in rng.integers(0, len(pairs), len(pairs))]
        v = slope(sub, key)
        if v is not None and np.isfinite(v):
            vals.append(v)
    return ([round(float(np.percentile(vals, 2.5)), 5),
             round(float(np.percentile(vals, 97.5)), 5)] if len(vals) >= 10 else [None, None])


def summarise(pairs, reps=200, seed=0):
    if len(pairs) < 10:
        return {"n": len(pairs)}
    out = {"games": len(pairs),
           "dS_mean": round(float(np.mean([p["dS"] for p in pairs])), 5),
           "dS_sd": round(float(np.std([p["dS"] for p in pairs])), 5),
           "silent_rows": int(sum(p["silent"] for p in pairs))}
    # **手数の差が勝敗を説明していないか**——説明していれば `ΔS` は席順を拾っている
    out["dn_mean"] = round(float(np.mean([p["dn"] for p in pairs])), 3)
    out["dn_auc"] = (round(auc([p["dn"] for p in pairs], [p["z"] for p in pairs]), 4)
                     if auc([p["dn"] for p in pairs], [p["z"] for p in pairs]) is not None
                     else None)
    # **T28-c**: 余裕で払えた守りの行だけで引き直す（貧しい席の減点を除く）
    cf = [p for p in pairs if p.get("dS_grdc") is not None and p.get("n_grdc")]
    if len(cf) >= 30:
        for p in cf:
            a, b = p["n_grdc"]
            p["dS_grdc_pr"] = p["dS_grdc"] - 0.0      # 既に 1 手あたり
        out["grd_comfortable_only"] = {
            "games": len(cf),
            "auc": (round(auc([p["dS_grdc"] for p in cf], [p["z"] for p in cf]), 4)
                    if auc([p["dS_grdc"] for p in cf], [p["z"] for p in cf]) is not None
                    else None),
            "slope": (round(slope(cf, "dS_grdc"), 5) if slope(cf, "dS_grdc") else None),
            "slope_ci95": _boot(cf, reps, seed, "dS_grdc")}
    for key in ("dS", "dS_per_row", "dS_atk", "dS_grd"):
        out[key] = {"auc": (round(auc([p[key] for p in pairs],
                                      [p["z"] for p in pairs]), 4)
                            if auc([p[key] for p in pairs], [p["z"] for p in pairs])
                            is not None else None),
                    "slope": (round(slope(pairs, key), 5)
                              if slope(pairs, key) is not None else None),
                    "slope_ci95": _boot(pairs, reps, seed, key)}
    # **P1 の検査**——帯ごとに傾きが揃えば `w` の変動は実害なし
    out["by_band"] = {}
    for nm in ("close", "mid", "decided"):
        sub = [p for p in pairs if p["v0"] is not None and band_of(p["v0"]) == nm]
        if len(sub) >= 50:
            out["by_band"][nm] = {"n": len(sub), "slope": (round(slope(sub), 5)
                                                           if slope(sub) else None)}
    a = out["dS"]["auc"]
    ci = out["dS_per_row"]["slope_ci95"]        # **正規化した版で判定する**
    # **事前登録**: `ΔS` が勝敗を予測するか（傾きの CI が 0 を含まないか）
    if None in ci:
        out["verdict"] = "undecided"
    elif ci[0] > 0:
        out["verdict"] = "bridge_holds"
    elif ci[1] < 0:
        out["verdict"] = "bridge_inverted"
    else:
        out["verdict"] = "no_link"
    out["auc_all"] = a
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="src", nargs="+", required=True, help="n_records のディレクトリ")
    ap.add_argument("--limit-games", type=int, default=0)
    ap.add_argument("--theta", type=float, default=THETA, help="**P3** の暫定値（§0.4）")
    ap.add_argument("--theta-mode", default="const", choices=("const", "board", "max"))
    ap.add_argument("--nu-targets", default="leader", choices=("leader", "board"))
    ap.add_argument("--silent", default="zero", choices=SILENT_MODES,
                    help="**P2** の暫定値——`zero` は無言の行を取りこぼし 0 として母数に入れる／"
                         "`exclude` は母数にも入れない")
    ap.add_argument("--margin-comfort", type=float, default=MARGIN_COMFORT,
                    help="**T28-c** の暫定値——守る力が来る攻撃をこれだけ上回れば「余裕で払えた」")
    ap.add_argument("--cond-unknown", type=float, default=1.0,
                    help="**判らない条件の係数**（§0.4 の感度。1.0＝上限・0.0＝下限）")
    ap.add_argument("--boot-reps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)

    try:
        import condition_value as CV
        CV.set_unknown_factor(a.cond_unknown)
    except Exception:
        pass
    t0 = time.time()
    per, stats = collect(a.src, a.limit_games, a.theta, MU, a.theta_mode, a.nu_targets,
                         a.silent, a.margin_comfort)
    pairs = pair_games(per, a.silent)
    res = {"stats": stats,
           "provisional": {"P3_theta": a.theta, "P2_silent": a.silent,
                           "T28c_margin": a.margin_comfort,
                           "note": "§0.4 の暫定値。感度を付けて読む"},
           "summary": summarise(pairs, a.boot_reps, a.seed),
           # **T28-b: 行ごとに帯で切ってから足した版**（判定の主はこちら）
           "per_band": {nm: summarise(pair_by_band(per, nm), a.boot_reps, a.seed)
                        for nm in ("close", "mid", "decided")},
           "seconds": round(time.time() - t0, 1)}
    txt = json.dumps(res, ensure_ascii=False, indent=2)
    print(txt)
    if a.out:
        with open(os.path.expanduser(a.out), "w", encoding="utf-8") as fh:
            fh.write(txt + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
