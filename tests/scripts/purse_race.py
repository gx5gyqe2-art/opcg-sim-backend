#!/usr/bin/env python3
"""**財布をレースで割る**（T111・ユーザの問い 2026-09-19「1本にまとめた定数はどんな意味を持つ？」の答え）。

## 問い

T109（速さ `A` の財布）と T110（耐久 `Θ` の予算）で**どちらもドンの金額は規則どおり**になったが、
**同じ席の `A` と `Θ` が互いを見ていない**。1 本のナップサックにまとめるには
`価値 = A の分 + ρ × Θ の分` の `ρ` が要る——が、

```
ρ の次元 = [価格/ターン] ÷ [価格] = [1/ターン]        ← ρ は「率」であって無次元ではない
ρ = (∂D/∂Θ_me) ÷ (∂D/∂A_me) = A_me² / (A_opp·Θ_opp) = (1/τ_me) · (A_me/A_opp)
対称な局面では ρ = A/Θ = 1/τ                          ← 「耐久 1 単位 ＝ 速さ 1 単位 ÷ 残りターン」
```

**`ρ` は局面の量**（どちらが速い側か・決め手がどれだけ近いか）なので、**定数として置くと
打ち筋を式に入れる**ことになる。しかも `ρ` は**微分＝線形近似**で、財布の選択は離散なので合わない。

## 式（**新定数ゼロ・`ρ` は現れない**）

```
1. 財布の組（手札の札・場の攻め手への付与）から、予算 P での (A の分, Θ の分) のパレート境界を出す
2. 境界の各点で D = τ_opp − τ_me = Θ_me/A_opp − Θ_opp/A_me を直に測り、最大の点を採る
```

`D` は両軸で単調増なので最適点は必ず境界上に在る＝**境界を出せば交換レートは要らない**。

## 本器が測ること（T111 の予告 (a)）

**速い側は速さに寄せ、遅い側は耐久に寄せるか**——出なければ式が間違っている。
`A_opp`・`Θ_opp` は**その席の行から読める分**（相手の盤面の攻め手・相手のライフ・手札・体）を使う
＝**1 パス近似**（両席の固定点は次の段）。

使い方:

    python tests/scripts/purse_race.py --in <records_dir> [--games N] [--json out.json]
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
import deck_refill as DR  # noqa: E402
import don_ledger as DL  # noqa: E402
import guard_afford as GA  # noqa: E402
import hand_plan as HP  # noqa: E402
import theory_order as TO  # noqa: E402
from theory_bridge import POL_COLS, ROW_COLS, _extra  # noqa: E402


#: **器は `crossing_bridge` に 1 本**（`opp_board_slope`／`free_cuttable_g`／`race_alloc`）。
#: 本器はそれを**記録の上で回して配分の内訳を集計するだけ**（1 トピック = 1 か所）。
opp_board_slope = CB.opp_board_slope
free_cuttable_g = CB.free_cuttable_g


def row_split(sc, tok, ci_row, idx2cid, cards, deck_me=None, deck_opp=None,
              theta=None, mu=None):
    """**1 行ぶんの配分**＝`{base_a, base_th, a_opp, th_opp, front_n, pick, maxa, gain}`。

    `pick`＝`D` を最大にする点 `(A の分, Θ の分)`・`maxa`＝**今の形**（全部速さ）の点・
    `gain`＝`D(pick) − D(maxa)`。"""
    theta = TO.THETA if theta is None else theta
    mu = TO.MU if mu is None else mu
    sc = np.asarray(sc)
    olp = float(sc[TO.SC_OPP_LEADER_POWER]) * 1e4 or 5000.0
    mlp = float(sc[TO.SC_MY_LEADER_POWER]) * 1e4 or 5000.0
    r = max(1.0, min(5.0, float(sc[TO.SC_OPP_LIFE])))
    don = float(sc[TO.SC_MY_DON])
    items = HP.hand_items(tok, ci_row, idx2cid, cards, olp, r) or []

    # --- 財布の外に在る土台（配分では動かない分）
    lead, chars = CB.theory_slope_parts(tok, olp, theta, mu, with_don=False)
    base_a = lead + chars
    if deck_me:
        base_a += float(DR.a_of(deck_me, olp, don, theta, mu))
        base_a += float(DR.e_of(deck_me, mlp, r, don))
    base_th = float(CB.threshold_of_me(sc, tok, g_hand=free_cuttable_g(items, mu)))
    th_life, th_hand, th_body = CB.threshold_parts(sc, tok, g_hand=mu, hand_blocker=0.0)
    th_opp = float(th_life + th_hand + th_body)
    a_opp = opp_board_slope(tok, mlp, theta, mu)
    if deck_opp:
        a_opp += float(DR.a_of(deck_opp, mlp, don, theta, mu))

    # --- 財布の中の選択肢（速さ 2 種・耐久 2 種）
    groups = CB.hand_groups(items, cards, olp, theta, mu, mlp, r, with_don=False)
    groups += CB.attach_groups(tok, olp, theta, mu)
    groups += CB.theta_groups(items, cards, olp, mu, don_left=None)
    front = CB.purse_pareto(groups, don)
    pick_a, pick_th, _parts, pick_d = CB.choose_by_race(front, base_a, base_th, a_opp, th_opp)
    maxa = max(front, key=lambda p: p[0]) if front else (0.0, 0.0, {})
    maxa_d = CB.race_margin(base_a + maxa[0], base_th + maxa[1], a_opp, th_opp)
    return {"base_a": base_a, "base_th": base_th, "a_opp": a_opp, "th_opp": th_opp,
            "don": don, "front_n": len(front),
            "pick_a": pick_a, "pick_th": pick_th, "pick_d": pick_d,
            "maxa_a": maxa[0], "maxa_th": maxa[1], "maxa_d": maxa_d,
            "gain": pick_d - maxa_d, "faster": base_a > a_opp,
            "paid": float(_parts.get("paid") or 0.0)}


def _agg():
    return {"n": 0, "pick_a": 0.0, "pick_th": 0.0, "maxa_a": 0.0, "maxa_th": 0.0,
            "gain": 0.0, "differs": 0, "th_chosen": 0, "front_n": 0,
            # **処方が打ち回しを説明するか**（T111 の予告 (c)）: そのターン実際に使い残したドン。
            # **配分が「耐久を買え」と言う行で、CPU も実際に多く残していたか**を見る。
            "idle": 0.0, "idle_th": 0.0, "n_th": 0, "idle_a": 0.0, "n_a": 0,
            "paid_theory": 0.0, "spent_real": 0.0}


def _add(d, row):
    d["n"] += 1
    for k in ("pick_a", "pick_th", "maxa_a", "maxa_th", "gain", "front_n"):
        d[k] += float(row[k])
    d["differs"] += int(abs(row["pick_a"] - row["maxa_a"]) > 1e-9
                        or abs(row["pick_th"] - row["maxa_th"]) > 1e-9)
    buys_th = row["pick_th"] > 1e-9
    d["th_chosen"] += int(buys_th)
    d["paid_theory"] += float(row.get("paid") or 0.0)
    if row.get("idle") is not None:
        d["idle"] += float(row["idle"])
        d["spent_real"] += float(row.get("spent") or 0.0)
        if buys_th:
            d["idle_th"] += float(row["idle"]); d["n_th"] += 1
        else:
            d["idle_a"] += float(row["idle"]); d["n_a"] += 1


def _fin(d):
    n = max(1, d["n"])
    return {"n": d["n"], "pick_a": round(d["pick_a"] / n, 5), "pick_th": round(d["pick_th"] / n, 5),
            "maxa_a": round(d["maxa_a"] / n, 5), "maxa_th": round(d["maxa_th"] / n, 5),
            "gain": round(d["gain"] / n, 5), "front_n": round(d["front_n"] / n, 2),
            "differs_share": round(d["differs"] / n, 4),
            "theta_chosen_share": round(d["th_chosen"] / n, 4),
            "paid_theory": round(d["paid_theory"] / n, 3),
            "spent_real": round(d["spent_real"] / n, 3),
            "idle_real": round(d["idle"] / n, 3),
            # **予告 (c)**: 配分が「耐久を買え」と言う行 対 「速さを買え」と言う行で、
            # **CPU が実際に使い残したドン**。処方が打ち回しを説明するなら前者が大きい。
            "idle_when_theta": round(d["idle_th"] / max(1, d["n_th"]), 3),
            "idle_when_speed": round(d["idle_a"] / max(1, d["n_a"]), 3),
            "n_theta": d["n_th"], "n_speed": d["n_a"]}


def collect(dirs, limit_games=0, theta=None, mu=None):
    cards = PL.Cards()
    idx2cid = {i: c for c, i in GA._vocab().items()}
    seat_decks = DR.decks_by_seed(dirs)
    out = {"all": _agg(), "faster": _agg(), "slower": _agg()}
    by_turn = {}
    games = 0
    for rows, pol, ex, L, ptr, idx in PL.iter_games(dirs, row_cols=ROW_COLS, pol_cols=POL_COLS,
                                                    extra_fn=_extra):
        games += 1
        if limit_games and games > limit_games:
            break
        seed_g = int(rows["seed"][idx[0]]) if len(idx) else -1
        decks = seat_decks.get(seed_g) or (None, None)
        first, last = {}, {}
        for i in idx:
            if int(rows["kind"][i]) != 0:
                continue
            w, t = int(rows["who"][i]), int(rows["turn"][i])
            if t < 1 or not PL.is_own_turn(w, t):
                continue
            first.setdefault((w, t), i)
            last[(w, t)] = i
        for (w, t), i in first.items():
            row = row_split(ex["sc"][i], ex["tok"][i], ex["ci"][i], idx2cid, cards,
                            deck_me=decks[w], deck_opp=decks[1 - w], theta=theta, mu=mu)
            # **実際の使い残し**＝そのターン最後の行のアクティブ（`don_ledger` と同じ読み）
            i2 = last[(w, t)]
            idle = float(np.asarray(ex["sc"][i2])[DL.SC_MY_ACTIVE])
            row["idle"] = idle
            row["spent"] = max(0.0, row["don"] - idle)
            _add(out["all"], row)
            _add(out["faster" if row["faster"] else "slower"], row)
            j = (t - 1) // 2
            by_turn.setdefault(min(j, 8), _agg())
            _add(by_turn[min(j, 8)], row)
    res = {"games": games, "all": _fin(out["all"]),
           "faster": _fin(out["faster"]), "slower": _fin(out["slower"]),
           "by_own_turn": {str(k): _fin(v) for k, v in sorted(by_turn.items())}}
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description="財布をレース（D）で割る（T111・交換レートを置かない）")
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
