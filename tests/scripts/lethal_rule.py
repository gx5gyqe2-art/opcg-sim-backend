#!/usr/bin/env python3
"""**「今このターンで殺せるか」を規則から数える**（T117・2026-09-19・ユーザ指示「それらの全課題についての修正、検証をお願いします」）。

## 何を測るか

理論には**「今殺せる」判定が 1 つも無い**。規則からは書ける（攻撃の本数・ライフの残り・ブロッカー・
カウンターで止まる枚数）。それがどれだけ当たるかを測る。

**素の規則**（通せる本数 ≥ 相手の残りライフ）は**適合率 0.905・再現率 0.576**（実）——
取りこぼしの最大の理由は**ドンを付与していないこと**（財布を入れると再現率 0.73〜0.76）。

## 式（新定数ゼロ・規則だけ）

* **通る本数** … 自分の攻撃手（`own_attackers_of` の `x = パワー − 相手リーダー`）のうち、
  **相手のアクティブなブロッカーに横取りされない本数**（ブロッカーは安い攻撃から止める・`shield_rate_of` と同じ規則）
* **止められる本数** … 守り手の手札で `c(x)` 枚ひと組を作れる回数（`cuttable_share` の枚数 ÷ `c(x)`）。
  **「相手は切らない」と仮定しない**（それは打ち筋）——規則どおり**切れる枚数の上限**まで切ると読む。
* **ドン付与** … `purse_plan`（T109 の財布）で**アクティブなドンを攻め手に配る**。`x` が上がると `c(x)` が増える
  ＝**止めるのに要る枚数が増える**（規則）。
* **判定** … `通る本数 − 止められる本数 ≥ 相手の残りライフ` なら「殺せる」。

**真値**はその局がそのターンに終わったか（記録の `z`）。**真値は「詰みが在ったか」より狭い**
——詰みが在っても CPU がそのターンに決めないことが在るので、**適合率の上限はこの真値の作り方が決めている**。

## 未解決の食い違い（**この器の数をそのまま「判定の質」と読まないこと**）

**本器の適合率は 0.23〜0.31**（実 300 局・n=3,847 行・真値 300）で、
検証側のプローブ（`docs/reports/2026-09-19_verify_seven.md` §5）の **0.905** と**大きく食い違う**。
原因の候補は 1 つに絞れている——**守り手が切れる枚数の読み方**:

* 本器は**自席の行からしか読めない**ので `相手の手札枚数 × そのデッキの切れる割合`（T91 と同じ規約）で読む。
* 検証側は**相手の手札のカウンター値そのもの**を使っていた（＝**相手席の行を読んでいる**）。

**どちらが正しいかは決まっていない**——完全情報で組む方針（§0.05）なら相手の行も読んでよいが、
**1 行の器としては読めない**。**この食い違いを解く前に「殺せる判定」を理論の項にしてはいけない**。
本器は**その食い違いを明示するために置く**（数は下の実測どおりで、隠さない）。

使い方:

    python tests/scripts/lethal_rule.py --in <records_dir> [--games N] [--don on|off] [--json out.json]
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
import guard_afford as GA  # noqa: E402
import hand_plan as HP  # noqa: E402
from theory_bridge import POL_COLS, ROW_COLS, _extra  # noqa: E402
from theory_order import (MU, SC_MY_DON, SC_MY_LEADER_POWER, SC_OPP_HAND, SC_OPP_LIFE,  # noqa: E402
                          SLOT_OPP_FIELD, THETA, c_of, own_attackers_of)


def stops_of(n_hand_cut, xs):
    """**守り手が止められる本数**（規則・安い攻撃から止めるのが守り手の最良）。

    `c(x)` 枚ひと組でしか止まらない（T61／T99）ので、**安い順に `c(x)` 枚ずつ払って何本止まるか**。
    `n_hand_cut` は切れる札の枚数（`cuttable_share × 手札`）。"""
    left = float(n_hand_cut)
    n = 0
    for x in sorted(float(v) for v in (xs or ())):
        need = c_of(x)
        if need <= 0.0 or left < need - 1e-9:
            break
        left -= need
        n += 1
    return n


def lethal_of_row(sc, tok, ci_row, idx2cid, cards, with_don=True, cut_share=None,
                  theta=THETA, mu=MU):
    """1 行の判定（`(殺せるか, 内訳)`）。**新定数ゼロ・打ち筋を仮定しない**。"""
    sc = np.asarray(sc); tok = np.asarray(tok)
    olp = float(sc[CB.SC_OPP_LEADER_POWER]) * 1e4 or 5000.0
    life = float(sc[SC_OPP_LIFE])
    xs = list(own_attackers_of(tok, olp))
    if with_don:
        # **ドンを財布から攻め手に配る**（T109）＝`x` を上げて止めるのに要る枚数を増やす
        items = HP.hand_items(tok, ci_row, idx2cid, cards, olp,
                              max(1.0, min(5.0, life))) if cards is not None else []
        don = float(sc[SC_MY_DON])
        gained = CB.attach_groups(tok, olp, theta, mu)
        pl = CB.purse_plan(gained + CB.hand_groups(items, cards, olp, theta, mu,
                                                   float(sc[SC_MY_LEADER_POWER]) * 1e4 or 5000.0,
                                                   max(1.0, min(5.0, life)), with_don=False), don)
        # 付与は 1 枚 1000 パワー（規則）＝配れた枚数だけ一番安い攻撃の `x` を持ち上げる
        k = int(round(min(don, 4.0 * max(1, len(xs)))))
        if k > 0 and xs:
            xs = sorted(xs)
            i = 0
            while k > 0 and i < len(xs):
                take = min(4, k)                      # 1 体に付けられるのは 4 枚まで（規則）
                xs[i] += 1000.0 * take
                k -= take
                i += 1
        _ = pl                                        # 財布の内訳は内訳表示用（判定には枚数だけ使う）
    n_block = CB._opp_active_blockers(tok)
    through = max(0, len(xs) - int(n_block))          # ブロッカーは安い攻撃から横取りする
    xs_through = sorted(xs)[int(n_block):] if n_block else sorted(xs)
    # 守り手の切れる枚数は**枚数 × デッキの切れる割合**（1 行からは相手の手札の中身が読めないので
    # **そのデッキの割合**で読む＝T91 の `r` と同じ規約・打ち筋は入らない）。割合が引けなければ 1.0（悲観側）。
    n_cut = float(sc[SC_OPP_HAND]) * (1.0 if cut_share is None else float(cut_share))
    stops = stops_of(n_cut, xs_through)
    hits = max(0, through - stops)
    return (hits >= life and life > 0.0), {"xs": len(xs), "blockers": int(n_block), "through": through,
                                           "stops": stops, "hits": hits, "life": life,
                                           "n_cut": round(n_cut, 3)}


def collect(dirs, limit_games=0, with_don=True):
    """記録を 1 度読んで適合率・再現率・F1 と取りこぼしの内訳を出す。"""
    cards = PL.Cards()
    idx2cid = {i: c for c, i in GA._vocab().items()}
    import deck_refill as DR
    refill = DR.shares_by_seed(dirs)                  # seed -> (席 0 の切れる割合, 席 1 の…)
    tp = fp = fn = rows = truth = 0
    games = 0
    fn_reasons = {"don_would_do_it": 0, "blockers": 0, "stops": 0, "other": 0}
    for r, pol, ex, L, ptr, idx in PL.iter_games(dirs, row_cols=ROW_COLS, pol_cols=POL_COLS,
                                                 extra_fn=_extra):
        games += 1
        if limit_games and games > limit_games:
            break
        seed_g = int(r["seed"][idx[0]]) if len(idx) else -1
        sh = refill.get(seed_g)
        # その局の最後の自席ターン（＝そのターンに終わった）を席ごとに拾う
        last_turn = {}
        for i in idx:
            w, t = int(r["who"][i]), int(r["turn"][i])
            if int(r["kind"][i]) != 0 or not PL.is_own_turn(w, t):
                continue
            if float(r["z"][i]) > 0.5:
                last_turn[w] = max(last_turn.get(w, -1), t)
        win_seat = max(last_turn, key=lambda k: last_turn[k]) if last_turn else None
        seen = set()
        for i in idx:
            w, t = int(r["who"][i]), int(r["turn"][i])
            if int(r["kind"][i]) != 0 or not PL.is_own_turn(w, t) or (w, t) in seen:
                continue
            seen.add((w, t))
            rows += 1
            is_kill = bool(win_seat is not None and w == win_seat and t == last_turn[w])
            truth += int(is_kill)
            cs = float(sh[1 - w]) if sh is not None else None
            pred, d = lethal_of_row(ex["sc"][i], ex["tok"][i], ex["ci"][i], idx2cid, cards,
                                    with_don, cs)
            if pred and is_kill:
                tp += 1
            elif pred and not is_kill:
                fp += 1
            elif is_kill and not pred:
                fn += 1
                if not with_don:
                    p2, _ = lethal_of_row(ex["sc"][i], ex["tok"][i], ex["ci"][i], idx2cid,
                                          cards, True, cs)
                    fn_reasons["don_would_do_it" if p2 else "other"] += 1
                elif d["blockers"] > 0:
                    fn_reasons["blockers"] += 1
                elif d["stops"] > 0:
                    fn_reasons["stops"] += 1
                else:
                    fn_reasons["other"] += 1
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    return {"games": games, "rows": rows, "truth": truth, "with_don": with_don,
            "n_pred": tp + fp, "tp": tp, "fp": fp, "fn": fn,
            "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(2 * prec * rec / max(1e-9, prec + rec), 4),
            "fn_reasons": fn_reasons}


def main(argv=None):
    ap = argparse.ArgumentParser(description="「今殺せるか」の判定を規則から測る（T117）")
    ap.add_argument("--in", dest="src", nargs="+", required=True)
    ap.add_argument("--games", type=int, default=0)
    ap.add_argument("--don", default="on", choices=("on", "off"))
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)
    out = collect(a.src, a.games, a.don == "on")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
