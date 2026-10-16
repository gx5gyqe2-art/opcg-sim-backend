#!/usr/bin/env python3
"""**T137c**（2026-09-23）: `two_curves.py`／`two_curves_metrics.py`（T137a/b）が測った
**曲線のずれ**（`G(t) − R(t)`）が、**盤面の状態量から読めるか**を測る。**表を出すだけ**——
新しい定数は作らず、当てはめ（回帰・係数の調整）もしない。

## 問い

T137b は**ズレが終盤ほど開く**ことを測った（局面別の距離・T107 と整合）。だが「終盤」は
`j`（自席ターン番号）でしか見ていない。**同じ `j` でも状態が違えば理論の粗さは違うはず**——
`Θ_me`／`Θ_opp`（両席の耐久）・`A_me`／`A_opp`（両席の速さ）・ライフ・手札の枚数を、
**同じ残差**（`G_j − R_j`・符号つき＝理論が実現を上回れば正）と突き合わせる。

## 式（新定数ゼロ・既存の量を並べるだけ）

* **状態量**は `kappa_vector.py`（T121・T128）が既に持つ 3 つの部品をそのまま呼ぶ:
  `rate_of_row`（`A`）・`g_of_row`（手札 1 枚あたりの価格）・`state_of_row`（`Θ_me`／`Θ_opp`）。
  **`deck_ids` を必ず渡す**（T128 の規約——渡さないと `A` の流入・効果の項が消える）。
  ライフ・手札枚数は `sc` から直接読む（`SC_MY_LIFE`／`SC_OPP_LIFE`／`SC_MY_HAND`／`SC_OPP_HAND`）。
* **残差**は `two_curves.py --dump` が書いた `(turns, g, r)` から `resid_j = g[j] − r[j]`
  （**曲線を作り直さない**——読むだけ）。
* **相関**は `rate_tracking.corr`（生）と `rate_tracking.demean_by` で `j` の影を抜いたもの
  （T128／T137b と同じ道具の再利用）——**`j` 自体が両方を動かす交絡**（T137b で確認済み）なので、
  抜かないと「状態が悪いから残差が大きい」と「ターンが進んだから両方大きい」を取り違える。

## 測るもの

両記録（実／合成）で、8 つの状態量 × {生の相関, `j` を抜いた相関} の表を出す。**当てはめはしない**
（表を読んで次の T を決めるのはユーザ）。

使い方（`two_curves.py --dump` を先に作っておくこと・同じ `--in`／`--games` で読む）:

    python tests/scripts/two_curves.py --in <dir> --dump tc.json
    python tests/scripts/two_curves_state.py --in <dir> --dump tc.json [--games N] [--json out.json]
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
import kappa_vector as KV  # noqa: E402
import theory_rs as TR  # noqa: E402
from theory_bridge import POL_COLS, ROW_COLS, _extra  # noqa: E402
from theory_rs import MU, THETA  # noqa: E402


# `corr`・`demean_by` は T128 の計器 `rate_tracking.py` から写した（その器は移植の段 0〔2026-10-06〕で退役）。
def corr(a, b):
    """相関（どちらかが定数なら `None`）。"""
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.size < 2 or a.std() < 1e-12 or b.std() < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def demean_by(xs, js):
    """`j`（自席ターン番号）ごとの平均を引く＝**ターン番号の影を抜く**。

    **これをやらないと「守りが効いた」と「ターンが進んだ」を取り違える**——
    ライフも手札もブロッカーも `j` とともに動くので、生の相関は `j` の傾向をそのまま拾う
    （2026-09-20 の実測: `d_life` の生の相関は実 +0.164・合成 −0.021 と**符号が割れた**）。"""
    xs = np.asarray(xs, float); js = np.asarray(js, int)
    out = xs.astype(float).copy()
    for j in np.unique(js):
        m = js == j
        out[m] = xs[m] - xs[m].mean()
    return out


#: 状態量の名前（順序は表の並びに使う）
STATE_KEYS = ("th_me", "th_opp", "a_me", "a_opp", "life_me", "life_opp", "hand_me", "hand_opp")


def state_by_turn(game, seat_decks, seed_g, theta=THETA, mu=MU, with_parts=False):
    """局 1 本ぶんの `(w, t)`（自席ターン・その最初の `kind=0` 行）→ 状態 dict（8 量）。

    **`kappa_vector.collect` の一次通過と同じ規約**——`A` はその席のターンの最初の行から
    （`rate_of_row` の注記どおり）・相手の `A`／手札価格は**相手の直近の自席ターン**から読む
    （`_opp_at` と同じ形）。相手がまだ 1 ターンも打っていない席は state が無い（`None` を返さず省く）。
    計算は Rust の局の駆動（`two_curves_state`・`core::drv_t18`）。

    **T143**: `with_parts=True` なら `th_opp` を**同じ引数の `threshold_parts`** で 3 項
    （`th_opp_life`／`th_opp_hand`／`th_opp_body`）にも割って添える（**足すと `th_opp` に戻る**）。"""
    pin = {"decks": TR.deck_list(KV._deck_pair(seat_decks, seed_g)), "with_parts": bool(with_parts)}
    res = TR.game_call("two_curves_state", game, {"cfg": TR.cfg(theta, mu), "in": pin, "stats": {}, "carry": {}})
    return {(int(w), int(t)): st for w, t, st in res["states"]}


def collect(dump, dirs, limit_games=0, theta=THETA, mu=MU):
    """`dump`（`two_curves.py --dump` の中身）× `dirs`（同じ記録）から残差×状態量の表を作る。"""
    seat_decks = KV._seat_decks(dirs)
    by_seed_w = {(int(row["seed"]), int(row["w"])): row for row in dump}

    residuals, js = [], []
    state_lists = {k: [] for k in STATE_KEYS}
    games = 0
    for game in PL.iter_games(dirs, row_cols=ROW_COLS, pol_cols=POL_COLS, extra_fn=_extra):
        games += 1
        if limit_games and games > limit_games:
            break
        seed_g = TR.seed_of(game)
        st_by_turn = state_by_turn(game, seat_decks, seed_g, theta, mu)
        for w in (0, 1):
            d = by_seed_w.get((seed_g, w))
            if not d:
                continue
            for n, t in enumerate(d["turns"]):
                st = st_by_turn.get((w, t))
                if st is None:
                    continue
                residuals.append(float(d["g"][n]) - float(d["r"][n]))
                js.append(n)
                for k in STATE_KEYS:
                    state_lists[k].append(st[k])

    resid = np.asarray(residuals, float)
    resid_dm = demean_by(resid, js) if residuals else resid
    by_state = {}
    for k in STATE_KEYS:
        xs = np.asarray(state_lists[k], float)
        raw = corr(xs, resid)
        dm = corr(demean_by(xs, js), resid_dm) if residuals else None
        by_state[k] = {"n": int(xs.size), "corr_raw": round(raw, 4) if raw is not None else None,
                       "corr_demeaned": round(dm, 4) if dm is not None else None}
    return {"n": len(residuals), "resid_mean": round(float(resid.mean()), 4) if residuals else None,
           "by_state": by_state}


def build_parser():
    ap = argparse.ArgumentParser(description="曲線のズレ×状態量の表（T137c・当てはめはしない）")
    ap.add_argument("--in", dest="src", nargs="+", required=True)
    ap.add_argument("--dump", required=True, help="two_curves.py --dump が書いた JSON（同じ --in / --games で作ったもの）")
    ap.add_argument("--games", type=int, default=0)
    ap.add_argument("--json", default="")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    with open(a.dump, encoding="utf-8") as f:
        dump = json.load(f)
    out = collect(dump, a.src, a.games)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
