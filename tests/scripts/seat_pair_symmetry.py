#!/usr/bin/env python3
"""**両席の行を対にして、勝率の読み `p` が席で反対称かを測る**（T151-1・2026-09-24）。

## 問い

完全情報（`docs/cpu_theory_gap.md` §0.05）では、同じ局面を両席から読んだ勝率は足して 1 になる
（`V_me = −V_opp`・`docs/game_theory.md` §2-2）。交点の橋の行（`crossing_bridge.collect` の `rows_out`）は
席 `w` の自席ターン `t` の開始点で `p_w(t) = W(τ_opp − τ_me)` を出す。**相手の次の行** `(1−w, t+1)` は同じ局面を
1 手（`w` のターン）だけ後から反対側で読んだものなので、**平均では** `p_w(t) + p_{1−w}(t+1) ≈ 1` のはず
（`w` の 1 ターンで局面は動くが、理論が正しければ動きの期待値は 0＝マルチンゲール）。

保存済みの測定（T149g・実 w41／合成 w39+w42）は**両席の行をまとめた mean p が 0.568／0.594**・優勢側
（`p>0.5`）の行の比率が **60%／63%**——対称なら 0.5／50% でなければならない。本器はその非対称を
**対の単位で**数字にする。

## 式（新定数ゼロ・当てはめない）

行 `(seed, w, t)` の相手は `(seed, 1−w, t+1)`（先手の `j` 番目の自席ターンは `t = 2j+1`・後手は `t = 2j+2`）。

* `sum_minus_1 = p_a + p_b − 1`（対ごと・平均が自席びいきの大きさ・正なら手番の席に甘い）
* `both_fav`＝両方 `p > 0.5`（反対称なら 0 件）／`both_dog`＝両方 `p < 0.5`
* **恒等式**（現行の読み方の署名）: 行 `a=(w,t)` の `τ_me` は `per_seat[(w,t)]`・行 `b=(1−w,t+1)` の `τ_opp` は
  **同じ** `per_seat[(w,t)]`（`crossing_bridge.py` の `op = per_seat[(1-w, prev_o[-1])]`）なので
  `τ_me(a) == τ_opp(b)` が**厳密に**成り立ち、`d_a + d_b = τ_opp(a) − τ_me(b)`＝**対のずれは相手の時計の古さそのもの**
  （`τ_opp(a)` は相手の前ターン開始・`τ_me(b)` は相手の今のターン開始）。相手の時計を同じ瞬間から読む切替
  （T151-2）では `τ_me(a) != τ_opp(b)` になる＝`shared_term_share` が 1 から 0 へ落ちるのが切替の署名。

## 検算の予告（測る前に書く・T151）

1. **現既定**: `mean(sum_minus_1)` は正で、**≈ +0.14（実）／+0.19（合成）**（mean p 0.568／0.594 の 2 倍 − 1 から
   導いた値・新しい定数ではない）。`shared_term_share = 1.0`。
2. **切替後（同じ瞬間から読む）**: `mean(sum_minus_1)` → 0.00±0.03・`both_fav` の割合 → ほぼ 0・
   `shared_term_share` → 0。**殺す基準**: 切替後も mean p が 0.55 を超えたまま＝古さは原因ではない。

## 取りこぼしの開示

相手の次の行が無い行（局の最後の行・`--pre-settle` で相手側だけ落ちた行）は対にならない。数を `unpaired` に
出す（理由の内訳は行だけからは分からないので、`--pre-settle off` と比べて読む）。

使い方:

    python tests/scripts/seat_pair_symmetry.py --in <records_dir> [...] [--games N] [--pre-settle on|game|off] \
        [--sigma-rel VALUE] [--json out.json]
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

import crossing_bridge as CB  # noqa: E402
import pre_settle_asymmetry as PA  # noqa: E402
from theory_order import MU, THETA  # noqa: E402

#: 恒等式（`τ_me(a) == τ_opp(b)`）の判定の刻み。浮動小数の同値は `==` で見る（同じ値の写しなので誤差は 0）が、
#: JSON を経由した行でも読めるように微小な幅を許す。
EPS = 1e-9


def partner_key(seed, who, t):
    """行 `(seed, w, t)` の相手＝`(seed, 1−w, t+1)`（相手の**次の**自席ターン）。"""
    return (seed, 1 - int(who), int(t) + 1)


def pair_rows(rows):
    """`pre_settle_asymmetry.rows_with_p` の行を対にする。返り値 `(pairs, unpaired)`。
    `pairs[i]` は `{"seed","t","who","stage","p_a","p_b","z_a","z_b","sum_minus_1","both_fav","both_dog",
    "d_a","d_b","tau_me_a","tau_opp_a","tau_me_b","tau_opp_b"}`。`unpaired` は相手の次の行が無かった行数。"""
    by_key = {}
    for r in rows:
        if r.get("t") is None or r.get("seed") is None or r.get("who") is None:
            continue
        by_key[(r["seed"], int(r["who"]), int(r["t"]))] = r
    pairs, unpaired = [], 0
    for r in rows:
        if r.get("t") is None or r.get("seed") is None or r.get("who") is None:
            unpaired += 1
            continue
        b = by_key.get(partner_key(r["seed"], r["who"], r["t"]))
        if b is None:
            unpaired += 1
            continue
        pa, pb = float(r["p"]), float(b["p"])
        pairs.append({"seed": r["seed"], "t": int(r["t"]), "who": int(r["who"]), "stage": r.get("stage"),
                      "p_a": pa, "p_b": pb, "z_a": float(r["z"]), "z_b": float(b["z"]),
                      "sum_minus_1": pa + pb - 1.0,
                      "both_fav": bool(pa > 0.5 and pb > 0.5), "both_dog": bool(pa < 0.5 and pb < 0.5),
                      "d_a": float(r["d"]), "d_b": float(b["d"]),
                      "tau_me_a": r.get("tau_me"), "tau_opp_a": r.get("tau_opp"),
                      "tau_me_b": b.get("tau_me"), "tau_opp_b": b.get("tau_opp")})
    return pairs, unpaired


def _summ(pairs):
    if not pairs:
        return {"n": 0}
    s = np.asarray([q["sum_minus_1"] for q in pairs], float)
    return {"n": len(pairs),
            "mean_sum_minus_1": float(s.mean()),
            "se_sum_minus_1": float(s.std(ddof=1) / np.sqrt(len(s))) if len(s) > 1 else None,
            "both_fav_share": float(np.mean([q["both_fav"] for q in pairs])),
            "both_dog_share": float(np.mean([q["both_dog"] for q in pairs])),
            "mean_p_a": float(np.mean([q["p_a"] for q in pairs])),
            "mean_p_b": float(np.mean([q["p_b"] for q in pairs]))}


def identity_check(pairs):
    """恒等式の検算: `shared_term_share`＝`τ_me(a) == τ_opp(b)` が成り立つ対の割合（現行の読み方なら 1.0）。
    `identity_resid`＝`|(d_a + d_b) − (τ_opp(a) − τ_me(b))|` の平均（現行なら 0）。τ が無い行は数えない。"""
    have = [q for q in pairs if all(q.get(k) is not None for k in ("tau_me_a", "tau_opp_a", "tau_me_b", "tau_opp_b"))]
    if not have:
        return {"n": 0}
    shared = [abs(float(q["tau_me_a"]) - float(q["tau_opp_b"])) <= EPS for q in have]
    resid = [abs((q["d_a"] + q["d_b"]) - (float(q["tau_opp_a"]) - float(q["tau_me_b"]))) for q in have]
    return {"n": len(have), "shared_term_share": float(np.mean(shared)), "identity_resid": float(np.mean(resid))}


def pair_table(pairs, unpaired=0):
    """対の表: 全体・段別（行 `a` の段）・席別（行 `a` の席）＋恒等式。"""
    out = {"all": _summ(pairs), "unpaired": int(unpaired), "identity": identity_check(pairs),
           "by_stage": {st: _summ([q for q in pairs if q["stage"] == st]) for st in PA.STAGES},
           "by_who": {str(w): _summ([q for q in pairs if q["who"] == w]) for w in (0, 1)}}
    return out


def collect(dirs, limit_games=0, pre_settle="on", slope="theory", sigma_rel=None, w_err="rel"):
    """記録を 1 度読み、行に `p` を付け、対の表を返す。`pre_settle` は `crossing_bridge.PRE_SETTLE_MODES`。"""
    old = CB.PRE_SETTLE_MODE
    try:
        CB.set_pre_settle_mode(pre_settle)
        rows_out, _ledger, stats, _th, _tc = CB.collect(dirs, limit_games, THETA, MU, "const")
    finally:
        CB.set_pre_settle_mode(old)
    if sigma_rel is None:
        sigma_rel = CB.sigma_rel_for(dirs)
    rows = PA.rows_with_p(rows_out, slope, sigma_rel, w_err)
    pairs, unpaired = pair_rows(rows)
    return {"games": stats.get("games"), "n_rows": len(rows), "pre_settle": pre_settle, "sigma_rel": sigma_rel,
            "opp_clock": getattr(CB, "OPP_CLOCK_MODE", None),
            "overall": {"mean_p": float(np.mean([r["p"] for r in rows])) if rows else None,
                        "mean_z": float(np.mean([r["z"] for r in rows])) if rows else None,
                        "favorite_share": (sum(1 for r in rows if r["p"] > 0.5) / len(rows)) if rows else None},
            "pairs": pair_table(pairs, unpaired)}


def main(argv=None):
    ap = argparse.ArgumentParser(description="両席の行を対にして p の反対称を測る（T151-1）")
    ap.add_argument("--in", dest="src", nargs="+", required=True)
    ap.add_argument("--games", type=int, default=0)
    ap.add_argument("--pre-settle", default="on", choices=CB.PRE_SETTLE_MODES)
    ap.add_argument("--slope", default="theory")
    ap.add_argument("--sigma-rel", type=float, default=None)
    ap.add_argument("--w-err", default="rel", choices=("abs", "rel"))
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)
    out = collect(a.src, a.games, a.pre_settle, a.slope, a.sigma_rel, a.w_err)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
