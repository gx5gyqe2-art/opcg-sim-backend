#!/usr/bin/env python3
"""**T137d**（2026-09-23）: 曲線の上に**決着の時刻**（T136・規則）を印し、**その瞬間に累積の損害が
相手の耐久 `Θ_opp(t)` に届いているか**を測る——**2 つの独立に組んだ橋**（T137a〜c の累積の橋と
T136 の規則の橋）を初めて直接突き合わせる。

## 問い

T136 は「相手が最善で守っても通る本数 ≥ 相手の残りライフ＋1」という**規則だけ**で決着（詰み）を判定する。
T137a〜c は「実際に与えた損害の累積 `R(t)`」「選んだ手の理論値を積んだ `G(t)`」「相手の今の耐久
`Θ_opp(t)`」を持っている（`Θ_opp` は `crossing_bridge` の交点の橋が使っているのと同じ式）。**2 つは
独立に組んだのに同じ『相手を仕留めた』という事実を指しているはず**——規則が「もう詰んでいる」と
宣言した瞬間、累積の損害は相手の耐久にどれだけ近いか（届いている／まだ足りない／通り過ぎている）。

## 式（新定数ゼロ・既存の 3 つの器の再利用のみ）

* **決着の時刻 `t_declare`**: `lethal_rule.settled_map(dirs)` の `{(seed,w,t): bool}` から、
  **勝った席**の自席ターンを昇順に見て最初に `True` になった `t`（無ければこの局は測れない・除外して数える）。
* **`R(t_declare)`／`G(t_declare)`**: `two_curves.py --dump` の `(turns, r, g)` から、`t_declare`
  **以下で最後の**ターンの累積値（階段関数として読む——`t_declare` に決定行が無くても壊れない）。
* **`Θ_opp(t_declare)`**: `two_curves_state.py` の `state_by_turn`（`kappa_vector.state_of_row` の
  そのままの再利用）が同じ `(w, t_declare)` に対して返す `th_opp`。**時々刻々の値**（交点の橋が
  各ターンで比べているのと同じ量）——固定の基準点ではなく、宣言のその瞬間の値を読む。

**主の比較は `R` 対 `Θ_opp`**（`crossing_bridge` の `F_end/Θ_start` と同じ単位＝「実際に与えた損害」
対「残りの耐久」）。`G` 対 `Θ_opp` は**単位が違う副次の数**として添える——`G` は攻撃だけでなく出す・
効果・付与も全部積んだ理論の総額で、Θ を削らない手（T137b で play 0.21〜0.25・effect ほぼ無相関）を
含むので、ゲームが進むほど `Θ` から離れて大きくなる（1 に近づく理由が無い）。**最初の実測（測る前に
`G` を主に置いて予告を書いたら比が 6 前後で外れ、`R` は既に累積してきた実現の損害なので `Θ` と
直接比べられると気づいて主従を入れ替えた**——この経緯は報告に残す。

## 予告（測る前に書く・`R` 対 `Θ_opp` について）

T137c は「理論の過大（T89）は相手の残り体力が大きいときに大きい」ことを見つけた。**決着の宣言は
相手の体力がほぼ尽きた局面でしか起きない**ので、**`R(t_declare)` は `Θ_opp(t_declare)` の近くに
来るはず**（比が 1 に近い）——T137c の読みが正しければ、終盤の一点である決着の瞬間には実現の損害と
耐久のズレが小さくなっているはずだから。

使い方（`two_curves.py --dump` を先に作っておく・同じ `--in`／`--games` で読む）:

    python tests/scripts/two_curves.py --in <dir> --dump tc.json
    python tests/scripts/two_curves_settle.py --in <dir> --dump tc.json [--games N] [--json out.json]
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
import guard_afford as GA  # noqa: E402
import kappa_vector as KV  # noqa: E402
import lethal_rule as LR  # noqa: E402
import two_curves_state as TS  # noqa: E402
from theory_bridge import POL_COLS, ROW_COLS, _extra  # noqa: E402
from theory_order import MU, THETA  # noqa: E402


def declared_turns_by_seed_w(settled):
    """`settled_map` の `{(seed,w,t): bool}` から、`declared=True` のターンだけを
    `{(seed,w): [t, ...]}`（昇順）にまとめる。"""
    out = {}
    for (seed_g, w, t), declared in settled.items():
        if declared:
            out.setdefault((seed_g, w), []).append(t)
    for k in out:
        out[k].sort()
    return out


def g_at_or_before(turns, g_list, t):
    """`turns`（昇順の自席ターン番号）／`g_list`（その累積値）から、`t` **以下で最後**のものを返す
    （階段関数——`t` そのものに値付けできた決定が無くても、直前の累積値をそのまま使う）。無ければ 0.0。"""
    val = 0.0
    for tt, gg in zip(turns, g_list):
        if tt <= t:
            val = gg
        else:
            break
    return val


def collect(dump, dirs, limit_games=0, theta=THETA, mu=MU):
    cards = PL.Cards()
    idx2cid = {i: c for c, i in GA._vocab().items()}
    seat_decks = KV._seat_decks(dirs)
    settled = LR.settled_map(dirs, limit_games)
    declared_ts = declared_turns_by_seed_w(settled)
    by_seed_w = {(int(row["seed"]), int(row["w"])): row for row in dump}

    g_ratios, g_diffs = [], []
    r_ratios, r_diffs = [], []
    n_winners = n_no_declare = n_no_state = 0
    games = 0
    for rows, pol, ex, L, ptr, idx in PL.iter_games(dirs, row_cols=ROW_COLS, pol_cols=POL_COLS, extra_fn=_extra):
        games += 1
        if limit_games and games > limit_games:
            break
        seed_g = int(rows["seed"][idx[0]]) if len(idx) else -1
        st_by_turn = TS.state_by_turn(rows, ex, idx, cards, idx2cid, seat_decks, seed_g, theta, mu)
        for w in (0, 1):
            d = by_seed_w.get((seed_g, w))
            if not d or d.get("winner") != w:
                continue                              # 勝った席だけ測る（決着＝その席が仕留めた瞬間）
            n_winners += 1
            ts = declared_ts.get((seed_g, w)) or []
            if not ts:
                n_no_declare += 1
                continue
            t_declare = ts[0]
            st = st_by_turn.get((w, t_declare))
            if st is None:
                n_no_state += 1
                continue
            th_opp = st["th_opp"]
            g_td = g_at_or_before(d["turns"], d["g"], t_declare)
            r_td = g_at_or_before(d["turns"], d["r"], t_declare)
            g_diffs.append(g_td - th_opp)
            r_diffs.append(r_td - th_opp)
            if th_opp > 1e-9:
                g_ratios.append(g_td / th_opp)
                r_ratios.append(r_td / th_opp)

    def _summary(ratios, diffs):
        ratios = np.asarray(ratios, float); diffs = np.asarray(diffs, float)
        return {"ratio": {"n": int(ratios.size),
                          "mean": round(float(ratios.mean()), 4) if ratios.size else None,
                          "median": round(float(np.median(ratios)), 4) if ratios.size else None,
                          "p25": round(float(np.percentile(ratios, 25)), 4) if ratios.size else None,
                          "p75": round(float(np.percentile(ratios, 75)), 4) if ratios.size else None},
                "diff": {"n": int(diffs.size),
                         "mean": round(float(diffs.mean()), 4) if diffs.size else None,
                         "median": round(float(np.median(diffs)), 4) if diffs.size else None}}

    out = {"n_winners": n_winners, "n_no_declare": n_no_declare, "n_no_state": n_no_state,
           "n_measured": len(r_diffs),
           # **主**: 実現の累積 `R(t_declare)` 対 `Θ_opp`（`crossing_bridge` の `F/Θ_start` と同じ単位——
           # どちらも「実際に与えた損害」対「残りの耐久」なので直接比べられる）。
           "r_vs_theta": _summary(r_ratios, r_diffs),
           # **副**: 理論の累積 `G(t_declare)`（攻撃だけでなく出す・効果・付与も全部積んだ値）対 `Θ_opp`。
           # **単位が違う**（`G` は手の種類を問わず全部を積む・`Θ` は残りの耐久 1 つ）ので、
           # 1 に近いことを期待する数ではない——「決着までに払った理論の総額は残り耐久の何倍か」という
           # 別の問い（`G` の中身の大半は耐久を削らない・T137b の型別相関を参照）。
           "g_vs_theta": _summary(g_ratios, g_diffs)}
    return out


def build_parser():
    ap = argparse.ArgumentParser(description="決着の瞬間に G(t) が Θ_opp(t) に届いているか（T137d）")
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
