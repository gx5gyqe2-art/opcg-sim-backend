#!/usr/bin/env python3
"""**T143**（2026-09-23・ユーザ指示「その分解はお願いします。」）: **決着の瞬間の `Θ_opp` を
ライフ／手札／体に分解し、そのターンから終局までに実際に要った損害と部品ごとに突き合わせる**。

## 問い

T137d は「決着の瞬間でも `Θ_opp` が約 0.26 残る」と報告した。**その 0.26 は何でできているか**——
そして**そもそも 0 になるべき量だったのか**。

## 測る前に気づいた T137d の比較の誤り

T137d は**開始からの累積** `R(t_declare)` を**その瞬間の残りの的** `Θ_opp(t_declare)` で割り、
「1 に近いはず」と予告した。だが `Θ(t)` は「**ここから先に要る**損害」の見積もり（T96 の規約:
`Θ(t)` 対 `要(t) = F_end − F_before(t)`）で、**宣言はそのターンの開始時に立つ**——とどめを刺す
ターンの仕事はまだ残っているので、**`Θ` は 0 でなくてよい**。**比べるべきは `Θ(t_declare)` 対
`要(t_declare)`**（累積の `R` を分子に置いた比は、そもそも 1 になる理由が無かった）。

## 2 つの `Θ`（同じ局・同じ席・同じ宣言ターン）

* **1 行の器**（`kappa_vector.state_of_row`＝`crossing_bridge.threshold`・T137c／T137d と線形の帳簿が使う）
* **交点の橋**（`crossing_bridge.collect` の `per_seat`＝`theta_check`）——1 行の器に無い**補正が 3 つ**掛かる:
  ① **手札の窓**（T116・`THETA_HAND_WINDOW=horizon`＝`min(手札の項, SR·τ0)`）
  ② **手札のブロッカー**（T106・`THETA_HAND_BLOCKER_MODE=on`・体の項に足す）
  ③ **手札 1 枚あたりの価格 `g` の読み方**（橋はターン最後の行・使い残しのドンでイベントも切れる〔T110〕／
     1 行の器はターン最初の行・ドン無し）

## 式（新定数ゼロ・既存の部品の再利用のみ）

* 1 行の器の 3 項: `two_curves_state.state_by_turn(..., with_parts=True)`（`threshold_parts` を `th_opp` と同じ引数で）
* 橋の 3 項と内訳: `crossing_bridge.collect` の `theta_check`（`th_life`／`th_hand`／`th_hand_raw`〔窓の前〕／
  `th_body`／`th_hb`〔体のうち手札のブロッカー〕／`need`）
* 要った損害の 3 項: `two_curves.py --dump` の `r_life`／`r_hand`／`r_body`（`harm_of` の 3 項）で
  `要_部品 = 累積(終局) − 累積(宣言ターンの直前)`
* 決着の時刻: `lethal_rule.settled_map` の勝った席の最初の `True`（T137d と同じ）

## 予告（測る前に書く）

1. **恒等式**: 1 行の器の 3 項の和 = `th_opp`（ビット一致）・橋の 3 項の和 = `theta`・**同じ行ならライフの項は
   2 つの `Θ` で一致**・**体の項は手札のブロッカーを引けば一致**・**dump から作った要 = 橋の要**
   （別々に書いた 2 つの実装・T137a で総額は一致済み）・要の 3 項の和 = 要。
2. **ライフ**: `要_ライフ ≈ Θ_ライフ`（どちらも `λ × 枚数`・残りのライフは全部奪われる）。比は 0.9〜1.0
   （1 を下回るなら最後の括りがとどめを取りこぼしている）。
3. **手札**: `要_手札` は**負**（奪ったライフの札は守り手の手札に入る＝`h·μ ≈ 0.049`／枚ぶん実現の損害から引かれる）
   ——**`Θ` の手札の項（≥ 0）は丸ごと要を超える**。
4. **超過の主犯は手札側**（`Θ` の手札の項 ＋ ライフの札が手札に戻る分）。**1 行の器の手札の項は橋より大きい**
   （窓が無い・T116 は残り 1 ターンの手札の項を 0.2433 → 0.0442〔実〕に縮めたと報告している）。
5. **体**は予告しない——`Θ` は**アクティブなブロッカー**（橋は＋手札のブロッカー）、要は**全キャラの `ν` の減少**で
   定義が違う（向きを読むだけ）。

使い方（`two_curves.py --dump` を先に作り直しておく——`r_life` 等が無い古い dump では落ちる）:

    python tests/scripts/two_curves.py --in <dir> --dump tc.json
    python tests/scripts/theta_at_settle.py --in <dir> --dump tc.json [--games N] [--json out.json]
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
import kappa_vector as KV  # noqa: E402
import lethal_rule as LR  # noqa: E402
import two_curves_settle as TD  # noqa: E402
import two_curves_state as TS  # noqa: E402
from theory_bridge import POL_COLS, ROW_COLS, _extra  # noqa: E402
from theory_order import MU, THETA  # noqa: E402

PARTS = ("life", "hand", "body")
#: dump の系列名 → 要の部品名（`total` は `r` そのもの）
_SERIES = (("r", "total"), ("r_life", "life"), ("r_hand", "hand"), ("r_body", "body"))


def need_parts(seat, t):
    """`要(t)`＝**宣言ターンから終局までに実際に与えた損害**（総額と 3 部品）。
    `累積(終局) − 累積(t より前の最後のターン)`（`t` より前に何も無ければ 0 から）。
    **`r_life` 等が無い古い dump は黙って 0 にせず落とす**（T143 以前の `two_curves.py` で作った dump）。"""
    turns = seat["turns"]
    out = {}
    for key, name in _SERIES:
        series = seat.get(key)
        if series is None:
            raise ValueError("dump に %r が無い——two_curves.py --dump を作り直す（T143 以前の dump）" % (key,))
        end = float(series[-1]) if len(series) else 0.0
        out[name] = end - float(TD.g_at_or_before(turns, series, int(t) - 1))
    return out


def join_row(a, b, nd):
    """1 行の器の状態 `a`（`with_parts=True`）・橋の `theta_check` の行 `b`・要 `nd` を 1 行に並べる。"""
    return {"t_left": int(b["t_left"]),
            "a_life": float(a["th_opp_life"]), "a_hand": float(a["th_opp_hand"]),
            "a_body": float(a["th_opp_body"]), "a_total": float(a["th_opp"]),
            "b_life": float(b["th_life"]), "b_hand": float(b["th_hand"]),
            "b_hand_raw": float(b["th_hand_raw"]), "b_body": float(b["th_body"]),
            "b_hb": float(b["th_hb"]), "b_total": float(b["theta"]),
            "need_total": float(nd["total"]), "need_life": float(nd["life"]),
            "need_hand": float(nd["hand"]), "need_body": float(nd["body"]),
            "need_bridge": float(b["need"]),
            "life_opp": float(a["life_opp"]), "hand_opp": float(a["hand_opp"])}


def _ratio(num, den):
    return round(float(num) / float(den), 4) if abs(float(den)) > 1e-12 else None


def invariants(rows, eps=1e-9):
    """**予告 1 の恒等式**を行ごとに検算する（最大の食い違いと、`eps` を超えた行の数）。"""
    checks = {
        "sum_one_row": lambda r: r["a_life"] + r["a_hand"] + r["a_body"] - r["a_total"],
        "sum_bridge": lambda r: r["b_life"] + r["b_hand"] + r["b_body"] - r["b_total"],
        "life_same_row": lambda r: r["a_life"] - r["b_life"],
        "body_blockers_same_row": lambda r: r["a_body"] - (r["b_body"] - r["b_hb"]),
        # 橋の `need` は **0 で下を切る**（`max(0, F_end − F_before)`）——dump の要も同じく切ってから比べる
        # （切りが効いた行の数は `summarise` の `n_need_total_negative` に別に出す）
        "need_two_impls": lambda r: max(0.0, r["need_total"]) - r["need_bridge"],
        "need_parts_sum": lambda r: r["need_life"] + r["need_hand"] + r["need_body"] - r["need_total"],
        # 窓は切るだけ（足さない）——正なら窓が手札の項を増やしている＝器の誤り
        "window_only_cuts": lambda r: max(0.0, r["b_hand"] - r["b_hand_raw"]),
    }
    out = {}
    for name, fn in checks.items():
        devs = [abs(float(fn(r))) for r in rows]
        out[name] = {"max_abs": round(max(devs), 12) if devs else None,
                     "n_bad": int(sum(1 for d in devs if d > eps))}
    return out


def block(rows):
    """母数 `rows` の平均の表（**比は平均どうしの比**＝行ごとの比の平均は `要` が小さい行に振り回される）。"""
    n = len(rows)
    if not n:
        return {"n": 0}
    keys = rows[0].keys()
    m = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    rnd = lambda x: round(float(x), 4)                                   # noqa: E731
    return {
        "n": n,
        "one_row": {"life": rnd(m["a_life"]), "hand": rnd(m["a_hand"]), "body": rnd(m["a_body"]),
                    "total": rnd(m["a_total"])},
        "bridge": {"life": rnd(m["b_life"]), "hand": rnd(m["b_hand"]), "hand_before_window": rnd(m["b_hand_raw"]),
                   "body": rnd(m["b_body"]), "body_hand_blocker": rnd(m["b_hb"]), "total": rnd(m["b_total"])},
        "need": {"life": rnd(m["need_life"]), "hand": rnd(m["need_hand"]), "body": rnd(m["need_body"]),
                 "total": rnd(m["need_total"])},
        "one_row_over_need": _ratio(m["a_total"], m["need_total"]),
        "bridge_over_need": _ratio(m["b_total"], m["need_total"]),
        "need_life_over_theta_life": _ratio(m["need_life"], m["a_life"]),
        # **超過**＝`Θ` の部品 − 要の部品（正なら `Θ` が多い）
        "excess_one_row": dict({p: rnd(m["a_" + p] - m["need_" + p]) for p in PARTS},
                               total=rnd(m["a_total"] - m["need_total"])),
        "excess_bridge": dict({p: rnd(m["b_" + p] - m["need_" + p]) for p in PARTS},
                              total=rnd(m["b_total"] - m["need_total"])),
        # **1 行の器 − 橋** を 3 つの補正に割る（和は `total` に一致する）
        "one_row_minus_bridge": {"g_reading": rnd(m["a_hand"] - m["b_hand_raw"]),
                                 "window": rnd(m["b_hand_raw"] - m["b_hand"]),
                                 "hand_blocker": rnd(-m["b_hb"]),
                                 "life": rnd(m["a_life"] - m["b_life"]),
                                 "body_blockers": rnd(m["a_body"] - (m["b_body"] - m["b_hb"])),
                                 "total": rnd(m["a_total"] - m["b_total"])},
        "share_need_hand_negative": round(float(np.mean([r["need_hand"] < 0.0 for r in rows])), 4),
        "life_opp_mean": rnd(m["life_opp"]), "hand_opp_mean": rnd(m["hand_opp"]),
    }


def summarise(rows):
    """全体と、宣言ターンが最後の自席ターンか（`t_left=1`）それより前か（`2+`）に分けた表。"""
    return {"all": block(rows),
            "by_t_left": {"1": block([r for r in rows if r["t_left"] == 1]),
                          "2+": block([r for r in rows if r["t_left"] >= 2])},
            "invariants": invariants(rows),
            # 要（総額）が負の行＝橋の `need` の 0 切りが効いた行（表の平均は切らない dump の要で取る）
            "n_need_total_negative": int(sum(1 for r in rows if r["need_total"] < 0.0))}


def collect(dump, dirs, limit_games=0, theta=THETA, mu=MU):
    cards = PL.Cards()
    idx2cid = {i: c for c, i in GA._vocab().items()}
    seat_decks = KV._seat_decks(dirs)
    settled = LR.settled_map(dirs, limit_games)
    declared = TD.declared_turns_by_seed_w(settled)
    by_seed_w = {(int(r["seed"]), int(r["w"])): r for r in dump}

    # 1 行の器（T137d と同じ `Θ_opp` を同じ引数で 3 項に割る）
    one_row = {}
    games = 0
    for rows, pol, ex, L, ptr, idx in PL.iter_games(dirs, row_cols=ROW_COLS, pol_cols=POL_COLS, extra_fn=_extra):
        games += 1
        if limit_games and games > limit_games:
            break
        seed_g = int(rows["seed"][idx[0]]) if len(idx) else -1
        targets = []
        for w in (0, 1):
            d = by_seed_w.get((seed_g, w))
            ts = declared.get((seed_g, w)) or []
            if d and d.get("winner") == w and ts:
                targets.append((w, ts[0]))
        if not targets:
            continue
        st = TS.state_by_turn(rows, ex, idx, cards, idx2cid, seat_decks, seed_g, theta, mu, with_parts=True)
        for w, t_d in targets:
            if (w, t_d) in st:
                one_row[(seed_g, w, t_d)] = st[(w, t_d)]

    # 交点の橋（出荷既定の補正込み）
    _ro, _lg, _st, _th, theta_check = CB.collect(dirs, limit_games, theta, mu, "const")
    bridge = {(int(r["seed"]), int(r["who"]), int(r["t"])): r for r in theta_check}

    joined = []
    counts = {"n_winners": 0, "n_no_declare": 0, "n_no_one_row": 0, "n_no_bridge": 0}
    for (seed_g, w), d in sorted(by_seed_w.items()):
        if d.get("winner") != w:
            continue
        counts["n_winners"] += 1
        ts = declared.get((seed_g, w)) or []
        if not ts:
            counts["n_no_declare"] += 1
            continue
        t_d = ts[0]
        a = one_row.get((seed_g, w, t_d))
        if a is None:
            counts["n_no_one_row"] += 1
            continue
        b = bridge.get((seed_g, w, t_d))
        if b is None:
            counts["n_no_bridge"] += 1
            continue
        joined.append(join_row(a, b, need_parts(d, t_d)))
    out = summarise(joined)
    out.update(counts)
    out["n_measured"] = len(joined)
    return out


def build_parser():
    ap = argparse.ArgumentParser(description="決着の瞬間の Θ_opp をライフ／手札／体に分解する（T143）")
    ap.add_argument("--in", dest="src", nargs="+", required=True)
    ap.add_argument("--dump", required=True, help="two_curves.py --dump が書いた JSON（T143 以降・同じ --in／--games）")
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
