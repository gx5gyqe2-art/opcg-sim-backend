"""**組み上げた理論はホールドアウトの棋譜を説明するか**（フェーズの中間ゲート・読み取り専用）。

ユーザ決定 2026-09-14「理論の完成とは、既存の棋譜に対して勝敗が説明できるかどうか」を、
**循環しない形に直したもの**（`docs/cpu_theory_gap.md` §0.1）。

## なぜ素朴な「棋譜で勝敗を説明できるか」ではいけないか

**4 つの価格はその棋譜への回帰で測った**。同じ棋譜で当てはまりを見るのは
**当てはめたデータで当てはまりを測る**ことになる。しかも**項を足せば当てはまりは必ず改善する**
ので、改善したことが証拠にならない（`cpu_theory_gap.md` §7.4 の「足す項を選べば必ず閉じられる」
と同じ形）。

## 本器が課す 4 つの条件

| 条件 | 実装 |
|---|---|
| **1. 当てはめた棋譜では測らない** | `--in` と `--construction` の重なりを**実行前に拒否する** |
| **2. 定数は凍結し、出力に刻む** | `frozen` 欄に値と出所を出す。**本器は 1 つも当てはめない** |
| **3. 自由度ゼロの指標だけ使う** | **AUC（順位のみ）**。較正も回帰もしない＝**ホールドアウトから値をもらう経路が無い** |
| **4. 勝敗だけでなく順序も見る** | 半 B（`theory_order` の順序一致）を同時に回す |

## 半 A——勝敗（**3 つを同じ行で比べる**）

```
base   = 自ライフ − 相手ライフ                      「当たり前」の下限
theory = λΔライフ + μΔ手札 + δΔドン + Σν(自場) − Σν(相手場)   凍結した理論
net    = pol_v0（ネットが見た根の価値）              **天井の目安**
```

**天井の目安を置くのが要点**——「勝敗には引きとトリガーの偶然が入るので、
何 % 説明できたら完成かに錨が無い」という問題への手当て。**ネットはこの課題そのもので
訓練されている**ので、**実際に取り出せる量の経験的な目安**になる。

```
position = (AUC_theory − AUC_base) / (AUC_net − AUC_base)
```

**0 = 当たり前のことしか言えていない・1 = ネットと同じだけ説明できている。**

## 読み方（事前登録）

- **`position` は監視の数字であって、最適化の目標にしてはいけない**——目標にした瞬間に
  条件 2 が破れる（残差を見て項を選ぶことになる）。
- **`position` が上がっても順序（半 B）が動かないことは有り得る**。
  理論の仕事は**手の順序**なので、**半 B が落ちたまま半 A だけ上がったら疑う**。
- `net` が `base` を下回る帯があれば、その帯は**天井の目安として使えない**（`position` を出さない）。

**限界**: 観察であって因果ではない（因果は T18＝フェーズの出口）。
`net` は上限ではなく**目安**——ネットも最適ではない。

実行例:
  OPCG_LOG_SILENT=1 python tests/scripts/theory_gate.py \\
    --in ~/w41 --construction ~/w39 ~/w42 --out ~/theory_gate.json
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
import theory_order as TO  # noqa: E402
from theory_order import (LAM, MU, PWR_EPS, S_IS_BLOCKER, S_IS_CHAR, S_POWER,  # noqa: E402
                          SC_MY_DON, SC_MY_HAND, SC_MY_LEADER_POWER, SC_MY_LIFE,
                          SC_OPP_LEADER_POWER, SC_OPP_LIFE, SLOT_OPP_FIELD,
                          SLOT_OWN_FIELD, THETA, nu_of, opp_chars_of)

ROW_COLS = ("who", "turn", "seed", "z", "kind", "step", "pol_len", "pol_chosen", "pol_v0")
#: `δ`（ドン 1 つの価格・`game_theory.md` §18）。`theory_order` は攻撃の値付けしか
#: しないので定数を持っていない——**本器が状態を採点するために要る**。
DELTA = 0.0277
#: 相手の手札・ドンの列（`scalars.rs`）。自分側と枠の列は `theory_order` から来る。
SC_OPP_HAND = 7
SC_OPP_DON = 4
#: 比べる予測子。**`net` は天井の目安**であって上限ではない。
PREDICTORS = ("base", "theory", "net")


def _extra(dd, n):
    return {"sc": np.asarray(dd["scalars"])[:n].astype(np.float32),
            "tok": np.asarray(dd["tokens"])[:n].astype(np.float32)}


def _round10(x):
    """`1e4` を掛けて戻した量は 10 の単位に丸める（`measurement.md` 罠 14-5）。"""
    return round(float(x) / 10.0) * 10.0


def side_nu(tok_row, slots, opp_leader_power, r_turns, my_leader_power, theta, mu,
            opp_chars=None):
    """片側の場の `ν` の合計（**凍結した `nu_of` をそのまま呼ぶ**）。"""
    tot = 0.0
    for s in range(slots.start, slots.stop):
        if float(tok_row[s, S_IS_CHAR]) <= 0.5:
            continue
        pw = _round10(float(tok_row[s, S_POWER]) * 1e4)
        if pw < -PWR_EPS:
            continue
        tot += nu_of(pw, opp_leader_power, r_turns, theta, mu,
                     is_blocker=float(tok_row[s, S_IS_BLOCKER]) > 0.5,
                     opp_chars=opp_chars, my_leader_power=my_leader_power)
    return float(tot)


def state_score(sc_row, tok_row, theta=THETA, mu=MU, lam=LAM, delta=DELTA,
                nu_targets="leader"):
    """**凍結した理論の盤面評価**（4 通貨の差の和）。当てはめは一切しない。"""
    my_l, opp_l = float(sc_row[SC_MY_LIFE]), float(sc_row[SC_OPP_LIFE])
    mlp = _round10(float(sc_row[SC_MY_LEADER_POWER]) * 1e4)
    olp = _round10(float(sc_row[SC_OPP_LEADER_POWER]) * 1e4)
    # `ν` の残りターン数は相手のライフで近似する（`theory_order.collect` と同じ約束）
    r_me = max(1.0, min(5.0, opp_l))
    r_opp = max(1.0, min(5.0, my_l))
    chars = opp_chars_of(tok_row) if nu_targets == "board" else None
    own = side_nu(tok_row, SLOT_OWN_FIELD, olp, r_me, mlp, theta, mu, chars)
    # **相手の場を採点するときは席を入れ替える**——相手から見た「相手のリーダー」は自分。
    opp = side_nu(tok_row, SLOT_OPP_FIELD, mlp, r_opp, olp, theta, mu, None)
    return (lam * (my_l - opp_l)
            + mu * (float(sc_row[SC_MY_HAND]) - float(sc_row[SC_OPP_HAND]))
            + delta * (float(sc_row[SC_MY_DON]) - float(sc_row[SC_OPP_DON]))
            + own - opp)


def collect(dirs, limit_games=0, theta=THETA, mu=MU, nu_targets="leader"):
    """自席ターン最初の main 行 → 3 つの予測子と勝敗。"""
    recs = []
    games = 0
    for rows, pol, ex, L, ptr, idx in PL.iter_games(dirs, row_cols=ROW_COLS,
                                                    extra_fn=_extra):
        games += 1
        if limit_games and games > limit_games:
            break
        seed = int(rows["seed"][idx[0]])
        seen = set()
        for i in idx:
            w, t = int(rows["who"][i]), int(rows["turn"][i])
            if t < 1 or int(rows["kind"][i]) != 0 or not PL.is_own_turn(w, t):
                continue
            if (w, t) in seen:
                continue
            seen.add((w, t))
            z = float(rows["z"][i])
            if z == 0.0:
                continue
            sc, tok = ex["sc"][i], ex["tok"][i]
            recs.append({
                "seed": seed, "turn": t,
                "base": float(sc[SC_MY_LIFE]) - float(sc[SC_OPP_LIFE]),
                "theory": state_score(sc, tok, theta, mu, nu_targets=nu_targets),
                "net": float(rows["pol_v0"][i]),
                "z": 1.0 if z > 0 else 0.0})
    return recs, games


def auc(scores, labels):
    """順位の AUC（**当てはめない**＝ホールドアウトから値をもらう経路が無い）。

    同値は 0.5 として数える（順位の平均を使う標準の扱い）。
    """
    s = np.asarray(scores, np.float64); y = np.asarray(labels, np.float64)
    pos, neg = (y > 0.5), (y <= 0.5)
    n_p, n_n = int(pos.sum()), int(neg.sum())
    if n_p == 0 or n_n == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), np.float64)
    sorted_s = s[order]
    i = 0
    while i < len(s):                     # 同値は平均順位に潰す
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[pos].sum() - n_p * (n_p + 1) / 2.0) / (n_p * n_n))


def position(a):
    """**天井の目安に対して理論がどこに居るか**。`net` が `base` を超えないと出さない。"""
    b, t, n = a.get("base"), a.get("theory"), a.get("net")
    if None in (b, t, n) or (n - b) <= 1e-9:
        return None
    return float((t - b) / (n - b))


def _aucs(recs):
    y = [r["z"] for r in recs]
    return {k: auc([r[k] for r in recs], y) for k in PREDICTORS}


def _boot(recs, reps=200, seed=0):
    """**対局を復元抽出**して AUC と `position` の CI（`measurement.md` §14-15）。"""
    if reps <= 0:
        return {}, [None, None]
    by = {}
    for r in recs:
        by.setdefault(r["seed"], []).append(r)
    gids = list(by)
    if len(gids) < 3:
        return {}, [None, None]
    rng = np.random.default_rng(seed)
    acc = {k: [] for k in PREDICTORS}
    pos = []
    for _ in range(int(reps)):
        pick = rng.integers(0, len(gids), len(gids))
        sub = [r for k in pick for r in by[gids[k]]]
        a = _aucs(sub)
        for k, v in a.items():
            if v is not None:
                acc[k].append(v)
        p = position(a)
        if p is not None and np.isfinite(p):
            pos.append(p)

    def ci(v):
        return ([round(float(np.percentile(v, 2.5)), 4),
                 round(float(np.percentile(v, 97.5)), 4)] if len(v) >= 10 else [None, None])
    return {k: ci(v) for k, v in acc.items()}, ci(pos)


def summarise(recs, reps=200, seed=0, by_turn=True):
    a = _aucs(recs)
    ci, pos_ci = _boot(recs, reps, seed)
    out = {"n": len(recs), "games": len({r["seed"] for r in recs}),
           "auc": {k: (round(v, 4) if v is not None else None) for k, v in a.items()},
           "auc_ci95": ci, "position": position(a), "position_ci95": pos_ci}
    if out["position"] is not None:
        out["position"] = round(out["position"], 4)
    if by_turn:
        # **帯ごとにも出す**——序盤は誰にも判らないので、全体の数字だけ見ると天井を見誤る
        out["by_turn"] = {}
        for name, lo, hi in (("early", 1, 4), ("mid", 5, 8), ("late", 9, 99)):
            sub = [r for r in recs if lo <= r["turn"] <= hi]
            if len(sub) < 200:
                continue
            aa = _aucs(sub)
            p = position(aa)
            out["by_turn"][name] = {
                "n": len(sub),
                "auc": {k: (round(v, 4) if v is not None else None) for k, v in aa.items()},
                "position": (round(p, 4) if p is not None else None)}
    return out


def order_half(dirs, limit_games=0, nu_targets="leader"):
    """**半 B——手の順序**（`theory_order` をそのまま呼ぶ・凍結した定数で）。

    **半 A だけ上がって半 B が動かない**ことは有り得る——勝敗の説明は**盤面の価値**の話で、
    理論の仕事は**手の順序**だから。**両方を同じ棋譜で並べて出す**のが本器の役目。

    ホールドアウト全局を使う（`holdout_mod=0`）——**`--in` 自体が既にホールドアウト**なので、
    ここで更に間引く理由が無い。
    """
    recs, stats = TO.collect(dirs, holdout_mod=0, limit_games=limit_games,
                             nu_targets=nu_targets)
    b = TO.block(recs)
    return {"stats": stats, "all": b, "verdict": TO.verdict(b)}


def overlap(src, construction):
    """**当てはめた棋譜で測るのを実行前に拒否する**（条件 1）。"""
    a = {os.path.realpath(os.path.expanduser(p)) for p in src}
    b = {os.path.realpath(os.path.expanduser(p)) for p in (construction or ())}
    return sorted(a & b)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="src", nargs="+", required=True,
                    help="**ホールドアウトの** n_records ディレクトリ")
    ap.add_argument("--construction", nargs="+", default=[],
                    help="**理論を組み上げた棋譜**。重なっていたら実行を拒否する（条件 1）")
    ap.add_argument("--limit-games", type=int, default=0)
    ap.add_argument("--nu-targets", default="leader", choices=("leader", "board"))
    ap.add_argument("--no-order", action="store_true",
                    help="半 B（手の順序）を飛ばす。**既定では回す**——勝敗だけ見ると"
                         "「価値は当たるが順序は当たらない」を見落とす")
    ap.add_argument("--boot-reps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)

    bad = overlap(a.src, a.construction)
    if bad:
        print(json.dumps({"error": "holdout_overlaps_construction", "dirs": bad},
                         ensure_ascii=False, indent=2))
        return 2

    t0 = time.time()
    recs, games = collect(a.src, a.limit_games, nu_targets=a.nu_targets)
    res = {"games": games, "rows": len(recs),
           # **凍結した定数を出力に刻む**（条件 2）——後から黙って引き直せないように
           "frozen": {"lambda": LAM, "mu": MU, "delta": DELTA, "theta": THETA,
                      "nu_targets": a.nu_targets,
                      "source": "docs/game_theory.md §18（合成デッキの棋譜から測った値）"},
           "holdout": [os.path.realpath(os.path.expanduser(p)) for p in a.src],
           "construction": [os.path.realpath(os.path.expanduser(p)) for p in a.construction],
           "summary": summarise(recs, a.boot_reps, a.seed),
           "seconds": round(time.time() - t0, 1)}
    if not a.no_order:
        res["order"] = order_half(a.src, a.limit_games, a.nu_targets)
        res["seconds"] = round(time.time() - t0, 1)
    txt = json.dumps(res, ensure_ascii=False, indent=2)
    print(txt)
    if a.out:
        with open(os.path.expanduser(a.out), "w", encoding="utf-8") as fh:
            fh.write(txt + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
