"""兄弟手の**順序**精度——接戦帯で方策は「どちらが良いか」を並べられているか
（ユーザ指摘 2026-08-13「接戦帯で較正を見る」の順序版・`docs/measurement.md` §8・読み取り専用）。

較正（ECE）は「勝率の当て方」の物差しで、**決定は順序で決まる**。同じ判断点の兄弟手のうち
どれが良いかを方策の事前分布が並べられていなければ、探索が毎回それを打ち消す仕事をしている
＝方策は決定に効いていない。記録には判断点ごとに候補の **事前分布 `pol_p`・訪問 `pol_n`・
行動価値 `pol_q`・根の価値 `pol_v0`** が全部入っている（dump v4・§20.6.1）ので、
**ネットを回さずに**測れる。

測る量（すべて「同じ行の兄弟手のペア」について）:

```
order_acc(p, n) = P[ (p_i − p_j) と (n_i − n_j) の符号が一致 ]     方策 vs 探索の結論
order_acc(p, q) = P[ (p_i − p_j) と (q_i − q_j) の符号が一致 ]     方策 vs 行動価値
                  （q は訪問が `--n-min` 以上の候補だけ＝推定が定まっている手に限る）
top1(p, n)      = P[ argmax p == argmax n ]
q_cost          = E[ q(argmax n) − q(argmax p) ]                   順序を誤った分の損（Q の単位）
```

`order_acc = 0.5` は**順序の情報が無い**（コイン）。`q_cost` は
「方策だけで打ったら探索の結論よりどれだけ悪い手を選ぶか」＝後悔
（`measurement.md` §2 の `R`＝実測 0.033〜0.084）と同じ単位で読める。

帯（`|pol_v0|`＝根の価値の絶対値。**接戦帯が決定に効く帯**）:
  `close`（|v0| ≤ --close）／`mid`／`decided`（|v0| > --decided）

出すもの: 帯ごとに rows／pairs／order_acc／top1／q_cost と、候補数・事前分布のエントロピー。

**交絡・限界**（読むときに添える）:
- `n` と `q` は**同じ探索の産物**なので、これは「方策が探索に追いついているか」であって
  「方策が正しいか」ではない。探索そのものの誤りは測れない（それは `worlds` と被覆の線）。
- 訪問が少ない候補の `q` は当てにならないので切る。しきい値は
  **`max(--n-min, ceil(--n-min-frac × その行の総訪問))`**＝既定は「総訪問の 5%」。
  絶対数だけ（旧案の 20＝sims/8）では残るペアが 1 行あたり数本しか無かった（2026-09-13 実測）。
- 窓（カウンター・対象選択）の行は候補を持たない（「窓・コミットは訪問を配らない」）ので
  **自席の main 行だけ**が母数。守りの順序はここでは測れない。

実行例:
  OPCG_LOG_SILENT=1 python tests/scripts/order_acc.py \\
    --in ~/n32_wave/w*/n_records --holdout-mod 7 --out ~/order_w32.json
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

from opcg_sim.learned.train import plan_labels as PL  # noqa: E402

ROW_COLS = ("who", "turn", "seed", "z", "kind", "step", "pol_len", "pol_chosen", "pol_v0")
POL_COLS = ("pol_n", "pol_q", "pol_p")
BANDS = ("close", "mid", "decided")


def band_of(v0, close=0.2, decided=0.6):
    a = abs(float(v0))
    return "close" if a <= close else ("decided" if a > decided else "mid")


def pair_agree(x, y, x_eps=0.0, y_eps=0.0):
    """並び順の一致（同符号のペア数, 判定できたペア数）。**同値のペアは母数から外す**。"""
    x = np.asarray(x, np.float64); y = np.asarray(y, np.float64)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    iu = np.triu_indices(len(x), 1)
    dx = dx[iu]; dy = dy[iu]
    ok = (np.abs(dx) > x_eps) & (np.abs(dy) > y_eps)
    if not ok.any():
        return 0, 0
    agree = int(((dx[ok] > 0) == (dy[ok] > 0)).sum())
    return agree, int(ok.sum())


def q_floor(n, n_min=5, n_min_frac=0.05):
    """`q` を信じる訪問の下限。**その行の総訪問の割合**と絶対数のうち大きい方。

    絶対数だけで切ると駄目だった（2026-09-13 実測）: 根は候補 10 本前後に 160 訪問を配るので、
    20 訪問を超える候補は 1〜3 本しか無く、ペアがほとんど残らない。割合で切ると
    どの sims でも同じ意味（＝「まともに見られた手」）になる。
    """
    total = float(np.asarray(n, np.float64).sum())
    return max(float(n_min), float(np.ceil(n_min_frac * total)))


def row_stats(n, q, p, n_min=5, q_eps=0.02, p_eps=1e-4, n_min_frac=0.05):
    """1 行（判断点）の順序統計。"""
    out = {"k": len(n)}
    a, t = pair_agree(p, n, p_eps, 0.0)
    out["pn_agree"], out["pn_pairs"] = a, t
    n_min = q_floor(n, n_min, n_min_frac)
    keep = np.asarray(n) >= n_min
    if keep.sum() >= 2:
        a, t = pair_agree(np.asarray(p)[keep], np.asarray(q)[keep], p_eps, q_eps)
        out["pq_agree"], out["pq_pairs"] = a, t
    else:
        out["pq_agree"], out["pq_pairs"] = 0, 0
    bn = int(np.argmax(n)); bp = int(np.argmax(p))
    out["top1_pn"] = bool(bn == bp)
    # 順序を誤った分の損（探索が選ぶ手の Q − 方策が選ぶ手の Q）。訪問が足りない手は測らない。
    out["q_cost"] = (float(q[bn]) - float(q[bp])) if (n[bn] >= n_min and n[bp] >= n_min) else None
    pp = np.asarray(p, np.float64)
    s = pp.sum()
    if s > 0:
        pp = pp / s
        nz = pp[pp > 0]
        out["prior_entropy"] = float(-(nz * np.log(nz)).sum())
    else:
        out["prior_entropy"] = None
    return out


def collect(dirs, holdout_mod=7, limit_games=0, n_min=5, q_eps=0.02,
            close=0.2, decided=0.6, n_min_frac=0.05):
    """holdout の自席 main 行 → 行ごとの順序統計（帯つき）。"""
    recs = []
    games = 0
    skipped_small = 0
    for rows, pol, _ex, L, ptr, idx in PL.iter_games(dirs, row_cols=ROW_COLS, pol_cols=POL_COLS):
        seed = int(rows["seed"][idx[0]])
        if holdout_mod and seed % holdout_mod != 0:
            continue
        games += 1
        if limit_games and games > limit_games:
            break
        for i in idx:
            if int(rows["kind"][i]) != 0:
                continue
            w = int(rows["who"][i]); t = int(rows["turn"][i])
            if t < 1 or not PL.is_own_turn(w, t):
                continue
            k = int(L[i])
            if k < 2:
                skipped_small += 1
                continue
            b = int(ptr[i])
            n = np.asarray(pol["pol_n"][b:b + k], np.float64)
            q = np.asarray(pol["pol_q"][b:b + k], np.float64)
            p = np.asarray(pol["pol_p"][b:b + k], np.float64)
            st = row_stats(n, q, p, n_min, q_eps, n_min_frac=n_min_frac)
            st["band"] = band_of(rows["pol_v0"][i], close, decided)
            st["turn"] = t
            st["v0"] = float(rows["pol_v0"][i])
            recs.append(st)
    return recs, games, skipped_small


def block(recs):
    if not recs:
        return None
    pn_a = sum(r["pn_agree"] for r in recs); pn_t = sum(r["pn_pairs"] for r in recs)
    pq_a = sum(r["pq_agree"] for r in recs); pq_t = sum(r["pq_pairs"] for r in recs)
    costs = [r["q_cost"] for r in recs if r["q_cost"] is not None]
    ent = [r["prior_entropy"] for r in recs if r["prior_entropy"] is not None]
    return {
        "rows": len(recs),
        "k_mean": round(float(np.mean([r["k"] for r in recs])), 2),
        "pairs_pn": pn_t, "order_acc_pn": round(pn_a / pn_t, 4) if pn_t else None,
        "pairs_pq": pq_t, "order_acc_pq": round(pq_a / pq_t, 4) if pq_t else None,
        "top1_pn": round(float(np.mean([r["top1_pn"] for r in recs])), 4),
        "q_cost_mean": round(float(np.mean(costs)), 4) if costs else None,
        "q_cost_n": len(costs),
        "prior_entropy_mean": round(float(np.mean(ent)), 4) if ent else None,
        # コインとの差（`order_acc − 0.5`）を 95% CI つきで（ペアは独立でないので**目安**）
        "pn_edge": round(pn_a / pn_t - 0.5, 4) if pn_t else None,
        "pn_edge_se": round(float(np.sqrt(0.25 / pn_t)), 4) if pn_t else None,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src", nargs="+", required=True, help="n_records のディレクトリ")
    ap.add_argument("--holdout-mod", type=int, default=7, help="seed%%N==0 だけ読む（0 で全部）")
    ap.add_argument("--limit-games", type=int, default=0)
    ap.add_argument("--n-min", type=int, default=5, help="q を信じる訪問の下限（絶対数）")
    ap.add_argument("--n-min-frac", type=float, default=0.05,
                    help="同・その行の総訪問に対する割合（下限は絶対数との大きい方）")
    ap.add_argument("--q-eps", type=float, default=0.02, help="これ以下の q 差は同値として外す")
    ap.add_argument("--close", type=float, default=0.2, help="|v0| がこれ以下＝接戦帯")
    ap.add_argument("--decided", type=float, default=0.6, help="|v0| がこれ超＝決着帯")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    t0 = time.time()
    recs, games, small = collect(args.src, args.holdout_mod, args.limit_games,
                                 args.n_min, args.q_eps, args.close, args.decided,
                                 args.n_min_frac)
    out = {"games": games, "rows": len(recs), "rows_single_candidate": small,
           "src": list(args.src),
           "params": {"n_min": args.n_min, "n_min_frac": args.n_min_frac, "q_eps": args.q_eps,
                      "close": args.close, "decided": args.decided},
           "all": block(recs)}
    for b in BANDS:
        out[b] = block([r for r in recs if r["band"] == b])
    out["seconds"] = round(time.time() - t0, 1)
    txt = json.dumps(out, ensure_ascii=False, indent=1)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(txt + "\n")
    print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
