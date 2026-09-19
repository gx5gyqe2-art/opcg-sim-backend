"""決着までの時間の**分布**——`P(T = t | s)` を記録から出す
（ユーザ指摘 2026-09-12「離散・連続は時系列に対しての話」・`docs/measurement.md` §3・読み取り専用）。

V は盤面に勝率 1 個を返すだけで「いつ」を持たない。分解すると

```
P(win | s) = Σ_t P(T = t | s) · P(win | s, T = t)
```

`2026-09-13_time_head.md` は **`E[T|s]` を測った**（誤差約 1.0 ターン）が、**分布は未測定**だった。
分布が要るのは「どちらにしろ負け濃厚なら無茶をする」の判断が
**「短い T の尾」がどれだけ在るか**で決まるため（期待値が同じでも尾の厚さで最適が変わる）。

本器は記録だけで `P(T=t|s)` を推定する（ネットも訓練も要らない）:

- `T` … その行から数えた**残りの自席ターン数**（`time_labels.label_game` の `t_left`・真値）
- 帯 `s` … `(自分のライフ, 相手のライフ)` と、任意で `(手札, 相手の場のキャラ数)`
- 出すもの: 帯ごとの `P(T=t)`（t=1..6+）・`E[T]`・`sd`・**ハザード** `h(t) = P(T=t | T≥t)`・
  そして**帯ごとの勝率**（`P(win|s)` と `P(win|s,T=t)` の両方＝上の分解の右辺がそのまま読める）

### なぜハザードで読むか

`h(t)` が `t` に依らず一定なら「決着はいつでも同じ確率で来る」（記憶のない過程）＝盤面から
時計を読む意味が薄い。`h(1)` が大きく `h` が急に落ちるなら**今が勝負**で、
`h` が上がっていくなら**まだ先**。**時間軸ヘッドが学ぶべき形はこの `h(t)`** で、
平均 1 個では表せない。

### 近似・**`all` を読んではいけない理由**

- **周辺分布 `P(T=t)`（`all`）は作りからほぼ一様**: 長さ n の対局は T=n, n−1, …, 1 の行を
  ちょうど 1 本ずつ出すので、全行を混ぜると一様に近づく（ハザードも自動的に増加する）。
  **意味があるのは帯ごとの条件付き分布 `bands`** で、`all` は母数と健全性の確認用。
- `T` は**自席ターン**の数（相手のターンは数えない）＝`time_labels` の定義そのまま。
- 引き分け（`z=0`）と最後まで決着しない局は `mask=0` で落ちる（`time_labels` の規約）。
- 帯は観測された盤面の分布に依存する＝**因果ではない**（「そのライフにした打ち回し」の条件付き）。
- `t>=6` は 1 つの箱にまとめる（尾の形は見えるが点推定はしない）。

実行例:
  OPCG_LOG_SILENT=1 python tests/scripts/hazard_curve.py \\
    --in ~/w32/n_records/n32_w* --holdout-mod 0 --out ~/hazard_w32.json
"""
import argparse
import collections
import json
import os
import sys
import time

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from opcg_sim.learned.train import plan_labels as PL  # noqa: E402
from opcg_sim.learned.train import time_labels as TL  # noqa: E402

ROW_COLS = ("who", "turn", "seed", "z", "kind", "step", "pol_len")
T_MAX = 6                      # これ以上は 1 つの箱（"6+"）
SC_L, SC_OPP_L, SC_H, SC_OPP_FIELD = 0, 1, 6, 9
C_TLEFT = 0


def _extra(dd, n):
    sc = np.asarray(dd["scalars"])[:n, :14].astype(np.float32)
    return {"sc": sc, "lives": np.asarray(dd["scalars"])[:n, 0:2].astype(np.float32)}


def t_bucket(t):
    """残りの自席ターン数 → 箱（1..5 と "6+"）。"""
    k = int(round(float(t)))
    if k < 1:
        k = 1
    return str(k) if k < T_MAX else f"{T_MAX}+"


def band_key(sc_row, wide=False):
    """帯。既定は (自ライフ, 相手ライフ)・`wide` で手札と相手の場も足す。"""
    parts = [int(round(float(sc_row[SC_L]))), int(round(float(sc_row[SC_OPP_L])))]
    if wide:
        parts += [int(round(float(sc_row[SC_H]))), int(round(float(sc_row[SC_OPP_FIELD])))]
    return "|".join(str(p) for p in parts)


def collect(dirs, holdout_mod=0, limit_games=0, wide=False, own_only=True):
    """holdout（既定は全局）→ 行ごとの (帯, T, 勝敗)。"""
    recs = []
    games = 0
    no_true = 0
    for rows, _pol, ex, L, ptr, idx in PL.iter_games(dirs, row_cols=ROW_COLS, pol_cols=(),
                                                     extra_fn=_extra):
        seed = int(rows["seed"][idx[0]])
        if holdout_mod and seed % holdout_mod != 0:
            continue
        games += 1
        if limit_games and games > limit_games:
            break
        tm, tmask = TL.label_game(rows, ex["lives"], idx)
        zs = {int(rows["who"][i]): float(rows["z"][i]) for i in idx}
        seen = set()
        for n, i in enumerate(idx):
            w = int(rows["who"][i]); t = int(rows["turn"][i])
            if t < 1 or int(rows["kind"][i]) != 0:
                continue
            if own_only and not PL.is_own_turn(w, t):
                continue
            if (w, t) in seen:                 # 1 ターン 1 行（自席ターンの最初の main 行）
                continue
            seen.add((w, t))
            if not tmask[n]:
                no_true += 1
                continue
            z = zs.get(w, 0.0)
            if z == 0.0:
                continue
            recs.append({"band": band_key(ex["sc"][i], wide),
                         "T": float(tm[n][C_TLEFT]) * TL.T_SCALE,
                         "z": 1.0 if z > 0 else 0.0,
                         "turn": t})
    return recs, games, no_true


def buckets():
    return [str(k) for k in range(1, T_MAX)] + [f"{T_MAX}+"]


def dist_of(recs):
    """1 帯 → `P(T=t)`・`E[T]`・sd・ハザード `h(t)`・帯の勝率・`P(win|T=t)`。"""
    n = len(recs)
    cnt = collections.Counter(t_bucket(r["T"]) for r in recs)
    keys = buckets()
    p = {k: round(cnt.get(k, 0) / n, 4) for k in keys}
    # ハザード h(t) = P(T=t | T>=t)（"6+" は残り全部なので 1.0）
    haz = {}
    left = n
    for k in keys:
        c = cnt.get(k, 0)
        haz[k] = round(c / left, 4) if left > 0 else None
        left -= c
    ts = np.array([min(float(r["T"]), float(T_MAX)) for r in recs], np.float64)
    win_by_t = {}
    for k in keys:
        sub = [r["z"] for r in recs if t_bucket(r["T"]) == k]
        win_by_t[k] = round(float(np.mean(sub)), 4) if len(sub) >= 20 else None
    return {"n": n, "p_T": p, "hazard": haz,
            "E_T": round(float(ts.mean()), 3), "sd_T": round(float(ts.std()), 3),
            "winrate": round(float(np.mean([r["z"] for r in recs])), 4),
            "winrate_by_T": win_by_t}


def flat_hazard_score(d):
    """ハザードが平ら（記憶のない過程）か。`max h − min h`（n が薄い箱は除く）。"""
    xs = [v for k, v in d["hazard"].items() if v is not None and k != f"{T_MAX}+"]
    return round(max(xs) - min(xs), 4) if len(xs) >= 2 else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src", nargs="+", required=True)
    ap.add_argument("--holdout-mod", type=int, default=0)
    ap.add_argument("--limit-games", type=int, default=0)
    ap.add_argument("--wide", action="store_true", help="帯に手札と相手の場も入れる（細かい）")
    ap.add_argument("--min-n", type=int, default=200, help="帯を出す最小行数")
    ap.add_argument("--top", type=int, default=16, help="出す帯の数（行数の多い順）")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    t0 = time.time()
    recs, games, no_true = collect(args.src, args.holdout_mod, args.limit_games, args.wide)
    by = collections.defaultdict(list)
    for r in recs:
        by[r["band"]].append(r)
    bands = sorted(by.items(), key=lambda kv: -len(kv[1]))[: args.top]
    out = {"games": games, "rows": len(recs), "rows_without_truth": no_true,
           "src": list(args.src), "wide": bool(args.wide),
           "all": dist_of(recs) if recs else None,
           "bands": {k: dist_of(v) for k, v in bands if len(v) >= args.min_n}}
    if out["all"]:
        out["all"]["hazard_spread"] = flat_hazard_score(out["all"])
    for k, v in out["bands"].items():
        v["hazard_spread"] = flat_hazard_score(v)
    out["seconds"] = round(time.time() - t0, 1)
    txt = json.dumps(out, ensure_ascii=False, indent=1)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(txt + "\n")
    print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
