"""**帯に入れるべき変数を機械的に洗う**（`ν` の交絡スクリーニング・読み取り専用）。

ユーザ指示 2026-09-14「今あるエンジンの変数から効きそうなものを洗えたりする？」。

`game_theory.md` §17.1.5c で「進行」を定義したが、**どの列を帯に入れるかは人が選んでいた**
（v1 は勘・v2 は時計から導出・v3 はユーザ指摘でパワーとブロッカーを追加）。
**本器は選ぶのをやめて、全列を同じ基準で採点する。**

## 採点の基準——交絡の定義そのもの

ある列 `c` を帯から落としたとき `β`（＝`ν`）に乗るバイアスは、およそ

```
bias ≈ （c が z を予測する強さ） × （c が説明変数と相関する強さ）
```

**両方が要る**——勝敗を予測するだけの列は、説明変数と無相関ならバイアスを生まない。
説明変数と相関するだけの列も、勝敗と無関係なら生まない。**だから積で採点する**。

計算は**今の帯の中で中心化**してから行う（既に帯に入っている列は自動的に 0 に潰れる＝
**帯が効いていることの検算**にもなる）。

```
r_z    = corr( c, z | 帯 )
r_x[k] = corr( c, 帯 k の体数 | 帯 )
score  = max_k | r_z · r_x[k] |
```

## 読み方（事前登録）

- **`score` が大きい列が「帯に入れるべき候補」**。
- **既に帯に入っている列は 0 付近に潰れる**——潰れなければ帯の実装がおかしい。
- **`own_board` 印の列は入れられない**（自分の場＝説明変数そのもの）。
  **高い score が出ても、それは交絡ではなく処置である**。この区別を器が持つ。

**限界**:

- **線形・単変量のスクリーニング**。交互作用や非線形の交絡は拾えない。
- **相関であって因果ではない**——「入れるべき」の候補を出すだけで、
  実際に入れて `ν` が動くかは `nu_measure --band` で確かめる（本器は絞り込み）。
- **`score` の絶対値に意味は無い**（順位を見る）。

実行例:
  OPCG_LOG_SILENT=1 python tests/scripts/band_screen.py --in ~/w39 --top 25
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
from opcg_sim.learned import n_rel_feat as F  # noqa: E402
from nu_measure import (BAND_MODES, ROW_COLS, SC_OPP_LEADER_POWER, _extra,  # noqa: E402
                        band_key, categories)

#: `scalars` の列名（`rust/opcg_engine/src/encode/scalars.rs` の対応表）。
#: **`own_board` は自分の場に由来する列**＝説明変数そのものなので帯に入れられない。
BASE_NAMES = {
    0: "my_life", 1: "opp_life",
    2: "my_don_active", 3: "my_don_rested", 4: "opp_don_active", 5: "opp_don_rested",
    6: "my_hand", 7: "opp_hand",
    8: "my_field_n", 9: "opp_field_n",
    10: "turn", 11: "is_my_turn",
    12: "my_leader_power", 13: "opp_leader_power",
    14: "my_leader_don", 15: "opp_leader_don",
    16: "my_deck", 17: "opp_deck", 18: "my_trash", 19: "opp_trash",
    54: "my_playable_chars", 66: "my_don_deck", 67: "opp_don_deck",
    68: "my_deck_apex_power", 69: "my_deck_apex_cost",
}
#: 自分の場に由来する列（**帯に入れられない＝処置であって交絡ではない**）。
#: `scalars` の固定位置ぶんと、`EXTRA_COLS` の名前で拾うぶん（列数から位置を出す）。
OWN_BOARD_BASE = {8, 63, 64, 65} | set(range(22, 34)) | set(range(34, 46))
#: **`attackers_left` は自分の場**（まだ攻撃できる自分のキャラ数）＝説明変数そのもの。
#: 実測で `my_field_n` と `r_x` が 0.693 対 0.691 とほぼ一致し、初版は交絡候補の 4 位に
#: 出してしまった（2026-09-14）。**名前で拾って印を付ける**。
OWN_BOARD_EXTRA = ("attackers_left",)


def own_board_cols(width):
    """自分の場に由来する列の index 集合（`scalars` の幅から `EXTRA` の位置を出す）。"""
    off = width - len(F.EXTRA_COLS)
    return OWN_BOARD_BASE | {off + F.EXTRA_COLS.index(n) for n in OWN_BOARD_EXTRA
                             if n in F.EXTRA_COLS}


def col_name(i, width):
    if i in BASE_NAMES:
        return BASE_NAMES[i]
    off = width - len(F.EXTRA_COLS)
    if i >= off:
        return "extra.%s" % F.EXTRA_COLS[i - off]
    for lo, hi, nm in ((20, 22, "koed"), (22, 34, "ability_used"), (34, 46, "newly_played"),
                       (46, 51, "my_deck_agg"), (51, 54, "opp_field_agg"),
                       (55, 60, "my_hand_agg"), (60, 63, "onplay_scan"),
                       (63, 66, "my_field_agg"), (70, 94, "leader_pair")):
        if lo <= i < hi:
            return "%s[%d]" % (nm, i - lo)
    return "col%d" % i


def _corr(a, b):
    sa, sb = a.std(), b.std()
    if sa <= 1e-12 or sb <= 1e-12:
        return 0.0
    return float(np.mean(a * b) / (sa * sb))


def collect(dirs, limit_games=0, band_mode="progress_full"):
    """自席ターン最初の main 行 → (帯, 全 scalars, 帯別の体数, 勝敗)。"""
    keys = ("lt_leader", "leader_to_sat", "over_sat")
    bands, scs, xs, zs = [], [], [], []
    games = 0
    for rows, pol, ex, L, ptr, idx in PL.iter_games(dirs, row_cols=ROW_COLS,
                                                    extra_fn=_extra):
        games += 1
        if limit_games and games > limit_games:
            break
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
            opl = float(sc[SC_OPP_LEADER_POWER]) * 1e4 or 5000.0
            cat = categories(tok, opl, "power")
            bands.append(band_key(sc, tok_row=tok, mode=band_mode))
            scs.append(np.asarray(sc, np.float64))
            xs.append([float(cat.get(k, 0.0)) for k in keys])
            zs.append(1.0 if z > 0 else 0.0)
    return bands, np.array(scs), np.array(xs), np.array(zs), games


def screen(bands, scs, xs, zs, min_rows=2):
    """帯の中で中心化して、列ごとに `|r_z · r_x|` を出す。"""
    by = {}
    for i, b in enumerate(bands):
        by.setdefault(b, []).append(i)
    keep = [i for b, ii in by.items() if len(ii) >= min_rows for i in ii]
    if len(keep) < 10:
        return None
    S = scs[keep].copy(); X = xs[keep].copy(); Z = zs[keep].copy()
    # 帯ごとに平均を引く（帯の固定効果を落とす＝帯に入っている列は 0 に潰れる）
    uniq = {}
    for i, src in enumerate(keep):
        uniq.setdefault(bands[src], []).append(i)
    for ii in uniq.values():
        S[ii] -= S[ii].mean(0)
        X[ii] -= X[ii].mean(0)
        Z[ii] -= Z[ii].mean()
    keys = ("lt_leader", "leader_to_sat", "over_sat")
    own = own_board_cols(S.shape[1])
    rows = []
    for c in range(S.shape[1]):
        col = S[:, c]
        r_z = _corr(col, Z)
        r_x = [_corr(col, X[:, k]) for k in range(X.shape[1])]
        score = max(abs(r_z * r) for r in r_x)
        rows.append({"col": c, "name": col_name(c, S.shape[1]),
                     "own_board": bool(c in own),
                     "r_z": round(r_z, 4),
                     "r_x": {k: round(r, 4) for k, r in zip(keys, r_x)},
                     "score": round(score, 5)})
    rows.sort(key=lambda r: -r["score"])
    return {"rows_used": len(keep), "bands_used": len(uniq), "cols": rows}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="src", nargs="+", required=True, help="n_records のディレクトリ")
    ap.add_argument("--limit-games", type=int, default=0)
    ap.add_argument("--band", dest="band_mode", default="progress_full", choices=BAND_MODES,
                    help="**この帯の中で**採点する（既に入っている列は 0 に潰れる）")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)

    t0 = time.time()
    bands, scs, xs, zs, games = collect(a.src, a.limit_games, a.band_mode)
    res = screen(bands, scs, xs, zs)
    out = {"games": games, "band_mode": a.band_mode,
           "rows_used": (res or {}).get("rows_used"),
           "bands_used": (res or {}).get("bands_used"),
           "top": (res or {}).get("cols", [])[:a.top],
           "seconds": round(time.time() - t0, 1)}
    txt = json.dumps(out, ensure_ascii=False, indent=2)
    print(txt)
    if a.out:
        with open(os.path.expanduser(a.out), "w", encoding="utf-8") as fh:
            fh.write(json.dumps(dict(out, all_cols=(res or {}).get("cols", [])),
                                ensure_ascii=False, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
