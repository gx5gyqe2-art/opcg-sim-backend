"""**アクティブなドン 1 個の価格 `δ_A`**——総在庫を固定した帯の中で、勝敗に対する「今アクティブなドン」の傾きを測る
（T57・2026-09-16・ユーザ指摘「ドンのアクティブかレストかの価値って今考慮されてない？」→「測ってみてください」・読み取り専用）。

`δ = 0.0277` は**総在庫**（アクティブ＋レスト＋付与）1 個の価格で、アクティブ／レストを区別しない。
アクティブなドンはそのターンの**使用権**を持つ（付与・登場・イベント）ので、在庫の価格に上乗せがあるはず:

```
δ_A = ∂P(win) / ∂(アクティブなドン)   帯 = (総在庫, 自ライフ, 相手ライフ, ターン帯, 手札, 自場の体数) を固定
```

2 つの場面で測る（どちらも `hand_value_slope.within_slope` の within 推定・局クラスタのブートストラップ）:

| 場面 | 行 | 読み |
|---|---|---|
| `mid`（自席ターンの途中） | 自席ターンの 2 番目以降の判断点（開始時は全部アクティブなので傾きが無い） | **このターンの残りの使用権**の価値 |
| `opp_turn`（相手のターン開始） | 相手の最初の判断点から**自分の**ドン（scalars 4/5）を読む・z は自分の側 | **相手のターンに払える**（カウンターイベント・ブロッカー）価値 |

**回帰しない・当てはめない**（帯の中の傾きを読むだけ）。**説明変数（アクティブ数）は帯に入れない**。

実行例:
  OPCG_LOG_SILENT=1 python tests/scripts/don_active_price.py --in ~/w41 --out ~/don_active.json
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
from hand_value_slope import turn_band, within_slope  # noqa: E402
from price_realised import S_ATTACHED_DON, don_stock  # noqa: E402
from theory_bridge import POL_COLS, ROW_COLS, _extra  # noqa: E402
from theory_order import S_IS_CHAR, SLOT_OPP_FIELD, SLOT_OWN_FIELD  # noqa: E402

DELTA = 0.0277
SC_MY_LIFE, SC_OPP_LIFE = 0, 1
SC_MY_ACT, SC_MY_REST, SC_OPP_ACT, SC_OPP_REST = 2, 3, 4, 5
SC_MY_HAND, SC_OPP_HAND = 6, 7
SC_MY_FIELD, SC_OPP_FIELD = 8, 9
SC_TURN = 10
SCENES = ("mid", "opp_turn")


def _attached(tok, slots):
    return sum(float(tok[s, S_ATTACHED_DON]) * 5.0 for s in range(slots.start, slots.stop)
               if float(tok[s, S_IS_CHAR]) > 0.5)


def row_mid(sc, tok, z):
    """自席ターンの途中の行——自分のアクティブなドンと、総在庫を含む帯。"""
    total = don_stock(sc, tok, "me")
    return {"active": float(sc[SC_MY_ACT]), "z": float(z),
            "band": "|".join(str(x) for x in (int(round(total)), int(round(float(sc[SC_MY_LIFE]))),
                                              int(round(float(sc[SC_OPP_LIFE]))),
                                              turn_band(int(round(float(sc[SC_TURN])))),
                                              int(round(float(sc[SC_MY_HAND]))),
                                              int(round(float(sc[SC_MY_FIELD])))))}


def row_opp_turn(sc, tok, z_me):
    """相手のターン開始の行（`who` は相手）から**自分の**ドンを読む。`z_me` は自分の側の勝敗。"""
    total = don_stock(sc, tok, "opp")
    return {"active": float(sc[SC_OPP_ACT]), "z": float(z_me),
            "band": "|".join(str(x) for x in (int(round(total)), int(round(float(sc[SC_OPP_LIFE]))),
                                              int(round(float(sc[SC_MY_LIFE]))),
                                              turn_band(int(round(float(sc[SC_TURN])))),
                                              int(round(float(sc[SC_OPP_HAND]))),
                                              int(round(float(sc[SC_OPP_FIELD])))))}


def collect(dirs, limit_games=0):
    rows_by = {s: [] for s in SCENES}
    stats = {"games": 0, "mid_rows": 0, "opp_turn_rows": 0}
    games = 0
    for rows, pol, ex, L, ptr, idx in PL.iter_games(dirs, row_cols=ROW_COLS, pol_cols=POL_COLS, extra_fn=_extra):
        games += 1
        if limit_games and games > limit_games:
            break
        stats["games"] += 1
        order = list(idx)
        z_of = {}
        for i in order:
            z = float(rows["z"][i])
            if z != 0.0:
                z_of[int(rows["who"][i])] = 1.0 if z > 0 else 0.0
        if len(z_of) < 2:
            continue
        seen_first = set()
        for n, i in enumerate(order):
            w, t = int(rows["who"][i]), int(rows["turn"][i])
            if t < 1 or not PL.is_own_turn(w, t) or int(rows["kind"][i]) != 0:
                continue
            sc, tok = ex["sc"][i], ex["tok"][i]
            if (w, t) not in seen_first:
                seen_first.add((w, t))
                # 相手のターン開始＝相手（w）の最初の判断点から、自分（1−w）のドンを読む
                r = row_opp_turn(sc, tok, z_of[1 - w]); r["game"] = games
                rows_by["opp_turn"].append(r); stats["opp_turn_rows"] += 1
                continue
            r = row_mid(sc, tok, z_of[w]); r["game"] = games
            rows_by["mid"].append(r); stats["mid_rows"] += 1
    return rows_by, stats


def _boot(rows, reps=200, seed=0):
    rng = np.random.default_rng(seed)
    games = sorted(set(r["game"] for r in rows))
    by = {}
    for r in rows:
        by.setdefault(r["game"], []).append(r)
    vals = []
    for _ in range(int(reps)):
        pick = rng.choice(games, size=len(games), replace=True)
        sample = [r for g in pick for r in by[g]]
        w = within_slope(sample, "z", "active")["within"]
        if w is not None:
            vals.append(w)
    return ([round(float(np.percentile(vals, 2.5)), 5), round(float(np.percentile(vals, 97.5)), 5)]
            if len(vals) >= 10 else [None, None])


def summarise(rows_by, reps=200, seed=0):
    out = {"delta_stock": DELTA, "scenes": {}}
    for scene, rows in rows_by.items():
        if len(rows) < 50:
            continue
        w = within_slope(rows, "z", "active")
        act = np.array([r["active"] for r in rows])
        out["scenes"][scene] = {"n": len(rows), "bands_used": w["bands_used"], "rows_used": w["rows_used"],
                                "active_mean": round(float(act.mean()), 3), "active_sd": round(float(act.std()), 3),
                                "delta_active_within": (round(w["within"], 5) if w["within"] is not None else None),
                                "within_se": (round(w["within_se"], 5) if w["within_se"] is not None else None),
                                "pooled": (round(w["pooled"], 5) if w["pooled"] is not None else None),
                                "ci95_games": _boot(rows, reps, seed),
                                "ratio_to_delta": (round(w["within"] / DELTA, 3) if w["within"] is not None else None)}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="src", nargs="+", required=True, help="n_records のディレクトリ")
    ap.add_argument("--limit-games", type=int, default=0)
    ap.add_argument("--boot-reps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    t0 = time.time()
    rows_by, stats = collect(a.src, a.limit_games)
    res = {"stats": stats, "summary": summarise(rows_by, a.boot_reps, a.seed), "seconds": round(time.time() - t0, 1)}
    txt = json.dumps(res, ensure_ascii=False, indent=2)
    print(txt)
    if a.out:
        with open(os.path.expanduser(a.out), "w", encoding="utf-8") as fh:
            fh.write(txt + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
