"""**相手の場の分布を記録から採る**（T46・2026-09-16・A 層・読み取り専用）。

`ν` の攻撃項に**「将来現れる相手の体を倒せる潜在価値」**（ユーザ決定 2026-09-16）を入れるための土台。
**今の場ではなく分布**で評価する（登場の価値が局面でブレない）——分布は**自席ターンの入口で相手の場に
居た体**（パワー・ブロッカーか）を、**判断点で判る量＝残りターン `R`（相手の残りライフの近似）**で条件付けて採る
（ターン数では切らない・§17.1 の 2 本の時計と同じ原理）。

出力は `tests/fixtures/opp_boards.json`——`theory_order.option_value` が読む。
**式（`nu_of`）はここでは一切使わない**（分布だけ・新定数なし）。

実行例:
  OPCG_LOG_SILENT=1 python tests/scripts/opp_board_dist.py --in ~/w39 ~/w42 --out tests/fixtures/opp_boards.json
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
from theory_bridge import POL_COLS, ROW_COLS, _extra  # noqa: E402
from theory_order import (S_IS_BLOCKER, S_IS_CHAR, S_POWER, SC_MY_LEADER_POWER, SC_OPP_LIFE,  # noqa: E402
                          SLOT_OPP_FIELD)

#: 既定の置き場（`theory_order` が読む）
DEFAULT_OUT = os.path.join(_ROOT, "tests", "fixtures", "opp_boards.json")
#: `R` の刻み（`r_turns = clamp(相手ライフ, 1, 5)` と同じ）
R_BUCKETS = (1, 2, 3, 4, 5)
#: 1 つの `R` に残す盤面の数（無作為抽出・再現可能）——分布の形を保ちつつ fixture を小さく保つ
SAMPLE_PER_R = 400


def r_bucket(opp_life):
    return int(max(1, min(5, round(float(opp_life)))))


def collect(dirs, limit_games=0):
    """`R` ごとに (自リーダーのパワー, [(体のパワー, ブロッカーか), …]) の列。"""
    boards = {r: [] for r in R_BUCKETS}
    stats = {"games": 0, "turns": 0}
    games = 0
    for rows, pol, ex, L, ptr, idx in PL.iter_games(dirs, row_cols=ROW_COLS,
                                                    pol_cols=POL_COLS, extra_fn=_extra):
        games += 1
        if limit_games and games > limit_games:
            break
        stats["games"] += 1
        seen = set()
        for i in idx:
            w, t = int(rows["who"][i]), int(rows["turn"][i])
            if t < 1 or not PL.is_own_turn(w, t) or int(rows["kind"][i]) != 0 or (w, t) in seen:
                continue
            seen.add((w, t))
            stats["turns"] += 1
            sc, tok = ex["sc"][i], ex["tok"][i]
            bodies = []
            for s in range(SLOT_OPP_FIELD.start, SLOT_OPP_FIELD.stop):
                if float(tok[s, S_IS_CHAR]) <= 0.5:
                    continue
                bodies.append([int(round(float(tok[s, S_POWER]) * 1e4 / 100.0) * 100),
                               bool(float(tok[s, S_IS_BLOCKER]) > 0.5)])
            mlp = int(round(float(sc[SC_MY_LEADER_POWER]) * 1e4 / 100.0) * 100) or 5000
            boards[r_bucket(sc[SC_OPP_LIFE])].append([mlp, bodies])
    return boards, stats


def sample(boards, per_r=SAMPLE_PER_R, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for r, bs in boards.items():
        if len(bs) <= per_r:
            out[str(r)] = bs
        else:
            k = rng.choice(len(bs), size=per_r, replace=False)
            out[str(r)] = [bs[int(j)] for j in sorted(k)]
    return out


def summarise(boards):
    out = {}
    for r, bs in boards.items():
        n = len(bs)
        if not n:
            continue
        sizes = [len(b[1]) for b in bs]
        pw = [p for b in bs for p, _ in b[1]]
        out[str(r)] = {"boards": n, "empty_share": round(sum(1 for s in sizes if s == 0) / n, 4),
                       "bodies_mean": round(float(np.mean(sizes)), 3),
                       "power_mean": round(float(np.mean(pw)), 1) if pw else None,
                       "blocker_share": (round(sum(1 for b in bs for _, k in b[1] if k) / len(pw), 4)
                                         if pw else None)}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="src", nargs="+", required=True, help="n_records のディレクトリ")
    ap.add_argument("--limit-games", type=int, default=0)
    ap.add_argument("--sample", type=int, default=SAMPLE_PER_R)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=DEFAULT_OUT)
    a = ap.parse_args(argv)
    t0 = time.time()
    boards, stats = collect(a.src, a.limit_games)
    res = {"source": [os.path.basename(os.path.abspath(d)) for d in a.src], "stats": stats,
           "sample_per_r": a.sample, "seed": a.seed,
           "summary": summarise(boards), "by_r": sample(boards, a.sample, a.seed),
           "seconds": round(time.time() - t0, 1)}
    txt = json.dumps(res, ensure_ascii=False)
    with open(os.path.expanduser(a.out), "w", encoding="utf-8") as fh:
        fh.write(txt + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "by_r"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
