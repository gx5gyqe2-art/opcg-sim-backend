"""価値データの読込＋**試合単位 train/val split**（リーク防止・全トレーナ共通）。

同じ試合（"g"＝seed）のサンプルを train と val に跨がせない＝val を「未知の試合」にする＝val_acc を
正直な汎化指標にする。`load_rows`+`split` を logreg/GBDT/MLP の全トレーナが共有し、**同一 seed/val_frac で
同一の val 試合集合**を使う＝3 モデルを公平に比較できる。"g" 無しデータは行単位 split にフォールバック。
"""
import json
import random
from typing import Any, Dict, List, Tuple

import conftest  # noqa: F401
from opcg_sim.src.core import cpu_features


def load_rows(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if len(r.get("f", [])) == cpu_features.N_FEATURES:
                rows.append(r)
    return rows


def split(rows: List[Dict[str, Any]], val_frac: float = 0.15, seed: int = 0
          ) -> Tuple[list, list, list, list, Dict[str, Any]]:
    """(Xtr, Ytr, Xva, Yva, meta) を返す。"g" があれば**試合単位**、無ければ行単位で分割。"""
    has_g = bool(rows) and all("g" in r for r in rows)
    if has_g:
        games = sorted({r["g"] for r in rows})
        random.Random(seed).shuffle(games)
        nval = max(1, int(val_frac * len(games)))
        val_games = set(games[:nval])
        tr = [r for r in rows if r["g"] not in val_games]
        va = [r for r in rows if r["g"] in val_games]
        meta = {"mode": "game", "n_games": len(games), "val_games": nval}
    else:
        idx = list(range(len(rows)))
        random.Random(seed).shuffle(idx)
        nval = max(1, int(val_frac * len(rows)))
        va = [rows[i] for i in idx[:nval]]
        tr = [rows[i] for i in idx[nval:]]
        meta = {"mode": "row", "n_games": None, "val_games": None}
    Xtr = [r["f"] for r in tr]; Ytr = [float(r["y"]) for r in tr]
    Xva = [r["f"] for r in va]; Yva = [float(r["y"]) for r in va]
    meta.update({"n_train": len(Xtr), "n_val": len(Xva)})
    return Xtr, Ytr, Xva, Yva, meta
