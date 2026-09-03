"""探索設定の A/B プローブ（v14）: ネットを固定したまま**探索の使い方**だけを変えて対戦させる。

背景（v9〜v13 の確定）: gen6 への学習系の介入は12回すべて対gen6 アリーナ 0.41-0.49 で頭打ち。
容量を1.7倍にして held-out MAE を 28% 改善しても実戦は不変（0.425）＝**教師ラベルへの当てはまりと
実戦力が相関していない**。一方 **探索の使い方（sims・c_puct・root 読み出し・aux 粘り項）は
このラインで一度も調整していない**。ネット非依存で実戦力を動かせる唯一の未検証領域を潰す。

同一ネット（既定 gen6）で「設定を変えた席」と「既定の席」を CRN ペア（同 seed・席入替）で
対戦させ、勝率を測る。0.5 から有意にずれれば、その設定差がそのまま実戦力の差。

実行例:
  OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/search_config_probe.py \
    --variants "sims320:sims=320" "cpuct25:c_puct=2.5" --pairs 20
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import math
import multiprocessing as mp
import time

import os as _os, sys as _sys  # noqa: E402  test bootstrap
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap  # noqa: E402,F401

_G = {}


def _parse_kv(spec):
    """'sims=320,c_puct=2.5' → {'sims':320,'c_puct':2.5}（inf/true/false も解釈）。"""
    out = {}
    for part in spec.split(","):
        if not part.strip():
            continue
        k, v = part.split("=")
        k = k.strip(); v = v.strip()
        if v in ("inf", "+inf"):
            out[k] = math.inf
        elif v.lower() in ("true", "false"):
            out[k] = (v.lower() == "true")
        elif k in ("sims",):
            out[k] = int(v)
        else:
            out[k] = float(v)
    return out


def _init_pool(kv):
    from cpu_arena import _load_db
    from opcg_sim.src.core.cpu_learned import LearnedEngine
    _G["db"] = _load_db()
    _G["var"] = LearnedEngine(**kv)      # 設定を変えた側（ネットは既定＝gen6）
    _G["base"] = LearnedEngine()         # 既定の側


def _play_pair(seed):
    from cpu_arena import play_game
    a = play_game(seed, _G["db"], "learned", "learned", p1_engine=_G["var"], p2_engine=_G["base"])
    b = play_game(seed, _G["db"], "learned", "learned", p1_engine=_G["base"], p2_engine=_G["var"])
    return (1.0 if a["winner"] == "p1" else 0.0) + (1.0 if b["winner"] == "p2" else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", required=True,
                    help="'名前:k=v[,k=v]' の並び（例 sims320:sims=320）")
    ap.add_argument("--pairs", type=int, default=20, help="seed ペア数（×2局）")
    ap.add_argument("--seed0", type=int, default=61000)
    ap.add_argument("--workers", type=int, default=3)
    ARGS = ap.parse_args()

    seeds = list(range(ARGS.seed0, ARGS.seed0 + ARGS.pairs))
    rows = []
    for spec in ARGS.variants:
        name, kvs = spec.split(":", 1)
        kv = _parse_kv(kvs)
        t0 = time.time()
        with mp.Pool(ARGS.workers, initializer=_init_pool, initargs=(kv,)) as pool:
            wins = sum(pool.imap_unordered(_play_pair, seeds))
        games = 2 * len(seeds)
        wr = wins / games
        se = math.sqrt(max(wr * (1 - wr), 1e-9) / games)
        rows.append({"name": name, "kv": {k: str(v) for k, v in kv.items()},
                     "wins": wins, "games": games, "wr": round(wr, 4),
                     "ci95": [round(wr - 1.96 * se, 3), round(wr + 1.96 * se, 3)],
                     "sec": int(time.time() - t0)})
        print(f"{name:<12} {kvs:<28} {wins:>5.1f}/{games}  wr={wr:.3f} "
              f"CI[{wr - 1.96 * se:.3f},{wr + 1.96 * se:.3f}]  ({time.time() - t0:.0f}s)",
              flush=True)
    print(f"SEARCH_PROBE_RESULT {json.dumps(rows, ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    _sys.exit(main())
