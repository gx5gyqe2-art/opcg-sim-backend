"""価値関数の自己対戦データ収集を**コア並列**で回す（NNUE step2・dev専用）。

`collect_value_data.collect_game`（hard α-β 自己対戦＝本番方策）を `multiprocessing.Pool` で並列実行。
各局は seed で決定論＝並列でも同結果。hard は1局 ~30-50s と重いので、コア並列で壁時計を縮める。
出力は単一 JSONL（`{"f":[...],"y":0/1}`）＝`train_value.py`/`train_gbdt.py` がそのまま読む。

実行例:
    OPCG_LOG_SILENT=1 python tests/collect_value_parallel.py --games 600 --real-decks --out tests/value_hard.jsonl
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
from typing import Any, Dict, List

import conftest  # noqa: F401
from collect_value_data import collect_game
from cpu_selfplay import _load_db, DEFAULT_MAX_STEPS
from opcg_sim.src.core import cpu_features, cpu_ai

_DB = None
_CFG: Dict[str, Any] = {}


def _init_worker(cfg: Dict[str, Any]):
    global _DB, _CFG
    _DB = _load_db()
    _CFG = cfg
    # 高速化（既定で適用）: 探索予算を下げる。思考時間A/Bで予算75≈600＝**hard は評価律速で予算↓でも強さ不変**
    # （docs/reports/... のレバー検証）＝データ品質を保ったまま ~4x 速い。set_budget_override で全 decide に適用。
    cpu_ai.set_budget_override(cfg.get("budget"))


def _collect_one(seed: int) -> List[Dict[str, Any]]:
    try:
        return collect_game(seed, _DB, _CFG["difficulty"], _CFG["max_steps"],
                            real_decks=_CFG["real_decks"])
    except Exception:
        return []


def _default_workers() -> int:
    return max(1, (os.cpu_count() or 2) - 1)


def main(argv=None):
    ap = argparse.ArgumentParser(description="価値データ収集（コア並列・hard 自己対戦）")
    ap.add_argument("--games", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--difficulty", choices=["hard"], default="hard")
    ap.add_argument("--real-decks", action="store_true", help="deckgen 実デッキ（検証済リーダー巡回）")
    ap.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    ap.add_argument("--workers", type=int, default=0, help="0=自動（コア数-1）")
    ap.add_argument("--budget", type=int, default=75,
                    help="探索予算/手（既定75＝本番相当・評価律速で強さ不変のまま~4x高速。Noneで既定300）")
    ap.add_argument("--out", default="tests/value_hard.jsonl")
    args = ap.parse_args(argv)

    cfg = {"difficulty": args.difficulty, "max_steps": args.max_steps, "real_decks": args.real_decks,
           "budget": args.budget}
    workers = args.workers or _default_workers()
    seeds = [args.seed + g for g in range(args.games)]

    n_rows = n_games = 0
    with open(args.out, "w", encoding="utf-8") as f:
        with mp.Pool(workers, initializer=_init_worker, initargs=(cfg,)) as pool:
            for i, rows in enumerate(pool.imap_unordered(_collect_one, seeds), 1):
                if rows:
                    n_games += 1
                    for r in rows:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                        n_rows += 1
                if i % 50 == 0:
                    print(f"  {i}/{args.games} games … valid={n_games} rows={n_rows}", flush=True)
    print(f"done: {n_games}/{args.games} valid games, {n_rows} rows → {args.out} "
          f"(features={cpu_features.N_FEATURES}, workers={workers})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
