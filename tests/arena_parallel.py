"""自己対戦アリーナの**並列実行**（検証基盤の高速化・dev専用）。

1 局 ~30-48s が律速。各局は独立（seed で決定論）なので、`multiprocessing` で**コア並列**して壁時計を縮める
（4コアで ~3-4x）。方策・評価・結果は不変＝**純粋な高速化**（同 seed は同結果）。

評価 v2 の係数は**ワーカープロセスごとに spec の `coeffs` から設定**する（プロセス分離なので相互汚染なし）。
SPSA の f(θ) 評価では θ を coeffs として渡せば、その θ で並列対戦できる。

対照ペア（antithetic）: 各 seed を両席で 1 回ずつ（同 game-seed・`separate_policy_rng=True`）。
"""
import math
import multiprocessing as mp
import os
from typing import Any, Dict, List, Optional

import conftest  # noqa: F401
from cpu_arena import _load_db, play_game, DEFAULT_MAX_STEPS, elo_delta
from opcg_sim.src.core import cpu_eval_v2

_DB = None


def _pair_level_ci(pair_scores):
    """ペア単位スコア（{0,0.5,1}）の平均と 95% CI（正規近似）→ 勝率と Elo 区間。

    対照ペア設計の正しい CI＝ペアを独立試行として扱う（2×pairs 局を独立 Bernoulli 扱いする素朴 CI より狭い）。
    （深さ/ブレンド/思考時間アリーナが共有する集計ヘルパ。）
    """
    n = len(pair_scores)
    mean = sum(pair_scores) / n
    var = sum((s - mean) ** 2 for s in pair_scores) / max(1, n - 1)
    half = 1.96 * math.sqrt(var / n)
    lo, hi = max(0.0, mean - half), min(1.0, mean + half)
    return {"win_rate": mean, "lo": lo, "hi": hi,
            "elo": elo_delta(mean), "elo_lo": elo_delta(lo), "elo_hi": elo_delta(hi)}


def _init_worker():
    """ワーカー起動時に 1 回だけカード DB をロード（局ごとの再ロードを避ける）。"""
    global _DB
    _DB = _load_db()


def _play_one(spec: Dict[str, Any]) -> Dict[str, Any]:
    """1 局を実行して勝者を返す。spec.coeffs があれば cpu_eval_v2 係数を上書きしてから対戦。

    例外（InvariantError 等）はワーカー内で握り潰して error 文字列で返す＝1局の失敗でプール全体を
    ハングさせない（pickle 不能例外の転送失敗対策）。失敗局は winner=None・error 付きで親が集計から除外。
    """
    try:
        res = play_game(spec["seed"], _DB, spec["p1d"], spec["p2d"],
                        max_steps=spec.get("max_steps", DEFAULT_MAX_STEPS),
                        p1_search=spec.get("p1_search"), p2_search=spec.get("p2_search"),
                        p1_budget=spec.get("p1_budget"), p2_budget=spec.get("p2_budget"),
                        p1_alpha=spec.get("p1_alpha"), p2_alpha=spec.get("p2_alpha"),
                        p1_pimc=spec.get("p1_pimc", 1), p2_pimc=spec.get("p2_pimc", 1),
                        p1_coeffs=spec.get("p1_coeffs"), p2_coeffs=spec.get("p2_coeffs"),
                        separate_policy_rng=True)
        return {"pair": spec["pair"], "seat": spec["seat"], "winner": res["winner"]}
    except Exception as e:
        return {"pair": spec["pair"], "seat": spec["seat"], "winner": None,
                "error": f"seed={spec['seed']} {type(e).__name__}: {e}"}


def _default_workers() -> int:
    return max(1, (os.cpu_count() or 2) - 1)


def paired_play(pairs: int, seed0: int = 0, max_steps: int = DEFAULT_MAX_STEPS,
                challenger_coeffs: Optional[Dict[str, float]] = None,
                baseline_coeffs: Optional[Dict[str, float]] = None,
                workers: Optional[int] = None,
                challenger_search=None, baseline_search=None,
                challenger_budget=None, baseline_budget=None,
                challenger_difficulty: str = "hard", baseline_difficulty: str = "hard",
                challenger_alpha=None, baseline_alpha=None,
                challenger_pimc: int = 1, baseline_pimc: int = 1) -> Dict[str, Any]:
    """対照ペアを**並列**で実行し、ペア単位スコア（{0,0.5,1}）と勝率（challenger 視点）を返す。

    評価は L1 単一系統（`cpu_eval_v2`）。両者とも難易度 hard（既定）。
    `challenger_coeffs`/`baseline_coeffs`（任意）= L1 係数の**席別**上書き（SPSA の候補θ vs 凍結基準）。
    手書きeval 撤去後は両側 L1 なので、係数差を席別に与えないと有意な A/B にならない（同係数＝50%）。
    `challenger_search`/`baseline_search`（任意・深さA/B用）= `(horizon, max_ply)` で席別に探索深さを
    上書き（None で既定）。探索深さだけを振れば「深さの伸びしろ」を測れる。
    `challenger_pimc`/`baseline_pimc`・`challenger_budget`/`baseline_budget` で hard の
    PIMC 世界数・予算を席別に振れる（α-β の係数/構成 A/B 用）。
    """
    workers = workers or _default_workers()
    cd, bd = challenger_difficulty, baseline_difficulty
    # 各ペア＝2 局（席A: challenger=p1 / 席B: challenger=p2・同 seed）。難易度も席で入れ替える。
    specs: List[Dict[str, Any]] = []
    for k in range(pairs):
        seed = seed0 + k
        specs.append({"pair": k, "seat": "A", "seed": seed, "p1d": cd, "p2d": bd,
                      "p1_search": challenger_search, "p2_search": baseline_search,
                      "p1_budget": challenger_budget, "p2_budget": baseline_budget,
                      "p1_alpha": challenger_alpha, "p2_alpha": baseline_alpha,
                      "p1_pimc": challenger_pimc, "p2_pimc": baseline_pimc,
                      "p1_coeffs": challenger_coeffs, "p2_coeffs": baseline_coeffs,
                      "max_steps": max_steps})
        specs.append({"pair": k, "seat": "B", "seed": seed, "p1d": bd, "p2d": cd,
                      "p1_search": baseline_search, "p2_search": challenger_search,
                      "p1_budget": baseline_budget, "p2_budget": challenger_budget,
                      "p1_alpha": baseline_alpha, "p2_alpha": challenger_alpha,
                      "p1_pimc": baseline_pimc, "p2_pimc": challenger_pimc,
                      "p1_coeffs": baseline_coeffs, "p2_coeffs": challenger_coeffs,
                      "max_steps": max_steps})

    if workers <= 1:
        _init_worker()
        results = [_play_one(s) for s in specs]
    else:
        with mp.Pool(workers, initializer=_init_worker) as pool:
            results = pool.map(_play_one, specs)

    # 失敗局（error）を抽出。両局が成功したペアだけを採点対象にする（片側失敗ペアは集計から除外）。
    errors = [r["error"] for r in results if r.get("error")]
    ok_by_pair: Dict[int, int] = {}
    for r in results:
        if not r.get("error"):
            ok_by_pair[r["pair"]] = ok_by_pair.get(r["pair"], 0) + 1
    scored_pairs = [k for k in range(pairs) if ok_by_pair.get(k, 0) == 2]

    # ペアごとに challenger の勝点を集計（席A: p1勝ち / 席B: p2勝ち）。
    by_pair: Dict[int, float] = {}
    for r in results:
        if r.get("error"):
            continue
        chal_won = (r["winner"] == "p1") if r["seat"] == "A" else (r["winner"] == "p2")
        by_pair[r["pair"]] = by_pair.get(r["pair"], 0.0) + (1.0 if chal_won else 0.0)
    pair_scores = [by_pair[k] / 2.0 for k in scored_pairs]
    n = len(pair_scores)
    wr = (sum(pair_scores) / n) if n else 0.5
    return {"win_rate": wr, "elo": elo_delta(wr), "pair_scores": pair_scores,
            "pairs": n, "games": 2 * n, "workers": workers,
            "errors": errors, "failed_games": len(errors)}


if __name__ == "__main__":
    import argparse
    import time
    ap = argparse.ArgumentParser(description="並列アリーナ（L1 自己対戦・対照ペア／SPSA coeffs 用）")
    ap.add_argument("--pairs", type=int, default=15)
    ap.add_argument("--workers", type=int, default=0, help="0=自動（コア数-1）")
    ap.add_argument("--seed0", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    res = paired_play(args.pairs, seed0=args.seed0, workers=(args.workers or None))
    dt = time.time() - t0
    print(f"勝率 = {res['win_rate']:.3f} (Elo {res['elo']:+.0f}) | "
          f"{res['games']}局 / {res['workers']}並列 / {dt:.1f}s "
          f"({dt/res['games']:.1f}s/局)")
