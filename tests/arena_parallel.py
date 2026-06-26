"""自己対戦アリーナの**並列実行**（検証基盤の高速化・dev専用）。

1 局 ~30-48s が律速。各局は独立（seed で決定論）なので、`multiprocessing` で**コア並列**して壁時計を縮める
（4コアで ~3-4x）。方策・評価・結果は不変＝**純粋な高速化**（同 seed は同結果）。

評価 v2 の係数は**ワーカープロセスごとに spec の `coeffs` から設定**する（プロセス分離なので相互汚染なし）。
SPSA の f(θ) 評価では θ を coeffs として渡せば、その θ で並列対戦できる。

対照ペア（antithetic）: 各 seed を両席で 1 回ずつ（同 game-seed・`separate_policy_rng=True`）。
"""
import multiprocessing as mp
import os
from typing import Any, Dict, List, Optional

import conftest  # noqa: F401
from cpu_arena import _load_db, play_game, DEFAULT_MAX_STEPS, elo_delta
from opcg_sim.src.core import cpu_eval_v2

_DB = None


def _init_worker():
    """ワーカー起動時に 1 回だけカード DB をロード（局ごとの再ロードを避ける）。"""
    global _DB
    _DB = _load_db()


def _play_one(spec: Dict[str, Any]) -> Dict[str, Any]:
    """1 局を実行して勝者を返す。spec.coeffs があれば cpu_eval_v2 係数を上書きしてから対戦。

    例外（InvariantError 等）はワーカー内で握り潰して error 文字列で返す＝1局の失敗でプール全体を
    ハングさせない（pickle 不能例外の転送失敗対策）。失敗局は winner=None・error 付きで親が集計から除外。
    """
    coeffs = spec.get("coeffs")
    if coeffs:
        for k, v in coeffs.items():
            setattr(cpu_eval_v2, k, v)
    try:
        res = play_game(spec["seed"], _DB, spec["p1d"], spec["p2d"],
                        max_steps=spec.get("max_steps", DEFAULT_MAX_STEPS),
                        p1_eval_v2=spec.get("p1_v2"), p2_eval_v2=spec.get("p2_v2"),
                        p1_search=spec.get("p1_search"), p2_search=spec.get("p2_search"),
                        p1_budget=spec.get("p1_budget"), p2_budget=spec.get("p2_budget"),
                        p1_mcts=spec.get("p1_mcts"), p2_mcts=spec.get("p2_mcts"),
                        p1_alpha=spec.get("p1_alpha"), p2_alpha=spec.get("p2_alpha"),
                        separate_policy_rng=True)
        return {"pair": spec["pair"], "seat": spec["seat"], "winner": res["winner"]}
    except Exception as e:
        return {"pair": spec["pair"], "seat": spec["seat"], "winner": None,
                "error": f"seed={spec['seed']} {type(e).__name__}: {e}"}


def _default_workers() -> int:
    return max(1, (os.cpu_count() or 2) - 1)


def paired_play(pairs: int, seed0: int = 0, max_steps: int = DEFAULT_MAX_STEPS,
                coeffs: Optional[Dict[str, float]] = None, workers: Optional[int] = None,
                challenger_eval_v2: bool = True, baseline_eval_v2: bool = False,
                challenger_search=None, baseline_search=None,
                challenger_budget=None, baseline_budget=None,
                challenger_difficulty: str = "hard", baseline_difficulty: str = "hard",
                challenger_mcts=None, baseline_mcts=None,
                challenger_alpha=None, baseline_alpha=None) -> Dict[str, Any]:
    """対照ペアを**並列**で実行し、ペア単位スコア（{0,0.5,1}）と勝率を返す。

    challenger = 評価v2 ON（既定）／baseline = 評価v2 OFF（成熟J値）。両者とも難易度 hard（既定）。
    coeffs（任意）= 評価 v2 の係数上書き（SPSA の θ 評価用）。workers=1 で逐次（デバッグ用）。
    `challenger_search`/`baseline_search`（任意・L1外の深さA/B用）= `(horizon, max_ply)` で席別に探索深さを
    上書き（None で既定）。eval_v2 を両側 OFF にして探索深さだけを振れば「深さの伸びしろ」を測れる。
    `challenger_difficulty`/`baseline_difficulty`（既定 hard）と `challenger_mcts`/`baseline_mcts`（dict・
    iters/horizon/worlds/determinize）で **expert(MCTS)** を片側/両側に置ける＝expert vs hard や
    expert の worlds/horizon A/B（L1外・MCTS側の伸びしろ）を同じ対照ペア基盤で測る。
    """
    workers = workers or _default_workers()
    cd, bd = challenger_difficulty, baseline_difficulty
    # 各ペア＝2 局（席A: challenger=p1 / 席B: challenger=p2・同 seed）。難易度も席で入れ替える。
    specs: List[Dict[str, Any]] = []
    for k in range(pairs):
        seed = seed0 + k
        specs.append({"pair": k, "seat": "A", "seed": seed, "p1d": cd, "p2d": bd,
                      "p1_v2": challenger_eval_v2, "p2_v2": baseline_eval_v2,
                      "p1_search": challenger_search, "p2_search": baseline_search,
                      "p1_budget": challenger_budget, "p2_budget": baseline_budget,
                      "p1_mcts": challenger_mcts, "p2_mcts": baseline_mcts,
                      "p1_alpha": challenger_alpha, "p2_alpha": baseline_alpha,
                      "max_steps": max_steps, "coeffs": coeffs})
        specs.append({"pair": k, "seat": "B", "seed": seed, "p1d": bd, "p2d": cd,
                      "p1_v2": baseline_eval_v2, "p2_v2": challenger_eval_v2,
                      "p1_search": baseline_search, "p2_search": challenger_search,
                      "p1_budget": baseline_budget, "p2_budget": challenger_budget,
                      "p1_mcts": baseline_mcts, "p2_mcts": challenger_mcts,
                      "p1_alpha": baseline_alpha, "p2_alpha": challenger_alpha,
                      "max_steps": max_steps, "coeffs": coeffs})

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
    ap = argparse.ArgumentParser(description="並列アリーナ（評価v2 ON vs OFF・対照ペア）")
    ap.add_argument("--pairs", type=int, default=15)
    ap.add_argument("--workers", type=int, default=0, help="0=自動（コア数-1）")
    ap.add_argument("--seed0", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    res = paired_play(args.pairs, seed0=args.seed0, workers=(args.workers or None))
    dt = time.time() - t0
    print(f"v2 勝率 = {res['win_rate']:.3f} (Elo {res['elo']:+.0f}) | "
          f"{res['games']}局 / {res['workers']}並列 / {dt:.1f}s "
          f"({dt/res['games']:.1f}s/局)")
