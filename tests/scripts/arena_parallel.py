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

import os as _os, sys as _sys  # noqa: E402  test bootstrap (sys.path + google stub)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap  # noqa: E402,F401
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
                        p1_pimc=spec.get("p1_pimc", 1), p2_pimc=spec.get("p2_pimc", 1),
                        p1_coeffs=spec.get("p1_coeffs"), p2_coeffs=spec.get("p2_coeffs"),
                        p1_sims=spec.get("p1_sims", 160), p2_sims=spec.get("p2_sims", 160),
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
                challenger_pimc: int = 1, baseline_pimc: int = 1,
                challenger_sims: int = 160, baseline_sims: int = 160) -> Dict[str, Any]:
    """対照ペアを**並列**で実行し、ペア単位スコア（{0,0.5,1}）と勝率（challenger 視点）を返す。

    難易度は hard(L1) と **learned(Gen2)** を混在可（A1）。`challenger_difficulty="learned"` で Gen2 の強度 A/B。
    `challenger_sims`/`baseline_sims` は learned の MCTS 探索数（既定=本番 160）。以下 L1 系ノブ:
    両者とも難易度 hard（既定）。
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
                      "p1_pimc": challenger_pimc, "p2_pimc": baseline_pimc,
                      "p1_coeffs": challenger_coeffs, "p2_coeffs": baseline_coeffs,
                      "p1_sims": challenger_sims, "p2_sims": baseline_sims,
                      "max_steps": max_steps})
        specs.append({"pair": k, "seat": "B", "seed": seed, "p1d": bd, "p2d": cd,
                      "p1_search": baseline_search, "p2_search": challenger_search,
                      "p1_budget": baseline_budget, "p2_budget": challenger_budget,
                      "p1_pimc": baseline_pimc, "p2_pimc": challenger_pimc,
                      "p1_coeffs": baseline_coeffs, "p2_coeffs": challenger_coeffs,
                      "p1_sims": baseline_sims, "p2_sims": challenger_sims,
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
    from cpu_arena import elo_ci
    from opcg_sim.src.core import cpu_ai

    ap = argparse.ArgumentParser(
        description="並列アリーナ（L1 自己対戦・対照ペア）。挑戦者の探索深さ／思考予算を席別に振り、"
                    "『深さ／思考時間の伸びしろ』の Elo を対照ペアで測る（旧 depth_arena / thinktime_arena を統合）。")
    ap.add_argument("--pairs", type=int, default=15, help="対照ペア数（総局数=2×pairs）")
    ap.add_argument("--workers", type=int, default=0, help="0=自動（コア数-1）")
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    # 席別 A/B（両側とも葉は出荷 J値固定。探索量だけを振る）。既定は挑戦者=ベースライン＝本番hard（≈50%）。
    ap.add_argument("--challenger-horizon", type=int, default=None,
                    help="挑戦者の探索ホライズン（旧 depth/thinktime の --horizon）。指定時 max-ply も要検討")
    ap.add_argument("--challenger-max-ply", type=int, default=None, help="挑戦者の総ply上限（horizon 比例で）")
    ap.add_argument("--challenger-budget", type=int, default=None, help="挑戦者の深掘り1手あたり clone 予算")
    ap.add_argument("--challenger-pimc", type=int, default=1, help="挑戦者の PIMC 世界数")
    ap.add_argument("--baseline-horizon", type=int, default=None, help="ベースラインのホライズン（既定=本番hard）")
    ap.add_argument("--baseline-max-ply", type=int, default=None, help="ベースラインの ply上限")
    ap.add_argument("--baseline-budget", type=int, default=None, help="ベースラインの予算（既定=本番hard）")
    ap.add_argument("--baseline-pimc", type=int, default=1, help="ベースラインの PIMC 世界数")
    # A1: learned(Gen2) の強度 A/B。例: --challenger-difficulty learned で「Gen2 vs 出荷hard」を測る。
    ap.add_argument("--challenger-difficulty", choices=["hard", "learned"], default="hard")
    ap.add_argument("--baseline-difficulty", choices=["hard", "learned"], default="hard")
    ap.add_argument("--challenger-sims", type=int, default=160, help="learned 挑戦者の MCTS 探索数")
    ap.add_argument("--baseline-sims", type=int, default=160, help="learned ベースラインの MCTS 探索数")
    ap.add_argument("--time", action="store_true", help="壁時計 latency を併記（深さ／思考時間のコスト可視化）")
    args = ap.parse_args()

    chal_search = ((args.challenger_horizon, args.challenger_max_ply)
                   if (args.challenger_horizon or args.challenger_max_ply) else None)
    base_search = ((args.baseline_horizon, args.baseline_max_ply)
                   if (args.baseline_horizon or args.baseline_max_ply) else None)

    t0 = time.time()
    res = paired_play(args.pairs, seed0=args.seed0, max_steps=args.max_steps,
                      workers=(args.workers or None),
                      challenger_difficulty=args.challenger_difficulty,
                      baseline_difficulty=args.baseline_difficulty,
                      challenger_sims=args.challenger_sims, baseline_sims=args.baseline_sims,
                      challenger_search=chal_search, baseline_search=base_search,
                      challenger_budget=args.challenger_budget, baseline_budget=args.baseline_budget,
                      challenger_pimc=args.challenger_pimc, baseline_pimc=args.baseline_pimc)
    dt = time.time() - t0
    ci = _pair_level_ci(res["pair_scores"]) if res["pair_scores"] else None
    print(f"勝率 = {res['win_rate']:.3f} (Elo {res['elo']:+.0f}) | "
          f"{res['games']}局 / {res['workers']}並列 / {dt:.1f}s "
          f"({dt/max(1,res['games']):.1f}s/局)")
    if ci:
        naive = elo_ci(res["win_rate"] * res["games"], res["games"])
        print(f"  ペア単位CI（分散低減・正）  : Elo95% [{ci['elo_lo']:+.0f}, {ci['elo_hi']:+.0f}]")
        print(f"  素朴Bernoulli CI（参考・広い）: Elo95% [{naive['elo_lo']:+.0f}, {naive['elo_hi']:+.0f}]")
        print("判定: " + ("挑戦者が有意に強い" if ci["elo_lo"] > 0 else
                         "挑戦者が有意に弱い" if ci["elo_hi"] < 0 else "互角（有意差なし）"))
    if res.get("failed_games"):
        print(f"⚠ 失敗局 {res['failed_games']}（除外）。例: " + " | ".join(res.get("errors", [])[:2]))
