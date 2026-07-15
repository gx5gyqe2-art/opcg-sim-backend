"""CPU 性能ゲート（運用ワンコマンド・dev/定期実行・perf計画 A2）。

本番既定 CPU＝**Gen2(learned)** の非退行を1コマンドで PASS/FAIL 判定する。凍結ベースラインは
**L1(hard)**（決定論・不変の物差し。net-vs-net の凍結Gen2 比較は cpu_learned のシングルトン解消＝A3 待ち）。

測るもの:
  1. 絶対強度アンカー: learned(Gen2) vs hard を対照ペア並列で戦わせ、勝率→Elo＋ペア単位 CI。
  2. レイテンシ: learned の 1 手思考時間（median/max・1手1秒予算の遵守）。
  3. 非退行: 全局決着（failed=0）＝クラッシュ/無限ループが無い。
  4. どのネットを測ったか: gen2_*.npz のハッシュを記録（結果の再現・追跡）。

判定は `evaluate_gate`（純関数）に集約＝`tests/test_perf_gate.py` が高速に固定する。

実行例:
    OPCG_LOG_SILENT=1 python tests/scripts/perf_gate.py --quick     # 少ペア・低sims（疎通/CI用の軽い確認）
    OPCG_LOG_SILENT=1 python tests/scripts/perf_gate.py --full      # 本走（pairs=40・sims=160・手動/定期）
    OPCG_LOG_SILENT=1 python tests/scripts/perf_gate.py --pairs 20 --sims 160 --max-latency-ms 1200
"""
import argparse
import hashlib
import os
import sys
import time
from typing import Any, Dict, List, Optional

import os as _os, sys as _sys  # noqa: E402  test bootstrap (sys.path + google stub)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap  # noqa: E402,F401

from game_driver import load_db, build_deck, DEFAULT_MAX_STEPS
from opcg_sim.src.core import action_api

_MODELS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                       "opcg_sim", "data", "learned")


# --- 判定ロジック（純関数・テストが固定する） -------------------------------

def evaluate_gate(ci: Optional[Dict[str, float]], latency: Dict[str, float],
                  failed_games: int, min_elo_lo: float, max_latency_ms: float) -> Dict[str, Any]:
    """収集済みの指標から PASS/FAIL を決める（対局を回さない純関数＝高速にテストできる）。

    合格条件（すべて満たす）:
      - 有効ペアがあり、Gen2 が hard に**有意に強い**（ペア単位 CI の下限 `elo_lo` > `min_elo_lo`）。
      - レイテンシ median が予算 `max_latency_ms` 以内。
      - 失敗局ゼロ（クラッシュ/無限ループ無し）。
    """
    reasons: List[str] = []
    if ci is None or ci.get("pairs", 0) <= 0:
        reasons.append("有効ペアが0（対局が全滅・データ不足）")
        return {"passed": False, "reasons": reasons}
    if ci["elo_lo"] <= min_elo_lo:
        reasons.append(f"強度不足: elo_lo={ci['elo_lo']:+.0f} <= 閾値 {min_elo_lo:+.0f}"
                       f"（Gen2 が hard に有意に勝てていない＝退行の疑い）")
    if latency["median_ms"] > max_latency_ms:
        reasons.append(f"レイテンシ超過: median={latency['median_ms']:.0f}ms > 予算 {max_latency_ms:.0f}ms")
    if failed_games > 0:
        reasons.append(f"失敗局 {failed_games}（クラッシュ/無限ループ）")
    return {"passed": not reasons, "reasons": reasons}


def model_hash() -> Dict[str, str]:
    """測ったネットの同定用に同梱 gen*_*.npz の SHA1（先頭12桁）を返す。"""
    out: Dict[str, str] = {}
    for name in ("gen2_value.npz", "gen2_policy.npz", "gen3_value.npz", "gen3_policy.npz",
                 "gen4_value.npz", "gen4_policy.npz", "gen5_value.npz", "gen5_policy.npz"):
        p = os.path.join(_MODELS, name)
        if os.path.exists(p):
            with open(p, "rb") as f:
                out[name] = hashlib.sha1(f.read()).hexdigest()[:12]
        else:
            out[name] = "<missing>"
    return out


# --- レイテンシ計測 ----------------------------------------------------------

def measure_latency(db, sims: int, n_decisions: int = 24, seed: int = 0) -> Dict[str, float]:
    """learned の 1 手思考時間（ms）を計測。ウォームアップ（net ロード）は初手を捨てて除外する。"""
    import random
    from opcg_sim.src.core import cpu_learned
    from opcg_sim.src.core.gamestate import GameManager, Player
    random.seed(seed)
    l1, c1 = build_deck(db, "p1")
    l2, c2 = build_deck(db, "p2")
    m = GameManager(Player("p1", c1, l1), Player("p2", c2, l2))
    m.start_game()
    pid_key = action_api.CONST.get("PENDING_REQUEST_PROPERTIES", {}).get("PLAYER_ID", "player_id")
    cpu_learned._lazy_init()   # ウォームアップ（初回ネットロードを計測から外す）
    times: List[float] = []
    steps = 0
    while m.winner is None and len(times) < n_decisions and steps < DEFAULT_MAX_STEPS:
        pend = m.get_pending_request()
        if not pend:
            break
        actor = m.p1 if m.p1.name == pend[pid_key] else m.p2
        t0 = time.perf_counter()
        mv = cpu_learned.decide_learned(m, actor, sims=sims)
        times.append((time.perf_counter() - t0) * 1000.0)
        if mv is None:
            break
        m.action_events = []
        if mv["kind"] == "battle":
            action_api.apply_battle_action(m, actor, mv["action_type"], mv.get("card_uuid"))
        else:
            action_api.apply_game_action(m, actor, mv["action_type"], mv.get("payload", {}))
        steps += 1
    times.sort()
    n = len(times)
    return {"median_ms": times[n // 2] if n else 0.0, "max_ms": times[-1] if n else 0.0, "n": float(n)}


# --- ゲート本体 --------------------------------------------------------------

def run_gate(pairs: int, sims: int, workers: Optional[int], min_elo_lo: float,
             max_latency_ms: float, seed0: int = 0) -> Dict[str, Any]:
    """learned(Gen2) vs hard の対照ペア並列＋レイテンシ計測＋判定をまとめて返す。"""
    from arena_parallel import paired_play, _pair_level_ci
    res = paired_play(pairs, seed0=seed0, workers=workers,
                      challenger_difficulty="learned", baseline_difficulty="hard",
                      challenger_sims=sims)
    ci = _pair_level_ci(res["pair_scores"]) if res["pair_scores"] else None
    if ci is not None:
        ci["pairs"] = res["pairs"]
    latency = measure_latency(db=load_db(), sims=sims)
    verdict = evaluate_gate(ci, latency, res.get("failed_games", 0), min_elo_lo, max_latency_ms)
    return {"ci": ci, "latency": latency, "arena": res, "verdict": verdict,
            "models": model_hash(), "sims": sims, "pairs": pairs}


def main(argv=None):
    ap = argparse.ArgumentParser(description="CPU 性能ゲート（Gen2 非退行・PASS/FAIL）")
    ap.add_argument("--quick", action="store_true", help="軽い確認: pairs=6・sims=40")
    ap.add_argument("--full", action="store_true", help="本走: pairs=40・sims=160")
    ap.add_argument("--pairs", type=int, default=None)
    ap.add_argument("--sims", type=int, default=None)
    ap.add_argument("--workers", type=int, default=0, help="0=自動（コア数-1）")
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--min-elo-lo", type=float, default=0.0,
                    help="合格に要する Elo CI 下限（既定0＝hard に有意に勝つ。Gen2 は本来 +数百）")
    ap.add_argument("--max-latency-ms", type=float, default=1200.0, help="1手思考時間 median の予算")
    args = ap.parse_args(argv)

    if args.full:
        pairs, sims = args.pairs or 40, args.sims or 160
    elif args.quick:
        pairs, sims = args.pairs or 6, args.sims or 40
    else:
        pairs, sims = args.pairs or 20, args.sims or 160

    t0 = time.time()
    rep = run_gate(pairs, sims, args.workers or None, args.min_elo_lo, args.max_latency_ms, args.seed0)
    dt = time.time() - t0
    ci, lat, res, v = rep["ci"], rep["latency"], rep["arena"], rep["verdict"]

    print(f"\n=== CPU 性能ゲート: learned(Gen2) vs hard（{res['pairs']}ペア={res['games']}局・sims={sims}） ===")
    print(f"models: " + " ".join(f"{k}={h}" for k, h in rep["models"].items()))
    if ci:
        print(f"勝率(Gen2視点) = {ci['win_rate']:.3f}  Elo = {ci['elo']:+.0f}  "
              f"[CI95 {ci['elo_lo']:+.0f}, {ci['elo_hi']:+.0f}]")
    print(f"レイテンシ(learned 1手): median={lat['median_ms']:.0f}ms max={lat['max_ms']:.0f}ms (n={int(lat['n'])})")
    if res.get("failed_games"):
        print(f"⚠ 失敗局 {res['failed_games']}: " + " | ".join(res.get("errors", [])[:2]))
    print(f"壁時計: {dt:.1f}s / {res['games']}局 / {res['workers']}並列")
    print("\n判定: " + ("✅ PASS" if v["passed"] else "❌ FAIL"))
    for r in v["reasons"]:
        print("  - " + r)
    return 0 if v["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
