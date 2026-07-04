"""CPU 検証基盤（フェーズ0）の**絶対強度メトリクス**: 凍結ベースライン Elo ＋ regret ログ
（docs/SPEC.md §2.5.3「2026-06 外部レビュー収束／検証基盤」）。

自己対戦＋インバリアントは自己参照的で、評価関数の改善が「強くなった」のか「相対順位が入れ替わった
だけ」なのかを区別できない。本ハーネスは2つの安価な絶対指標を与える:

  1. 凍結ベースライン Elo（`arena`）: **固定の参照相手**（既定 easy＝正直な 1-ply 貪欲・チューニングで
     変化しない安定相手）に対する挑戦者（normal/hard）の勝率を測り、Elo 差へ変換する。版 N を改善した
     とき、固定相手への勝率が上がる＝絶対的に強くなった、という単調指標。先手有利を相殺するため**席を
     交互に入れ替える**（偶数 seed は p1=挑戦者・奇数 seed は p2=挑戦者）。

  2. regret ログ（`regret`）: 各意思決定で `cpu_ai.decide_with_regret` が返す greedy regret
     ＝ deep_value(深掘り最善手) − deep_value(1-ply 貪欲手) を 1 ゲーム分集計する。大きい regret は
     「浅い読みなら崖に落ちる」局面＝評価/探索が効いている所、恒常的に 0 なら深掘りが効いていない兆候。

いずれも `cpu_selfplay` の決定論ランナーと同じコアパス（action_api）で進行し、本番挙動と乖離しない。
pytest スイートには**機械の健全性のみ**を高速・有界に固定する（`tests/test_cpu_arena.py`）。実ゲームは
低速（normal ≈ 1 手/秒）なので、版間 Elo の本走はこのスクリプトを手動/定期実行する想定:

    OPCG_LOG_SILENT=1 python tests/cpu_arena.py arena --challenger normal --baseline easy --games 20
    OPCG_LOG_SILENT=1 python tests/cpu_arena.py regret --difficulty normal --seed 0
"""
import argparse
import math
import random
import sys
from typing import Any, Dict, List, Optional

import os as _os, sys as _sys  # noqa: E402  test bootstrap (sys.path + google stub)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap  # noqa: E402,F401

from opcg_sim.src.core import action_api, cpu_ai  # regret/realize が CONST 参照＋decide_* を使う

# 対局ループ・席生成・共有部品は game_driver（設計⑥）へ集約。play_game/regret/realize はこれを差し込む。
# `_load_db`/`build_deck`/`DEFAULT_MAX_STEPS`/`InvariantError` は arena_parallel 等の後方互換のため再エクスポート。
from game_driver import (  # noqa: F401
    load_db as _load_db,
    build_deck,
    DEFAULT_MAX_STEPS,
    InvariantError,
    make_seat,
    run_game,
)


# --- Elo 変換 -----------------------------------------------------------------

def elo_delta(win_rate: float) -> float:
    """勝率 → Elo 差（挑戦者 − ベースライン）。0.5→0・0.76→+200・0.24→-200。

    端（0/1）は ±inf を避けて有限の小/大サンプル境界へクランプする。
    """
    p = min(max(win_rate, 1e-4), 1.0 - 1e-4)
    return -400.0 * math.log10(1.0 / p - 1.0)


def win_rate(wins: float, games: int) -> float:
    """引き分けは 0.5 勝として `wins` に半端で含める想定。games==0 は 0.5（無情報）。"""
    return 0.5 if games <= 0 else wins / games


# --- 信頼区間（Phase 0・分散低減アリーナの合否判定器） -------------------------

def wilson_interval(wins: float, games: int, z: float = 1.96) -> Dict[str, float]:
    """勝率の Wilson スコア信頼区間（小標本・連勝/連敗でも縮退しない）。

    正規近似（mean±z·SE）は連勝（分散 0）で半幅 0＝「完全確信」と誤報し、合否ゲートを lucky sweep で
    自明に通してしまう。Wilson は p̂=0/1 でも有限の妥当な区間を返すのでゲートに適する。引き分けを 0.5 勝で
    含む `wins`（端数可）も総試行 `games` に対する比率として扱う（厳密な二項ではないが保守的近似）。
    games==0 は (0,1)・mean 0.5（無情報）。
    """
    if games <= 0:
        return {"mean": 0.5, "games": 0.0, "lo": 0.0, "hi": 1.0, "half_width": float("inf")}
    n = float(games)
    p = wins / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    spread = (z / denom) * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    lo, hi = max(0.0, center - spread), min(1.0, center + spread)
    return {"mean": p, "games": n, "lo": lo, "hi": hi, "half_width": (hi - lo) / 2.0}


def elo_ci(wins: float, games: int, z: float = 1.96) -> Dict[str, float]:
    """勝率 Wilson CI → Elo 点推定と Elo 信頼区間（端点を Elo へ写像）。

    elo_delta は非線形なので Elo 区間は点推定まわりに**非対称**。点推定 `elo`＝elo_delta(勝率)、区間は
    [elo_lo, elo_hi]＝端点写像。`elo_half_width`＝(elo_hi−elo_lo)/2 は**区間幅の半分**（点推定からの片側
    対称幅ではない＝非対称区間のため elo±half は [elo_lo,elo_hi] を再現しない）。Phase 0 合格ゲート
    （Elo 区間幅 半分 < 15）の判定に使う。games<1 では半幅 inf。
    """
    w = wilson_interval(wins, games, z)
    if games <= 0:
        return {"elo": 0.0, "elo_lo": elo_delta(0.0), "elo_hi": elo_delta(1.0),
                "elo_half_width": float("inf"), "win_rate": 0.5, "games": 0.0}
    elo = elo_delta(w["mean"])
    lo_e, hi_e = elo_delta(w["lo"]), elo_delta(w["hi"])
    return {"elo": elo, "elo_lo": lo_e, "elo_hi": hi_e,
            "elo_half_width": (hi_e - lo_e) / 2.0, "win_rate": w["mean"], "games": w["games"]}


# --- 非対称（挑戦者 vs ベースライン）対局ランナー -----------------------------

def _arena_seat(difficulty, policy, rng, pimc, budget, search, coeffs, sims):
    """arena の 1 席を作る。`difficulty=="learned"` は Gen2 学習型（本番既定 CPU）＝L1 の席別ノブは持たず
    `sims`（MCTS 探索数）のみ。CRN（`rng`）は L1 席専用（learned は global random 由来＝PR-D2 で seed 再現）。
    それ以外（hard 等）は L1（α-β＋ビーム＋PIMC）席で、情報方針/CRN/PIMC/予算/深さ/L1係数を席別に掛ける。
    """
    if difficulty == "learned":
        return make_seat(kind="learned", sims=sims)
    return make_seat(difficulty, kind="arena", info_policy=policy, policy_rng=rng,
                     pimc_worlds=pimc, budget=budget, search=search, coeffs=coeffs)


def play_game(seed: int, db, p1_difficulty: str, p2_difficulty: str,
              max_steps: int = DEFAULT_MAX_STEPS,
              p1_policy: str = cpu_ai.DEFAULT_INFO_POLICY,
              p2_policy: str = cpu_ai.DEFAULT_INFO_POLICY,
              separate_policy_rng: bool = False,
              p1_pimc: int = 1, p2_pimc: int = 1,
              p1_budget=None, p2_budget=None,
              p1_search=None, p2_search=None,
              p1_coeffs=None, p2_coeffs=None,
              p1_sims: int = 160, p2_sims: int = 160) -> Dict[str, Any]:
    """p1/p2 に別難易度を割り当てて 1 ゲームを決定論的に完走させ、勝者を返す。

    対局ループは `game_driver.run_game`（全ハーネス共通）で回し、席（seat）だけ非対称にする。
    `difficulty` は L1 系（hard）と **learned（Gen2 学習型・本番既定 CPU）** を混在できる（A1: 強度 A/B の
    learned 対応）。learned 席は `p1_sims`/`p2_sims`（MCTS 探索数・既定=本番 160）を使い、L1 の席別ノブ
    （policy/pimc/budget/search/coeffs・CRN rng）は無視する。`p1_policy`/`p2_policy` は L1 の情報方針（fair/cheat）。

    `separate_policy_rng=True`（Phase 0）で**方策のタイブレーク乱数を game 乱数（global random）から分離**
    する（各 L1 席に seed 派生の独立 `random.Random`）。これは方策タイブレークの決定性を game 乱数から
    切り離すだけ＝**完全な配りレベル CRN ではない**（learned 席には未適用＝global random 由来）。
    どちらも同一 seed なら決定論再現する（learned は PR-D2、L1 は既存の決定論）。
    """
    p1_rng = random.Random(seed * 2 + 1) if separate_policy_rng else None
    p2_rng = random.Random(seed * 2 + 2) if separate_policy_rng else None
    seats = {
        "p1": _arena_seat(p1_difficulty, p1_policy, p1_rng, p1_pimc, p1_budget, p1_search, p1_coeffs, p1_sims),
        "p2": _arena_seat(p2_difficulty, p2_policy, p2_rng, p2_pimc, p2_budget, p2_search, p2_coeffs, p2_sims),
    }
    # arena は各手番で get_legal_actions を事前呼びしない（seat 内で解決＝乱数消費順の保存）。
    result = run_game(seed, db, seats=seats, max_steps=max_steps,
                      legal_moves="skip", invariants="raise")
    return {"seed": seed, "winner": result.winner, "steps": result.steps, "turns": result.turns}


def arena(db, challenger: str, baseline: str, games: int, seed0: int = 0,
          max_steps: int = DEFAULT_MAX_STEPS,
          challenger_policy: str = cpu_ai.DEFAULT_INFO_POLICY,
          baseline_policy: str = cpu_ai.DEFAULT_INFO_POLICY,
          challenger_sims: int = 160, baseline_sims: int = 160) -> Dict[str, Any]:
    """挑戦者 vs 固定ベースラインを `games` 局。**席を交互に入替**して先手有利を相殺し、勝率と Elo を返す。

    偶数 i: p1=挑戦者 / 奇数 i: p2=挑戦者。引き分け（あれば）は 0.5 勝で計上。
    `challenger`/`baseline` は `learned`（Gen2）も可（A1）。`challenger_sims`/`baseline_sims` は learned の MCTS 探索数。
    `challenger_policy`/`baseline_policy`（Phase -1・fair/cheat）は L1 の情報方針 A/B（席入替に追従）。
    """
    wins = 0.0
    decided = 0
    detail: List[Dict[str, Any]] = []
    for i in range(games):
        seed = seed0 + i
        chal_is_p1 = (i % 2 == 0)
        p1d, p2d = (challenger, baseline) if chal_is_p1 else (baseline, challenger)
        p1p, p2p = ((challenger_policy, baseline_policy) if chal_is_p1
                    else (baseline_policy, challenger_policy))
        p1s, p2s = (challenger_sims, baseline_sims) if chal_is_p1 else (baseline_sims, challenger_sims)
        res = play_game(seed, db, p1d, p2d, max_steps=max_steps, p1_policy=p1p, p2_policy=p2p,
                        p1_sims=p1s, p2_sims=p2s)
        chal_seat = "p1" if chal_is_p1 else "p2"
        won = (res["winner"] == chal_seat)
        wins += 1.0 if won else 0.0
        decided += 1
        detail.append({"seed": seed, "challenger_seat": chal_seat, "winner": res["winner"],
                       "challenger_won": won, "turns": res["turns"]})
    wr = win_rate(wins, decided)
    return {"challenger": challenger, "baseline": baseline,
            "challenger_policy": challenger_policy, "baseline_policy": baseline_policy,
            "games": decided, "challenger_wins": wins, "win_rate": wr,
            "elo_delta": elo_delta(wr), "detail": detail}


def arena_paired(db, challenger: str, baseline: str, pairs: int, seed0: int = 0,
                 max_steps: int = DEFAULT_MAX_STEPS,
                 challenger_policy: str = cpu_ai.DEFAULT_INFO_POLICY,
                 baseline_policy: str = cpu_ai.DEFAULT_INFO_POLICY,
                 challenger_pimc: int = 1, baseline_pimc: int = 1,
                 challenger_budget=None, baseline_budget=None,
                 challenger_sims: int = 160, baseline_sims: int = 160) -> Dict[str, Any]:
    """分散低減アリーナ（Phase 0・antithetic 席ペアリング）。各 seed を**両席で 1 回ずつ**戦わせ、
    挑戦者の 2 局の勝敗を集計する＝**先手有利（席順）をペア内で相殺**する。

    範囲（正直な明記）: 相殺できるのは**席順のみ**。デッキは決定論ミラー（両者同一構成）だが、
    `start_game` は p1→p2 の順に独立シャッフルするため、挑戦者が p1 の局と p2 の局では**引く山が異なる**
    ＝配り運（ドロー分散）は相殺しない。さらに mulligan は対局ループ中に決まり、α-β 探索はクローン上の
    効果解決で global `random` を方策依存に消費するため、**同一 game-seed でも方策が違えば配りは厳密には
    一致しない**（`separate_policy_rng=True` が分離するのは方策タイブレーク乱数のみ）。完全な配りレベル
    CRN は GameManager への乱数注入（クローンが独自 rng を持ち探索が本譜の乱数を汚さない）が必要＝
    フォローアップ（Phase 0b）。それでも席相殺だけで独立 N 局よりは分散が落ちる。

    CI は総試行（2×pairs 局）の挑戦者勝率に対する **Wilson 区間**（連勝でも縮退しない・小標本に頑健）。
    合格ゲート＝Elo 区間幅 半分 < 15。1 ペアの勝点 {0,0.5,1} は (A=挑戦者 p1 勝 + B=挑戦者 p2 勝)/2。
    """
    wins = 0.0
    detail: List[Dict[str, Any]] = []
    for k in range(pairs):
        seed = seed0 + k
        # 席A: 挑戦者=p1。席B: 同一 game-seed で挑戦者=p2（席だけ反転）。
        a = play_game(seed, db, challenger, baseline, max_steps=max_steps,
                      p1_policy=challenger_policy, p2_policy=baseline_policy,
                      separate_policy_rng=True, p1_pimc=challenger_pimc, p2_pimc=baseline_pimc,
                      p1_budget=challenger_budget, p2_budget=baseline_budget,
                      p1_sims=challenger_sims, p2_sims=baseline_sims)
        b = play_game(seed, db, baseline, challenger, max_steps=max_steps,
                      p1_policy=baseline_policy, p2_policy=challenger_policy,
                      separate_policy_rng=True, p1_pimc=baseline_pimc, p2_pimc=challenger_pimc,
                      p1_budget=baseline_budget, p2_budget=challenger_budget,
                      p1_sims=baseline_sims, p2_sims=challenger_sims)
        chal_a = 1.0 if a["winner"] == "p1" else 0.0
        chal_b = 1.0 if b["winner"] == "p2" else 0.0
        wins += chal_a + chal_b
        detail.append({"seed": seed, "chal_as_p1_won": chal_a, "chal_as_p2_won": chal_b,
                       "pair_score": (chal_a + chal_b) / 2.0})
    games = 2 * pairs
    ci = elo_ci(wins, games)
    return {"challenger": challenger, "baseline": baseline,
            "challenger_policy": challenger_policy, "baseline_policy": baseline_policy,
            "pairs": pairs, "games": games, "win_rate": ci["win_rate"], "elo_delta": ci["elo"],
            "elo_lo": ci["elo_lo"], "elo_hi": ci["elo_hi"],
            "elo_half_width": ci["elo_half_width"], "detail": detail}


# --- regret ログ --------------------------------------------------------------

def regret_trace(db, seed: int, difficulty: str = "hard",
                 max_steps: int = DEFAULT_MAX_STEPS) -> Dict[str, Any]:
    """1 ゲームを自己対戦し、各 MAIN 意思決定の greedy regret を集計する（mean/max/count/p95）。

    regret は `cpu_ai.decide_with_regret`（deep_value(深掘り最善) − deep_value(1-ply 貪欲)）。
    実際に手を進めるのは返り値の move（＝通常の対局と同じ進行）なので、トレースは本番方策の軌跡上で取る。
    対局ループは `run_game`（invariants="skip"＝インバリアント検査なし・スタック/None で break・
    apply 例外は素通し）で回し、seat が MAIN_ACTION のみ `decide_with_regret` へ差し替えて regret を集める。
    """
    KEY_ACTION = action_api.CONST.get('PENDING_REQUEST_PROPERTIES', {}).get('ACTION', 'action')
    mems: Dict[str, Any] = {"p1": {}, "p2": {}}
    regrets: List[float] = []

    def seat(ctx):
        if ctx.pending.get(KEY_ACTION) == "MAIN_ACTION":
            move, regret = cpu_ai.decide_with_regret(ctx.manager, ctx.actor, difficulty, random)
            regrets.append(regret)
            return move
        return cpu_ai.decide_guarded(ctx.manager, ctx.actor, difficulty, random, mems[ctx.actor.name])

    run_game(seed, db, seats={"p1": seat, "p2": seat}, max_steps=max_steps,
             legal_moves="skip", invariants="skip")

    n = len(regrets)
    s = sorted(regrets)
    return {
        "seed": seed, "difficulty": difficulty, "decisions": n,
        "mean": (sum(regrets) / n) if n else 0.0,
        "max": max(regrets) if n else 0.0,
        "p95": (s[min(n - 1, int(0.95 * n))] if n else 0.0),
        "nonzero": sum(1 for r in regrets if r > 0.0),
    }


# --- value-realization gap（ターン内 楽観崩落・§2.5.3） ----------------------

def realize_trace(db, seed: int, difficulty: str = "hard",
                  max_steps: int = DEFAULT_MAX_STEPS) -> Dict[str, Any]:
    """1 ゲームを自己対戦し、**value-realization gap**（ターン内の楽観崩落）を集計する。

    各 MAIN 意思決定で `decide_with_regret(out=...)` から採用手の深掘りスコア（`chosen_deep`）を取り、
    (player, turn) ごとに時系列で並べる。ターン頭でドン/盤面に過剰コミットし、手番が進んで初めて
    「読み切れなかった代償」が露見すると、採用手の深掘り値は**ターン内で単調に崩落**する（実ケース:
    付与時 +4798 → 攻撃時 -91）。1 ターンの gap = max(そのターンの chosen_deep) − 最終決定の chosen_deep。
    大きい gap が頻発する＝探索が予算地平線の外を楽観視して資源を溶かしている兆候（B が縮める対象）。
    対局ループは `run_game`（invariants="skip"）で回し、seat が MAIN_ACTION の採用手深掘り値を収穫する。
    """
    KEY_ACTION = action_api.CONST.get('PENDING_REQUEST_PROPERTIES', {}).get('ACTION', 'action')
    mems: Dict[str, Any] = {"p1": {}, "p2": {}}
    series: Dict[tuple, List[float]] = {}   # (player, turn) -> [chosen_deep, ...]

    def seat(ctx):
        if ctx.pending.get(KEY_ACTION) == "MAIN_ACTION":
            info: Dict[str, Any] = {}
            move, _r = cpu_ai.decide_with_regret(ctx.manager, ctx.actor, difficulty, random, out=info)
            if "chosen_deep" in info:
                series.setdefault((ctx.actor.name, ctx.manager.turn_count), []).append(info["chosen_deep"])
            return move
        return cpu_ai.decide_guarded(ctx.manager, ctx.actor, difficulty, random, mems[ctx.actor.name])

    run_game(seed, db, seats={"p1": seat, "p2": seat}, max_steps=max_steps,
             legal_moves="skip", invariants="skip")

    gaps = [max(v) - v[-1] for v in series.values() if len(v) >= 2]
    n = len(gaps)
    s = sorted(gaps)
    return {
        "seed": seed, "difficulty": difficulty, "turns_scored": n,
        "mean_gap": (sum(gaps) / n) if n else 0.0,
        "max_gap": max(gaps) if n else 0.0,
        "p95_gap": (s[min(n - 1, int(0.95 * n))] if n else 0.0),
        "big_gaps": sum(1 for g in gaps if g >= 2000.0),   # ライフ ~1/3 相当以上の崩落ターン数
    }


# --- CLI ----------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description="CPU 検証基盤: 凍結ベースライン Elo ＋ regret ログ")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pa = sub.add_parser("arena", help="挑戦者 vs 固定ベースラインの勝率→Elo")
    pa.add_argument("--challenger", choices=["hard", "learned"], default="hard")
    pa.add_argument("--baseline", choices=["hard", "learned"], default="hard")
    # Phase -1: 情報方針の A/B（既定＝挑戦者 fair vs ベースライン cheat＝フェア化の損失量を測る）。
    pa.add_argument("--challenger-policy", choices=["fair", "cheat"], default="fair")
    pa.add_argument("--baseline-policy", choices=["fair", "cheat"], default="cheat")
    pa.add_argument("--challenger-sims", type=int, default=160, help="learned 挑戦者の MCTS 探索数")
    pa.add_argument("--baseline-sims", type=int, default=160, help="learned ベースラインの MCTS 探索数")
    pa.add_argument("--games", type=int, default=10)
    pa.add_argument("--seed", type=int, default=0)
    pa.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)

    pp = sub.add_parser("arena-paired", help="分散低減アリーナ（antithetic+CRN・Elo CI つき。learned=Gen2 可）")
    pp.add_argument("--challenger", choices=["hard", "learned"], default="hard")
    pp.add_argument("--baseline", choices=["hard", "learned"], default="hard")
    pp.add_argument("--challenger-policy", choices=["fair", "cheat"], default="fair")
    pp.add_argument("--baseline-policy", choices=["fair", "cheat"], default="cheat")
    pp.add_argument("--challenger-pimc", type=int, default=1, help="挑戦者の PIMC 世界数（>=2 で決定化・L1のみ）")
    pp.add_argument("--baseline-pimc", type=int, default=1, help="ベースラインの PIMC 世界数（L1のみ）")
    pp.add_argument("--challenger-budget", type=int, default=None, help="挑戦者の深掘り予算（L1のみ）")
    pp.add_argument("--baseline-budget", type=int, default=None, help="ベースラインの深掘り予算（L1のみ）")
    pp.add_argument("--challenger-sims", type=int, default=160, help="learned 挑戦者の MCTS 探索数")
    pp.add_argument("--baseline-sims", type=int, default=160, help="learned ベースラインの MCTS 探索数")
    pp.add_argument("--pairs", type=int, default=50)
    pp.add_argument("--seed", type=int, default=0)
    pp.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)

    pr = sub.add_parser("regret", help="自己対戦 1 局の greedy regret 集計")
    pr.add_argument("--difficulty", choices=["hard"], default="hard")
    pr.add_argument("--seed", type=int, default=0)
    pr.add_argument("--games", type=int, default=1)
    pr.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)

    pz = sub.add_parser("realize", help="自己対戦の value-realization gap（ターン内 楽観崩落）集計")
    pz.add_argument("--difficulty", choices=["hard"], default="hard")
    pz.add_argument("--seed", type=int, default=0)
    pz.add_argument("--games", type=int, default=1)
    pz.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)

    args = ap.parse_args(argv)
    db = _load_db()

    if args.cmd == "arena":
        rep = arena(db, args.challenger, args.baseline, args.games, args.seed, args.max_steps,
                    challenger_policy=args.challenger_policy, baseline_policy=args.baseline_policy,
                    challenger_sims=args.challenger_sims, baseline_sims=args.baseline_sims)
        for d in rep["detail"]:
            print(f"  seed={d['seed']} challenger={d['challenger_seat']} winner={d['winner']} "
                  f"{'WIN' if d['challenger_won'] else 'loss'} turns={d['turns']}")
        print(f"\narena: {rep['challenger']}[{rep['challenger_policy']}] vs "
              f"{rep['baseline']}[{rep['baseline_policy']}]  "
              f"{rep['challenger_wins']:.1f}/{rep['games']}  win_rate={rep['win_rate']:.3f}  "
              f"Elo={rep['elo_delta']:+.0f}")
        return 0

    if args.cmd == "arena-paired":
        rep = arena_paired(db, args.challenger, args.baseline, args.pairs, args.seed, args.max_steps,
                           challenger_policy=args.challenger_policy, baseline_policy=args.baseline_policy,
                           challenger_pimc=args.challenger_pimc, baseline_pimc=args.baseline_pimc,
                           challenger_budget=args.challenger_budget, baseline_budget=args.baseline_budget,
                           challenger_sims=args.challenger_sims, baseline_sims=args.baseline_sims)
        for d in rep["detail"]:
            print(f"  seed={d['seed']} p1won={d['chal_as_p1_won']:.0f} p2won={d['chal_as_p2_won']:.0f} "
                  f"pair={d['pair_score']:.2f}")
        gate = "PASS" if rep["elo_half_width"] < 15.0 else "WIDE"
        cl = f"{rep['challenger_policy']}{'+pimc%d' % args.challenger_pimc if args.challenger_pimc > 1 else ''}"
        bl = f"{rep['baseline_policy']}{'+pimc%d' % args.baseline_pimc if args.baseline_pimc > 1 else ''}"
        print(f"\narena-paired: {rep['challenger']}[{cl}] vs "
              f"{rep['baseline']}[{bl}]  pairs={rep['pairs']}  "
              f"win_rate={rep['win_rate']:.3f}  Elo={rep['elo_delta']:+.0f} "
              f"[{rep['elo_lo']:+.0f}, {rep['elo_hi']:+.0f}] half={rep['elo_half_width']:.0f} ({gate})")
        return 0

    if args.cmd == "realize":
        for i in range(args.games):
            rep = realize_trace(db, args.seed + i, args.difficulty, args.max_steps)
            print(f"realize seed={rep['seed']} {rep['difficulty']}: turns={rep['turns_scored']} "
                  f"mean_gap={rep['mean_gap']:.1f} p95_gap={rep['p95_gap']:.1f} "
                  f"max_gap={rep['max_gap']:.1f} big_gaps={rep['big_gaps']}")
        return 0

    # regret
    for i in range(args.games):
        rep = regret_trace(db, args.seed + i, args.difficulty, args.max_steps)
        print(f"regret seed={rep['seed']} {rep['difficulty']}: decisions={rep['decisions']} "
              f"mean={rep['mean']:.1f} p95={rep['p95']:.1f} max={rep['max']:.1f} nonzero={rep['nonzero']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
