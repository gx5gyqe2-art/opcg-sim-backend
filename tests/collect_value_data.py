"""学習価値関数（§2.5.7 残5）の **自己対戦データ収集**（オフライン・dev専用）。

両者 AI（`decide_guarded`）で決定論的に対局し、**各ターン境界**で両プレイヤー視点の特徴ベクトル
（`cpu_features.extract_features`）を記録、対局終了後に**そのプレイヤーが勝ったか**(0/1)でラベル付けして
JSONL（1行 `{"f":[...],"y":0/1}`）へ書き出す。出荷物には含めない（ローカル/Cloud Run Jobs で実行）。

実行例:
    OPCG_LOG_SILENT=1 python tests/collect_value_data.py --games 200 --difficulty normal --out /tmp/value_data.jsonl
"""
import argparse
import json
import random
import sys
from typing import Any, Dict, List

import conftest  # noqa: F401

from opcg_sim.src.core.gamestate import GameManager, Player
from opcg_sim.src.core import action_api, cpu_ai, cpu_features, cpu_value_data
from opcg_sim.src.core.invariants import check_invariants

from cpu_selfplay import _load_db, build_deck, DEFAULT_MAX_STEPS
import deckgen

_VERIFIED = list(deckgen.VERIFIED_LEADERS.values())


def _build_decks(seed: int, db, real_decks: bool):
    """合成（同色キャラ50）or 実デッキ（`deckgen`・検証済リーダーを巡回）でデッキを組む。"""
    if not real_decks:
        l1, c1 = build_deck(db, "p1")
        l2, c2 = build_deck(db, "p2")
        return l1, c1, l2, c2
    l1id = _VERIFIED[seed % len(_VERIFIED)]
    l2id = _VERIFIED[(seed + 1) % len(_VERIFIED)]
    L1, c1 = deckgen.build_realistic_deck(db, "p1", l1id, random.Random(seed * 2))
    L2, c2 = deckgen.build_realistic_deck(db, "p2", l2id, random.Random(seed * 2 + 1))
    return L1, c1, L2, c2


def _make_decider(policy: str, iters: int, horizon: int):
    """policy に応じた1手決定関数を返す（局ごとに新規生成＝mem をリセット）。

    hard＝本番方策 α-β（`decide_guarded`）＝学習分布を本番に一致させる。`iters`/`horizon` は後方互換で残すが不使用。
    """
    mem: Dict[str, Dict[str, Any]] = {}
    def decide(m, actor):
        return cpu_ai.decide_guarded(m, actor, policy, random, mem.setdefault(actor.name, {}))
    return decide


def collect_game(seed: int, db, difficulty: str, max_steps: int,
                 real_decks: bool = False, iters: int = 40, horizon: int = 2) -> List[Dict[str, Any]]:
    """1 局を自己対戦し、ターン境界の (特徴, プレイヤー) を集めて最終勝敗でラベル付けして返す。"""
    random.seed(seed)
    l1, c1, l2, c2 = _build_decks(seed, db, real_decks)
    if not l1 or not l2:
        return []
    m = GameManager(Player("p1", c1, l1), Player("p2", c2, l2))
    m.start_game()
    decide = _make_decider(difficulty, iters, horizon)
    pid_key = action_api.CONST.get("PENDING_REQUEST_PROPERTIES", {}).get("PLAYER_ID", "player_id")

    samples: List[Dict[str, Any]] = []   # {"f":[...], "p":"p1"}
    prev_turn = m.turn_count
    step = 0
    while m.winner is None and step < max_steps:
        pending = m.get_pending_request()
        if not pending:
            break
        actor = m.p1 if m.p1.name == pending[pid_key] else m.p2
        move = decide(m, actor)
        if move is None:
            break
        m.action_events = []
        try:
            if move["kind"] == "battle":
                action_api.apply_battle_action(m, actor, move["action_type"], move.get("card_uuid"))
            else:
                action_api.apply_game_action(m, actor, move["action_type"], move.get("payload", {}))
        except Exception:
            return []   # 異常局は捨てる
        if check_invariants(m):
            return []
        # ターン境界で両者視点のスナップショットを採る（葉=ターン境界と同じ観測点）。
        # 実対局のライブ採取（api/app.py）と同一の観測点・ラベル規約を共有する（cpu_value_data）。
        if m.turn_count != prev_turn:
            prev_turn = m.turn_count
            samples.extend(cpu_value_data.turn_boundary_samples(m))
        step += 1

    if m.winner is None:
        return []   # 未決着は捨てる（ラベル付け不能）
    rows = cpu_value_data.label_samples(samples, m.winner)
    for r in rows:
        r["g"] = seed   # 試合ID（=seed）。試合単位 train/val split のリーク防止に使う。
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description="価値関数の自己対戦データ収集")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--difficulty", choices=["hard"], default="hard",
                    help="自己対戦の方策（hard＝α-β＝本番方策）")
    ap.add_argument("--real-decks", action="store_true", help="deckgen の実デッキ（検証済リーダー巡回）で対戦")
    ap.add_argument("--iters", type=int, default=40, help="（後方互換・未使用）")
    ap.add_argument("--horizon", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    ap.add_argument("--out", default="/tmp/value_data.jsonl")
    args = ap.parse_args(argv)

    db = _load_db()
    n_rows = n_games = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for g in range(args.games):
            rows = collect_game(args.seed + g, db, args.difficulty, args.max_steps,
                                real_decks=args.real_decks, iters=args.iters, horizon=args.horizon)
            if not rows:
                continue
            n_games += 1
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                n_rows += 1
            if (g + 1) % 20 == 0:
                print(f"  {g+1}/{args.games} games … rows={n_rows}")
    print(f"done: {n_games} valid games, {n_rows} rows → {args.out} "
          f"(features={cpu_features.N_FEATURES})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
