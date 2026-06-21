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
from opcg_sim.src.core import action_api, cpu_ai, cpu_features
from opcg_sim.src.core.invariants import check_invariants

from cpu_selfplay import _load_db, build_deck, DEFAULT_MAX_STEPS


def collect_game(seed: int, db, difficulty: str, max_steps: int) -> List[Dict[str, Any]]:
    """1 局を自己対戦し、ターン境界の (特徴, プレイヤー) を集めて最終勝敗でラベル付けして返す。"""
    random.seed(seed)
    l1, c1 = build_deck(db, "p1")
    l2, c2 = build_deck(db, "p2")
    if not l1 or not l2:
        return []
    m = GameManager(Player("p1", c1, l1), Player("p2", c2, l2))
    m.start_game()
    mem = {"p1": {}, "p2": {}}
    pid_key = action_api.CONST.get("PENDING_REQUEST_PROPERTIES", {}).get("PLAYER_ID", "player_id")

    samples: List[Dict[str, Any]] = []   # {"f":[...], "p":"p1"}
    prev_turn = m.turn_count
    step = 0
    while m.winner is None and step < max_steps:
        pending = m.get_pending_request()
        if not pending:
            break
        actor = m.p1 if m.p1.name == pending[pid_key] else m.p2
        move = cpu_ai.decide_guarded(m, actor, difficulty, random, mem.setdefault(actor.name, {}))
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
        if m.turn_count != prev_turn:
            prev_turn = m.turn_count
            for name in ("p1", "p2"):
                samples.append({"f": cpu_features.extract_features(m, name), "p": name})
        step += 1

    if m.winner is None:
        return []   # 未決着は捨てる（ラベル付け不能）
    return [{"f": s["f"], "y": 1 if m.winner == s["p"] else 0} for s in samples]


def main(argv=None):
    ap = argparse.ArgumentParser(description="価値関数の自己対戦データ収集")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--difficulty", choices=["easy", "normal", "hard"], default="normal")
    ap.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    ap.add_argument("--out", default="/tmp/value_data.jsonl")
    args = ap.parse_args(argv)

    db = _load_db()
    n_rows = n_games = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for g in range(args.games):
            rows = collect_game(args.seed + g, db, args.difficulty, args.max_steps)
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
