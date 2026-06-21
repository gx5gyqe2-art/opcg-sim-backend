"""学習価値葉の **自己対戦 Elo 検証**（§2.5.7 残5・dev専用）。

expert(MCTS) 同士で **challenger（葉ブレンド α>0）vs baseline（α=0）** を席交互で対戦させ、challenger の
勝率＝「価値葉を入れて現状以上か」を測る。**>50%（+Elo）でなければ本番 ON にしない**ためのゲート。
α は `cpu_value_model.set_alpha_override` で**手番ごとに切替**（env は全体共通のため使えない）。

実行例:
    OPCG_LOG_SILENT=1 python tests/value_arena.py --games 24 --alpha 0.3 --iters 80
"""
import argparse
import math
import random
import sys

import conftest  # noqa: F401

from opcg_sim.src.core.gamestate import GameManager, Player
from opcg_sim.src.core import action_api, cpu_mcts, cpu_value_model
from opcg_sim.src.core.invariants import check_invariants
from cpu_selfplay import _load_db, DEFAULT_MAX_STEPS
import deckgen

_VERIFIED = list(deckgen.VERIFIED_LEADERS.values())
_PIDK = action_api.CONST.get("PENDING_REQUEST_PROPERTIES", {}).get("PLAYER_ID", "player_id")


def play_game(seed, db, alpha, iters, horizon, challenger_seat):
    """1 局を expert 同士で対戦。`challenger_seat`（"p1"/"p2"）だけ α を効かせる。勝者名を返す。"""
    random.seed(seed)
    l1id = _VERIFIED[seed % len(_VERIFIED)]
    l2id = _VERIFIED[(seed + 1) % len(_VERIFIED)]
    L1, c1 = deckgen.build_realistic_deck(db, "p1", l1id, random.Random(seed * 2))
    L2, c2 = deckgen.build_realistic_deck(db, "p2", l2id, random.Random(seed * 2 + 1))
    m = GameManager(Player("p1", c1, L1), Player("p2", c2, L2))
    m.start_game()
    caches = {"p1": {}, "p2": {}}
    step = 0
    while m.winner is None and step < DEFAULT_MAX_STEPS:
        pend = m.get_pending_request()
        if not pend:
            break
        actor = m.p1 if m.p1.name == pend[_PIDK] else m.p2
        cpu_value_model.set_alpha_override(alpha if actor.name == challenger_seat else 0.0)
        mv = cpu_mcts.decide_mcts_macro(m, actor, "hard", random,
                                        cache=caches[actor.name], iterations=iters, horizon=horizon)
        cpu_value_model.set_alpha_override(None)
        if mv is None:
            break
        m.action_events = []
        try:
            if mv["kind"] == "battle":
                action_api.apply_battle_action(m, actor, mv["action_type"], mv.get("card_uuid"))
            else:
                action_api.apply_game_action(m, actor, mv["action_type"], mv.get("payload", {}))
        except Exception:
            return None
        if check_invariants(m):
            return None
        step += 1
    return m.winner


def main(argv=None):
    ap = argparse.ArgumentParser(description="学習価値葉の自己対戦 Elo 検証")
    ap.add_argument("--games", type=int, default=24)
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--iters", type=int, default=80)
    ap.add_argument("--horizon", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1000)
    args = ap.parse_args(argv)
    if not cpu_value_model.is_available():
        print("value_model.json が無い/不一致＝検証不能"); return 1

    db = _load_db()
    wins = draws = losses = 0
    for g in range(args.games):
        seat = "p1" if g % 2 == 0 else "p2"   # 席交互＝デッキ非対称を相殺
        w = play_game(args.seed + g, db, args.alpha, args.iters, args.horizon, seat)
        if w is None:
            continue
        if w == seat:
            wins += 1
        else:
            losses += 1
        if (g + 1) % 4 == 0:
            print(f"  {g+1}/{args.games}: challenger {wins}W-{losses}L")
    n = wins + losses
    wr = wins / n if n else 0.0
    elo = 400 * math.log10(wr / (1 - wr)) if 0 < wr < 1 else float("nan")
    print(f"\nα={args.alpha} iters={args.iters}: challenger {wins}W-{losses}L / {n}  "
          f"勝率={wr:.3f}  ≈{elo:+.0f} Elo")
    print("→ 勝率が明確に >0.5 なら本番 ON 候補・~0.5/負けなら α=0 据え置き（無害）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
