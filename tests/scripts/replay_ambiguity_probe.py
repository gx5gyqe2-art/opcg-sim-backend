"""実対局リプレイ R0: 記録アクション（card_id/ラベル基準）の一意復元可否を計測する（dev・計画根拠）。

`docs/replay_verification_plan.md` §3/§6: 人間アクションは `_describe_move`（card_id/ラベル基準・uuid 非依存）で
記録される。再生時はこれを合法手へ逆写像する必要があり、**同 card_id の複製カード**で曖昧化しうる。
本プローブは自己対戦の各意思決定点で「採用手の記述子に一致する合法手が何個あるか」を数え、曖昧率を
アクション種別ごとに集計する＝(A) 決定論タイブレーク逆引きで足りるか／(B) 記録の高解像度化が要るか、の判断材料。

実行例:
    OPCG_LOG_SILENT=1 python tests/scripts/replay_ambiguity_probe.py --games 10 --policy ai
    OPCG_LOG_SILENT=1 python tests/scripts/replay_ambiguity_probe.py --games 30 --policy random   # 高速・多数
"""
import argparse
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List

import os as _os, sys as _sys  # noqa: E402  test bootstrap
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap  # noqa: E402,F401

from game_driver import load_db, make_seat, run_game, leader_deck_builder
from opcg_sim.src.core import cpu_ai
import heldout_decks as HD


class _AmbiguityObserver:
    """各意思決定で採用手の記述子に一致する合法手数を数える（>1＝ラベルだけでは一意復元できない）。"""

    def __init__(self):
        self.total = 0
        self.ambiguous = 0
        self.by_action = defaultdict(lambda: [0, 0])   # action_type -> [total, ambiguous]
        self.match_hist = Counter()                    # 一致合法手数の分布
        self.examples: List[Dict[str, Any]] = []

    def on_decision(self, ctx, move):
        m = ctx.manager
        legal = m.get_legal_actions(ctx.actor)
        desc = cpu_ai._describe_move(m, move)
        matches = sum(1 for mv in legal if cpu_ai._describe_move(m, mv) == desc)
        at = (desc or {}).get("action_type", "?")
        self.total += 1
        self.by_action[at][0] += 1
        self.match_hist[matches] += 1
        if matches > 1:
            self.ambiguous += 1
            self.by_action[at][1] += 1
            if len(self.examples) < 12:
                self.examples.append({"turn": ctx.turn, "player": ctx.actor.name,
                                      "desc": desc, "matches": matches, "legal": len(legal)})


def main(argv=None):
    ap = argparse.ArgumentParser(description="実対局リプレイ R0: 記録アクションの一意復元可否を計測")
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--policy", choices=["ai", "random"], default="ai",
                    help="ai=L1(現実的な軌跡・低速) / random=高速・多数（位置の曝露は広い）")
    ap.add_argument("--difficulty", choices=["hard"], default="hard")
    ap.add_argument("--real-decks", action="store_true",
                    help="held-out 実デッキ（4-of 複製あり）で計測＝曖昧性の実態。既定=合成デッキ(distinct card_id)")
    args = ap.parse_args(argv)

    db = load_db()
    obs = _AmbiguityObserver()
    finished = 0
    deck_ids = HD.deck_ids() if args.real_decks else None

    def _deck_builder(_db, seed):
        # seed でデッキ対戦を巡回（実デッキは複製ありで曖昧性の源を含む）。
        ids = deck_ids
        l1, c1 = HD.build(_db, ids[seed % len(ids)], "p1")
        l2, c2 = HD.build(_db, ids[(seed + 1) % len(ids)], "p2")
        return l1, c1, l2, c2

    for i in range(args.games):
        seats = {pid: make_seat(args.difficulty, kind="ai" if args.policy == "ai" else "random",
                                mem={}) for pid in ("p1", "p2")}
        try:
            run_game(args.seed0 + i, db, seats=seats, observers=[obs], legal_moves="check",
                     deck_builder=(_deck_builder if args.real_decks else leader_deck_builder()))
            finished += 1
        except Exception as e:
            print(f"  game {i}: {type(e).__name__}: {e}")

    decks_label = "held-out実デッキ(複製あり)" if args.real_decks else "合成デッキ(distinct)"
    print(f"\n=== 実対局リプレイ R0: 記録アクションの一意復元可否"
          f"（{finished}/{args.games}局・policy={args.policy}・{decks_label}） ===")
    print(f"意思決定点: {obs.total}")
    rate = (obs.ambiguous / obs.total) if obs.total else 0.0
    print(f"ラベルだけで**一意復元できない**（同記述子の合法手が2つ以上）: {obs.ambiguous} ({rate:.2%})")
    print("一致合法手数の分布: " + ", ".join(f"{k}個={v}" for k, v in sorted(obs.match_hist.items())))
    print("\nアクション種別ごとの曖昧率:")
    for at, (tot, amb) in sorted(obs.by_action.items(), key=lambda kv: -kv[1][1]):
        print(f"  {at:<22} 曖昧 {amb}/{tot} ({(amb/tot if tot else 0):.1%})")
    if obs.examples:
        print("\n曖昧な意思決定の例:")
        for e in obs.examples[:8]:
            print(f"  t{e['turn']} {e['player']} {e['desc']} → 一致{e['matches']}手/合法{e['legal']}手")
    print("\n判定材料: 曖昧率が低い（数%以下）→ (A) 決定論タイブレーク逆引きで実用充分。"
          "\n          特定アクション種に集中 → その種だけ (B) 記録高解像度化。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
