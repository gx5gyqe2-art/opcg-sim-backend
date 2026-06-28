"""④ settle-PASS 自己リーサル過大検出の計装（dev・§H/§I）。

settle（探索打ち切り）は戦闘応答を両側 PASS で畳む。地平線外で「相手が無防備な前提」のまま勝者が
確定する＝攻め側は自分のリーサルを過大検出（ノーガード特攻の温床）。本スクリプトは **単一プロセス**で
自己対戦を回し、`_settle_eval` のグローバル計数を集計して「settle ループが勝者を生成した頻度」を実測する。

判定: loop_winner_*（PASS整流が生成した勝者）が calls に対し稀なら過大検出はほぼ起きていない＝幽霊。
無視できない頻度なら、続けて mulligan/settle 同型の A/B（neutralize）で Elo 影響を測る価値がある。

実行: OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/settle_pass_instrument.py --games 30
"""
import argparse
import time

import conftest  # noqa: F401
from cpu_arena import _load_db, play_game
from opcg_sim.src.core import cpu_ai
import deckgen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--pimc", type=int, default=1)
    ap.add_argument("--real-decks", action="store_true", help="deckgen 実構築デッキで対戦")
    args = ap.parse_args()
    db = _load_db()
    ids = list(deckgen.VERIFIED_LEADERS.values())

    cpu_ai.set_settle_stats(True)
    cpu_ai.reset_settle_stats()
    t0 = time.time()
    failed = 0
    for g in range(args.games):
        rl = None
        if args.real_decks:
            lid = ids[g % len(ids)]
            rl = (lid, lid)
        try:
            play_game(args.seed0 + g, db, "hard", "hard", p1_pimc=args.pimc, p2_pimc=args.pimc,
                      realistic_leaders=rl, separate_policy_rng=True)
        except Exception:
            failed += 1
    cpu_ai.set_settle_stats(False)
    s = cpu_ai.get_settle_stats()
    dt = time.time() - t0

    calls = max(1, s["calls"])
    loop_me = s["loop_winner_me"]; loop_opp = s["loop_winner_opp"]
    loop_total = loop_me + loop_opp
    print("\n=== ④ settle-PASS 計装 ===")
    deck_tag = "実構築デッキ" if args.real_decks else "合成同色50枚"
    print(f"{args.games}局 / {deck_tag} / pimc={args.pimc} / {dt:.0f}s / 失敗{failed}局")
    print(f"_settle_eval 呼び出し: {s['calls']:,}")
    print(f"  entry_winner（探索が読み切った真の勝敗）   : {s['entry_winner']:,}  ({s['entry_winner']/calls:.3%})")
    print(f"  loop_winner_me （PASS整流が root 勝利生成・攻め過大検出 suspect）: {loop_me:,}  ({loop_me/calls:.3%})")
    print(f"  loop_winner_opp（PASS整流が相手勝利生成・守り過大検出 suspect）: {loop_opp:,}  ({loop_opp/calls:.3%})")
    print(f"  うち敗者の戦闘応答を PASS で畳んだ局面       : {s['loop_winner_passdef']:,}")
    print(f"\nループ生成勝者の総率 = {loop_total/calls:.3%}（calls 比）")
    print("解釈: 稀（<~0.5%）なら過大検出は幽霊＝④打ち切り。無視できない頻度なら neutralize A/B で Elo 影響を測る。")


if __name__ == "__main__":
    main()
