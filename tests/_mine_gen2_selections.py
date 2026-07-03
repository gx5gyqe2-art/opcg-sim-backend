"""検証①-②(実戦版): 出荷Gen2 が黒ひげを操縦する実戦で、効果選択の実際の中身を集計。
active_interaction.source_card_uuid で発生元カードを特定し、選んだ payload の
accepted / selected_uuids を記録。OP16-080(リダイレクト任意) と OP16-119(ライフ追加up-to) を追う。
"""
import argparse
import collections
import random
import numpy as np
import conftest  # noqa: F401
import heldout_decks as HD
from cpu_selfplay import _load_db
from opcg_sim.src.core.gamestate import GameManager, Player
from opcg_sim.src.core import cpu_ai, cpu_learned

DECK = "blackbeard_black_yellow"


def card_id_of(m, uuid):
    if not uuid:
        return None
    try:
        c = m._find_card_by_uuid(uuid)
    except Exception:
        return None
    if c is None:
        return None
    mm = getattr(c, "master", None)
    return getattr(mm, "card_id", None) or getattr(c, "card_id", None) or getattr(mm, "number", None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=6)
    ap.add_argument("--sims", type=int, default=160)
    ap.add_argument("--pimc", type=int, default=4)
    ap.add_argument("--ply-cap", type=int, default=550)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    db = _load_db()
    l1rng = random.Random(args.seed + 777)
    nrng = np.random.default_rng(args.seed)

    # (card_id, action_type) -> Counter of choice-labels
    tally = collections.defaultdict(collections.Counter)

    for pair in range(args.pairs):
        for seat in ("p1", "p2"):
            _l1, c1 = HD.build(db, DECK, "p1"); _l2, c2 = HD.build(db, DECK, "p2")
            m = GameManager(Player("p1", c1, _l1), Player("p2", c2, _l2)); m.start_game()
            ply = 0
            while ply < args.ply_cap and m.winner is None:
                pa = m.pending_actor_action()
                if pa is None:
                    break
                nm = pa[0]
                actor = m.p1 if m.p1.name == nm else m.p2
                if nm == seat:
                    pend = getattr(m, "active_interaction", None)
                    src_id = at = None
                    if pend:
                        at = pend.get("action_type")
                        src_id = card_id_of(m, pend.get("source_card_uuid"))
                    mv = cpu_learned.decide_learned(m, actor, sims=args.sims, rng=nrng)
                    if pend and mv and at in ("CONFIRM_OPTIONAL", "SELECT_TARGET", "FIELD_OVERFLOW_TRASH"):
                        pl = mv.get("payload") or {}
                        if at == "CONFIRM_OPTIONAL":
                            lab = "accept" if pl.get("accepted") else "decline"
                        else:
                            lab = f"select{len(pl.get('selected_uuids') or [])}"
                        tally[(src_id, at)][lab] += 1
                else:
                    mv = cpu_ai.decide(m, actor, rng=l1rng, info_policy="fair", pimc_worlds=args.pimc)
                if mv is None:
                    break
                try:
                    cpu_ai._apply_move_inplace(m, nm, mv)
                except Exception:
                    break
                ply += 1

    print(f"\n=== 出荷Gen2 の効果選択（黒ひげ実戦 {args.pairs}ペア・sims{args.sims}） ===")
    if not tally:
        print("  選択対話に遭遇せず（試行増が必要）")
    for (cid, at), cnt in sorted(tally.items(), key=lambda x: -sum(x[1].values())):
        total = sum(cnt.values())
        detail = "  ".join(f"{k}={v}" for k, v in cnt.most_common())
        print(f"  {str(cid):10s} {at:16s} n={total:3d}  {detail}")
    print("\n注目:")
    print("  OP16-080 CONFIRM_OPTIONAL … accept=リダイレクト発動 / decline=見送り")
    print("  OP16-119 SELECT_TARGET   … select1=ライフ追加 / select0=見送り")


if __name__ == "__main__":
    main()
