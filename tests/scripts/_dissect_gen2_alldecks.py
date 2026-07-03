"""検証①堅牢化: 出荷Gen2 vs 本番L1(PIMC4) を全3 held-out実デッキで測る。
出荷設定（決定化=相手手札のみ再サンプル）での勝率を席替えで測定する。
"""
import argparse
import random
import numpy as np
import os as _os, sys as _sys  # noqa: E402  test bootstrap (sys.path + google stub)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap  # noqa: E402,F401
import heldout_decks as HD
from cpu_selfplay import _load_db
from opcg_sim.src.core.gamestate import GameManager, Player
from opcg_sim.src.core import cpu_ai, cpu_learned


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

    cpu_learned._lazy_init()
    print(f"determinize=出荷(相手手札のみ) sims={args.sims} pimc={args.pimc} pairs={args.pairs}\n", flush=True)

    for did in HD.deck_ids():
        w = n = 0
        for pair in range(args.pairs):
            for seat in ("p1", "p2"):
                _l1, c1 = HD.build(db, did, "p1"); _l2, c2 = HD.build(db, did, "p2")
                m = GameManager(Player("p1", c1, _l1), Player("p2", c2, _l2)); m.start_game()
                ply = 0
                while ply < args.ply_cap and m.winner is None:
                    pa = m.pending_actor_action()
                    if pa is None:
                        break
                    nm = pa[0]
                    actor = m.p1 if m.p1.name == nm else m.p2
                    if nm == seat:
                        mv = cpu_learned.decide_learned(m, actor, sims=args.sims, rng=nrng)
                    else:
                        mv = cpu_ai.decide(m, actor, rng=l1rng, info_policy="fair", pimc_worlds=args.pimc)
                    if mv is None:
                        break
                    try:
                        cpu_ai._apply_move_inplace(m, nm, mv)
                    except Exception:
                        break
                    ply += 1
                if m.winner is None:
                    continue
                n += 1
                if m.winner == seat:
                    w += 1
        p = w / n if n else float("nan")
        print(f"  {did:26s} Gen2勝率={p:.3f} ({w}/{n})", flush=True)


if __name__ == "__main__":
    main()
