"""透視切り分け: 出荷Gen2 の MCTS決定化を『公平版(自分の山札順・両者ライフ中身も再サンプル
＝透視禁止)』に差し替えても、1手が正常時間で返るか（過去のハング再検証）＋
Gen2 vs L1 の勝率が保たれるか。fair det はテスト側実装（本番は不変）。
"""
import argparse
import random
import time
import numpy as np
import os as _os, sys as _sys  # noqa: E402  test bootstrap (sys.path + google stub)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap  # noqa: E402,F401
import heldout_decks as HD
from cpu_selfplay import _load_db
from opcg_sim.src.core.gamestate import GameManager, Player
from opcg_sim.src.core import cpu_ai, cpu_learned
from opcg_sim.src.learned.adapter import OPCGGame

DECK = "blackbeard_black_yellow"


def _resample_hidden(pl, rng, include_hand):
    life = pl.life
    fd_idx = [i for i, c in enumerate(life) if not getattr(c, "is_face_up", False)]
    pool = list(pl.deck) + [life[i] for i in fd_idx]
    if include_hand:
        pool += list(pl.hand)
    if not pool:
        return
    rng.shuffle(pool)
    k = 0
    if include_hand:
        n_hand = len(pl.hand)
        pl.hand[:] = pool[k:k + n_hand]; k += n_hand
    for i in fd_idx:
        c = pool[k]; k += 1
        try:
            c.is_face_up = False
        except Exception:
            pass
        life[i] = c
    pl.deck[:] = pool[k:]


class FairGame(OPCGGame):
    """determinize を『両者の隠匿情報を再サンプル（透視禁止）』へ差し替えた研究用アダプタ。"""
    def determinize(self, state, me_name, rng):
        clone = state.clone()
        me = clone.p1 if clone.p1.name == me_name else clone.p2
        opp = clone.p2 if clone.p1.name == me_name else clone.p1
        _resample_hidden(me, rng, include_hand=False)
        _resample_hidden(opp, rng, include_hand=True)
        return clone


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "match"], default="smoke")
    ap.add_argument("--pairs", type=int, default=6)
    ap.add_argument("--sims", type=int, default=160)
    ap.add_argument("--pimc", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    db = _load_db()
    nrng = np.random.default_rng(args.seed)
    l1rng = random.Random(args.seed + 777)
    cpu_learned._lazy_init()
    cpu_learned._STATE["game"] = FairGame()   # ← Gen2 の探索を公平決定化に

    if args.mode == "smoke":
        # 中盤局面まで進めて1手の所要時間を測る（ハング検出）
        _l1, c1 = HD.build(db, DECK, "p1"); _l2, c2 = HD.build(db, DECK, "p2")
        m = GameManager(Player("p1", c1, _l1), Player("p2", c2, _l2)); m.start_game()
        for _ in range(12):
            pa = m.pending_actor_action()
            if pa is None:
                break
            nm = pa[0]; actor = m.p1 if m.p1.name == nm else m.p2
            mv = cpu_ai.decide(m, actor, rng=l1rng, info_policy="fair", pimc_worlds=1)
            if mv is None:
                break
            cpu_ai._apply_move_inplace(m, nm, mv)
        pa = m.pending_actor_action(); nm = pa[0]; actor = m.p1 if m.p1.name == nm else m.p2
        t0 = time.time()
        mv = cpu_learned.decide_learned(m, actor, sims=args.sims, rng=nrng)
        print(f"公平決定化での Gen2 1手: {time.time() - t0:.2f}s  move={mv and mv.get('action_type')}", flush=True)
        print("→ 数秒で返ればハングは解消／過去のハングは別要因", flush=True)
        return

    # match: Gen2(公平決定化) vs L1(出荷pimc) 全3デッキ
    print(f"Gen2=公平決定化(透視禁止) vs L1=pimc{args.pimc}  sims={args.sims} pairs={args.pairs}\n", flush=True)
    for did in HD.deck_ids():
        w = n = 0
        for pair in range(args.pairs):
            for seat in ("p1", "p2"):
                _l1, c1 = HD.build(db, did, "p1"); _l2, c2 = HD.build(db, did, "p2")
                m = GameManager(Player("p1", c1, _l1), Player("p2", c2, _l2)); m.start_game()
                ply = 0
                while ply < 550 and m.winner is None:
                    pa = m.pending_actor_action()
                    if pa is None:
                        break
                    nm = pa[0]; actor = m.p1 if m.p1.name == nm else m.p2
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
        print(f"  {did:26s} Gen2(公平)勝率={w/n if n else float('nan'):.3f} ({w}/{n})", flush=True)


if __name__ == "__main__":
    main()
