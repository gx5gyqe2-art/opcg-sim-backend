"""最終確認ゲート: 凍結 warm-start value + MCTS(160) が **本番 L1（α-β+PIMC4）** に
held-out 実デッキで勝てるか。これまでの代理(greedy-L1)ではなく製品同等の強い相手で取り直す。

- learned = warm-start value（L1評価蒸留・凍結）を再生成 + MCTS(160)/uniform prior。
- 相手 = cpu_ai.decide(pimc_worlds=4, info_policy=fair)＝製品の α-β+PIMC。
- 席替え（p1/p2 半々）・Wilson CI・N 増量で CI を締める。
"""
import argparse
import random
import numpy as np
import conftest  # noqa: F401
import rl_fingerprint as FP
import rl_encoder as E
from rl_effective_state import encode_v3, DIM_V3, make_value_fn_for
from pre_flight4_mcts import mask_fps, COLOR
from mini_set_trial import MLP
from deck_generator import DeckGenerator
from heldout_gate import gen_dataset_parametric
import heldout_decks as HD
from cpu_selfplay import _load_db
from opcg_sim.src.core.gamestate import GameManager, Player
from opcg_sim.src.core import cpu_ai
from opcg_sim.src.learned.adapter import OPCGGame
import selfplay_loop as SL


def wilson(w, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 1.0
    p = w / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return p, (c - m) / d, (c + m) / d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot-games", type=int, default=200)
    ap.add_argument("--sims", type=int, default=160)
    ap.add_argument("--pimc", type=int, default=4)
    ap.add_argument("--pairs", type=int, default=25)
    ap.add_argument("--ply-cap", type=int, default=550)
    ap.add_argument("--every", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed); nrng = np.random.default_rng(args.seed)
    l1rng = random.Random(args.seed + 777)
    db = _load_db(); vocab = E.build_vocab(db)
    fps = mask_fps(FP.build_fingerprints(db), [COLOR])
    gen = DeckGenerator(db, seed=args.seed)
    game = OPCGGame(fair_determinize=True)

    print(f"warm-start value 再生成: L1評価 {args.boot_games} games（凍結）...", flush=True)
    Xb, Yb = gen_dataset_parametric(gen, db, vocab, fps, args.boot_games, args.ply_cap,
                                    args.every, rng, encode_fn=encode_v3)
    vnet = MLP(DIM_V3, seed=args.seed); vnet.fit_norm(Xb, Yb)
    vnet.train(Xb, Yb, epochs=args.epochs, rng=nrng)
    vf = make_value_fn_for(vnet, vocab, fps, encode_v3)
    print(f"  states={len(Xb)}  相手=本番L1(α-β+PIMC{args.pimc}) sims={args.sims} pairs={args.pairs}", flush=True)

    learned_move = lambda m, me: SL._run_mcts(game, vf, None, m, me, args.sims, 0.0, nrng)[0]

    def l1_move(m, actor_name):
        pl = m.p1 if m.p1.name == actor_name else m.p2
        return cpu_ai.decide(m, pl, rng=l1rng, info_policy="fair", pimc_worlds=args.pimc)

    print(f"\n=== 最終ゲート: 凍結value+MCTS({args.sims}) vs 本番L1(PIMC{args.pimc}) / held-out実デッキ ===", flush=True)
    out = {}
    for did in HD.deck_ids():
        w = n = 0
        for pair in range(args.pairs):
            for learned_seat in ("p1", "p2"):
                _l1, c1 = HD.build(db, did, "p1"); _l2, c2 = HD.build(db, did, "p2")
                m = GameManager(Player("p1", c1, _l1), Player("p2", c2, _l2)); m.start_game()
                ply = 0
                while ply < args.ply_cap and m.winner is None:
                    pa = m.pending_actor_action()
                    if pa is None:
                        break
                    nm = pa[0]
                    mv = learned_move(m, nm) if nm == learned_seat else l1_move(m, nm)
                    if mv is None:
                        break
                    try:
                        cpu_ai._apply_move_inplace(m, nm, mv)
                    except Exception:
                        break
                    ply += 1
                if m.winner is not None:
                    n += 1
                    if m.winner == learned_seat:
                        w += 1
            if (pair + 1) % 5 == 0:
                print(f"    [{did}] {pair + 1}/{args.pairs} pairs: {w}/{n}", flush=True)
        p, lo, hi = wilson(w, n)
        out[did] = (p, lo, hi, n)
        print(f"  {did}: 勝率={p:.3f} CI[{lo:.3f},{hi:.3f}] (n={n})", flush=True)

    avg = float(np.mean([p for p, lo, hi, n in out.values()]))
    minlo = min(lo for p, lo, hi, n in out.values())
    print(f"\n判定: avg={avg:.3f}  minCI下端={minlo:.3f}  "
          f"（合格: avg≥0.60 かつ 全デッキCI下端≥0.40）", flush=True)
    print(f"  → {'PASS' if (avg >= 0.60 and minlo >= 0.40) else 'not yet'}", flush=True)


if __name__ == "__main__":
    main()
