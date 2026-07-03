"""検証①: 出荷 Gen2（gen2_*.npz・decide_learned）が黒ひげ(OP16-080)で本番L1に負けるかを
**実際の出荷ネットで**再現し、敗北局のトレースを収集して H0/H1 を判定する。

H0: 汎化不足の統計的劣化（負けるが手は妥当・value正気）
H1: 系統的バグ/退行（value符号反転・飽和・NaN／PASS集中／殴らない・リーサル見逃し／
    OP16-080固有効果との相互作用で悪手）

出力: 各局 W/L・敗北局の手ごとトレース（chosen/value Q/上位候補visit%・Q/l1食い違い）を
JSON 保存し、集計を print。
"""
import argparse
import json
import random
import numpy as np
import conftest  # noqa: F401
import heldout_decks as HD
from cpu_selfplay import _load_db
from opcg_sim.src.core.gamestate import GameManager, Player
from opcg_sim.src.core import cpu_ai, cpu_learned

DECK = "blackbeard_black_yellow"


def is_attack_available(legal):
    return any(m.get("action_type") in ("ATTACK", "BATTLE") or m.get("kind") == "battle"
              for m in (legal or []))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=6)
    ap.add_argument("--sims", type=int, default=160)
    ap.add_argument("--pimc", type=int, default=4)
    ap.add_argument("--ply-cap", type=int, default=550)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str,
                    default="/tmp/claude-0/-home-user/3bd3067e-48b2-52b6-99c2-33478ab4ab32/scratchpad/dissect_gen2.json")
    args = ap.parse_args()

    db = _load_db()
    l1rng = random.Random(args.seed + 777)
    nrng = np.random.default_rng(args.seed)

    all_values = []          # 全 learned 手の value Q
    degen_turn_end = 0       # 攻撃可能なのに TURN_END/PASS を選んだ回数
    learned_moves_total = 0
    l1_disagree = 0
    games = []
    w = n = 0

    for pair in range(args.pairs):
        for learned_seat in ("p1", "p2"):
            _l1, c1 = HD.build(db, DECK, "p1"); _l2, c2 = HD.build(db, DECK, "p2")
            m = GameManager(Player("p1", c1, _l1), Player("p2", c2, _l2)); m.start_game()
            traces = []
            ply = 0
            while ply < args.ply_cap and m.winner is None:
                pa = m.pending_actor_action()
                if pa is None:
                    break
                nm = pa[0]
                actor = m.p1 if m.p1.name == nm else m.p2
                if nm == learned_seat:
                    legal = m.get_legal_actions(actor)
                    tr = {}
                    mv = cpu_learned.decide_learned(m, actor, sims=args.sims, rng=nrng, trace=tr)
                    learned_moves_total += 1
                    if "value" in tr:
                        all_values.append(tr["value"])
                    if tr.get("l1_disagrees"):
                        l1_disagree += 1
                    ae = is_attack_available(legal)
                    chosen_at = (tr.get("chosen") or {}).get("action_type")
                    if ae and chosen_at in ("TURN_END", "PASS"):
                        degen_turn_end += 1
                    tr["_attack_available"] = ae
                    traces.append(tr)
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
            won = (m.winner == learned_seat)
            if won:
                w += 1
            games.append({"pair": pair, "learned_seat": learned_seat, "won": won,
                          "winner": m.winner, "plies": ply, "traces": traces})
            print(f"  game {n}: learned={learned_seat} winner={m.winner} "
                  f"{'WIN' if won else 'LOSS'} plies={ply}", flush=True)

    vals = np.array([v for v in all_values if v is not None], dtype=float)
    nan_ct = int(np.isnan(vals).sum()) if len(vals) else 0
    with open(args.out, "w") as f:
        json.dump({"args": vars(args), "record": f"{w}/{n}", "games": games}, f, ensure_ascii=False)

    print(f"\n=== 検証① 出荷Gen2 vs 本番L1(PIMC{args.pimc}) / 黒ひげ ===", flush=True)
    print(f"  戦績: {w}/{n}  勝率={w/n if n else float('nan'):.3f}", flush=True)
    print(f"  learned 手数={learned_moves_total}", flush=True)
    if len(vals):
        print(f"  value Q: mean={vals.mean():+.3f} std={vals.std():.3f} "
              f"min={vals.min():+.3f} max={vals.max():+.3f} NaN={nan_ct} "
              f"(飽和|Q|>0.95={int((np.abs(vals)>0.95).sum())}/{len(vals)})", flush=True)
    print(f"  退行: 攻撃可能なのにTURN_END/PASS = {degen_turn_end}/{learned_moves_total}", flush=True)
    print(f"  L1と食い違う手 = {l1_disagree}/{learned_moves_total} "
          f"({100.0*l1_disagree/max(learned_moves_total,1):.0f}%)", flush=True)
    print(f"  トレース保存: {args.out}", flush=True)


if __name__ == "__main__":
    main()
