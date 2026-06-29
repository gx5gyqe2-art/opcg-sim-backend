"""学習evalスパイク D-2: データ生成（dev・docs/reports/cpu_learned_eval_spike_design_20260629.md §A/D）。

現行CPU(教師)の自己対戦で局面を採取し **outcome（最終勝敗）** でラベル付け。Dual-Net の value 教師信号。
**ノイズ注入**（確率 eps でランダム合法手）で教師の局所最適から state を散らす（模倣の天井対策＝レビュー論点2）。
各採取局面は `rl_encoder.encode` で to-move 視点に符号化。value = +1(to-move が勝者) / -1(敗者)。

注意: 本走（10^6〜10^7局・3世代）は外部計算資源。ここは**少量での正しさ検証＋npz保存基盤**。
実行(スモーク): OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/rl_datagen.py --games 4 --eps 0.25 --out /tmp/d.npz
"""
import argparse
import random

import numpy as np

import conftest  # noqa: F401
from opcg_sim.src.core import action_api, cpu_ai
from opcg_sim.src.core.gamestate import GameManager, Player
from cpu_selfplay import build_deck, _load_db
import rl_encoder as E


def _apply(m, actor, mv):
    if mv["kind"] == "battle":
        action_api.apply_battle_action(m, actor, mv["action_type"], mv.get("card_uuid"))
    else:
        action_api.apply_game_action(m, actor, mv["action_type"], mv.get("payload", {}))


def generate(db, vocab, n_games, eps, max_steps, seed0, sample_every=1):
    """ノイズ付き自己対戦で (encoding, to_move, value) を集める。返り値は stack 済み dict。"""
    S, F, I, Y = [], [], [], []
    cpu_ai.set_budget_override(40)
    try:
        for g in range(n_games):
            seed = seed0 + g
            random.seed(seed)
            l1, c1 = build_deck(db, "p1")
            l2, c2 = build_deck(db, "p2")
            m = GameManager(Player("p1", c1, l1), Player("p2", c2, l2))
            m.start_game()
            prng = random.Random(seed * 13 + 1)
            snaps = []   # (scalars, field, idx, to_move)
            step = 0
            while m.winner is None and step < max_steps:
                pa = m.pending_actor_action()
                if not pa:
                    break
                pid, _ = pa
                actor = m.p1 if m.p1.name == pid else m.p2
                # 意思決定局面のみ採取（マリガン等の自明手も含むが害は小）。
                if step % sample_every == 0:
                    enc = E.encode(m, pid, vocab)
                    snaps.append((enc["scalars"], enc["field"], enc["card_idx"], pid))
                # ノイズ注入: eps でランダム合法手・残りは教師(CPU)。
                if prng.random() < eps:
                    moves = m.get_legal_actions(actor)
                    if not moves:
                        break
                    mv = prng.choice(moves)
                else:
                    try:
                        mv = cpu_ai.decide_guarded(m, actor, "hard", prng, pimc_worlds=1)
                    except Exception:
                        break
                if mv is None:
                    break
                try:
                    _apply(m, actor, mv)
                except Exception:
                    break
                step += 1
            if m.winner is None:
                continue   # 未決着（max_steps）＝ラベル付け不能で破棄
            for sc, fl, ix, who in snaps:
                S.append(sc); F.append(fl); I.append(ix)
                Y.append(1.0 if who == m.winner else -1.0)
    finally:
        cpu_ai.set_budget_override(None)
    if not S:
        return None
    return {"scalars": np.stack(S), "field": np.stack(F),
            "card_idx": np.stack(I), "value": np.array(Y, dtype=np.float32)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=4)
    ap.add_argument("--eps", type=float, default=0.25, help="ノイズ注入率(ランダム手の確率)")
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--sample-every", type=int, default=2)
    ap.add_argument("--out", default=None, help="npz 保存先")
    args = ap.parse_args()
    db = _load_db()
    vocab = E.build_vocab(db)
    data = generate(db, vocab, args.games, args.eps, args.max_steps, args.seed0, args.sample_every)
    if data is None:
        print("採取0（全局 max_steps 未決着）"); return
    n = len(data["value"])
    pos = float((data["value"] > 0).mean())
    print(f"採取局面: {n}  value+1率={pos:.2f}  scalars{data['scalars'].shape} "
          f"field{data['field'].shape} idx{data['card_idx'].shape}")
    if args.out:
        np.savez_compressed(args.out, **data)
        print(f"保存: {args.out}")


if __name__ == "__main__":
    main()
