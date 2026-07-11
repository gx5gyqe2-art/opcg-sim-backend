"""蒸留v2 データ生成（再開可能シャード方式・司令塔 2026-07-09）。

出荷v1（value+policy・enc v1）の自己対戦を**97リーダー多様分布**で回し、各局面に
（v3符号化, 教師=出荷v1のvalue生予測）を記録して npz シャードに逐次保存する。
シャード単位で resume 可能（存在するシャードはスキップ）＝コンテナ再起動に耐える。

実行: OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/scripts/distill2_gen.py \
        --games 10000 --shard 250 --outdir /home/user/distill2_data
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import multiprocessing as mp
import time

import numpy as np

import os as _os, sys as _sys  # noqa: E402  test bootstrap
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import _bootstrap  # noqa: E402,F401
import rl_encoder as E
import rl_net as RN
from az_policy import PolicyScorer
from opcg_sim.src.core import cpu_ai
from opcg_sim.src.learned.config import SELFPLAY_TEMP_MOVES
from az_mcts_tree import TreeMCTS
from opcg_game import OPCGGame
from cpu_selfplay import _load_db
from deckgen import all_leader_ids
import p3_loop as P

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SHIP_V = os.path.join(REPO, "opcg_sim", "data", "learned", "gen2_value.npz")
SHIP_P = os.path.join(REPO, "opcg_sim", "data", "learned", "gen2_policy.npz")
SIMS, EPS = 40, 0.25
_W = {}


def _init():
    db = _load_db()
    vocab = E.build_vocab(db)
    ship_v = RN.ValueNet.load(SHIP_V)
    _W.update(db=db, vocab=vocab, game=OPCGGame(), ship_v=ship_v,
              vf=P.value_fn_of(ship_v, vocab, 1),
              pf=P.priors_fn_of(PolicyScorer.load(SHIP_P), vocab, 1),
              leaders=all_leader_ids(db))


def _one(seed):
    db, vocab, game = _W["db"], _W["vocab"], _W["game"]
    rng = np.random.default_rng(seed)
    m = game.new_game(db, int(rng.integers(1 << 30)), leaders=_W["leaders"])
    e1, e3, steps = [], [], 0
    while game.winner(m) is None and not game.is_terminal(m) and steps < 400:
        name = game.current_player(m)
        if name is None:
            break
        mc = TreeMCTS(game, value_fn=_W["vf"], priors_fn=_W["pf"], n_sims=SIMS,
                      determinize_fn=lambda s, r: game.determinize(s, name, r),
                      rng=rng, dirichlet_eps=EPS)
        move, N, legal = mc.run(m)
        if move is None or N is None or N.sum() == 0:
            break
        e1.append(E.encode(m, name, vocab, version=1))
        e3.append(E.encode(m, name, vocab, version=3))
        a = int(np.argmax(N)) if steps >= SELFPLAY_TEMP_MOVES else int(rng.choice(len(N), p=N / N.sum()))
        try:
            cpu_ai._apply_move_inplace(m, name, legal[a])
        except Exception:
            break
        steps += 1
    if not e1:
        return None
    b1 = {k: np.stack([r[k] for r in e1]) for k in ("scalars", "field", "card_idx")}
    y = _W["ship_v"].predict(b1).astype(np.float32)
    return (np.stack([r["scalars"] for r in e3]), np.stack([r["field"] for r in e3]),
            np.stack([r["card_idx"] for r in e3]), y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=10000)
    ap.add_argument("--shard", type=int, default=250)
    ap.add_argument("--outdir", default="/home/user/distill2_data")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    n_shards = args.games // args.shard
    pool = mp.Pool(args.workers, initializer=_init)
    t0 = time.perf_counter()
    for si in range(n_shards):
        path = os.path.join(args.outdir, f"shard_{si:04d}.npz")
        if os.path.exists(path):
            continue                                   # resume: 既存シャードはスキップ
        seeds = [61000 + si * args.shard + i for i in range(args.shard)]
        parts = [p for p in pool.map(_one, seeds) if p is not None]
        S = np.concatenate([p[0] for p in parts]); F = np.concatenate([p[1] for p in parts])
        I = np.concatenate([p[2] for p in parts]); y = np.concatenate([p[3] for p in parts])
        np.savez(path + ".tmp.npz", scalars=S, field=F, card_idx=I, value=y)
        os.replace(path + ".tmp.npz", path)            # atomic: 中途半端なシャードを残さない
        print(f"shard {si+1}/{n_shards}: {len(parts)}ゲーム {len(y)}局面 "
              f"({time.perf_counter()-t0:.0f}s)", flush=True)
    pool.close(); pool.join()
    print("GENERATION_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    mp.set_start_method("fork", force=True)
    import sys
    sys.exit(main())
