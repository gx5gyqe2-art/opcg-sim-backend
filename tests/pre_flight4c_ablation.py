"""pre-flight ④c: 表現の“もつれ”アブレーション（レビュー反証の検証）。

レビュー仮説: 黄で corr≈0 まで崩れるのは、fingerprint が「色」次元へ過剰適合(entanglement)して
おり、色が未知になった瞬間に他の(trigger/action)次元を無視するから。もし真なら「色を隠す」と
非黄→黄のゼロショット転移(corr)が回復するはず。回復しなければ「決定的な相互作用が表現に欠落」。

方法(安い・MCTS不要): fingerprint の一部次元をゼロマスクして net0 を bootstrap し直し、
diag_A と同じ corr(net,L1) を 非黄/黄 で再計測。
  variants: baseline / no-color(色6次元をマスク) / behavior-only(trigger+action だけ残す)
"""
import argparse
import random

import numpy as np

import rl_fingerprint as FP
import rl_encoder as E
from rl_encoder_v2 import DIM
from probe_generalization import _leaders_split
from mini_set_trial import MLP, gen_dataset
from pre_flight4b_trace import sample_states
from cpu_selfplay import _load_db

COLOR = (7, 13)
NON_BEHAVIOR = (0, 21)   # static+color+type+keywords（trigger/action の前まで）


def mask_fps(fps, zero_slices):
    out = {}
    for k, v in fps.items():
        v2 = v.copy()
        for a, b in zero_slices:
            v2[a:b] = 0.0
        out[k] = v2
    return out


def corr_of(net, rows):
    X = np.array([r[0] for r in rows], np.float32)
    L1 = np.array([r[1] for r in rows], np.float32)
    v = np.array([net.value(x) for x in X])
    L1n = (L1 - net.ymu) / net.ysd
    return float(np.corrcoef(v, L1n)[0, 1]) if len(v) > 2 else float("nan")


def run_variant(name, fps, train_leaders, held_leaders, db, vocab, args, rng, nrng):
    Xb, Yb = gen_dataset(train_leaders, db, vocab, fps, args.boot_games, args.ply_cap, args.every, rng)
    net = MLP(DIM, seed=args.seed); net.fit_norm(Xb, Yb)
    net.train(Xb, Yb, epochs=args.epochs, rng=nrng)
    rows_tr = sample_states(train_leaders, db, vocab, fps, args.sample_games, args.ply_cap, args.every, rng)
    rows_h = sample_states(held_leaders, db, vocab, fps, args.sample_games, args.ply_cap, args.every, rng)
    c_tr, c_h = corr_of(net, rows_tr), corr_of(net, rows_h)
    print(f"  {name:16s} corr(net,L1): 非黄={c_tr:+.3f}  黄={c_h:+.3f}   (黄で回復={c_h:+.3f})")
    return c_tr, c_h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot-games", type=int, default=140)
    ap.add_argument("--sample-games", type=int, default=40)
    ap.add_argument("--ply-cap", type=int, default=550)
    ap.add_argument("--every", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    nrng = np.random.default_rng(args.seed)
    db = _load_db()
    vocab = E.build_vocab(db)
    fps = FP.build_fingerprints(db)
    train_leaders, held_leaders = _leaders_split(db, "color", 0, rng)

    variants = [
        ("baseline", fps),
        ("no-color", mask_fps(fps, [COLOR])),
        ("behavior-only", mask_fps(fps, [NON_BEHAVIOR])),
    ]
    print(f"seed={args.seed}  === corr(net,L1) 非黄→黄 のゼロショット転移 ===")
    res = {}
    for name, f in variants:
        res[name] = run_variant(name, f, train_leaders, held_leaders, db, vocab, args, rng, nrng)
    b_h = res["baseline"][1]
    nc_h = res["no-color"][1]
    bo_h = res["behavior-only"][1]
    print("\n判定:")
    if nc_h > b_h + 0.15 or bo_h > b_h + 0.15:
        print(f"  色/静的次元を隠すと黄corrが回復({b_h:+.2f}→no-color {nc_h:+.2f}/behav {bo_h:+.2f})"
              f"＝色への“もつれ”が主因（レビュー仮説◯・表現の脱もつれが本筋）")
    else:
        print(f"  隠しても黄corrは回復せず({b_h:+.2f}→{nc_h:+.2f}/{bo_h:+.2f})"
              f"＝もつれでなく相互作用/振る舞いの表現欠落か真の未学習（②被覆＋表現拡張が要）")


if __name__ == "__main__":
    main()
