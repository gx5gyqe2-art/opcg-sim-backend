"""P3 1シャードプローブ（疎通＋スループット実測・docs/.../cpu_rl_pilot_p3_harness_20260629.md）。

レビュー確定の着手前プローブ。測るもの:
  ① 自己対戦スループット（games/sec・sims別）→ シャードサイズの決定（回収窓に収める）。
  ② オンライン更新の loss 挙動（**学習率を下げて**忘却/振動のサニティ）。
  ③ チェックポイントのファイルサイズ・書込時間（commit コストの実測）。
※ 勝率は見ない（疎通とスループットのみ・レビュー確定）。

実行: OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/p3_probe.py --sims 40 --games 8
"""
import argparse
import os
import time

import numpy as np

import conftest  # noqa: F401
import rl_encoder as E
import rl_net as RN
from az_policy import PolicyScorer, train_policy
from opcg_game import OPCGGame
from cpu_selfplay import _load_db
import p3_loop as P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, nargs="+", default=[40, 100])
    ap.add_argument("--games", type=int, default=6, help="各sims水準の自己対戦数")
    ap.add_argument("--sl-net", default="tests/p2_sl_net.npz")
    ap.add_argument("--lr", type=float, default=5e-4, help="オンライン更新の学習率(意図的に低め)")
    ap.add_argument("--dirichlet-eps", type=float, default=0.25)
    args = ap.parse_args()

    P._DB = _load_db()
    vocab = E.build_vocab(P._DB)
    game = OPCGGame()
    v0 = RN.ValueNet.load(args.sl_net)
    print(f"Gen0 value net: {args.sl_net}", flush=True)

    # value_fn に Dirichlet ノイズ付き自己対戦をさせるため、generate に eps を通す版を即席で。
    def gen_with_noise(n, sims, rng):
        S, F, I, Y, pol = [], [], [], [], []
        from az_mcts_tree import TreeMCTS
        from opcg_action import legal_action_matrix
        from az_policy import state_context
        from opcg_sim.src.core import cpu_ai
        vf = P.value_fn_of(v0, vocab)
        for g in range(n):
            m = game.new_game(P._DB, int(rng.integers(1 << 30)))
            recs_v, recs_p = [], []
            steps = 0
            while game.winner(m) is None and not game.is_terminal(m) and steps < 400:
                name = game.current_player(m)
                if name is None:
                    break
                mc = TreeMCTS(game, value_fn=vf, priors_fn=None, n_sims=sims,
                              determinize_fn=lambda s, r: game.determinize(s, name, r),
                              rng=rng, dirichlet_eps=args.dirichlet_eps)
                move, N, legal = mc.run(m)
                if move is None or N is None or N.sum() == 0:
                    break
                enc = E.encode(m, name, vocab)
                recs_v.append((enc, name))
                recs_p.append((state_context(m, name, vocab),
                               legal_action_matrix(m, legal, name), N / N.sum()))
                a = int(np.argmax(N)) if steps >= 8 else int(rng.choice(len(N), p=(N / N.sum())))
                try:
                    cpu_ai._apply_move_inplace(m, name, legal[a])
                except Exception:
                    break
                steps += 1
            w = game.winner(m)
            if w is None:
                continue
            for enc, who in recs_v:
                S.append(enc["scalars"]); F.append(enc["field"]); I.append(enc["card_idx"])
                Y.append(1.0 if who == w else -1.0)
            for ctx, am, visit in recs_p:
                pol.append((ctx, am, visit))
        if not S:
            return None, None
        return ({"scalars": np.stack(S), "field": np.stack(F), "card_idx": np.stack(I),
                 "value": np.array(Y, dtype=np.float32)}, pol)

    print("\n=== ① 自己対戦スループット ===", flush=True)
    last = None
    for sims in args.sims:
        rng = np.random.default_rng(100 + sims)
        t0 = time.perf_counter()
        vdata, pol = gen_with_noise(args.games, sims, rng)
        dt = time.perf_counter() - t0
        if vdata is None:
            print(f"  sims={sims}: 採取0"); continue
        npos = len(vdata["value"])
        gps = args.games / dt
        print(f"  sims={sims:>3}: {args.games}局 {dt:5.0f}s  {gps:.3f} games/s  "
              f"{npos}局面 ({npos/args.games:.0f}/局)  1コア", flush=True)
        # 回収窓を target 分に収めるシャードサイズの目安。
        for win_min in (15, 30):
            print(f"      → {win_min}分窓・1コア: {int(gps*win_min*60)}局/シャード "
                  f"（4コア概算 ×3.5={int(gps*win_min*60*3.5)}局）")
        last = (vdata, pol)

    if last is None:
        print("採取0で終了"); return 1
    vdata, pol = last

    print("\n=== ② オンライン更新の loss 挙動（LR下げ・忘却/振動サニティ） ===", flush=True)
    vmse0 = float(((v0.predict(vdata) - vdata["value"]) ** 2).mean())
    RN.train(v0, vdata, epochs=3, lr=args.lr, batch=128, val_frac=0.1)
    vmse1 = float(((v0.predict(vdata) - vdata["value"]) ** 2).mean())
    print(f"  value MSE: {vmse0:.3f} → {vmse1:.3f}（lr={args.lr}・3ep）", flush=True)
    pnet = PolicyScorer(hidden=128, seed=0)
    ce0 = train_policy(pnet, pol, epochs=1, lr=args.lr)
    ce1 = train_policy(pnet, pol, epochs=3, lr=args.lr)
    print(f"  policy CE: {ce0:.3f} → {ce1:.3f}（lr={args.lr}）", flush=True)

    print("\n=== ③ チェックポイント書込コスト ===", flush=True)
    os.makedirs("/tmp/p3ckpt", exist_ok=True)
    t0 = time.perf_counter(); v0.save("/tmp/p3ckpt/v.npz"); pnet.save("/tmp/p3ckpt/p.npz")
    dt = time.perf_counter() - t0
    sz = os.path.getsize("/tmp/p3ckpt/v.npz") + os.path.getsize("/tmp/p3ckpt/p.npz")
    print(f"  value+policy 保存: {dt*1000:.0f}ms  合計 {sz/1024:.0f}KB", flush=True)
    print("\nプローブ完了。①でシャードサイズ確定・②でlossが暴れないこと確認・③でcommitコスト把握。"
          "（勝率は判定に使わない＝疎通とスループットのみ）")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
