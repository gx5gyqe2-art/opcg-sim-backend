"""P3: AZ自己対戦RLループ（OPCG・value+pointer policy）＋世代間クロス評価。

docs/.../cpu_rl_pilot_plan_20260629.md P3。Gen0(=P2のSL価値net＋uniform prior)で自己対戦し、
(局面, MCTS訪問分布, 最終勝敗) を採取→ value net(outcome)＋policy(訪問分布) を学習して Gen1…と進める。
policy は uniform から **RL で育てる**（P2でL1模倣policyを足さない＝模倣の天井回避・レビュー確定）。

判定はクロス評価（gen N+1 vs gen N／対Gen0）。損切り（レビュー確定）:
  Gen1 vs Gen0 ≥0.55 で続行・以降の対前世代は 0.51〜0.52(後退でないこと)・
  Gen3までに「Gen_k vs Gen0」が0.55未達なら停止。**本走は N=400 CRN・常設CPU VM**。
  本環境＝ハーネス＋インフラ試走（疎通のみ・勝率は無視）。

スモーク: OPCG_LOG_SILENT=1 PYTHONPATH=tests python tests/p3_loop.py --smoke
"""
import argparse
import time

import numpy as np

import conftest  # noqa: F401
from opcg_sim.src.core import cpu_ai
from opcg_game import OPCGGame
from az_mcts_tree import TreeMCTS
import rl_encoder as E
import rl_net as RN
from az_policy import PolicyScorer, state_context, train_policy
from opcg_action import legal_action_matrix
from cpu_selfplay import _load_db


# ---- ネットを MCTS の value_fn / priors_fn に変換 ----
def value_fn_of(net, vocab):
    def value(state, to_move):
        if state.winner is not None:
            return 1.0 if state.winner == to_move else -1.0
        enc = E.encode(state, to_move, vocab)
        batch = {k: enc[k][None, ...] for k in ("scalars", "field", "card_idx")}
        return float(net.predict(batch)[0])
    return value


def priors_fn_of(policy, vocab):
    if policy is None:
        return None
    def priors(state, legal):
        me = state.pending_actor_action()[0]
        ctx = state_context(state, me, vocab)
        am = legal_action_matrix(state, legal, me)
        p = policy.priors(ctx, am)
        return p if p.shape[0] == len(legal) else None
    return priors


def _sample(counts, rng, temp):
    if temp <= 1e-6:
        return int(np.argmax(counts))
    p = counts.astype(np.float64) ** (1.0 / temp)
    s = p.sum()
    return int(rng.choice(len(p), p=p / s)) if s > 0 else int(np.argmax(counts))


# ---- 自己対戦でデータ採取 ----
def selfplay_game(game, value_fn, priors_fn, vocab, sims, c_puct, rng, temp_moves=8, max_steps=400):
    m = game.new_game(db=_DB, seed=int(rng.integers(1 << 30)))
    val_recs, pol_recs = [], []   # (enc, who) / (ctx, am, visit, who)
    steps = 0
    while game.winner(m) is None and not game.is_terminal(m) and steps < max_steps:
        name = game.current_player(m)
        if name is None:
            break
        mcts = TreeMCTS(game, value_fn=value_fn, priors_fn=priors_fn, c_puct=c_puct,
                        n_sims=sims, determinize_fn=lambda s, r: game.determinize(s, name, r), rng=rng)
        move, N, legal = mcts.run(m)
        if move is None or N is None or N.sum() == 0:
            break
        visit = N.astype(np.float64) / N.sum()
        enc = E.encode(m, name, vocab)
        val_recs.append((enc, name))
        ctx = state_context(m, name, vocab)
        am = legal_action_matrix(m, legal, name)
        pol_recs.append((ctx, am, visit, name))
        a = _sample(N, rng, temp=1.0 if steps < temp_moves else 0.0)
        try:
            cpu_ai._apply_move_inplace(m, name, legal[a])
        except Exception:
            break
        steps += 1
    winner = game.winner(m)
    return val_recs, pol_recs, winner


def generate(game, value_fn, priors_fn, vocab, n_games, sims, c_puct, rng, log=print):
    S, F, I, Y = [], [], [], []
    pol = []
    for g in range(n_games):
        vr, pr, w = selfplay_game(game, value_fn, priors_fn, vocab, sims, c_puct, rng)
        if w is None:
            continue
        for enc, who in vr:
            S.append(enc["scalars"]); F.append(enc["field"]); I.append(enc["card_idx"])
            Y.append(1.0 if who == w else -1.0)
        for ctx, am, visit, who in pr:
            pol.append((ctx, am, visit))
        if (g + 1) % 5 == 0:
            log(f"  selfplay {g+1}/{n_games}（value局面{len(Y)} policy{len(pol)}）", flush=True)
    if not S:
        return None, None
    vdata = {"scalars": np.stack(S), "field": np.stack(F),
             "card_idx": np.stack(I), "value": np.array(Y, dtype=np.float32)}
    return vdata, pol


def train_generation(vocab, vdata, pol, d_emb=24, hidden=128, v_epochs=20, p_epochs=4, seed=0, log=print):
    vnet = RN.ValueNet(len(vocab), d_emb=d_emb, hidden=hidden, feat_dim=E.feature_dim(), seed=seed)
    tm, vm = RN.train(vnet, vdata, epochs=v_epochs, lr=2e-3, batch=256, val_frac=0.05)
    pnet = PolicyScorer(hidden=hidden, seed=seed)
    ce = train_policy(pnet, pol, epochs=p_epochs, lr=2e-3)
    log(f"  train: value mse={tm:.3f}/{vm:.3f}  policy ce={ce:.3f}", flush=True)
    return vnet, pnet


# ---- クロス評価（CRN・先後交互） ----
def _agent(game, vnet, pnet, vocab, sims, c_puct):
    vf = value_fn_of(vnet, vocab); pf = priors_fn_of(pnet, vocab)
    def act(state, name, rng):
        mcts = TreeMCTS(game, value_fn=vf, priors_fn=pf, c_puct=c_puct, n_sims=sims,
                        determinize_fn=lambda s, r: game.determinize(s, name, r), rng=rng)
        move, _, _ = mcts.run(state)
        if move is None:
            legal = game.legal_actions(state)
            move = legal[0] if legal else None
        return move
    return act


def cross_eval(game, agentA, agentB, pairs, seed0=3000):
    res = {"a_win": 0, "draw": 0, "a_loss": 0}
    for i in range(pairs):
        for a_is_p1 in (True, False):
            m = game.new_game(_DB, seed0 + i)
            rng = np.random.default_rng((seed0 + i) * 7 + (0 if a_is_p1 else 1))
            steps = 0
            while game.winner(m) is None and not game.is_terminal(m) and steps < 400:
                name = game.current_player(m)
                if name is None:
                    break
                ag = agentA if (name == "p1") == a_is_p1 else agentB
                mv = ag(m, name, rng)
                if mv is None:
                    break
                try:
                    cpu_ai._apply_move_inplace(m, name, mv)
                except Exception:
                    break
                steps += 1
            w = game.winner(m)
            if w is None:
                res["draw"] += 1
            else:
                res["a_win" if ((w == "p1") == a_is_p1) else "a_loss"] += 1
    res["games"] = pairs * 2
    return res


_DB = None


def main():
    global _DB
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="疎通のみ（勝率は無視・dev用）")
    ap.add_argument("--games", type=int, default=6)
    ap.add_argument("--sims", type=int, default=30)
    ap.add_argument("--gens", type=int, default=1)
    ap.add_argument("--eval-pairs", type=int, default=3)
    ap.add_argument("--sl-net", default=None, help="Gen0 value net（無ければ乱数初期化）")
    ap.add_argument("--c-puct", type=float, default=1.5)
    args = ap.parse_args()

    _DB = _load_db()
    vocab = E.build_vocab(_DB)
    game = OPCGGame()
    rng = np.random.default_rng(0)

    # Gen0: value=SL net(or 乱数)・policy=uniform(None)。
    if args.sl_net:
        v0 = RN.ValueNet.load(args.sl_net); print(f"Gen0 value net: {args.sl_net}", flush=True)
    else:
        v0 = RN.ValueNet(len(vocab), d_emb=24, hidden=128, feat_dim=E.feature_dim(), seed=0)
    gens = [(v0, None)]   # (value_net, policy or None)

    print(f"=== P3 {'SMOKE' if args.smoke else 'RUN'}: gens={args.gens} games/gen={args.games} "
          f"sims={args.sims} ===", flush=True)
    for g in range(args.gens):
        vnet, pnet = gens[-1]
        t0 = time.perf_counter()
        vdata, pol = generate(game, value_fn_of(vnet, vocab), priors_fn_of(pnet, vocab),
                              vocab, args.games, args.sims, args.c_puct, rng)
        if vdata is None:
            print("データ0（全局未決着）"); return 1
        nv, npnet = train_generation(vocab, vdata, pol, seed=g)
        gens.append((nv, npnet))
        # クロス評価: 新世代 vs 直前世代。
        a_new = _agent(game, nv, npnet, vocab, args.sims, args.c_puct)
        a_old = _agent(game, vnet, pnet, vocab, args.sims, args.c_puct)
        r = cross_eval(game, a_new, a_old, args.eval_pairs)
        wr = (r["a_win"] + 0.5 * r["draw"]) / r["games"]
        print(f"Gen{g+1} vs Gen{g}: 勝率={wr:.3f} {r}  ({time.perf_counter()-t0:.0f}s)", flush=True)
    if args.smoke:
        print("\nSMOKE: ループ疎通OK（自己対戦→value/policy学習→クロス評価が例外なく完走）。"
              "※勝率は乱数＝判定に使わない（レビュー確定）。")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
