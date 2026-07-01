"""AZ 自己対戦→学習→世代反復ループ＋クロス評価（GATE A〜パイロット共通部品）。

docs/.../cpu_rl_pilot_plan_20260629.md §3。自己対戦で (局面, MCTS訪問分布, 手番) を採取し、
最終勝敗を value 教師に。Dual-Net を学習し世代を進める。評価は固定相手/他世代との対戦勝率。
"""
import numpy as np

from az_mcts import MCTS
import az_net as AZ


def _sample(counts, rng, temp):
    """訪問分布から手をサンプル（temp>0）or argmax（temp==0）。"""
    if temp <= 1e-6:
        return int(np.argmax(counts))
    p = counts ** (1.0 / temp)
    s = p.sum()
    if s <= 0:
        legal = np.flatnonzero(counts >= 0)
        return int(rng.choice(legal))
    return int(rng.choice(len(p), p=p / s))


def selfplay_game(game, net, rng, n_sims, c_puct, temp_moves=4):
    """1ゲーム自己対戦。返り値: [(encoded, visit_policy, player)], winner。"""
    mcts = MCTS(game, net, c_puct=c_puct, n_sims=n_sims, rng=rng)
    state = game.initial_state()
    recs, ply = [], 0
    while not game.is_terminal(state):
        counts = mcts.run(state, add_noise=True)
        if counts.sum() == 0:
            break
        recs.append((game.encode(state), counts.copy(), game.current_player(state)))
        a = _sample(counts, rng, temp=1.0 if ply < temp_moves else 0.0)
        state = game.apply(state, a)
        ply += 1
    winner = game.winner(state)
    return recs, winner


def generate_data(game, net, n_games, rng, n_sims, c_puct, temp_moves=4):
    X, Pi, Y = [], [], []
    for _ in range(n_games):
        recs, winner = selfplay_game(game, net, rng, n_sims, c_puct, temp_moves)
        for enc, pol, player in recs:
            X.append(enc); Pi.append(pol)
            Y.append(0.0 if winner is None else (1.0 if winner == player else -1.0))
    if not X:
        return None
    return {"X": np.stack(X).astype(np.float64),
            "policy": np.stack(Pi).astype(np.float64),
            "value": np.array(Y, dtype=np.float64)}


def play_match(game, agent_a, agent_b, n_games, rng):
    """agent_x(state, rng)->action。a を先手/後手 交互に。返り値 dict(a_win,draw,a_loss)。"""
    res = {"a_win": 0, "draw": 0, "a_loss": 0}
    for g in range(n_games):
        a_first = (g % 2 == 0)
        state = game.initial_state()
        while not game.is_terminal(state):
            p = game.current_player(state)
            a_to_move = (p == 0) == a_first
            act = (agent_a if a_to_move else agent_b)(state, rng)
            state = game.apply(state, act)
        w = game.winner(state)
        if w is None:
            res["draw"] += 1
        else:
            a_is_winner = (w == 0) == a_first
            res["a_win" if a_is_winner else "a_loss"] += 1
    return res


def net_agent(game, net, n_sims, c_puct):
    """評価用エージェント（決定的＝ノイズ無し・argmax）。"""
    def act(state, rng):
        mcts = MCTS(game, net, c_puct=c_puct, n_sims=n_sims, rng=rng)
        counts = mcts.run(state, add_noise=False)
        return int(np.argmax(counts))
    return act


def random_agent(game):
    def act(state, rng):
        legal = game.legal_actions(state)
        return int(legal[rng.integers(len(legal))])
    return act


def run_generations(game, gens, games_per_gen, n_sims, c_puct,
                    hidden=64, epochs=12, lr=2e-3, seed=0, log=print):
    """世代ループ。各世代の net と学習データ規模を返す。"""
    rng = np.random.default_rng(seed)
    net = AZ.AZNet(game.feat_dim, game.n_actions, hidden=hidden, seed=seed)
    nets = [_clone(net)]   # gen0 = 未学習
    for g in range(gens):
        data = generate_data(game, net, games_per_gen, rng, n_sims, c_puct)
        if data is None:
            log(f"gen{g+1}: データ0"); continue
        vm, ce = AZ.train(net, data, epochs=epochs, lr=lr, seed=seed + g)
        log(f"gen{g+1}: 局面{len(data['value'])}  v_mse={vm:.3f} p_ce={ce:.3f}")
        nets.append(_clone(net))
    return nets


def _clone(net):
    import copy
    return copy.deepcopy(net)
