"""AZ部品（tictactoe/az_net/az_mcts）の単体検証＝GATE A 本走の土台が正しいことを保証。"""
import numpy as np

import conftest  # noqa: F401
import tictactoe as T
import az_net as AZ
from az_mcts import MCTS
from az_loop import play_match, net_agent, random_agent


def test_winner_and_terminal():
    g = T.TicTacToe()
    s = ((1, 1, 1, 2, 2, 0, 0, 0, 0), 1)
    assert g.winner(s) == 0 and g.is_terminal(s)
    draw = ((1, 2, 1, 1, 2, 2, 2, 1, 1), 0)
    assert g.winner(draw) is None and g.is_terminal(draw)
    start = g.initial_state()
    assert not g.is_terminal(start) and len(g.legal_actions(start)) == 9


def test_perfect_player_never_loses():
    """ミニマックス完全プレイ: 自己対戦は必ず引き分け・ランダムに負けない（オラクルの健全性）。"""
    g = T.TicTacToe()
    rng = np.random.default_rng(0)
    perfect = lambda s, r: T.perfect_action(g, s, r)
    res = play_match(g, perfect, perfect, 20, rng)
    assert res["a_win"] == 0 and res["a_loss"] == 0, f"完全vs完全が引分でない: {res}"
    res2 = play_match(g, perfect, random_agent(g), 60, rng)
    assert res2["a_loss"] == 0, f"完全がランダムに負けた: {res2}"


def test_aznet_overfits_tiny():
    """tiny集合を value/policy 両方過学習できる＝forward/backprop/Adam が正しい。"""
    rng = np.random.default_rng(1)
    n, A, F = 40, 9, 18
    X = rng.standard_normal((n, F))
    Y = rng.choice([-1.0, 1.0], size=n)
    # 鋭い(one-hot)ターゲット＝正しく学習できれば CE→0（soft乱数だと最小CE=ターゲットのエントロピーで頭打ち）。
    cls = rng.integers(0, A, size=n)
    P = np.eye(A)[cls]
    net = AZ.AZNet(F, A, hidden=64, seed=0)
    vm, ce = AZ.train(net, {"X": X, "policy": P, "value": Y},
                      epochs=400, lr=3e-3, batch=40, seed=0)
    assert vm < 0.05, f"value を過学習できない (v_mse={vm:.3f})"
    assert ce < 0.3, f"policy を過学習できない (p_ce={ce:.3f})"


def test_mcts_takes_winning_move():
    """終端報酬が伝播し、未学習netでも sims を増やせば即勝ち手を選ぶ＝探索が正しい。"""
    g = T.TicTacToe()
    net = AZ.AZNet(g.feat_dim, g.n_actions, hidden=32, seed=2)
    state = ((1, 1, 0, 2, 2, 0, 0, 0, 0), 0)   # player0 が cell2 で 1,1,1 勝ち
    mcts = MCTS(g, net, c_puct=1.5, n_sims=300, rng=np.random.default_rng(0))
    counts = mcts.run(state, add_noise=False)
    assert int(np.argmax(counts)) == 2, f"即勝ち手(2)を選べない: {counts}"


def test_mcts_blocks_opponent_win():
    g = T.TicTacToe()
    net = AZ.AZNet(g.feat_dim, g.n_actions, hidden=32, seed=3)
    state = ((1, 1, 0, 2, 0, 0, 0, 0, 0), 1)   # player1 は cell2 で相手の勝ちを防ぐ
    mcts = MCTS(g, net, c_puct=1.5, n_sims=400, rng=np.random.default_rng(0))
    counts = mcts.run(state, add_noise=False)
    assert int(np.argmax(counts)) == 2, f"相手の即勝ちをブロックできない: {counts}"
