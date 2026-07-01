"""P2 harness の高速単体検証（CI内）。重い対L1対戦は p2_gen0.py（手動・外部規模）。

学習や対戦はせず、SL価値の配線（encode→net→[-1,1]）と SL-MCTSエージェントが合法手を返すことを確認。
"""
import numpy as np

import conftest  # noqa: F401
import rl_encoder as E
import rl_net as RN
from opcg_game import OPCGGame
from cpu_selfplay import _load_db
from p2_gen0 import sl_value, mcts_sl_agent


def test_sl_value_in_range():
    db = _load_db()
    vocab = E.build_vocab(db)
    game = OPCGGame()
    net = RN.ValueNet(len(vocab), d_emb=24, hidden=128, feat_dim=E.feature_dim(), seed=0)
    m = game.new_game(db, 1)
    v = sl_value(game, net, vocab)(m, game.current_player(m))
    assert -1.0 <= v <= 1.0, f"SL価値が範囲外: {v}"


def test_sl_mcts_agent_returns_legal_move():
    db = _load_db()
    vocab = E.build_vocab(db)
    game = OPCGGame()
    net = RN.ValueNet(len(vocab), d_emb=24, hidden=128, feat_dim=E.feature_dim(), seed=0)
    m = game.new_game(db, 2)
    name = game.current_player(m)
    legal = game.legal_actions(m)
    agent = mcts_sl_agent(game, sl_value(game, net, vocab), sims=8)
    move = agent(m, name, np.random.default_rng(0))
    assert move in legal, "SL-MCTS が合法手を返さない"


def test_net_save_load_roundtrip(tmp_path):
    db = _load_db()
    vocab = E.build_vocab(db)
    net = RN.ValueNet(len(vocab), d_emb=24, hidden=128, feat_dim=E.feature_dim(), seed=0)
    p = str(tmp_path / "n.npz")
    net.save(p)
    net2 = RN.ValueNet.load(p)
    game = OPCGGame()
    m = game.new_game(db, 3)
    enc = E.encode(m, game.current_player(m), vocab)
    batch = {k: enc[k][None, ...] for k in ("scalars", "field", "card_idx")}
    assert abs(float(net.predict(batch)[0]) - float(net2.predict(batch)[0])) < 1e-9, "save/load で値が変わる"
