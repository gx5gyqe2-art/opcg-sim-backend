"""P2 harness の単体検証。重い対L1対戦は p2_gen0.py（手動・外部規模）。

学習や対戦はせず、SL価値の配線（encode→net→[-1,1]）と SL-MCTSエージェントが合法手を返すことを確認。
"""
import numpy as np

import conftest  # noqa: F401
import pytest
import rl_encoder as E
import rl_net as RN
from opcg_game import OPCGGame
from cpu_selfplay import _load_db
from p2_gen0 import sl_value, mcts_sl_agent, match, l1_agent_factory

pytestmark = pytest.mark.cpu_infra


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


def test_match_leaders_param_rotates_and_is_backward_compatible():
    """p3_vs_l1.py の --rotate-leaders 配管の土台: match(leaders=...) が new_game へ伝播する。

    未指定は従来どおり固定リーダー（後方互換）。指定時はプール抽選＋リアルデッキ化＝
    p3_run --rotate-leaders で学習した場合に評価分布を揃えられる。
    """
    from deckgen import all_leader_ids
    db = _load_db()
    vocab = E.build_vocab(db)
    game = OPCGGame()
    net = RN.ValueNet(len(vocab), d_emb=8, hidden=16, feat_dim=E.feature_dim(), seed=0)
    agent = mcts_sl_agent(game, sl_value(game, net, vocab), sims=4)
    l1_factory = lambda: l1_agent_factory("easy", pimc_worlds=1)

    r0 = match(game, db, agent, l1_factory, pairs=1, log=lambda *a, **k: None)
    assert r0["games"] == 2

    pool = all_leader_ids(db)
    r1 = match(game, db, agent, l1_factory, pairs=1, log=lambda *a, **k: None, leaders=pool)
    assert r1["games"] == 2
