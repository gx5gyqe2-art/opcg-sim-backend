"""P3部品（opcg_action / az_policy）の高速単体検証（CI内）。重い loop は p3_loop.py --smoke。"""
import numpy as np

import conftest  # noqa: F401
import rl_encoder as E
from opcg_game import OPCGGame
from cpu_selfplay import _load_db
import opcg_action as A
from az_policy import PolicyScorer, state_context, train_policy


def test_action_features_shape_and_onehot():
    db = _load_db(); game = OPCGGame()
    m = game.new_game(db, 1)
    name = game.current_player(m)
    legal = game.legal_actions(m)
    mat = A.legal_action_matrix(m, legal, name)
    assert mat.shape == (len(legal), A.ACTION_DIM), "action 行列の形が不正"
    # 各手は action_type の one-hot を1つ持つ（既知型のみ）。
    for mv, row in zip(legal, mat):
        at = mv.get("action_type")
        if at in A._AT_IDX:
            assert row[A._AT_IDX[at]] == 1.0, f"action_type one-hot 不正: {at}"


def test_action_key_distinguishes_moves():
    db = _load_db(); game = OPCGGame()
    m = game.new_game(db, 2)
    legal = game.legal_actions(m)
    keys = [A.action_key(mv) for mv in legal]
    assert len(set(keys)) == len(keys), "異なる合法手が同一キーに衝突（hash同一性が壊れる）"
    for k in keys:
        hash(k)   # hashable であること


def test_action_features_deterministic():
    db = _load_db(); game = OPCGGame()
    m = game.new_game(db, 3)
    name = game.current_player(m)
    legal = game.legal_actions(m)
    a = A.legal_action_matrix(m, legal, name)
    b = A.legal_action_matrix(m, legal, name)
    assert np.array_equal(a, b), "action 特徴が非決定的"


def test_policy_priors_normalized():
    pol = PolicyScorer(ctx_dim=E.feature_dim(), hidden=32, seed=0)
    ctx = np.random.default_rng(0).standard_normal(E.feature_dim())
    am = np.random.default_rng(1).standard_normal((5, A.ACTION_DIM))
    p = pol.priors(ctx, am)
    assert p.shape == (5,) and abs(p.sum() - 1.0) < 1e-9 and (p >= 0).all()


def test_policy_overfits_single_sample():
    """1局面でターゲット手に質量を寄せられる＝forward/softmax-CE/backprop が正しい。"""
    rng = np.random.default_rng(2)
    ctx = rng.standard_normal(E.feature_dim())
    am = rng.standard_normal((6, A.ACTION_DIM))
    target = np.zeros(6); target[3] = 1.0
    pol = PolicyScorer(ctx_dim=E.feature_dim(), hidden=32, seed=0)
    ce0 = -np.log(pol.priors(ctx, am)[3] + 1e-9)
    train_policy(pol, [(ctx, am, target)], epochs=300, lr=5e-3)
    p = pol.priors(ctx, am)
    assert int(p.argmax()) == 3, f"ターゲット手に収束しない: {p}"
    assert -np.log(p[3] + 1e-9) < ce0, "CE が下がっていない"


def test_state_context_dim():
    db = _load_db(); game = OPCGGame(); vocab = E.build_vocab(db)
    m = game.new_game(db, 4)
    ctx = state_context(m, game.current_player(m), vocab)
    assert ctx.shape == (E.feature_dim(),), "状態文脈の次元が feature_dim と不一致"
