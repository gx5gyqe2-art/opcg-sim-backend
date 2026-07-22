"""P3部品（opcg_action / az_policy）の単体検証。重い loop は p3_loop.py --smoke。"""
import numpy as np

import conftest  # noqa: F401
import pytest
import rl_encoder as E
import rl_net as RN
from opcg_game import OPCGGame
from cpu_selfplay import _load_db
import opcg_action as A
from az_policy import PolicyScorer, state_context, train_policy

pytestmark = pytest.mark.cpu_infra


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


def test_v10_appendonly_dims():
    """cpu_v10 append-only ラチェット: encoder v5=55（v4=51 は不変）・action ACTION_DIM=26。
    既存版の次元が変わる＝並べ替え/削除＝温スタートの恒等性が壊れるので検出する。"""
    assert E.scalars_dim(4) == 51, "v4 の次元が変わった（append-only 違反）"
    assert E.scalars_dim(5) == 55, "v5 = v4 + 相手場集約3 + 展開余力1"
    assert A.ACTION_DIM == 26, "v10 = 既存25 + ATTACH_DON 付与後パワー1"
    assert E._opp_field_aggregate([]) == [0.0, 0.0, 0.0], "空場は脅威ゼロ"
    db = _load_db(); game = OPCGGame(); m = game.new_game(db, 1)
    name = game.current_player(m); vocab = E.build_vocab(db)
    assert E.encode(m, name, vocab, version=5)["scalars"].shape[0] == 55
    # 温スタート拡張は焼き込み vocab を引き継ぐ（欠落すると serve が build_vocab へ
    # フォールバックし、DB 増加の途中挿入で index がズレ＝候補評価が壊れる・2026-07-22 実害）
    v = RN.ValueNet.load("opcg_sim/data/learned/gen5_value.npz")
    assert v.vocab_ids, "gen5 は vocab 焼き込み済みのはず"
    assert v.expanded(E.scalars_dim(4), 4).vocab_ids == list(v.vocab_ids), \
        "expanded() が vocab_ids を落としている（serve で index ズレ）"


def test_state_context_dim():
    db = _load_db(); game = OPCGGame(); vocab = E.build_vocab(db)
    m = game.new_game(db, 4)
    ctx = state_context(m, game.current_player(m), vocab)
    assert ctx.shape == (E.feature_dim(),), "状態文脈の次元が feature_dim と不一致"


def test_new_game_leader_rotation_uses_pool_and_realistic_decks():
    """穴B: new_game(leaders=...) は指定プールから両席のリーダーを抽選し、リアルデッキで組む。

    固定1リーダーのミラー戦だと【ドン‼×1】系リーダー効果（OP11-041 ナミの防御+2000 等）が
    自己対戦データに一度も現れず、v2 再学習でも学べない。ローテーションで盤面分布を広げる。
    """
    from deckgen import all_leader_ids
    db = _load_db(); game = OPCGGame()
    pool = all_leader_ids(db)
    assert len(pool) > 1
    # 同一 seed は決定論（CRN）。
    m_a = game.new_game(db, 7, leaders=pool)
    m_b = game.new_game(db, 7, leaders=pool)
    assert m_a.p1.leader.master.card_id == m_b.p1.leader.master.card_id
    assert m_a.p2.leader.master.card_id == m_b.p2.leader.master.card_id
    # 抽選されたリーダーはプール内。デッキは50枚（リアルデッキ）。
    assert m_a.p1.leader.master.card_id in pool
    assert m_a.p2.leader.master.card_id in pool
    # seed を変えるとリーダー組み合わせが分布する（複数 seed で2種以上の p1 リーダー）。
    seen = {game.new_game(db, s, leaders=pool).p1.leader.master.card_id for s in range(12)}
    assert len(seen) >= 2, "リーダーが分布していない（ローテーション不発）"


def test_new_game_default_is_backward_compatible():
    """leaders 未指定は従来挙動（build_deck・先頭リーダー固定）＝既存テスト/スモーク互換。"""
    db = _load_db(); game = OPCGGame()
    a = game.new_game(db, 3)
    b = game.new_game(db, 3)
    assert a.p1.leader.master.card_id == b.p1.leader.master.card_id
    # 未指定では両席とも同一（先頭）リーダー＝従来のミラー。
    assert a.p1.leader.master.card_id == a.p2.leader.master.card_id
