"""P3 ループの疎通検証（slow・CI除外）: 自己対戦→value/policy学習→クロス評価が例外なく完走。

レビュー確定どおり**勝率シグナルは見ない**（数局モデルの勝率は乱数）。本テストの目的は
「RLループ機械が OPCG 上で end-to-end に動く（テンソル不整合・パイプライン詰まりが無い）」のみ。
実行: OPCG_LOG_SILENT=1 PYTHONPATH=tests python -m pytest tests/test_p3_loop.py -q -s -m slow
"""
import numpy as np
import pytest

import conftest  # noqa: F401
import rl_encoder as E
import rl_net as RN
from opcg_game import OPCGGame
from cpu_selfplay import _load_db
import p3_loop as P


@pytest.mark.slow
def test_p3_loop_runs_end_to_end():
    P._DB = _load_db()
    vocab = E.build_vocab(P._DB)
    game = OPCGGame()
    rng = np.random.default_rng(0)
    v0 = RN.ValueNet(len(vocab), d_emb=24, hidden=128, feat_dim=E.feature_dim(), seed=0)

    # 自己対戦データ採取（uniform prior）。
    vdata, pol = P.generate(game, P.value_fn_of(v0, vocab), None, vocab,
                            n_games=2, sims=8, c_puct=1.5, rng=rng, log=lambda *a, **k: None)
    assert vdata is not None and len(vdata["value"]) > 0, "value データが採れない"
    assert len(pol) == len(vdata["value"]), "policy サンプル数が value 局面数と不一致"
    assert pol[0][1].shape[1] == P.legal_action_matrix.__globals__["ACTION_DIM"] \
        or pol[0][1].shape[1] > 0, "action 行列の次元が不正"

    # value+policy 学習。
    vnet, pnet = P.train_generation(vocab, vdata, pol, v_epochs=3, p_epochs=2,
                                    seed=0, log=lambda *a, **k: None)
    # 学習後 net が MCTS の value/priors として機能する。
    af = P.priors_fn_of(pnet, vocab)
    m = game.new_game(P._DB, 7)
    legal = game.legal_actions(m)
    pri = af(m, legal)
    assert pri is not None and abs(pri.sum() - 1.0) < 1e-6, "policy prior が正規化されない"

    # クロス評価が完走（勝率は判定に使わない＝疎通のみ）。
    a_new = P._agent(game, vnet, pnet, vocab, sims=8, c_puct=1.5)
    a_old = P._agent(game, v0, None, vocab, sims=8, c_puct=1.5)
    r = P.cross_eval(game, a_new, a_old, pairs=1)
    assert r["games"] == 2 and sum(r[k] for k in ("a_win", "draw", "a_loss")) == 2
