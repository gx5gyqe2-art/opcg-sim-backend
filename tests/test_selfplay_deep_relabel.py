"""v6 深探索再ラベル（`p3_loop.selfplay_game` の relabel_frac/relabel_sims）の機構健全性。

docs/reports/v5_adoption_20260715.md §4-2・mark_deep_probe_20260715.md（PRIOR_BOUND 対策）。
各決定点を確率 relabel_frac で「深い sims × prior 平坦化 × ノイズ無し」の教師探索にかけ、
**policy 教師（訪問分布）だけ**を差し替える。ここでは学習効果でなく配管を固定する:

  1. relabel ON の自己対戦が完走し、policy 教師が正規（Σ=1・legal と同数）のまま。
  2. 同一 seed → 同一結果（決定論＝再現可能な生成）。
  3. relabel_frac=0（既定）は乱数を追加消費しない＝従来の生成列と完全一致（後方互換）。
"""
import numpy as np
import pytest

import conftest  # noqa: F401
import p3_loop as P
import rl_encoder as E
import rl_net as RN
from cpu_selfplay import _load_db
from opcg_game import OPCGGame

pytestmark = pytest.mark.cpu_infra   # 基盤健全性（自己対戦生成の内部機構）


@pytest.fixture(scope="module")
def env():
    db = _load_db()
    vocab = E.build_vocab(db)
    vnet = RN.ValueNet(len(vocab), d_emb=8, hidden=16, feat_dim=E.feature_dim(1), seed=0)
    game = OPCGGame(prune_futile=False)   # 生成想定（v6 柱⑤）
    return db, vocab, P.value_fn_of(vnet, vocab, 1), game


def _run(env, seed, relabel_frac, relabel_sims, max_steps=25):
    db, vocab, vf, game = env
    rng = np.random.default_rng(seed)
    return P.selfplay_game(game, vf, None, vocab, sims=8, c_puct=1.5, rng=rng,
                           max_steps=max_steps, enc_version=1, db=db,
                           relabel_frac=relabel_frac, relabel_sims=relabel_sims)


def test_relabel_on_yields_valid_policy_targets(env):
    """relabel 全点適用（frac=1.0）でも自己対戦が完走し、教師分布が正規のまま。"""
    vr, pr, _w = _run(env, seed=11, relabel_frac=1.0, relabel_sims=16)
    assert pr, "policy 教師が採れていない"
    for ctx, am, visit in ((c, a, v) for c, a, v, _ in pr):
        assert visit.shape[0] == am.shape[0], "教師分布と合法手行列の件数不一致"
        assert abs(float(visit.sum()) - 1.0) < 1e-6, "教師分布が正規化されていない"
        assert float(visit.min()) >= 0.0


def test_relabel_is_deterministic(env):
    """同一 seed → 同一の教師分布列（決定論＝生成の再現可能性を壊さない）。"""
    _, pr1, _ = _run(env, seed=23, relabel_frac=1.0, relabel_sims=16)
    _, pr2, _ = _run(env, seed=23, relabel_frac=1.0, relabel_sims=16)
    assert len(pr1) == len(pr2)
    for (_, _, v1, _), (_, _, v2, _) in zip(pr1, pr2):
        assert np.array_equal(v1, v2)


def test_relabel_off_is_backward_compatible(env):
    """frac=0（既定）は判定乱数を引かない＝relabel 引数の有無で生成列が変わらない（後方互換）。"""
    db, vocab, vf, game = env
    rng1 = np.random.default_rng(31)
    vr1, pr1, _ = P.selfplay_game(game, vf, None, vocab, sims=8, c_puct=1.5, rng=rng1,
                                  max_steps=25, enc_version=1, db=db)   # 引数なし＝従来呼び出し
    vr2, pr2, _ = _run(env, seed=31, relabel_frac=0.0, relabel_sims=1600)
    assert len(vr1) == len(vr2) and len(pr1) == len(pr2)
    for (_, _, v1, _), (_, _, v2, _) in zip(pr1, pr2):
        assert np.array_equal(v1, v2)
