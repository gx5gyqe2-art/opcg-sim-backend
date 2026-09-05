"""N系 c10（効果構造符号化ネット・2026-09-03 採用・2026-09-05 に既定を a1 へ譲りロールバック先）の
serve 配線契約。既定（a1・NRel）の契約は `tests/test_nrel_default.py`。

**必須テスト**（`cpu_infra` ではない）: ここが壊れるとロールバック先の CPU がクラッシュする／
黙って別のネットで打つ（誤評価）ため、ゲームプレイ退行そのものを見逃す。

守る性質:
  1. `LearnedEngine(value_path=_C10_VALUE)` は同梱 `neff_c10.npz` を N系として読む（`vnet` はアダプタ・
     `pnet` 無し・`priors_override` あり・符号化 v12・出口ヘッド無し）。
  2. 同梱 npz は**ネット付属 vocab_ids** を持ち、gen15 系譜（N系の訓練 vocab）と同一。
     vocab_ids を持たない旧 N系 npz は同梱既定の vocab_ids へフォールバックする
     （`build_vocab` の現行 DB ソートには落とさない＝2026-07-15 の索引ズレ事故を再発させない）。
  3. アダプタの `predict`/`predict_exit` は `NEffNet.value` と一致（tanh 範囲・出口ヘッド無し）。
  4. 方策 priors は合法手上で確率（非負・和1・長さ一致）。
  5. `decide` は合法手を返し、同一 seed で決定論（同じ npz を明示パスで読んでも同じ手）。
  6. 表の前計算はプロセス内でパス単位に共有される（並列アリーナで重複しない）。
  7. G15 ペアを明示パスで読む経路（ロールバック先）は従来どおり G系配線のまま。
  8. `is_neff_npz` の判別: N系 npz は True・G系（gen15_value）は False。
"""
import argparse
import os
import random

import numpy as np
import pytest

import conftest  # noqa: F401
import _bootstrap  # noqa: F401

import coach_gate as CG
import counterfactual_referee as CR
import replay_reeval as RE
from cpu_selfplay import _load_db
from opcg_sim.src.core import cpu_learned as CL
from opcg_sim.src.core.cpu_learned import LearnedEngine
from opcg_sim.src.learned import encoder as E
from opcg_sim.src.learned import n_eff as NE


@pytest.fixture(scope="module")
def db():
    return _load_db()


@pytest.fixture(scope="module")
def m2(db):
    CR.ARGS = argparse.Namespace(true_board=True)
    CR.GAMES = {}
    raw = RE.load_replay_json(CG.REPLAYS_V2["m2"])
    rec = raw.get("replay", raw)
    CR.GAMES["m2"] = (rec, {f.get("action_index"): f for f in raw.get("frames") or []},
                      rec["actions"])
    return db


def _board(db, i):
    m, who = CR._restore_board(db, "m2", i)
    m.action_events = []
    name = who if isinstance(who, str) else who.name
    return m, name, (m.p1 if m.p1.name == name else m.p2)


def _batch(m, name, vocab):
    enc = E.encode(m, name, vocab, version=12)
    return {k: enc[k][None, ...] for k in ("scalars", "field", "card_idx")}


def _c10():
    return LearnedEngine(value_path=CL._C10_VALUE)


def test_c10_is_neff_and_rollback_target():
    assert os.path.basename(CL._C10_VALUE) == "neff_c10.npz" and CL._DEFAULT_POLICY is None
    assert NE.is_neff_npz(CL._C10_VALUE) and not NE.is_neff_npz(CL._G15_VALUE)
    assert not NE.is_neff_npz(CL._DEFAULT_VALUE), "既定は a1（NRel）＝N系 c の判別に掛からない"
    eng = _c10()
    assert isinstance(eng.vnet, NE.NEffValueAdapter)
    assert eng.pnet is None and eng.priors_override is not None
    assert eng.enc_version == 12 and eng.vnet.feat_dim == E.feature_dim(12)
    assert not eng.vnet.has_exit_head("battle") and not eng.vnet.has_exit_head("turn")
    assert eng._exit_value_fn("battle") is None
    assert CL.available()


def test_bundled_npz_carries_gen15_vocab_ids():
    from opcg_sim.src.learned.value_net import ValueNet
    ids = NE.default_vocab_ids()
    g15 = ValueNet.load(CL._G15_VALUE)
    assert ids == list(g15.vocab_ids), "c10 の vocab_ids は訓練時（gen15 系譜）と同一のはず"
    eng = _c10()
    assert eng.vnet.vocab_ids == ids
    assert max(eng.vocab.values()) == len(ids) == eng.vnet.tab.shape[0] - 1


def test_npz_without_vocab_ids_falls_back_to_default_ids(tmp_path):
    """旧 N系 npz（vocab_ids 無し）は同梱既定の vocab_ids で読む＝現行 DB ソートへは落ちない。"""
    d = np.load(CL._C10_VALUE, allow_pickle=True)
    p = str(tmp_path / "old_style.npz")
    np.savez_compressed(p, **{k: d[k] for k in d.files if k not in ("vocab_ids", "meta")})
    assert NE.is_neff_npz(p)
    eng = LearnedEngine(value_path=p)
    ref = _c10()
    assert eng.vocab == ref.vocab
    assert np.array_equal(eng.vnet.tab, ref.vnet.tab)
    # 現行 DB ソートとは違う（DB は訓練後に増えている）
    assert eng.vocab != E.build_vocab(CL._SHARED["db"])


def test_adapter_matches_net_value_and_has_no_exit_head(m2):
    eng = _c10()
    for i in (10, 30, 50):
        m, name, _ = _board(m2, i)
        b = _batch(m, name, eng.vocab)
        v = eng.vnet.predict(b)
        ci = np.asarray(b["card_idx"])
        if ci.shape[1] < NE.MAX_CI:
            ci = np.concatenate([ci, np.zeros((1, NE.MAX_CI - ci.shape[1]), ci.dtype)], 1)
        direct = eng.vnet.net.value(np.asarray(b["scalars"], np.float32), ci[:, :NE.MAX_CI])
        assert np.allclose(v, direct) and -1.0 <= float(v[0]) <= 1.0
        assert np.array_equal(eng.vnet.predict_exit(b, "battle"), v)
        va, aux = eng.vnet.predict_with_aux(b)
        assert np.array_equal(va, v) and float(aux[0]) == 0.0


def test_priors_are_probabilities_over_legal(m2):
    eng = _c10()
    seen = 0
    for i in (10, 30, 50, 70):
        m, name, _ = _board(m2, i)
        pa = m.pending_actor_action()
        if not pa:
            continue
        legal = eng.game.legal_actions(m)
        if not legal:
            continue
        p = eng._priors()(m, legal)
        assert p is not None and p.shape == (len(legal),)
        assert float(p.min()) >= 0.0 and abs(float(p.sum()) - 1.0) < 1e-4
        seen += 1
    assert seen >= 1


def test_decide_is_legal_and_deterministic_across_load_paths(m2, tmp_path):
    """同梱パスのロードとコピーのロード（同じ重み）は同一 seed で同一手。"""
    d = np.load(CL._C10_VALUE, allow_pickle=True)
    p = str(tmp_path / "copy.npz")
    np.savez_compressed(p, **{k: d[k] for k in d.files})
    a, b = _c10(), LearnedEngine(value_path=p)
    for i in (10, 50):
        m, name, actor = _board(m2, i)
        mvs = []
        for e in (a, b):
            random.seed(20260903)
            e._world_seeds = {}
            e._commits.clear()
            mvs.append(e.decide(m, actor, sims=8, rng=np.random.default_rng(i)))
        assert mvs[0] is not None and mvs[0].get("action_type")
        assert mvs[0] == mvs[1]


def test_tables_shared_per_path_but_vnet_per_engine():
    """表・重みはプロセス内共有（重複計算なし）、`vnet` はエンジンごとの別インスタンス
    （net-vs-net で席ごとに差し替えられる LearnedEngine の契約）。"""
    a, b = _c10(), _c10()
    assert a.vnet is not b.vnet
    assert a.vnet.net is b.vnet.net and a.vnet.tab is b.vnet.tab
    assert a.priors_override is b.priors_override


def test_g15_pair_explicit_path_keeps_g_wiring():
    from opcg_sim.src.learned.value_net import ValueNet
    eng = LearnedEngine(value_path=CL._G15_VALUE, policy_path=CL._G15_POLICY)
    assert isinstance(eng.vnet, ValueNet) and eng.pnet is not None
    assert eng.priors_override is None and eng.enc_version == 12
