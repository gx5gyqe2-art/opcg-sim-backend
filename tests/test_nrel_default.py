"""出荷既定 CPU＝NRel a1（N系 v3・R 遮断・相手デッキ知識あり・2026-09-05 採用）の serve 配線契約。

**必須テスト**（`cpu_infra` ではない）: ここが壊れると実プレイの CPU がクラッシュする／
黙って別のネットで打つ（誤評価）ため、ゲームプレイ退行そのものを見逃す。

守る性質:
  1. `LearnedEngine()` は同梱 `nrel_a1.npz` を NRel として読む（`vnet`=`NRelValueAdapter`・`pnet` 無し・
     `priors_override` あり・符号化 v13・出口ヘッド無し）。
  2. 同梱 npz は `ablate=["rel"]`（ユーザ決定 2026-09-05「R なし・相手デッキ知識は入れる」）＝serve は
     関係 R を計算せず、相手デッキ知識の列（`OPP_POOL_COLS`）は生きている。
  3. ネット付属 vocab_ids は c10／gen15 系譜と同一（訓練時 idx の固定・2026-07-15 の索引ズレ対策）。
  4. 葉価値は `predict_state`（盤面から直接・tanh 範囲）・priors は合法手上の確率。
  5. `decide` は合法手を返し、同一 seed で決定論（同じ npz を明示パスで読んでも同じ手）。
  6. 表の前計算はプロセス内でパス単位に共有され、`vnet` はエンジンごとの別インスタンス。
  7. ロールバック先 c10（`_C10_VALUE`）は N系 c の配線のまま（`tests/test_neff_default.py`）。
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
from opcg_sim.src.learned import n_rel as NL


@pytest.fixture(scope="module")
def m2():
    db = _load_db()
    CR.ARGS = argparse.Namespace(true_board=True)
    CR.GAMES = {}
    raw = RE.load_replay_json(CG.REPLAYS_V2["m2"])
    rec = raw.get("replay", raw)
    CR.GAMES["m2"] = (rec, {f.get("action_index"): f for f in raw.get("frames") or []}, rec["actions"])
    return db


def _board(db, i):
    m, who = CR._restore_board(db, "m2", i)
    m.action_events = []
    name = who if isinstance(who, str) else who.name
    return m, name, (m.p1 if m.p1.name == name else m.p2)


def test_default_is_nrel_a1():
    assert os.path.basename(CL._DEFAULT_VALUE) == "nrel_a1.npz" and CL._DEFAULT_POLICY is None
    assert NL.is_nrel_npz(CL._DEFAULT_VALUE) and not NE.is_neff_npz(CL._DEFAULT_VALUE)
    assert CL.available()
    eng = LearnedEngine()
    assert isinstance(eng.vnet, NL.NRelValueAdapter)
    assert eng.pnet is None and eng.priors_override is not None
    assert eng.enc_version == 13 and eng.vnet.feat_dim == E.feature_dim(13)
    assert not eng.vnet.has_exit_head("battle") and eng._exit_value_fn("battle") is None


def test_bundled_a1_ablates_rel_but_keeps_opp_pool(m2):
    eng = LearnedEngine()
    assert eng.vnet.net.ablate == {"rel"} and eng.vnet.net.meta.get("ablate") == ["rel"]
    assert eng.vnet.net.meta.get("kind") == "nrel-a"
    m, name, _ = _board(m2, 30)
    sc, ci, tok, rel_om, rel_oo, _R = eng.vnet.encode_state(m, name)
    assert not rel_om.any() and not rel_oo.any(), "R 遮断＝serve は関係を計算しない"
    # 相手デッキ知識（未見プールの要約列）は入力に残り、値を変えると評価が変わる
    v0 = eng.vnet.predict_state(m, name)
    sc2 = sc.copy(); sc2[:, list(NL.OPP_POOL_COLS)] += 0.5
    h, present = eng.vnet.net.tokens_forward(ci, tok, rel_om, rel_oo, eng.vnet.tab)
    e2 = eng.vnet.net.body(sc2, h, present)
    v2 = float(np.tanh((e2 @ eng.vnet.net.Wv + eng.vnet.net.bv)[0, 0]))
    assert -1.0 <= v0 <= 1.0 and abs(v0 - v2) > 1e-6


def test_bundled_a1_carries_c10_vocab_ids():
    eng = LearnedEngine()
    ids = NE.default_vocab_ids()                      # c10（gen15 系譜）の vocab_ids
    assert eng.vnet.vocab_ids == ids
    assert max(eng.vocab.values()) == len(ids) == eng.vnet.tab.shape[0] - 1
    assert eng.vocab.get("PRB01-001") == 2282 and "ST31-001" not in eng.vocab


def test_priors_are_probabilities_over_legal(m2):
    eng = LearnedEngine()
    seen = 0
    for i in (10, 30, 50, 70):
        m, name, _ = _board(m2, i)
        if not m.pending_actor_action():
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
    d = np.load(CL._DEFAULT_VALUE, allow_pickle=True)
    p = str(tmp_path / "copy.npz")
    np.savez_compressed(p, **{k: d[k] for k in d.files})
    a, b = LearnedEngine(), LearnedEngine(value_path=p)
    for i in (10, 50):
        m, name, actor = _board(m2, i)
        mvs = []
        for e in (a, b):
            random.seed(20260905)
            e._world_seeds = {}
            e._commits.clear()
            mvs.append(e.decide(m, actor, sims=8, rng=np.random.default_rng(i)))
        assert mvs[0] is not None and mvs[0].get("action_type")
        assert mvs[0] == mvs[1]


def test_tables_shared_per_path_but_vnet_per_engine():
    a, b = LearnedEngine(), LearnedEngine()
    assert a.vnet is not b.vnet
    assert a.vnet.net is b.vnet.net and a.vnet.tab is b.vnet.tab and a.vnet.rt is b.vnet.rt
    assert a.priors_override is b.priors_override


def test_c10_rollback_path_keeps_neff_wiring():
    eng = LearnedEngine(value_path=CL._C10_VALUE)
    assert isinstance(eng.vnet, NE.NEffValueAdapter) and eng.enc_version == 12
    assert eng.pnet is None and eng.priors_override is not None
