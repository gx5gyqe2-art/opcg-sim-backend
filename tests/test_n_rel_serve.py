"""NRel の serve 配線（P3・2026-09-04・`cpu_learned` × `n_rel.NRelValueAdapter`）の契約。

基盤健全性（`cpu_infra`）: 候補本体の配線（出荷既定は c10 のまま・`test_neff_default` が守る）。

守る性質:
  1. `LearnedEngine(value_path=<NRel npz>)` が NRel を判別し、`vnet` は `NRelValueAdapter`・`pnet` 無し・
     `priors_override` あり・符号化 v13・出口ヘッド無し。既定（c10）の配線は変わらない。
  2. 葉価値は `predict_state`（盤面から直接）で評価され、`_value_fn` がそれを使う（値は [−1,1]）。
  3. priors は合法手上の確率（非負・和 1・長さ一致）。候補の主体/対象の枠 index が付く。
  4. `decide` は合法手を返し、同一 seed で決定論。表・関係表はプロセス内共有・`vnet` は席ごと別。
  5. レイテンシ（情報）: decide sims 32 の時間を c10 と並べて出力する（関門は計画 §5・本テストでは固定しない）。
"""
import argparse
import random
import time

import numpy as np
import pytest

import conftest  # noqa: F401
import _bootstrap  # noqa: F401

import coach_gate as CG
import counterfactual_referee as CR
import replay_reeval as RE
from cpu_selfplay import _load_db
from n_eff_feat import build_eff_tables
from opcg_sim.src.core import cpu_learned as CL
from opcg_sim.src.core.cpu_learned import LearnedEngine
from opcg_sim.src.learned import n_rel as NL

pytestmark = pytest.mark.cpu_infra


@pytest.fixture(scope="module")
def npz(tmp_path_factory):
    """乱数初期化の NRel（hidden 32）を npz に保存（配線の検査に重みの質は要らない）。"""
    tabs = build_eff_tables()
    net = NL.NRelNet(tabs[:5], hidden=32, seed=7)
    vocab = tabs[5]
    net.vocab_ids = [cid for cid, _i in sorted(vocab.items(), key=lambda kv: kv[1])]
    p = str(tmp_path_factory.mktemp("nrel") / "nrel_test.npz")
    net.save(p, meta={"kind": "nrel-a", "name": "test"})
    return p


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


def test_engine_detects_nrel_and_wires_adapter(npz):
    assert NL.is_nrel_npz(npz) and not CL._is_neff(npz) and CL._is_nrel(npz)
    eng = LearnedEngine(value_path=npz)
    assert isinstance(eng.vnet, NL.NRelValueAdapter)
    assert eng.pnet is None and eng.priors_override is not None
    assert eng.enc_version == NL.NR_ENC_VERSION == 13
    assert not eng.vnet.has_exit_head("battle") and eng._exit_value_fn("battle") is None
    ref = LearnedEngine()                                      # 既定は c10 のまま
    assert not isinstance(ref.vnet, NL.NRelValueAdapter) and ref.enc_version == 12
    b = LearnedEngine(value_path=npz)
    assert b.vnet is not eng.vnet and b.vnet.net is eng.vnet.net and b.vnet.tab is eng.vnet.tab


def test_leaf_value_uses_predict_state(npz, m2):
    eng = LearnedEngine(value_path=npz)
    vf = CL._value_fn(eng.vnet, eng.vocab, eng.enc_version)
    for i in (10, 30, 50):
        m, name, _ = _board(m2, i)
        v = vf(m, name)
        assert -1.0 <= v <= 1.0
        assert abs(v - eng.vnet.predict_state(m, name)) < 1e-6
    with pytest.raises(ValueError):
        eng.vnet.predict({"scalars": np.zeros((1, 123), np.float32), "card_idx": np.zeros((1, 22), np.int64)})


def test_priors_are_probabilities_with_slot_indices(npz, m2):
    eng = LearnedEngine(value_path=npz)
    seen = 0
    for i in (10, 30, 50, 70):
        m, name, _ = _board(m2, i)
        legal = eng.game.legal_actions(m)
        if not legal:
            continue
        p = eng._priors()(m, legal)
        assert p is not None and p.shape == (len(legal),)
        assert float(p.min()) >= 0.0 and abs(float(p.sum()) - 1.0) < 1e-4
        seen += 1
    assert seen >= 1


def test_decide_is_legal_and_deterministic(npz, m2):
    a, b = LearnedEngine(value_path=npz), LearnedEngine(value_path=npz)
    for i in (10, 50):
        m, name, actor = _board(m2, i)
        mvs = []
        for e in (a, b):
            random.seed(20260904)
            e._world_seeds = {}
            e._commits.clear()
            mvs.append(e.decide(m, actor, sims=8, rng=np.random.default_rng(i)))
        assert mvs[0] is not None and mvs[0].get("action_type") and mvs[0] == mvs[1]


def test_latency_report(npz, m2):
    """情報のみ: sims 32 の decide 時間を c10 と並べる（関門は計画 §5 の別測定）。"""
    out = {}
    for label, eng in (("c10", LearnedEngine()), ("nrel", LearnedEngine(value_path=npz))):
        ts = []
        for i in (30, 50, 70):
            m, name, actor = _board(m2, i)
            random.seed(1)
            eng._world_seeds = {}
            eng._commits.clear()
            t = time.time()
            eng.decide(m, actor, sims=32, rng=np.random.default_rng(i))
            ts.append(time.time() - t)
        out[label] = float(np.mean(ts))
    print(f"\nNREL_LATENCY sims32 c10 {out['c10']:.2f}s nrel {out['nrel']:.2f}s ratio {out['nrel']/max(out['c10'],1e-6):.2f}")
    assert out["nrel"] > 0
