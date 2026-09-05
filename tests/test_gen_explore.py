"""生成の探索多様性の契約（純正AZ 2026-08-27・`n_record_gen` の探索プロファイル基盤）。

AZ の自己対戦は root Dirichlet ノイズ（ε=0.25）と序盤の温度サンプリング（τ=1）が
探索の多様性を担う。`LearnedEngine(dirichlet_eps=…, temp_turns=…)` の seam を固定する:
**serve 既定（両方 None）は完全に従来どおり**・temp_turns 有効時は序盤メイン窓の選択が
訪問分布からのサンプリングになり seed で分散する・選ばれる手は常に合法候補の一員。
"""
import argparse
import random

import numpy as np
import conftest  # noqa: F401
import pytest

import _bootstrap  # noqa: F401

import coach_gate as CG
import counterfactual_referee as CR
import replay_reeval as RE
from cpu_selfplay import _load_db
from opcg_sim.src.core.cpu_learned import LearnedEngine
from opcg_sim.src.learned.plan import move_sig

pytestmark = pytest.mark.cpu_infra


@pytest.fixture(scope="module")
def m2_game():
    CR.ARGS = argparse.Namespace(true_board=True)
    CR.GAMES = {}
    raw = RE.load_replay_json(CG.REPLAYS_V2["m2"])
    rec = raw.get("replay", raw)
    CR.GAMES["m2"] = (rec, {f.get("action_index"): f for f in raw.get("frames") or []},
                      rec["actions"])
    return _load_db()


def _board(db, i):
    m, who = CR._restore_board(db, "m2", i)
    m.action_events = []
    return m, (who if isinstance(who, str) else who.name)


def _decide(eng, m, name, seed, rec=None):
    random.seed(20260827)
    eng._world_seeds = {}
    eng._commits.clear()
    actor = m.p1 if m.p1.name == name else m.p2
    return eng.decide(m, actor, sims=8, rng=np.random.default_rng(seed), record=rec)


def test_serve_default_is_deterministic_argmax(m2_game):
    """serve 既定（seam 未指定）: 同一 seed で同一手＝従来どおり決定的。"""
    eng = LearnedEngine()
    assert eng.dirichlet_eps is None and eng.temp_turns is None
    m, name = _board(m2_game, 50)
    a = _decide(eng, m, name, 0)
    b = _decide(eng, m, name, 0)
    assert move_sig(a) == move_sig(b)


def test_temp_sampling_varies_and_stays_legal(m2_game):
    """temp_turns 有効: 選択は候補（record.groups）の一員のまま、seed により分散する。

    ネットは c10 を明示（`_C10_VALUE`）: 検査対象はサンプリングの機構であってネットの尖り方ではない。
    既定が a1 になった 2026-09-05 に m2@50 で 10 seed が全て同じ手に収束し（a1 の訪問分布が尖っている）
    偽の失敗を出したため、機構の検査は分布が割れる既知のネットで固定する。"""
    from opcg_sim.src.core.cpu_learned import _C10_VALUE
    eng = LearnedEngine(value_path=_C10_VALUE, temp_turns=99)   # 全ターン温度 ON（試験用）
    m, name = _board(m2_game, 50)
    sigs = set()
    for sd in range(10):
        rec = {}
        mv = _decide(eng, m, name, sd, rec=rec)
        assert mv is not None
        assert (rec["sig"], rec.get("k")) in [(g["sig"], g.get("k"))
                                              for g in rec["groups"]]
        sigs.add((rec["sig"], rec.get("k")))
    assert len(sigs) >= 2, "10 seed で全て同一手＝サンプリングが効いていない"


def test_dirichlet_seam_smoke(m2_game):
    """dirichlet_eps 有効でも合法手が返る（配線の煙試験）。"""
    eng = LearnedEngine(dirichlet_eps=0.25)
    m, name = _board(m2_game, 50)
    mv = _decide(eng, m, name, 3)
    assert mv is not None and mv.get("action_type")
