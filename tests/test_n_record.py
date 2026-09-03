"""棋譜ダンプの record 観測の契約（純正Nループ① 2026-08-26・`n_record_gen.py` の基盤）。

`LearnedEngine.decide(record=dict)` は**観測専用**——kind（main/window/commit）・sig（選択手の
move_sig・箱レベル）・main では groups（等価マージ後の全候補 {sig,n,q}＝`_merge_root_stats` と
同一集計）を書き、**手の選択には一切影響しない**ことを固定する。
"""
import argparse
import json
import random
import weakref

import numpy as np
import conftest  # noqa: F401
import pytest

import _bootstrap  # noqa: F401

import coach_gate as CG
import counterfactual_referee as CR
import replay_reeval as RE
from cpu_selfplay import _load_db
from opcg_sim.src.core.cpu_learned import LearnedEngine
from opcg_sim.src.learned.mcts import in_battle, in_dialog
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


@pytest.fixture(scope="module")
def eng():
    return LearnedEngine()


def _board(db, i):
    m, who = CR._restore_board(db, "m2", i)
    m.action_events = []
    return m, (who if isinstance(who, str) else who.name)


def _actor(m, name):
    return m.p1 if m.p1.name == name else m.p2


def _decide(eng, m, name, record=None, seed=20260826):
    """決定論の1手（global random / drng / エンジン内キャッシュを固定）。"""
    random.seed(seed)
    eng._world_seeds = {}
    eng._commits.clear()
    return eng.decide(m, _actor(m, name), sims=8,
                      rng=np.random.default_rng(0), record=record)


def test_record_main_contract(m2_game, eng):
    """main 窓: kind/sig/groups の契約（groups は n 降順・sig は候補の一員・n 合計 > 0）。"""
    m, name = _board(m2_game, 50)
    rec = {}
    mv = _decide(eng, m, name, record=rec)
    assert mv is not None
    assert rec["kind"] == "main" and rec["sims"] == 8
    gs = rec["groups"]
    assert gs and sum(g["n"] for g in gs) > 0
    assert all(gs[i]["n"] >= gs[i + 1]["n"] for i in range(len(gs) - 1))
    # 配分箱の k 違いは同 sig（move_sig は don_k 非含有）＝候補の同一性は (sig, k)
    assert (rec["sig"], rec["k"]) in [(g["sig"], g["k"]) for g in gs]
    assert len({(json.dumps(g["sig"]), g["k"]) for g in gs}) == len(gs)
    # 返る手は実対局契約どおり原始手化されるが、記録は箱レベル＝DON_BOX 選択時は sig が
    # 返り値と別になり得る。それ以外は一致する。
    if rec["sig"][0] != "DON_BOX":
        assert rec["sig"] == move_sig(mv)


def test_record_is_observation_only(m2_game, eng):
    """観測不変: record の有無で選択が変わらない（同一 seed・同一キャッシュ状態）。"""
    m, name = _board(m2_game, 50)
    mv_plain = _decide(eng, m, name)
    mv_rec = _decide(eng, m, name, record={})
    assert move_sig(mv_plain) == move_sig(mv_rec)


def test_record_commit_kind(m2_game, eng):
    """commit 消化: kind="commit"・sig は返った原始手の sig・groups は書かれない。"""
    m, name = _board(m2_game, 50)
    allocs = [x for x in eng.game.legal_actions(m)
              if x.get("action_type") == "DON_BOX" and not x["payload"].get("target_ids")
              and int(x["payload"].get("don_k") or 0) >= 2]
    assert allocs, "実盤面で配分箱（k>=2）が立たない"
    random.seed(20260826)
    eng._world_seeds = {}
    eng._commits.clear()
    eng._commits[eng._commit_key(m, name)] = (
        weakref.ref(m), [("__box__", move_sig(allocs[0]), 2)])
    rec = {}
    mv = eng.decide(m, _actor(m, name), sims=8, rng=np.random.default_rng(0), record=rec)
    assert rec["kind"] == "commit"
    assert rec["sig"] == move_sig(mv)
    assert "groups" not in rec


def test_record_window_kind(m2_game, eng):
    """窓の根畳み: 戦闘/対話窓の盤面では kind="window"・sig=選択手。"""
    for i in range(60):
        m, name = _board(m2_game, i)
        if in_battle(m) or in_dialog(m):
            break
    else:
        pytest.skip("m2 リプレイに戦闘/対話窓の復元点が無い")
    rec = {}
    mv = _decide(eng, m, name, record=rec)
    assert mv is not None
    assert rec["kind"] == "window"
    assert rec["sig"] == move_sig(mv)
