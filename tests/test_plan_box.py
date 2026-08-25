"""マクロ手化 P6-a＝ターン箱（プラン読み出し×箱語彙）の契約（2026-08-25）。

箱化フルセットのエンジンでは、プラン提案（`plan.select_plan`）は**箱の列**として立ち
（配分箱 DON_BOX 等が sig に現れる）、実対局への出力は常に原始手へ変換される。
対話窓は `dialog_box=True` で出口 value 埋め（v39 の quiesce 埋めを対話箱に置換）。
設計の正本は `docs/cpu_macro_plan.md` §5 P6。
"""
import argparse

import numpy as np
import conftest  # noqa: F401
import pytest

import _bootstrap  # noqa: F401

import coach_gate as CG
import counterfactual_referee as CR
import replay_reeval as RE
from cpu_selfplay import _load_db
from opcg_sim.src.core import cpu_learned as CL
from opcg_sim.src.learned import plan as PL

pytestmark = pytest.mark.cpu_infra

_BOX_KW = dict(macro_moves=True, defense_box=True, box_dialog=True,
               box_battle=True, battle_readout=True, quiesce=True)


@pytest.fixture(scope="module")
def m2_44():
    CR.ARGS = argparse.Namespace(true_board=True)
    CR.GAMES = {}
    raw = RE.load_replay_json(CG.REPLAYS_V2["m2"])
    rec = raw.get("replay", raw)
    CR.GAMES["m2"] = (rec, {f.get("action_index"): f for f in raw.get("frames") or []},
                      rec["actions"])
    db = _load_db()
    m, who = CR._restore_board(db, "m2", 44)
    return m, (who if isinstance(who, str) else who.name)


def test_select_plan_proposes_box_steps(m2_44):
    m, name = m2_44
    eng = CL.LearnedEngine(**_BOX_KW)
    vf = CL._value_fn(eng.vnet, eng.vocab, eng.enc_version)
    pf = CL._priors_fn(eng.pnet, eng.vocab, eng.enc_version)
    steps, diag = PL.select_plan(eng.game, m, name, vf, pf, np.random.default_rng(5),
                                 battle_value_fn=eng._battle_value_fn(),
                                 min_spread=0.0, dialog_box=True)
    assert steps, f"プランが立たない: {diag}"
    assert diag.get("n_plans", 0) >= 2
    # 箱語彙: 浮ドンのある m2@44 では配分箱（DON_BOX）が提案の語彙に入る
    assert any(sig[0] == "DON_BOX" for sig in steps), steps


def test_plan_readout_outputs_primitive(m2_44):
    m, name = m2_44
    eng = CL.LearnedEngine(plan_readout=True, **_BOX_KW)
    actor = m.p1 if m.p1.name == name else m.p2
    seen_plan = False
    for s in range(6):
        eng._world_seeds = {}
        eng._turn_plans.clear()
        tr = {}
        mv = eng.decide(m, actor, sims=16, rng=np.random.default_rng(100 + s), trace=tr)
        assert (mv or {}).get("action_type") != "DON_BOX"   # 出力は常に原始手（素通し事故ガード）
        if tr.get("readout") == "turn_plan":
            seen_plan = True
    assert seen_plan, "6シードで一度もプラン読み出しが採用されない"


def test_plan_step_completes_box_before_advancing(m2_44):
    """箱の半消化バグの再発ガード（2026-08-25・plan-box アリーナ 0.06 の根因）:
    k>0 の DON_BOX ステップは先頭原始手を返した後も**プランに残り**、箱を完走してから
    次のステップへ進む（pop すると付与だけして攻撃しない/配分が途切れる）。"""
    m, name = m2_44
    eng = CL.LearnedEngine(plan_readout=True, **_BOX_KW)
    actor = m.p1 if m.p1.name == name else m.p2
    legal = eng.game.legal_actions(m)
    # 原始手2回以上で完走する箱＝配分 k>=2 またはアタック形 k>=1（付与k＋攻撃1）
    def emits(x):
        p = x.get("payload") or {}
        k = int(p.get("don_k") or 0)
        return k + (1 if p.get("target_ids") else 0) if x.get("action_type") == "DON_BOX" else 0
    boxes = [x for x in legal if emits(x) >= 2]
    assert boxes, "複数原始手の箱が立たない盤面では検証できない"
    sig = PL.move_sig(boxes[0])
    n_total = emits(boxes[0])
    rng = np.random.default_rng(1)
    eng._world_seeds = {}
    # decide を1回呼んでプランキーを確立してから、プランを注入して差し替える
    eng.decide(m, actor, sims=8, rng=rng)
    assert eng._turn_plans, "プランキャッシュが作られていない"
    key = list(eng._turn_plans)[-1]
    eng._turn_plans[key] = [sig]
    mv = eng.decide(m, actor, sims=8, rng=rng)
    assert (mv or {}).get("action_type") == "ATTACH_DON"      # 先頭原始手＝付与
    st = eng._turn_plans[key][0]                              # カウントダウンで残る
    assert st[0] == "__box__" and st[1] == sig and st[2] == n_total - 1


def test_execute_plan_dialog_box_kw_accepted(m2_44):
    m, name = m2_44
    eng = CL.LearnedEngine(**_BOX_KW)
    vf = CL._value_fn(eng.vnet, eng.vocab, eng.enc_version)
    pf = CL._priors_fn(eng.pnet, eng.vocab, eng.enc_version)
    world = eng.game.determinize(m, name, np.random.default_rng(7))
    exit_mgr = PL.execute_plan(eng.game, world, name, [], vf, pf, dialog_box=True)
    assert exit_mgr is not None                              # 空プラン＝ターンを閉じた出口
