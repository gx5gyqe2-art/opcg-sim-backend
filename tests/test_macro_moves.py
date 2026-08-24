"""マクロ手化 P1＝配分箱の契約（ユーザ設計 2026-08-24）。

`cpu_ai.don_alloc_candidates`（「対象へk枚付与」を DON_BOX の target_ids=[] 形で合成）と
adapter の seam（macro_moves で原始 ATTACH_DON を配分箱に置換・既定 OFF は挙動不変）、
配分箱の適用展開（付与のみ・攻撃しない）、行動特徴（ATTACH_DON として特徴化）を固定する。
"""
import argparse
import types

import conftest  # noqa: F401
import pytest

import _bootstrap  # noqa: F401

import coach_gate as CG
import counterfactual_referee as CR
import replay_reeval as RE
from cpu_selfplay import _load_db
from opcg_sim.src.core import cpu_ai
from opcg_sim.src.learned import action as A
from opcg_sim.src.learned.adapter import OPCGGame

pytestmark = pytest.mark.cpu_infra


def _card(uuid, text="", attached=0):
    return types.SimpleNamespace(
        uuid=uuid, attached_don=attached,
        master=types.SimpleNamespace(card_id=uuid, effect_text=text))


def _mgr(field=(), don=3, leader=None):
    p1 = types.SimpleNamespace(name="p1", leader=leader, field=list(field),
                               don_active=[1] * don)
    return types.SimpleNamespace(p1=p1, p2=types.SimpleNamespace(name="p2"))


def _attach(uuid):
    return {"kind": "game", "action_type": "ATTACH_DON", "payload": {"uuid": uuid}}


def test_alloc_candidates_salient_ks():
    drake = _card("d", text="【ドン!!×3】効果", attached=1)     # 閾値開放 k=2
    vanilla = _card("v")
    m = _mgr(field=[drake, vanilla], don=4)
    out = cpu_ai.don_alloc_candidates(m, "p1", [_attach("d"), _attach("v")])
    ks = {(o["payload"]["uuid"], o["payload"]["don_k"]) for o in out}
    assert ("d", 1) in ks and ("d", 2) in ks and ("d", 4) in ks   # 1・閾値・全振り
    assert ("v", 1) in ks and ("v", 4) in ks
    assert all(o["action_type"] == "DON_BOX" and o["payload"]["target_ids"] == []
               for o in out)


def test_alloc_candidates_respects_budget_and_sources():
    m = _mgr(field=[_card("x")], don=1)
    out = cpu_ai.don_alloc_candidates(m, "p1", [_attach("x")])
    assert {(o["payload"]["uuid"], o["payload"]["don_k"]) for o in out} == {("x", 1)}
    assert cpu_ai.don_alloc_candidates(m, "p1", []) == []          # 原始手が無ければ箱も無い
    assert cpu_ai.don_alloc_candidates(_mgr(don=0), "p1", [_attach("x")]) == []


def test_action_features_alloc_box_as_attach():
    # 配分箱（target_ids=[]）は ATTACH_DON として特徴化＝素付与の prior を継承
    alloc = {"kind": "game", "action_type": "DON_BOX",
             "payload": {"uuid": None, "target_ids": [], "don_k": 2}}
    f = A.action_features(_mgr(), alloc, "p1")
    assert f[A._AT_IDX["ATTACH_DON"]] == 1.0
    assert f[A._AT_IDX["ATTACK"]] == 0.0


@pytest.fixture(scope="module")
def m2_44():
    """m2@44＝浮ドンありのメイン窓（実盤面）。"""
    CR.ARGS = argparse.Namespace(true_board=True)
    CR.GAMES = {}
    raw = RE.load_replay_json(CG.REPLAYS_V2["m2"])
    rec = raw.get("replay", raw)
    CR.GAMES["m2"] = (rec, {f.get("action_index"): f for f in raw.get("frames") or []},
                      rec["actions"])
    db = _load_db()
    m, who = CR._restore_board(db, "m2", 44)
    return m, (who if isinstance(who, str) else who.name)


def test_apply_alloc_box_attaches_without_attack(m2_44):
    m, name = m2_44
    g = OPCGGame(macro_moves=True)
    allocs = [x for x in g.legal_actions(m)
              if x.get("action_type") == "DON_BOX" and not x["payload"].get("target_ids")]
    assert allocs, "実盤面で配分箱が立たない"
    mv = max(allocs, key=lambda x: x["payload"]["don_k"])
    p = m.p1 if m.p1.name == name else m.p2
    before = len(p.don_active)
    nxt = g.apply(m, mv, name)
    assert nxt is not None
    p2 = nxt.p1 if nxt.p1.name == name else nxt.p2
    assert len(p2.don_active) == before - mv["payload"]["don_k"]   # k枚消費
    assert getattr(nxt, "active_battle", None) is None             # 攻撃はしていない


def test_macro_seam_replaces_primitive_attach(m2_44):
    m, _ = m2_44
    on = OPCGGame(macro_moves=True).legal_actions(m)
    off = OPCGGame(macro_moves=False).legal_actions(m)
    assert not any(x.get("action_type") == "ATTACH_DON" for x in on)   # 原始付与は撤廃
    assert any(x.get("action_type") == "ATTACH_DON" for x in off)      # OFF は従来のまま
    # OFF（既定）は配分箱を混ぜない
    assert not any(x.get("action_type") == "DON_BOX" and not x["payload"].get("target_ids")
                   for x in off)
