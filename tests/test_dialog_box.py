"""マクロ手化 P3/P5＝対話箱の契約（2026-08-25）。

`mcts.in_dialog`（効果対話窓の判定・外周は含めない）と、対話窓の箱読み出し
（`LearnedEngine._window_choice`＝窓の根畳み・`resolved_branch_values(window_pred=in_dialog)`）、
seam（`TREE_BOX_DIALOG`）を固定する。設計の正本は
`docs/cpu_macro_plan.md` §2（PLAY 辺=カード使用箱・ACTIVATE 辺=効果起動箱・応答窓=応答箱・
トリガー可否=CONFIRM_TRIGGER 窓を対話箱1機構で実現）。
"""
import argparse
import types

import numpy as np
import conftest  # noqa: F401
import pytest

import _bootstrap  # noqa: F401

import coach_gate as CG
import counterfactual_referee as CR
import replay_reeval as RE
from cpu_selfplay import _load_db
from opcg_sim.src.core.cpu_learned import LearnedEngine
from opcg_sim.src.learned.mcts import (TreeMCTS, in_battle, in_dialog,
                                       resolve_battle_inplace, resolved_branch_values)

pytestmark = pytest.mark.cpu_infra


def _mgr(pending):
    return types.SimpleNamespace(pending_actor_action=lambda: pending)


def test_in_dialog_vocabulary():
    for a in ("SEARCH_AND_SELECT", "CONFIRM_OPTIONAL", "CONFIRM_TRIGGER",
              "CHOICE", "DECLARE_COST"):
        assert in_dialog(_mgr(("p1", a))), a
    # 外周・メイン・戦闘・無窓は対話箱の対象外
    for a in ("MAIN_ACTION", "MULLIGAN", "ARRANGE_DECK", "SELECT_RESOURCE",
              "SELECT_BLOCKER", "SELECT_COUNTER"):
        assert not in_dialog(_mgr(("p1", a))), a
    assert not in_dialog(_mgr(None))


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
    return m, (who if isinstance(who, str) else who.name)


def test_real_dialog_window_detected(m2_game):
    m, _ = _board(m2_game, 22)                     # サーチ選択窓（非戦闘）
    assert in_dialog(m) and not in_battle(m)
    m6, _ = _board(m2_game, 6)                     # ARRANGE_DECK＝外周は畳まない
    assert not in_dialog(m6)


def test_dialog_readout_returns_legal_and_fast(m2_game):
    m, name = _board(m2_game, 22)
    actor = m.p1 if m.p1.name == name else m.p2
    on = LearnedEngine(box_dialog=True)
    off = LearnedEngine()
    tr = {}
    mv = on.decide(m, actor, sims=8, rng=np.random.default_rng(1), trace=tr)
    assert tr.get("readout") == "window_resolved"
    # 出力合法: ON の手は OFF（従来）の合法手集合に含まれる（対話箱は新手型を作らない）
    legal = off.game.legal_actions(m)
    assert any(mv == x for x in legal)


def test_dialog_branch_resolution_reaches_exit(m2_game):
    m, name = _board(m2_game, 22)
    eng = LearnedEngine(box_dialog=True)
    legal = eng.game.legal_actions(m)
    vals = resolved_branch_values(eng.game, m, name, legal,
                                  lambda mgr, n: 0.0, window_pred=in_dialog)
    assert any(v is not None for v in vals)        # 少なくとも1枝は出口へ到達
    # 解決規約そのもの: 適用→対話解決で窓の外へ出る
    c = m.clone()
    c.action_events = []
    from opcg_sim.src.core import cpu_ai
    cpu_ai._apply_move_inplace(c, name, legal[0], stop_at_select=True)
    resolve_battle_inplace(eng.game, c, window_pred=in_dialog)
    assert not in_dialog(c)


def test_seam_default_on_and_off_override():
    from opcg_sim.src.learned import config as C
    assert C.TREE_BOX_DIALOG is True               # 既定 ON（2026-08-25 箱化一括採用）
    t = TreeMCTS(game=types.SimpleNamespace(apply_inplace=None, unmake=None),
                 value_fn=lambda m, n: 0.0)
    assert t.box_dialog is True                    # 既定は config に従う
    t_off = TreeMCTS(game=types.SimpleNamespace(apply_inplace=None, unmake=None),
                     value_fn=lambda m, n: 0.0, box_dialog=False)
    assert t_off.box_dialog is False               # 席別 seam で OFF に戻せる
