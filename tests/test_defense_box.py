"""マクロ手化 P4-c＝防御箱 v1 の契約（2026-08-24）。

`cpu_ai.defense_battle_need`（戦闘を止める必要値の算術）と `cpu_ai.defense_box_prune`
（防御窓の D1'/D2' 支配則による候補整形）、adapter の seam（defense_box・既定 OFF は
挙動不変）を固定する。設計根拠は `docs/reports/2026-08-24_p4_defense_verdict.md`
（D族ヘッド再学習はゲート FAIL＝学習でなく候補整形として実装するという判定）。
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
from opcg_sim.src.learned.adapter import OPCGGame

pytestmark = pytest.mark.cpu_infra


def _unit(power):
    return types.SimpleNamespace(get_power=lambda attacking, _p=power: _p)


def _c(uuid, counter):
    return types.SimpleNamespace(uuid=uuid, current_counter=counter)


def _mgr(atk, tgt, hand, buff=0, battle=True):
    bat = {"attacker": _unit(atk), "target": _unit(tgt), "counter_buff": buff} \
        if battle else None
    p1 = types.SimpleNamespace(name="p1", hand=list(hand))
    return types.SimpleNamespace(p1=p1, p2=types.SimpleNamespace(name="p2"),
                                 active_battle=bat)


def _sc(uuid):
    return {"kind": "battle", "action_type": "SELECT_COUNTER", "card_uuid": uuid}


_PASS = {"kind": "battle", "action_type": "PASS"}


def test_defense_battle_need_arithmetic():
    assert cpu_ai.defense_battle_need(_mgr(6000, 5000, [])) == 2000
    assert cpu_ai.defense_battle_need(_mgr(5000, 5000, [])) == 1000   # 同値は攻撃側勝ち
    assert cpu_ai.defense_battle_need(_mgr(4000, 5000, [])) == 0      # 止まっている
    assert cpu_ai.defense_battle_need(_mgr(6000, 5000, [], buff=1000)) == 1000
    assert cpu_ai.defense_battle_need(_mgr(0, 0, [], battle=False)) is None


def test_prune_d1_total_insufficient_drops_counters():
    m = _mgr(9000, 5000, [_c("a", 1000), _c("b", 2000)])   # need=5000 > 総量3000
    out = cpu_ai.defense_box_prune(m, "p1", [_sc("a"), _sc("b"), _PASS])
    assert out == [_PASS]                                   # 素通しのみ残る


def test_prune_d2_already_stopped_drops_counters():
    m = _mgr(4000, 5000, [_c("a", 1000)])                   # need=0
    out = cpu_ai.defense_box_prune(m, "p1", [_sc("a"), _PASS])
    assert out == [_PASS]


def test_prune_keeps_affordable_window():
    m = _mgr(6000, 5000, [_c("a", 2000), _c("b", 1000)])    # need=2000 ≤ 総量3000
    moves = [_sc("a"), _sc("b"), _PASS]
    assert cpu_ai.defense_box_prune(m, "p1", moves) == moves


def test_prune_skips_mixed_or_nonprinted_windows():
    m = _mgr(9000, 5000, [_c("a", 1000)])
    mixed = [_sc("a"), {"kind": "battle", "action_type": "SELECT_BLOCKER",
                        "card_uuid": "x"}, _PASS]
    assert cpu_ai.defense_box_prune(m, "p1", mixed) == mixed     # 混在窓は触らない
    m0 = _mgr(9000, 5000, [_c("a", 0)])                          # 印字0の札が候補に混ざる
    zero = [_sc("a"), _PASS]
    assert cpu_ai.defense_box_prune(m0, "p1", zero) == zero


@pytest.fixture()
def m2_58():
    """m2@58＝実盤面の防御窓（総量不足・素通しが正のユーザ裁定点＝D1型の代表例）。"""
    CR.ARGS = argparse.Namespace(true_board=True)
    CR.GAMES = {}
    raw = RE.load_replay_json(CG.REPLAYS_V2["m2"])
    rec = raw.get("replay", raw)
    CR.GAMES["m2"] = (rec, {f.get("action_index"): f for f in raw.get("frames") or []},
                      rec["actions"])
    db = _load_db()
    m, who = CR._restore_board(db, "m2", 58)
    return m, (who if isinstance(who, str) else who.name)


def test_defense_box_seam_on_real_board(m2_58):
    m, _name = m2_58
    off = OPCGGame(defense_box=False).legal_actions(m)
    on = OPCGGame(defense_box=True).legal_actions(m)
    assert any(x.get("action_type") == "SELECT_COUNTER" for x in off)  # OFF は従来のまま
    assert not any(x.get("action_type") == "SELECT_COUNTER" for x in on)
    assert any(x.get("action_type") == "PASS" for x in on)             # 素通しは常に残る
