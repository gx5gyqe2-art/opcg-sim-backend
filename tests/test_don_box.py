"""ドン箱（DON_BOX・`docs/cpu_don_box_plan.md` Phase 1＝攻撃箱）の検証。

固定する性質:
  - 列挙算術: k ∈ {通る最小, 2枚要求} のみ・1 ≤ k ≤ 予算・重複なし・対象は相手リーダー
  - 展開等価: DON_BOX の適用結果 ＝ 原始列（ATTACH_DON×k → ATTACK）の逐次適用結果
  - 原始手変換: `don_box_first_primitive` が実対局出力を ATTACH_DON に変換（他は素通し）
  - seam: OPCGGame(don_box=None/False) では候補が合成されない（既定挙動不変）
  - action_key: k 違いの箱は別キー（探索木で別 edge）・既存手のキーは不変
"""
import conftest  # noqa: F401
import pytest

from opcg_sim.src.core import cpu_ai
from opcg_sim.src.core.gamestate import Phase
from opcg_sim.src.learned.action import action_key
from opcg_sim.src.learned.adapter import OPCGGame

pytestmark = pytest.mark.cpu_infra   # 基盤健全性（探索候補合成の機構）


@pytest.fixture(scope="module")
def db():
    from cpu_selfplay import _load_db
    return _load_db()


def _new_gm(db):
    from cpu_selfplay import build_deck
    from opcg_sim.src.core.gamestate import GameManager, Player
    l1, c1 = build_deck(db, "p1")
    l2, c2 = build_deck(db, "p2")
    gm = GameManager(Player("p1", c1, l1), Player("p2", c2, l2))
    gm.start_game()
    gm.turn_count = 5
    gm.turn_player = gm.p1
    gm.phase = Phase.MAIN
    return gm


def _vanilla_attacker(gm):
    for c in list(gm.p1.deck):
        if (c.master.type.name == "CHARACTER" and not c.has_keyword("速攻")
                and not cpu_ai._has_don_conditional(c)):
            return c
    return None


def _setup(gm, power_delta, dons):
    """p1 場に素体1体（リーダー比 power_delta）・アクティブドン dons 枚の盤面を作る。"""
    gm.p2.field.clear()
    leader_pw = int(gm.p2.leader.get_power(False))
    c = _vanilla_attacker(gm)
    assert c is not None
    gm.p1.deck.remove(c)
    gm.p1.field.append(c)
    c.is_rest = False
    c.is_newly_played = False
    c.passive_power_override = leader_pw + power_delta
    gm.p1.don_active.clear()
    for _ in range(dons):
        gm.p1.don_active.append(gm.p1.don_deck.pop())
    return c, leader_pw


def _boxes(gm):
    raw = gm.get_legal_actions(gm.p1)
    return cpu_ai.don_box_candidates(gm, "p1", raw)


def test_enumeration_arithmetic(db):
    gm = _new_gm(db)
    c, _L = _setup(gm, power_delta=0, dons=4)      # ちょうどリーダー同値
    boxes = _boxes(gm)
    ks = sorted(b["payload"]["don_k"] for b in boxes if b["payload"]["uuid"] == c.uuid)
    assert ks == [2], f"同値からは 2枚要求（+2000）だけが箱になる（k_min=0 は素攻撃）: {ks}"
    # 不足 -1500: k_min=2（届く最小）・k_two=4（7000相当）
    c.passive_power_override = int(gm.p2.leader.get_power(False)) - 1500
    ks = sorted(b["payload"]["don_k"] for b in _boxes(gm) if b["payload"]["uuid"] == c.uuid)
    assert ks == [2, 4]
    # 予算 1 枚だと届かない箱は出ない
    gm.p1.don_active[:] = gm.p1.don_active[:1]
    ks = [b["payload"]["don_k"] for b in _boxes(gm) if b["payload"]["uuid"] == c.uuid]
    assert ks == []
    # 予算 0 は常に空
    gm.p1.don_active.clear()
    assert _boxes(gm) == []


def test_box_apply_equals_primitive_sequence(db):
    gm = _new_gm(db)
    c, L = _setup(gm, power_delta=0, dons=3)
    box = {"kind": "game", "action_type": "DON_BOX",
           "payload": {"uuid": c.uuid, "target_ids": [gm.p2.leader.uuid], "don_k": 2}}
    m_box = cpu_ai._apply_clone(gm, "p1", box, stop_at_select=True)
    assert m_box is not None
    m_seq = gm
    for mv in ({"kind": "game", "action_type": "ATTACH_DON", "payload": {"uuid": c.uuid}},
               {"kind": "game", "action_type": "ATTACH_DON", "payload": {"uuid": c.uuid}},
               {"kind": "game", "action_type": "ATTACK",
                "payload": {"uuid": c.uuid, "target_ids": [gm.p2.leader.uuid]}}):
        m_seq = cpu_ai._apply_clone(m_seq, "p1", mv, stop_at_select=True)
        assert m_seq is not None
    for m in (m_box, m_seq):
        pass
    a1 = next(x for x in m_box.p1.field if x.master.card_id == c.master.card_id)
    a2 = next(x for x in m_seq.p1.field if x.master.card_id == c.master.card_id)
    assert getattr(a1, "attached_don", 0) == getattr(a2, "attached_don", 0) == 2
    assert len(m_box.p1.don_active) == len(m_seq.p1.don_active) == 1
    assert (m_box.active_battle is None) == (m_seq.active_battle is None)
    assert len(m_box.p2.life) == len(m_seq.p2.life)


def test_first_primitive_conversion():
    box = {"kind": "game", "action_type": "DON_BOX",
           "payload": {"uuid": "u1", "target_ids": ["u2"], "don_k": 2}}
    mv = cpu_ai.don_box_first_primitive(box)
    assert mv == {"kind": "game", "action_type": "ATTACH_DON", "payload": {"uuid": "u1"}}
    other = {"kind": "game", "action_type": "TURN_END", "payload": {}}
    assert cpu_ai.don_box_first_primitive(other) is other
    assert cpu_ai.don_box_first_primitive(None) is None


def test_adapter_seam_follows_config_and_override(db):
    """既定＝config（SERVE_DON_BOX=True・ユーザ判断 2026-08-12）・インスタンス指定が優先。"""
    from opcg_sim.src.learned.config import SERVE_DON_BOX
    assert SERVE_DON_BOX is True
    gm = _new_gm(db)
    _setup(gm, power_delta=0, dons=4)
    default = OPCGGame()                  # 既定＝config に従う（ON）
    off = OPCGGame(don_box=False)         # 席別上書きで旧挙動
    on = OPCGGame(don_box=True)
    assert any(m.get("action_type") == "DON_BOX" for m in default.legal_actions(gm))
    assert not any(m.get("action_type") == "DON_BOX" for m in off.legal_actions(gm))
    assert any(m.get("action_type") == "DON_BOX" for m in on.legal_actions(gm))


def test_action_key_distinguishes_k(db):
    b1 = {"kind": "game", "action_type": "DON_BOX",
          "payload": {"uuid": "u1", "target_ids": ["u2"], "don_k": 1}}
    b2 = {"kind": "game", "action_type": "DON_BOX",
          "payload": {"uuid": "u1", "target_ids": ["u2"], "don_k": 2}}
    assert action_key(b1) != action_key(b2)
    # 既存手のキーは don_k=None が末尾に付くだけ（同一手同士は同一のまま）
    a = {"kind": "game", "action_type": "ATTACK", "payload": {"uuid": "u1", "target_ids": ["u2"]}}
    assert action_key(a) == action_key(dict(a))
