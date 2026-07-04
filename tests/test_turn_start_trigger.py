"""ターン開始時トリガー（TURN_START）— 「自分のターン開始時、発動できる。」（OP11-040）。

従来はパーサが ACTIVATE_MAIN へフォールバックし、ターン開始時に自動誘発する経路が
無かった（＝一度も発動しない）。TURN_START への写像と switch_turn での発火を検証する。
公式裁定: 条件「自分の場のドン!!が8枚以上ある場合」は**ドン!!展開前**（ターン開始時点）
の枚数で判定する。

実行:
  OPCG_LOG_SILENT=1 python -m pytest tests/test_turn_start_trigger.py -q -s -p no:cacheprovider
"""
import os as _os, sys as _sys  # noqa: E402  test bootstrap (sys.path + google スタブ)
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "harness"))

from leader_test_helpers import build, make_char
from opcg_sim.src.models.models import DonInstance


def _setup(don_count):
    """p2手番から自分(p1)のターン開始を跨ぐ盤面。p1の場のドン!!を don_count 枚にする。"""
    gm, p1, p2, L = build("OP11-040")
    gm.turn_player = p2
    gm.opponent = p1
    p1.don_active = [DonInstance(owner_id=p1.name) for _ in range(don_count)]
    p1.don_rested = []
    return gm, p1, p2, L


def _accept_and_drive(gm, p1, pick_uuid=None, limit=6):
    steps = 0
    while gm.active_interaction and steps < limit:
        ia = gm.active_interaction
        at = ia.get("action_type")
        cands = ia.get("selectable_uuids") or [c.uuid for c in ia.get("candidates", [])]
        if at == "CONFIRM_TRIGGER":
            gm.resolve_interaction(p1, {"accepted": True})
        elif at == "SELECT_TARGET" and pick_uuid in cands:
            gm.resolve_interaction(p1, {"selected_uuids": [pick_uuid]})
        else:
            gm.resolve_interaction(p1, {"accepted": True, "selected_uuids": cands[:1],
                                        "position": "BOTTOM"})
        steps += 1


def test_op11_040_fires_confirmation_at_turn_start_with_8_don():
    """場のドン8枚でターン開始 → 発動確認（CONFIRM_TRIGGER）が立つ。"""
    gm, p1, p2, L = _setup(8)
    gm.end_turn()
    gm._advance_pending_triggers()
    ia = gm.active_interaction
    assert ia is not None and ia["action_type"] == "CONFIRM_TRIGGER"
    assert "モンキー・D・ルフィ" in ia.get("message", "")


def test_op11_040_accept_recovers_strawhat_from_top5():
    """受諾 → デッキ上5枚から《麦わらの一味》1枚を手札に加え、残りをデッキへ。"""
    gm, p1, p2, L = _setup(8)
    zoro = make_char(p1, name="ロロノア・ゾロ", traits=["麦わらの一味"])
    p1.deck.insert(2, zoro)  # 通常ドロー(トップ1枚)で引かれない位置
    hand_before = len(p1.hand)
    deck_before = len(p1.deck)
    gm.end_turn()
    gm._advance_pending_triggers()
    _accept_and_drive(gm, p1, pick_uuid=zoro.uuid)
    assert zoro in p1.hand
    assert len(p1.hand) == hand_before + 2      # 通常ドロー+1、回収+1
    assert len(p1.deck) == deck_before - 2      # ドロー-1、回収-1（残り4枚はデッキへ戻る）
    assert gm.active_interaction is None


def test_op11_040_decline_does_nothing():
    """拒否 → 通常ドロー以外は何も起きない。"""
    gm, p1, p2, L = _setup(8)
    hand_before = len(p1.hand)
    gm.end_turn()
    gm._advance_pending_triggers()
    gm.resolve_interaction(p1, {"accepted": False})
    assert gm.active_interaction is None
    assert len(p1.hand) == hand_before + 1      # 通常ドローのみ


def test_op11_040_does_not_fire_below_8_don():
    """場のドン7枚では発動しない（条件未達）。"""
    gm, p1, p2, L = _setup(7)
    gm.end_turn()
    gm._advance_pending_triggers()
    assert gm.active_interaction is None


def test_op11_040_don_counted_before_don_phase():
    """裁定: ドン8枚以上の判定はドン!!展開前。開始時6枚（展開後8枚）では発動しない。"""
    gm, p1, p2, L = _setup(6)
    gm.end_turn()   # ドン!!フェイズで+2され場は8枚になるが、判定は開始時の6枚
    gm._advance_pending_triggers()
    assert gm.active_interaction is None
