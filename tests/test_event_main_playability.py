"""イベントのメインフェイズ発動可否 — 【メイン】効果を持つイベントのみ手札から発動できる。

【カウンター】/【トリガー】のみのイベント（ゴムゴムの巨人 OP09-078 等）はメインフェイズに
手札から発動できない（カウンターは防御時の SELECT_COUNTER、トリガーはライフ公開時のみ）。
従来はコストさえ払えれば合法手に列挙され、自ターンに空撃ちできていた。

実行:
  OPCG_LOG_SILENT=1 python -m pytest tests/test_event_main_playability.py -q -s -p no:cacheprovider
"""
import os as _os, sys as _sys  # noqa: E402  test bootstrap (sys.path + google スタブ)
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "harness"))

import pytest

from leader_test_helpers import build, db
from opcg_sim.src.models.models import CardInstance, DonInstance
from opcg_sim.src.models.enums import Phase


def _setup(hand_ids):
    gm, p1, p2, L = build("OP11-040")  # 麦わらルフィ（青）
    gm.turn_player = p1
    gm.opponent = p2
    gm.phase = Phase.MAIN
    gm.turn_count = 3
    p1.hand = [CardInstance(db().get_card(cid), p1.name) for cid in hand_ids]
    while len(p1.don_active) < 6:
        p1.don_active.append(DonInstance(owner_id=p1.name))
    return gm, p1, p2, L


def _play_uuids(gm, p1):
    return {m["payload"]["uuid"] for m in gm.get_legal_actions(p1)
            if m.get("action_type") == "PLAY"}


# --- 合法手列挙（get_legal_actions） --------------------------------------

def test_counter_only_event_not_in_legal_plays():
    """OP09-078 ゴムゴムの巨人（【カウンター】専用）はメインの PLAY 合法手に出ない。"""
    gm, p1, p2, L = _setup(["OP09-078"])
    assert p1.hand[0].uuid not in _play_uuids(gm, p1)


def test_counter_plus_trigger_event_not_in_legal_plays():
    """OP06-059 ホワイトスネーク（【カウンター】+【トリガー】）もメインでは出ない。"""
    gm, p1, p2, L = _setup(["OP06-059"])
    assert p1.hand[0].uuid not in _play_uuids(gm, p1)


def test_main_plus_counter_event_in_legal_plays():
    """OP11-080 ギア2（【メイン】+【カウンター】）は【メイン】を持つので出る。"""
    gm, p1, p2, L = _setup(["OP11-080"])
    assert p1.hand[0].uuid in _play_uuids(gm, p1)


def test_main_plus_trigger_event_in_legal_plays():
    """OP08-076 しぬほど…おいしい♡（【メイン】+【トリガー】）も出る。"""
    gm, p1, p2, L = _setup(["OP08-076"])
    assert p1.hand[0].uuid in _play_uuids(gm, p1)


# --- 実行ガード（play_card_action） ---------------------------------------

def test_counter_only_event_rejected_on_play():
    """カウンター専用イベントを直接プレイしようとすると拒否される（盤面不変）。"""
    gm, p1, p2, L = _setup(["OP09-078"])
    gomu = p1.hand[0]
    hand_before = len(p1.hand)
    trash_before = len(p1.trash)
    don_before = len(p1.don_active)
    with pytest.raises(ValueError):
        gm.play_card_action(p1, gomu)
    assert gomu in p1.hand and len(p1.hand) == hand_before
    assert len(p1.trash) == trash_before        # トラッシュ送りもされない
    assert len(p1.don_active) == don_before      # コストも払われない


def test_main_event_still_playable():
    """【メイン】を持つイベント（OP11-080 ギア2）は従来どおり発動できる。"""
    gm, p1, p2, L = _setup(["OP11-080"])
    gear = p1.hand[0]
    gm.play_card_action(p1, gear)
    # イベントは発動後トラッシュへ（【メイン】解決が中断しなければ）
    assert gear not in p1.hand
