"""カウンターステップで【カウンター】イベントが選べること（実プレイの退行・2026-09-08）。

Rust 切替後、`pending.rs::counter_candidates` が「カウンター値を持つ手札」だけを候補にしていて、
**【カウンター】トリガのイベント**（コストをアクティブなドン!!で払える分）が `selectable_uuids` に
載らなかった＝フロントの対戦でカウンターイベントが使えなかった（ユーザ報告）。Python 版は
`interaction.get_pending_request` の BATTLE_COUNTER 分岐で (a) カウンター値 (b) 払えるイベント の
両方を出していた。golden の random 帯は防御側のドン!!が尽きた盤面ばかりでこの経路を通らないため、
ここで API と同じ `RsGame` を実際に駆動して固定する。

盤面の作り方: p1 のデッキを「コスト 1 の【カウンター】イベント EB01-009」で埋めて先攻にし、
両者の 1 ターン目（攻撃不可）と p1 の 2 ターン目を TURN_END で流す（p1 のドン!! 3 枚はアクティブの
まま）→ p2 がリーダーで p1 のリーダーを攻撃 → p1 のカウンター要求。
"""
import os

import pytest

import _bootstrap  # noqa: F401
from opcg_sim.api.engine_rs import RsGame

COUNTER_EVENT = "EB01-009"     # コスト 1・【カウンター】・カウンター値 0
VANILLA_CHAR = "EB01-005"      # 能力なし
LEADER = "EB01-001"


def _drive_to_counter_step(spend_don: bool = False):
    """p1 先攻。両者 1 ターン目は攻撃できないので、p1 の 2 ターン目（ドン!! 3 枚）を終えてから
    p2 がリーダーで p1 のリーダーを攻撃する。`spend_don` なら p1 は 2 ターン目にドン!!を全部
    リーダーへ付与してアクティブ 0 で防御に入る。"""
    p1_deck = [COUNTER_EVENT] * 30 + [VANILLA_CHAR] * 20
    p2_deck = [VANILLA_CHAR] * 50
    game = RsGame.create_from_ids("P1", "P2", LEADER, p1_deck, LEADER, p2_deck, first_player="p1")
    for _ in range(2):
        pending = game.get_pending_request(False)
        assert pending["action"] == "MULLIGAN", pending
        game.apply_game_action(pending["player_id"], "KEEP_HAND", {})
    board = game.board()
    p1_leader = board["players"]["p1"]["leader"]["uuid"]
    p2_leader = board["players"]["p2"]["leader"]["uuid"]
    for turn_player in ("P1", "P2", "P1"):
        pending = game.get_pending_request(False)
        assert pending["player_id"] == turn_player and pending["action"] == "MAIN_ACTION", pending
        if turn_player == "P1" and spend_don and game.turn_count >= 3:
            while game.board()["players"]["p1"]["don_active"]:
                game.apply_game_action("P1", "ATTACH_DON", {"card_id": p1_leader})
        game.apply_game_action(turn_player, "TURN_END", {})
    pending = game.get_pending_request(False)
    assert pending["player_id"] == "P2" and pending["action"] == "MAIN_ACTION", pending
    game.apply_game_action("P2", "ATTACK", {"card_id": p2_leader, "target_ids": [p1_leader]})
    return game


def test_counter_events_are_offered_and_playable():
    if not os.path.exists(os.path.join(os.path.dirname(__file__), "..", "opcg_sim", "data", "opcg_effects.json")):
        pytest.skip("効果 JSON（生成物）が無い")
    game = _drive_to_counter_step()
    pending = game.get_pending_request(False)
    assert pending["player_id"] == "P1" and pending["action"] == "SELECT_COUNTER", pending
    board = game.board()
    p1 = board["players"]["p1"]
    assert len(p1["don_active"]) == 3, "先攻 2 ターン目までのドン!! 3 枚がアクティブのまま"
    hand = p1["zones"]["hand"]
    events = [c["uuid"] for c in hand if c["card_id"] == COUNTER_EVENT]
    assert events, "手札にカウンターイベントがある盤面のはず"
    # (b) 払えるカウンターイベントが候補に載る（カウンター値 0 でも）。
    selectable = set(pending["selectable_uuids"])
    assert set(events) <= selectable, {"events": events, "selectable": sorted(selectable)}
    # 合法手にも同じ候補が載る（CPU・再生・serve が読む口）。
    legal = game.get_legal_actions("P1")
    offered = {m["card_uuid"] for m in legal if m["action_type"] == "SELECT_COUNTER"}
    assert set(events) <= offered
    # 実際に発動できる: コスト 1 を払い、効果が解決してトラッシュへ。
    events_out = game.apply_battle_action("P1", "SELECT_COUNTER", events[0])
    assert any(e.get("type") == "COUNTER" for e in events_out), events_out
    p1 = game.board()["players"]["p1"]
    assert len(p1["don_active"]) == 2, "コスト 1 を払う"
    assert events[0] in {c["uuid"] for c in p1["zones"]["trash"]}


def test_counter_events_need_the_don_to_pay():
    """ドン!!が足りないイベントは出さない（出すと `pay_cost` で例外＝Python 版と同じ判定）。"""
    if not os.path.exists(os.path.join(os.path.dirname(__file__), "..", "opcg_sim", "data", "opcg_effects.json")):
        pytest.skip("効果 JSON（生成物）が無い")
    game = _drive_to_counter_step(spend_don=True)
    pending = game.get_pending_request(False)
    assert pending["player_id"] == "P1" and pending["action"] == "SELECT_COUNTER", pending
    p1 = game.board()["players"]["p1"]
    assert not p1["don_active"], "ドン!!は全部リーダーに付与した"
    events = {c["uuid"] for c in p1["zones"]["hand"] if c["card_id"] == COUNTER_EVENT}
    assert events, "手札にカウンターイベントがある盤面のはず"
    assert not (events & set(pending["selectable_uuids"])), "払えないイベントは候補にしない"
    assert pending.get("can_skip") is True
