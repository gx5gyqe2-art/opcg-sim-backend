"""【メイン】効果を持つイベントが合法手（PLAY）に列挙されること（Rust 化後の退行・2026-09-08）。

`rules/legal.rs` が「イベントは常に登場不可」と列挙していたため、Rust 切替（2026-09-07）以降の
CPU（探索の候補は同じ列挙を使う）は【メイン】イベント（除去・ドロー・パンプ）を一度も打てなかった
（分岐点シナリオ 36 本で発見・`docs/reports/2026-09-08_scenario_analysis_02.md`・計画 §8.26）。
人間はフロントから `apply_game_action("PLAY")` を直接呼ぶので気付かない経路。Python 版
（`_event_has_main_play`）と同じく「ON_PLAY／ACTIVATE_MAIN を持つイベントは列挙、カウンター／
トリガー専用は列挙しない」を、API と同じ `RsGame` で固定する。
"""
import os

import pytest

import _bootstrap  # noqa: F401
from opcg_sim.api.engine_rs import RsGame

MAIN_EVENT = "OP07-096"        # 嵐脚: コスト 1・【メイン】カード 1 枚を引く（追加条件はトラッシュ 10 枚以上）
COUNTER_EVENT = "EB01-009"     # コスト 1・【カウンター】のみ
LEADER = "EB01-001"
VANILLA_CHAR = "EB01-005"


def _game_at_p1_main():
    if not os.path.exists(os.path.join(os.path.dirname(__file__), "..", "opcg_sim", "data", "opcg_effects.json")):
        pytest.skip("効果 JSON（生成物）が無い")
    p1_deck = [MAIN_EVENT] * 25 + [COUNTER_EVENT] * 25
    p2_deck = [VANILLA_CHAR] * 50
    game = RsGame.create_from_ids("P1", "P2", LEADER, p1_deck, LEADER, p2_deck, first_player="p1")
    for _ in range(2):
        pending = game.get_pending_request(False)
        assert pending["action"] == "MULLIGAN", pending
        game.apply_game_action(pending["player_id"], "KEEP_HAND", {})
    pending = game.get_pending_request(False)
    assert pending["player_id"] == "P1" and pending["action"] == "MAIN_ACTION", pending
    return game


def test_main_events_are_legal_and_counter_only_events_are_not():
    game = _game_at_p1_main()
    p1 = game.board()["players"]["p1"]
    assert len(p1["don_active"]) == 1, "先攻 1 ターン目のドン!! 1 枚（コスト 1 のイベントを払える）"
    hand = p1["zones"]["hand"]
    mains = {c["uuid"] for c in hand if c["card_id"] == MAIN_EVENT}
    counters = {c["uuid"] for c in hand if c["card_id"] == COUNTER_EVENT}
    assert mains or counters, "手札にイベントがある盤面のはず"
    legal = game.get_legal_actions("P1")
    plays = {m["payload"]["uuid"] for m in legal if m["action_type"] == "PLAY"}
    assert mains <= plays, {"mains": mains, "plays": plays}
    assert not (counters & plays), "【カウンター】専用のイベントはメインで列挙しない"
    # 列挙した手がそのまま適用できる（＝探索が選んだ手を実対局へ出せる）。
    if mains:
        uuid = next(iter(mains))
        hand_before = len(hand)
        game.apply_game_action("P1", "PLAY", {"uuid": uuid})
        p1 = game.board()["players"]["p1"]
        assert uuid in {c["uuid"] for c in p1["zones"]["trash"]}, "発動したイベントはトラッシュへ"
        assert len(p1["zones"]["hand"]) == hand_before, "1 枚使って 1 枚引く"
        assert not p1["don_active"], "コスト 1 を払う"
