"""対象選択の既定解決が「効果の種別」（利益／コスト）で分岐すること（WP `rs-select-fix`）。

`effects/interact.rs::choose_selection` は自分側の up_to 選択を一律「コスト系＝min 件」と
決め打ちしていた（かつ自分のリーダー＋キャラの混在を判別できず `None` → 空選択）ため、
万雷 (OP15-078) の【カウンター】「自分のリーダーかキャラ1枚までを、このバトル中、パワー+1000」や
神の裁き (OP15-075) の【メイン】「自分のリーダーかキャラ1枚までを、このターン中、パワー+1000」の
既定手が実際には**何も選ばない**になっていた（`docs/reports/2026-09-08_select_default_rca.md` §Q4・
`docs/rust_engine_plan.md` §8.27.2 案①）。本ファイルは API と同じ `RsGame` でこの 3 ケースを固定する:

1. 神の裁き（利益系・自分のリーダーかキャラ1枚まで＝リーダーのみが候補）→ 既定手はリーダーを選ぶ。
2. 万雷（利益系・COUNTER）→ 既定手の BUFF 選択は空でない。
3. コスト系（自分のキャラ1枚までを KO する）→ 既定手は今までどおり空のまま（回帰確認）。
"""
import os
import random

import pytest

import _bootstrap  # noqa: F401
from opcg_sim.api.engine_rs import RsGame

ENEL_LEADER = "OP15-058"       # エネル: 【起動メイン】…（本テストでは条件判定にしか使わない）
KAMI_NO_SABAKI = "OP15-075"    # 神の裁き: コスト0・【メイン】ドン‼-1：自分のリーダーかキャラ1枚まで+1000→KO
BANRAI = "OP15-078"            # 万雷: 【カウンター】自分のリーダーかキャラ1枚まで、このバトル中+1000
VANILLA_LEADER = "EB01-001"
VANILLA_CHAR = "EB01-005"
# コスト系代表: 影の集合地（OP06-095）【メイン】/【カウンター】
# 「自分のコスト2以下の特徴《スリラーバーク海賊団》を持つキャラを任意の枚数KOしてもよい」
# （own・FIELD・is_up_to の KO＝RCA Q4 の「コスト系」13 件と同じ形）。対象になるには場に
# 条件を満たすキャラが要るので、ペローナ（OP01-077・コスト1・スリラーバーク海賊団）を先に出す。
SHADOWS_STAGE = "OP06-095"
THRILLER_CHAR = "OP01-077"     # ペローナ: コスト1・特徴《スリラーバーク海賊団》を持つキャラ

EFFECTS_JSON = os.path.join(os.path.dirname(__file__), "..", "opcg_sim", "data", "opcg_effects.json")


def _skip_if_no_effects_json():
    if not os.path.exists(EFFECTS_JSON):
        pytest.skip("効果 JSON（生成物）が無い")


def _keep_hands(game: RsGame):
    for _ in range(2):
        pending = game.get_pending_request(False)
        assert pending["action"] == "MULLIGAN", pending
        game.apply_game_action(pending["player_id"], "KEEP_HAND", {})


def _resolve_default(game: RsGame, player_id: str):
    """`default_interaction_payload` で対話を 1 段だけ既定解決する（API と同じ経路）。"""
    pending = game.get_pending_request(False)
    payload = game.default_interaction_payload(pending)
    game.apply_game_action(player_id, "RESOLVE_EFFECT_SELECTION", payload)


def test_kami_no_sabaki_default_selects_the_leader():
    """神の裁き（コスト0・ドン‼-1）をエネルで打つと、自分への +1000 の候補はリーダーだけ
    （場にキャラがいない盤面）＝以前は `zones_in(["hand","field"])` に leader が含まれず None →
    空選択になっていた。既定手（`get_legal_actions` の唯一の候補）はリーダーを選ぶ。"""
    _skip_if_no_effects_json()
    p1_deck = [KAMI_NO_SABAKI] * 50
    p2_deck = [VANILLA_CHAR] * 50
    game = RsGame.create_from_ids(
        "P1", "P2", ENEL_LEADER, p1_deck, VANILLA_LEADER, p2_deck, first_player="p1"
    )
    _keep_hands(game)
    pending = game.get_pending_request(False)
    assert pending["player_id"] == "P1" and pending["action"] == "MAIN_ACTION", pending
    board = game.board()
    p1 = board["players"]["p1"]
    assert p1["don_active"], "先攻1ターン目のドン!!でドン‼-1のコストを払える盤面のはず"
    hand = p1["zones"]["hand"]
    uuid = next(c["uuid"] for c in hand if c["card_id"] == KAMI_NO_SABAKI)

    game.apply_game_action("P1", "PLAY", {"uuid": uuid})

    # ドン‼-1（RETURN_DON）のコスト選択（候補が1枚でも SelectResource は必ず立つ）。
    pending = game.get_pending_request(False)
    if pending is not None and pending.get("action") == "SELECT_RESOURCE":
        _resolve_default(game, "P1")

    pending = game.get_pending_request(False)
    assert pending is not None, "自分への+1000の対象選択が立つはず"
    assert pending["action"] == "SEARCH_AND_SELECT", pending
    leader_uuid = p1["leader"]["uuid"]
    assert pending["selectable_uuids"] == [leader_uuid], pending
    assert pending["intent"] == "BENEFIT", pending

    # `get_legal_actions` の既定手（唯一の候補）はリーダーを選ぶ（空にならない）。
    legal = game.get_legal_actions("P1")
    assert len(legal) == 1, legal
    move = legal[0]
    assert move["action_type"] == "RESOLVE_EFFECT_SELECTION", move
    assert move["payload"]["selected_uuids"] == [leader_uuid], move

    # 既定解決そのもの（`default_interaction_payload`）も同じ答えを返す。
    payload = game.default_interaction_payload(pending)
    assert payload["selected_uuids"] == [leader_uuid], payload


def _drive_to_p2_second_turn_attack(p1_deck, p2_deck, p1_leader=VANILLA_LEADER, p2_leader=VANILLA_LEADER):
    """p1 先攻。p1/p2/p1 の3ターンを TURN_END で流し、p2 の2ターン目に
    p2 のリーダーで p1 のリーダーを攻撃する（`test_counter_event_offer.py` と同じ組み立て）。"""
    game = RsGame.create_from_ids(
        "P1", "P2", p1_leader, p1_deck, p2_leader, p2_deck, first_player="p1"
    )
    _keep_hands(game)
    board = game.board()
    p1_leader_uuid = board["players"]["p1"]["leader"]["uuid"]
    p2_leader_uuid = board["players"]["p2"]["leader"]["uuid"]
    for turn_player in ("P1", "P2", "P1"):
        pending = game.get_pending_request(False)
        assert pending["player_id"] == turn_player and pending["action"] == "MAIN_ACTION", pending
        game.apply_game_action(turn_player, "TURN_END", {})
    pending = game.get_pending_request(False)
    assert pending["player_id"] == "P2" and pending["action"] == "MAIN_ACTION", pending
    game.apply_game_action("P2", "ATTACK", {"card_id": p2_leader_uuid, "target_ids": [p1_leader_uuid]})
    return game, p1_leader_uuid


def test_banrai_counter_default_buff_target_is_not_empty():
    """万雷を【カウンター】で使ったときの既定手（BUFF「自分のリーダーかキャラ1枚まで+1000」の
    対象選択）が空にならない（RCA (a): 以前は `merged_search_actions` の先頭が常にこの空選択
    だった＝万雷のカウンター価値が名目より弱かった）。"""
    _skip_if_no_effects_json()
    p1_deck = [BANRAI] * 50
    p2_deck = [VANILLA_CHAR] * 50
    game, p1_leader_uuid = _drive_to_p2_second_turn_attack(p1_deck, p2_deck)
    pending = game.get_pending_request(False)
    assert pending["player_id"] == "P1" and pending["action"] == "SELECT_COUNTER", pending
    hand = game.board()["players"]["p1"]["zones"]["hand"]
    counter_uuid = next(c["uuid"] for c in hand if c["card_id"] == BANRAI)
    assert counter_uuid in pending["selectable_uuids"], pending

    game.apply_battle_action("P1", "SELECT_COUNTER", counter_uuid)

    pending = game.get_pending_request(False)
    assert pending is not None, "自分への+1000の対象選択が立つはず"
    assert pending["player_id"] == "P1" and pending["action"] == "SEARCH_AND_SELECT", pending
    assert pending["selectable_uuids"] == [p1_leader_uuid], pending
    assert pending["intent"] == "BENEFIT", pending

    legal = game.get_legal_actions("P1")
    assert len(legal) == 1, legal
    move = legal[0]
    assert move["action_type"] == "RESOLVE_EFFECT_SELECTION", move
    assert move["payload"]["selected_uuids"], "利益系の既定手は空でない（RCA (a) の直接の回帰確認）"
    assert move["payload"]["selected_uuids"] == [p1_leader_uuid], move


def test_cost_type_self_selection_default_stays_empty():
    """コスト系（自分のキャラを任意の枚数KOしてもよい＝影の集合地）の既定手は今までどおり空のまま
    （利益系だけを変える・回帰確認）。

    盤面: p1 が①ペローナ（コスト1・スリラーバーク海賊団）を先に場に出し（1ターン目）、
    ③（p1 の2ターン目）に影の集合地（コスト2）を打つ。KO 候補はペローナ1枚だけになる。
    `random.seed` はこの2枚が両方とも初手＝マリガン後の手札に入る値を固定で使う
    （デッキはこの2種＋埋め草。シャッフルは `create_from_ids` が呼ぶ `random.getrandbits` に従う）。
    """
    _skip_if_no_effects_json()
    random.seed(3)
    p1_deck = [THRILLER_CHAR] * 10 + [SHADOWS_STAGE] * 10 + [VANILLA_CHAR] * 30
    p2_deck = [VANILLA_CHAR] * 50
    game = RsGame.create_from_ids(
        "P1", "P2", VANILLA_LEADER, p1_deck, VANILLA_LEADER, p2_deck, first_player="p1"
    )
    _keep_hands(game)

    hand = game.board()["players"]["p1"]["zones"]["hand"]
    char_uuid = next(c["uuid"] for c in hand if c["card_id"] == THRILLER_CHAR)
    game.apply_game_action("P1", "PLAY", {"uuid": char_uuid})
    # ペローナの【登場時】（デッキ上5枚を見て並べ替え）は既定解決で畳む（本テストの対象外）。
    _resolve_default(game, "P1")
    game.apply_game_action("P1", "TURN_END", {})
    pending = game.get_pending_request(False)
    assert pending["player_id"] == "P2" and pending["action"] == "MAIN_ACTION", pending
    game.apply_game_action("P2", "TURN_END", {})

    pending = game.get_pending_request(False)
    assert pending["player_id"] == "P1" and pending["action"] == "MAIN_ACTION", pending
    p1 = game.board()["players"]["p1"]
    assert len(p1["don_active"]) >= 2, "影の集合地（コスト2）を打てる盤面のはず"
    field_uuids = {c["uuid"] for c in p1["zones"]["field"] if c["card_id"] == THRILLER_CHAR}
    assert field_uuids, "1ターン目に出したペローナが場にいるはず"
    hand = p1["zones"]["hand"]
    stage_uuid = next(c["uuid"] for c in hand if c["card_id"] == SHADOWS_STAGE)

    game.apply_game_action("P1", "PLAY", {"uuid": stage_uuid})
    # 「効果を発動しますか？」（自分のリーダー+1000の後の「してもよい」KO 全体の確認）は受ける。
    pending = game.get_pending_request(False)
    assert pending["action"] == "CONFIRM_OPTIONAL", pending
    game.apply_game_action("P1", "RESOLVE_EFFECT_SELECTION", {"accepted": True})

    pending = game.get_pending_request(False)
    assert pending is not None, "任意 KO の対象選択が立つはず"
    assert pending["player_id"] == "P1" and pending["action"] == "SEARCH_AND_SELECT", pending
    assert set(pending["selectable_uuids"]) == field_uuids, pending
    assert pending["intent"] == "COST", pending

    legal = game.get_legal_actions("P1")
    assert len(legal) == 1, legal
    move = legal[0]
    assert move["action_type"] == "RESOLVE_EFFECT_SELECTION", move
    assert move["payload"]["selected_uuids"] == [], "コスト系の既定手は従来どおり空のまま"
