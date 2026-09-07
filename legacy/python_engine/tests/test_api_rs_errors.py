"""不正な行動のエラー文言が Python エンジンと一致する（Rust 化した対戦 API・計画 §15.3）。

**問い**: 対戦 API を Rust エンジンで動かしても、不正な行動に対して**フロントへ返る
`error.message` が Python 版とまったく同じ文字列**か。文言はフロントがそのまま表示する
契約（`/api/game/action` の `error.message`）なので、1 文字でも変われば UI が変わる。

検査の仕方（1 ケースにつき 3 つを突き合わせる）:

1. **Rust**: `engine_rs.RsGame` に不正な行動を投げて `ValueError` の文言を採る。
2. **Python（オラクル）**: 同じ盤面を記録 v5 の `hidden` から `GameManager` へ復元し
   （`legacy/python_engine/core/rs_bridge.py`）、`action_api` に同じ不正な行動を投げて文言を採る。
3. **転記**: Python の実装（`core/action_api.py` ／ `core/gamestate.py::_validate_action` ／
   `core/engine/turn_flow.py` ／ `core/engine/battle.py` ／ `core/engine/card_moves.py`）から
   **本ファイルへ書き写した literal**。

3 つが一致して合格。2 だけでは「両方同時に壊れた」を見逃し、3 だけでは「Python 側が変わった」を
見逃すため、両方を持つ。

実行: OPCG_LOG_SILENT=1 python -m pytest tests/test_api_rs_errors.py -q -s -p no:cacheprovider
"""
import conftest  # noqa: F401  (google スタブ注入 & sys.path 設定)

import pytest

from opcg_sim.api import engine_rs
from legacy.python_engine.core import action_api
from opcg_sim.src.models.models import CardInstance
from opcg_sim.src.utils.loader import CardLoader

pytestmark = pytest.mark.legacy

P1 = "P1"
P2 = "P2"


@pytest.fixture(scope="module")
def db():
    import os
    data = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "opcg_sim", "data", "opcg_cards.json")
    loader = CardLoader(data)
    loader.load()
    for cid in list(loader.raw_db.keys()):
        loader.get_card(cid)
    return loader


def _deck(db):
    """リーダー 1 枚＋同一キャラ 50 枚（`tests/test_api.py` のデッキ stub と同じ組み方）。

    キャラは**効果を持たないコスト 2 の 1 枚**を使う: ターン 1（ドン!!1 枚）では払えず、
    ターン 3（3 枚）では払える＝「コスト不足」と「登場したターンのキャラ」の両方を組める。
    効果を持たない＝登場で対話（中断）が立たないので、盤面の前提が読みどおりに揃う。
    """
    leader_master = next(c for c in db.cards.values() if c.type.name == "LEADER")
    char_master = next(c for c in db.cards.values()
                       if c.type.name == "CHARACTER" and c.cost == 2
                       and not c.abilities and not c.trigger_text)
    return leader_master, char_master


def _new_game(db, first_player="p1"):
    leader_master, char_master = _deck(db)
    leader1 = CardInstance(leader_master, P1)
    leader2 = CardInstance(leader_master, P2)
    cards1 = [CardInstance(char_master, P1) for _ in range(50)]
    cards2 = [CardInstance(char_master, P2) for _ in range(50)]
    return engine_rs.RsGame.create(P1, P2, leader1, cards1, leader2, cards2,
                                   first_player, card_db=db)


def _to_turn(game, turn: int):
    """TURN_END を繰り返して指定ターンまで進める。"""
    while game.turn_count < turn:
        game.apply_game_action(game.turn_player_name, "TURN_END", {})
    return game


def _to_main(game):
    """マリガンを両者キープで抜けて MAIN（ターン 1）にする。"""
    for _ in range(2):
        pending = game.get_pending_request(False)
        game.apply_game_action(pending["player_id"], "KEEP_HAND", {})
    assert game.phase.name == "MAIN"
    return game


def _msg_rust(game, kind, player_id, action_type, arg):
    """Rust エンジンへ不正な行動を投げ、`ValueError` の文言を返す（例外が出なければ失敗）。"""
    with pytest.raises(ValueError) as exc:
        if kind == "battle":
            game.apply_battle_action(player_id, action_type, arg)
        else:
            game.apply_game_action(player_id, action_type, arg)
    return str(exc.value)


def _py_manager(game, db):
    """Rust の盤面を記録 v5 の `hidden` から Python の `GameManager` へ復元する（オラクル側）。

    本番の `RsGame` はもうこの経路を持たない（2026-09-07・第 2 段 `rs-archive-cutover` で
    CPU の思考が Rust に移り、暫定経路 `py_manager()` を撤去した）。**この 3 者照合のためだけ**に
    ここで組む＝`legacy` マーカー付きのこのファイルだけが Python エンジンを触る。
    """
    from legacy.python_engine.core import rs_bridge
    manager = rs_bridge.manager_from_hidden(
        db, game.hidden(), suppress_pending=False,
        names={"p1": game.p1_name, "p2": game.p2_name})
    rs_bridge.attach_shallow_interaction(manager, game.get_pending_request(False))
    return manager


def _msg_python(game, db, kind, player_id, action_type, arg):
    """同じ盤面を Python の `GameManager` へ復元し、同じ行動の `ValueError` 文言を返す。"""
    manager = _py_manager(game, db)
    manager.action_events = []
    player = manager.p1 if manager.p1.name == player_id else manager.p2
    with pytest.raises(ValueError) as exc:
        if kind == "battle":
            action_api.apply_battle_action(manager, player, action_type, arg)
        else:
            action_api.apply_game_action(manager, player, action_type, arg)
    return str(exc.value)


def _check(game, db, kind, player_id, action_type, arg, expected):
    """Rust ＝ Python ＝ 転記した literal を突き合わせる（**Python を先に**採る＝Rust 側の
    部分適用が盤面へ残っても、オラクルは行動前の盤面で採れる）。"""
    py = _msg_python(game, db, kind, player_id, action_type, arg)
    rs = _msg_rust(game, kind, player_id, action_type, arg)
    assert py == expected, f"Python 側の文言が転記と違う: {py!r} != {expected!r}"
    assert rs == expected, f"Rust 側の文言が Python と違う: {rs!r} != {expected!r}"


# --- マリガン中 --------------------------------------------------------------

def test_mulligan_phase_rejects_a_main_action(db):
    """要求はマリガンなのに TURN_END（`_validate_action`）。"""
    game = _new_game(db)
    _check(game, db, "game", P1, "TURN_END", {},
           "不適切なアクションです。期待されているアクション: MULLIGAN")


def test_out_of_turn_player_during_mulligan(db):
    """先行（P1）の要求中に後攻（P2）が答える（`_validate_action` の手番違い）。

    文言に入るのは**プレイヤー名**（Python は `pending["player_id"]`）＝席名 "p1" ではない。
    """
    game = _new_game(db)
    _check(game, db, "battle", P2, "SELECT_BLOCKER", None,
           "現在は P1 のターン/フェイズです。")


def test_unknown_game_action(db):
    """辞書に無いアクション種別（`action_api` の else 分岐）。"""
    game = _new_game(db)
    _check(game, db, "game", P1, "NO_SUCH_ACTION", {},
           "不明なアクションです: NO_SUCH_ACTION")


def test_mulligan_twice(db):
    """同じプレイヤーが 2 度マリガンする（`turn_flow.do_mulligan`）。

    `do_mulligan` は `_validate_action` を通らない（フェイズと `mulligan_done` を直接見る）。
    """
    game = _new_game(db)
    game.apply_game_action(P1, "MULLIGAN", {})
    _check(game, db, "game", P1, "MULLIGAN", {}, "既にマリガンを実施済みです。")


# --- メインフェイズ ----------------------------------------------------------

def test_mulligan_after_main(db):
    """マリガンフェイズを抜けてからのマリガン（`turn_flow.do_mulligan`）。"""
    game = _to_main(_new_game(db))
    _check(game, db, "game", P1, "MULLIGAN", {}, "マリガンフェーズではありません。")


def test_keep_hand_after_main(db):
    """マリガンフェイズを抜けてからのキープ（`turn_flow.keep_hand`）。"""
    game = _to_main(_new_game(db))
    _check(game, db, "game", P1, "KEEP_HAND", {}, "マリガンフェーズではありません。")


def test_play_unknown_card(db):
    """手札に無い uuid をプレイ（`action_api` の PLAY 分岐）。"""
    game = _to_main(_new_game(db))
    _check(game, db, "game", P1, "PLAY", {"uuid": "no-such-uuid"},
           "対象のカードが手札にありません。")


def test_play_without_uuid(db):
    """uuid を渡さない PLAY（Python も手札の探索が空振りして同じ文言）。"""
    game = _to_main(_new_game(db))
    _check(game, db, "game", P1, "PLAY", {}, "対象のカードが手札にありません。")


def test_play_not_enough_don(db):
    """アクティブなドン!!より高いコストのカードをプレイ（`card_moves.pay_cost`）。"""
    game = _to_main(_new_game(db))
    board = game.board()
    hand = board["players"]["p1"]["zones"]["hand"]
    expensive = next((c for c in hand if c["cost"] > 1), None)
    assert expensive, "ターン 1 のドン!!（1 枚）で払えないカードが手札に要る"
    _check(game, db, "game", P1, "PLAY", {"uuid": expensive["uuid"]},
           "ドン!!が不足しています。")


def test_unknown_battle_action_during_block_step(db):
    """戦闘中に未知の戦闘アクション（`_validate_action`＝期待は要求どおりの種別）。"""
    game = _to_turn(_to_main(_new_game(db)), 3)
    p1_leader = game.board()["players"]["p1"]["leader"]["uuid"]
    p2_leader = game.board()["players"]["p2"]["leader"]["uuid"]
    game.apply_game_action(P1, "ATTACK", {"uuid": p1_leader, "target_ids": [p2_leader]})
    # ブロッカーが場に居ないので要求はカウンターステップまで進む（`engine/battle.py`）。
    assert game.get_pending_request(False)["action"] == "SELECT_COUNTER"
    _check(game, db, "battle", P2, "NO_SUCH_BATTLE_ACTION", None,
           "不適切なアクションです。期待されているアクション: SELECT_COUNTER")


def test_attack_self(db):
    """自分自身を攻撃対象にする（`action_api` の ATTACK 分岐）。"""
    game = _to_main(_new_game(db))
    leader = game.board()["players"]["p1"]["leader"]["uuid"]
    _check(game, db, "game", P1, "ATTACK",
           {"uuid": leader, "target_ids": [leader]},
           "自分自身を攻撃対象に選択することはできません。")


def test_attack_without_payload(db):
    """uuid も target_ids も無い ATTACK（両方 None＝「同じ」と判定される Python の順序）。"""
    game = _to_main(_new_game(db))
    _check(game, db, "game", P1, "ATTACK", {},
           "自分自身を攻撃対象に選択することはできません。")


def test_attack_unknown_attacker(db):
    """自分の場に居ない uuid でアタック（`action_api` の ATTACK 分岐）。"""
    game = _to_main(_new_game(db))
    target = game.board()["players"]["p2"]["leader"]["uuid"]
    _check(game, db, "game", P1, "ATTACK",
           {"uuid": "no-such-uuid", "target_ids": [target]},
           "アタックするカードが見つかりません。")


def test_attack_unknown_target(db):
    """相手の場に居ない uuid を攻撃対象にする（`action_api` の ATTACK 分岐）。"""
    game = _to_main(_new_game(db))
    leader = game.board()["players"]["p1"]["leader"]["uuid"]
    _check(game, db, "game", P1, "ATTACK",
           {"uuid": leader, "target_ids": ["no-such-uuid"]},
           "攻撃対象が見つかりません。")


def test_attack_on_the_first_turn(db):
    """各プレイヤーの最初のターン（`turn_count <= 2`）はアタックできない（`engine/battle.py`）。"""
    game = _to_main(_new_game(db))
    p1_leader = game.board()["players"]["p1"]["leader"]["uuid"]
    p2_leader = game.board()["players"]["p2"]["leader"]["uuid"]
    _check(game, db, "game", P1, "ATTACK",
           {"uuid": p1_leader, "target_ids": [p2_leader]},
           "最初のターンはアタックできません。")


def test_attach_don_unknown_card(db):
    """自分の場に居ない uuid へドン!!付与（`action_api` の ATTACH_DON 分岐）。"""
    game = _to_main(_new_game(db))
    _check(game, db, "game", P1, "ATTACH_DON", {"uuid": "no-such-uuid"},
           "ドン!!を付与する対象のカードが見つかりません。")


def test_attach_don_without_active_don(db):
    """アクティブなドン!!を使い切ってからの付与（`action_api` の ATTACH_DON 分岐）。"""
    game = _to_main(_new_game(db))
    leader = game.board()["players"]["p1"]["leader"]["uuid"]
    game.apply_game_action(P1, "ATTACH_DON", {"uuid": leader})   # ターン 1 のドン!!は 1 枚
    assert game.board()["players"]["p1"]["don_active"] == []
    _check(game, db, "game", P1, "ATTACH_DON", {"uuid": leader},
           "アクティブなドン!!が不足しています。")


def test_activate_main_unknown_card(db):
    """自分の場に居ない uuid の起動メイン（`action_api` の ACTIVATE_MAIN 分岐）。"""
    game = _to_main(_new_game(db))
    _check(game, db, "game", P1, "ACTIVATE_MAIN", {"uuid": "no-such-uuid"},
           "効果を発動するカードが見つかりません。")


def test_select_blocker_outside_battle(db):
    """戦闘中でないのにブロッカー選択（`_validate_action`）。"""
    game = _to_main(_new_game(db))
    _check(game, db, "battle", P1, "SELECT_BLOCKER", None,
           "不適切なアクションです。期待されているアクション: MAIN_ACTION")


def test_select_counter_outside_battle(db):
    """戦闘中でないのにカウンター選択（`_validate_action`）。"""
    game = _to_main(_new_game(db))
    _check(game, db, "battle", P2, "SELECT_COUNTER", None,
           "現在は P1 のターン/フェイズです。")


# --- 2 ターン目以降（アタックの前提条件）-------------------------------------


def test_attack_with_a_rested_attacker(db):
    """レスト中のカードでアタックする（`engine/battle.py`）。"""
    game = _to_turn(_to_main(_new_game(db)), 3)
    p1_leader = game.board()["players"]["p1"]["leader"]["uuid"]
    p2_leader = game.board()["players"]["p2"]["leader"]["uuid"]
    game.apply_game_action(P1, "ATTACK", {"uuid": p1_leader, "target_ids": [p2_leader]})
    # 攻撃してレストになったリーダーで、もう一度アタックする。
    while game.get_pending_request(False)["action"] != "MAIN_ACTION":
        pending = game.get_pending_request(False)
        game.apply_battle_action(pending["player_id"], "PASS", None)
    assert game.board()["players"]["p1"]["leader"]["is_rest"] is True
    _check(game, db, "game", P1, "ATTACK",
           {"uuid": p1_leader, "target_ids": [p2_leader]},
           "アタックするカードはアクティブ状態でなければなりません。")


def test_attack_an_active_character(db):
    """アクティブな相手キャラは攻撃できない（レスト状態のみ・`engine/battle.py`）。"""
    game = _to_turn(_to_main(_new_game(db)), 4)   # p2 のターン（キャラを 1 体出す）
    hand = game.board()["players"]["p2"]["zones"]["hand"]
    cheap = hand[0]
    game.apply_game_action(P2, "PLAY", {"uuid": cheap["uuid"]})
    game.apply_game_action(P2, "TURN_END", {})
    p1_leader = game.board()["players"]["p1"]["leader"]["uuid"]
    victim = game.board()["players"]["p2"]["zones"]["field"][0]
    assert victim["is_rest"] is False
    _check(game, db, "game", P1, "ATTACK",
           {"uuid": p1_leader, "target_ids": [victim["uuid"]]},
           "レスト状態のキャラクターのみ攻撃可能です。")


def test_attack_with_a_newly_played_character(db):
    """登場したターンのキャラはアタックできない（速攻を除く・`engine/battle.py`）。"""
    game = _to_turn(_to_main(_new_game(db)), 3)
    hand = game.board()["players"]["p1"]["zones"]["hand"]
    cheap = hand[0]
    game.apply_game_action(P1, "PLAY", {"uuid": cheap["uuid"]})
    played = game.board()["players"]["p1"]["zones"]["field"][0]
    assert played["is_rest"] is False
    p2_leader = game.board()["players"]["p2"]["leader"]["uuid"]
    _check(game, db, "game", P1, "ATTACK",
           {"uuid": played["uuid"], "target_ids": [p2_leader]},
           "登場したターンのキャラクターは攻撃できません（速攻を除く）。")


# --- HTTP 層（`error.message` がそのままレスポンスに載る）---------------------

def test_http_error_message_is_the_engine_message(db, monkeypatch):
    """`/api/game/action` の `error.message` がエンジンの文言をそのまま返す（契約）。"""
    from fastapi.testclient import TestClient
    from opcg_sim.api import app as A
    from opcg_sim.api import state
    from opcg_sim.api.services import decks as deck_svc

    leader_master, char_master = _deck(db)

    def _stub_load_deck_mixed(source_str, owner_id):
        return (CardInstance(leader_master, owner_id),
                [CardInstance(char_master, owner_id) for _ in range(50)])

    monkeypatch.setattr(deck_svc, "load_deck_mixed", _stub_load_deck_mixed)
    state.clear_all()
    try:
        with TestClient(A.app) as client:
            gid = client.post("/api/game/create", json={
                "p1_deck": "db:x", "p2_deck": "db:y", "p1_name": P1, "p2_name": P2,
            }).json()["game_id"]
            r = client.post("/api/game/action", json={
                "game_id": gid, "player_id": P1, "action": "TURN_END"})
            body = r.json()
            assert r.status_code == 200 and body["success"] is False
            assert body["error"]["message"] == \
                "不適切なアクションです。期待されているアクション: MULLIGAN"
    finally:
        state.clear_all()


# --- `hidden` の往復（暫定 CPU 経路とリプレイフレームの土台）-------------------

def test_hidden_json_round_trips_through_the_engine(db):
    """`hidden_json()` →`Game.from_hidden()` で**同じ盤面**が戻る（計画 §15.2）。

    `hidden`（記録 v5）は「盤面を完全に再構成できる内部状態」という契約で、暫定 CPU 経路
    （`rs_bridge.manager_from_hidden`）とリプレイフレームがこれに乗る。欄を 1 つ落としても
    盤面 dict には出ない（＝黙って壊れる）ことがあるため、往復で明示的に見る。

    **中断（対話）スタック・誘発待ち行列は `hidden` に無い**（件数だけ持つ）＝往復で失われる。
    ここでは中断の立っていない局面で見る（限界は `rs_bridge` の docstring）。
    """
    import json
    import opcg_engine

    game = _to_turn(_to_main(_new_game(db)), 3)
    hand = game.board()["players"]["p1"]["zones"]["hand"]
    game.apply_game_action(P1, "PLAY", {"uuid": hand[0]["uuid"]})   # 場・トラッシュ・ドン!!を動かす
    game.apply_game_action(P1, "ATTACH_DON", {"uuid": game.board()["players"]["p1"]["leader"]["uuid"]})
    assert game._game.has_interaction is False

    restored = opcg_engine.Game.from_hidden(json.dumps(game.hidden()), P1, P2)
    assert json.loads(restored.board_json()) == game.board()
    assert json.loads(restored.pending_json()) == game.get_pending_request(False)
    assert json.loads(restored.hidden_json()) == game.hidden()
