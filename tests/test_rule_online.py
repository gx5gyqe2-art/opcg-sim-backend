"""ルールモード・オンライン対戦（ルーム/ロビー + WebSocket 同期）のテスト。

Firestore に依存しないよう load_deck_mixed をモックし、実カード DB から
リーダー + キャラ 50 枚のデッキを構築して GameManager を起動する。
"""
import conftest  # noqa: F401  (google スタブ注入 & sys.path 設定)
import pytest
from fastapi.testclient import TestClient

from opcg_sim.api import app as appmod
from opcg_sim.src.models.models import CardInstance


def _build_deck(owner_id):
    leader, cards = None, []
    for cid in appmod.card_db.raw_db.keys():
        c = appmod.card_db.get_card(cid)
        if c is None:
            continue
        if leader is None and c.type.name == "LEADER":
            leader = CardInstance(c, owner_id)
        elif c.type.name == "CHARACTER" and len(cards) < 50:
            cards.append(CardInstance(c, owner_id))
        if leader and len(cards) >= 50:
            break
    return leader, cards


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(appmod, "load_deck_mixed", lambda src, owner: _build_deck(owner))
    monkeypatch.setattr(appmod, "_deck_preview", lambda deck_id, owner: {"leader_id": "L", "leader_name": "Leader"})
    # 各テストでレジストリを汚さないようにクリア
    appmod.RULE_ROOMS.clear()
    appmod.GAMES.clear()
    return TestClient(appmod.app)


def test_rule_room_lifecycle_and_ws_sync(client):
    # ルーム作成 → WAITING
    res = client.post("/api/rule/create", json={"room_name": "Test Room"}).json()
    assert res["success"] and res["status"] == "WAITING"
    gid = res["game_id"]

    # 一覧に出る
    listed = client.get("/api/rule/list").json()["games"]
    assert any(r["game_id"] == gid for r in listed)

    with client.websocket_connect(f"/ws/game/{gid}") as ws1, \
         client.websocket_connect(f"/ws/game/{gid}") as ws2:
        # 接続直後に WAITING 状態が届く
        init = ws1.receive_json()
        assert init["status"] == "WAITING" and init["game_state"] is None
        ws2.receive_json()

        # デッキ選択 → ready が立ち、両者へブロードキャスト
        client.post("/api/rule/action", json={"game_id": gid, "action_type": "SET_DECK", "player_id": "p1", "deck_id": "db:a"})
        assert ws1.receive_json()["ready_states"]["p1"] is True
        ws2.receive_json()
        client.post("/api/rule/action", json={"game_id": gid, "action_type": "SET_DECK", "player_id": "p2", "deck_id": "db:b"})
        ws1.receive_json(); ws2.receive_json()

        # 対局開始 → PLAYING + 実ゲーム状態 + 先攻へマリガン pending
        # 対戦モードの先行はランダム（コイントス）なので、先攻/後攻は固定せず
        # pending_request から動的に解決する。
        started = client.post("/api/rule/action", json={"game_id": gid, "action_type": "START", "player_id": "p1"}).json()
        assert started["status"] == "PLAYING"
        msg = ws1.receive_json()
        ws2.receive_json()
        assert msg["status"] == "PLAYING"
        assert msg["game_state"] is not None
        # マリガンは先攻プレイヤーから要求される（先攻はランダム）
        first = msg["pending_request"]["player_id"]
        assert first in ("p1", "p2")
        assert msg["pending_request"]["action"] == "MULLIGAN"
        second = "p2" if first == "p1" else "p1"

        # ゲームアクション（/api/game/action）でもルームへブロードキャストされる
        kept = client.post("/api/game/action", json={"game_id": gid, "action": "KEEP_HAND", "player_id": first, "payload": {}}).json()
        assert kept["success"]
        broadcast = ws1.receive_json()
        ws2.receive_json()
        # マリガン要求が後攻プレイヤーへ移る
        assert broadcast["pending_request"]["player_id"] == second


def test_game_state_fetch_resync(client):
    """/api/game/state はルーム対局の現在状態を読み取り専用で返す（WS取りこぼし時の
    再同期フォールバック）。冪等で、盤面/手番を一切変えない。"""
    gid = client.post("/api/rule/create", json={"room_name": "R"}).json()["game_id"]
    client.post("/api/rule/action", json={"game_id": gid, "action_type": "SET_DECK", "player_id": "p1", "deck_id": "db:a"})
    client.post("/api/rule/action", json={"game_id": gid, "action_type": "SET_DECK", "player_id": "p2", "deck_id": "db:b"})
    client.post("/api/rule/action", json={"game_id": gid, "action_type": "START", "player_id": "p1"})

    # 読み取り専用フェッチ: WS と同形（STATE_UPDATE / game_state / pending_request）。
    s1 = client.get(f"/api/game/state?game_id={gid}").json()
    assert s1["status"] == "PLAYING"
    assert s1["game_state"] is not None
    assert s1["pending_request"]["action"] == "MULLIGAN"
    pid = s1["pending_request"]["player_id"]

    # 冪等: 何度呼んでも手番（pending の宛先）は変化しない（副作用なし）。
    s2 = client.get(f"/api/game/state?game_id={gid}").json()
    assert s2["pending_request"]["player_id"] == pid
    assert s2["game_state"]["turn_info"]["turn_count"] == s1["game_state"]["turn_info"]["turn_count"]


def test_game_state_fetch_unknown_game(client):
    """未知の game_id はエラー（success=False）を返す。"""
    res = client.get("/api/game/state?game_id=nope").json()
    assert res["success"] is False


def test_rule_start_requires_both_ready(client):
    gid = client.post("/api/rule/create", json={"room_name": "R"}).json()["game_id"]
    # p1 のみ ready
    client.post("/api/rule/action", json={"game_id": gid, "action_type": "SET_DECK", "player_id": "p1", "deck_id": "db:a"})
    res = client.post("/api/rule/action", json={"game_id": gid, "action_type": "START", "player_id": "p1"}).json()
    assert res["success"] is False
    assert appmod.RULE_ROOMS[gid]["status"] == "WAITING"
