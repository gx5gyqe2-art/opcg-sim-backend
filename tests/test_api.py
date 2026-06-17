"""FastAPI HTTP/WS 層のスモークテスト（`opcg_sim/api/app.py`）。

エンジン挙動は他スイートが担保済みなので、ここでは**API 契約**（ステータス／
レスポンス形／セッションヘッダ／主要フローの疎通／エラーハンドリング）のみを検証する。

Firestore は本テスト環境で未初期化（`app.db is None`・conftest が google.cloud を
スタブ化）。デッキ読込 `load_deck_mixed` は Firestore に依存するため、ローカル
カード DB からデッキを組む stub に差し替えて対局生成系を疎通させる。デッキ CRUD
（Firestore 必須）は「DB 未初期化でも 200＋整形済みエラー/空応答」を確認する。

実行: OPCG_LOG_SILENT=1 python -m pytest tests/test_api.py -q -s -p no:cacheprovider
"""
import conftest  # noqa: F401  (google スタブ注入 & sys.path 設定)

import pytest
from fastapi.testclient import TestClient

from opcg_sim.api import app as A
from opcg_sim.src.models.models import CardInstance


# --- フィクスチャ -----------------------------------------------------------

def _load_card_db():
    """カード DB を全件マテリアライズし、リーダー1枚と適当なキャラ1枚を返す。"""
    db = A.card_db
    for cid in list(db.raw_db.keys()):
        db.get_card(cid)
    leader = next(c for c in db.cards.values() if c.type.name == "LEADER")
    char = next(c for c in db.cards.values() if c.type.name == "CHARACTER")
    return leader, char


@pytest.fixture
def client(monkeypatch):
    """`load_deck_mixed` を DB 非依存の stub に差し替えた TestClient。

    レジストリ（GAMES/SANDBOX_GAMES/RULE_ROOMS/CPU_GAMES）はモジュール共有なので
    各テストの前後でクリアし、テスト間の汚染を防ぐ。
    """
    leader_master, char_master = _load_card_db()

    def _stub_load_deck_mixed(source_str, owner_id):
        leader = CardInstance(leader_master, owner_id)
        cards = [CardInstance(char_master, owner_id) for _ in range(50)]
        return leader, cards

    monkeypatch.setattr(A, "load_deck_mixed", _stub_load_deck_mixed)

    for reg in (A.GAMES, A.SANDBOX_GAMES, A.RULE_ROOMS, A.CPU_GAMES):
        reg.clear()

    with TestClient(A.app) as c:
        yield c

    for reg in (A.GAMES, A.SANDBOX_GAMES, A.RULE_ROOMS, A.CPU_GAMES):
        reg.clear()


# --- 基盤 -------------------------------------------------------------------

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["constants_loaded"] is True


def test_session_header_roundtrip(client):
    """ミドルウェアが X-Session-ID をレスポンスへ反映する（指定時はそのまま）。"""
    r = client.get("/health", headers={"X-Session-ID": "test-sess-1"})
    assert r.headers.get("X-Session-ID") == "test-sess-1"
    # 未指定でも自動採番されて返る
    r2 = client.get("/health")
    assert r2.headers.get("X-Session-ID")


def test_get_cards(client):
    r = client.get("/api/cards")
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True
    assert isinstance(body["cards"], list) and len(body["cards"]) > 0
    assert "uuid" in body["cards"][0] and "name" in body["cards"][0]


def test_cards_etag_conditional_get(client):
    """/api/cards は ETag を返し、If-None-Match 一致時は本体なしの 304 を返す。"""
    r = client.get("/api/cards")
    assert r.status_code == 200
    etag = r.headers.get("ETag")
    assert etag and len(etag) > 0
    # 同じ ETag を提示すると 304（本体なし）
    r304 = client.get("/api/cards", headers={"If-None-Match": etag})
    assert r304.status_code == 304
    assert not r304.content
    # 異なる ETag なら通常どおり 200＋本体
    r200 = client.get("/api/cards", headers={"If-None-Match": '"stale"'})
    assert r200.status_code == 200 and r200.json()["success"] is True


def test_assets_version(client):
    """画像キャッシュ版数を返す（非空文字列・カードDB由来で安定）。"""
    r = client.get("/api/assets/version")
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True
    assert isinstance(body["v"], str) and len(body["v"]) > 0
    # 同一プロセス内では安定（リクエストごとに変わらない）
    assert client.get("/api/assets/version").json()["v"] == body["v"]


def test_log_single_and_batch(client):
    r1 = client.post("/api/log", json={"level": "info", "action": "t", "msg": "hi"})
    assert r1.status_code == 200 and r1.json()["mode"] == "single"
    r2 = client.post("/api/log", json=[{"sessionId": "s", "msg": "a"}])
    assert r2.status_code == 200 and r2.json()["mode"] == "batch"


# --- ルールゲーム（対局）フロー ---------------------------------------------

def _create_game(client, **extra):
    payload = {"p1_deck": "db:x", "p2_deck": "db:y", "p1_name": "P1", "p2_name": "P2"}
    payload.update(extra)
    return client.post("/api/game/create", json=payload)


def test_game_create_and_state(client):
    r = _create_game(client)
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True
    gid = body["game_id"]
    assert gid and gid in A.GAMES
    assert body["game_state"]["turn_info"]["current_phase"] == "MULLIGAN"

    # 読み取り専用 state が同一ゲームを返す（副作用なし）
    s = client.get("/api/game/state", params={"game_id": gid})
    assert s.status_code == 200
    sbody = s.json()
    assert sbody["success"] is True and sbody["game_id"] == gid


def test_game_full_flow_through_mulligan_and_turn_end(client):
    gid = _create_game(client).json()["game_id"]

    # pending（マリガン要求）を両プレイヤー分 KEEP_HAND でこなす
    body = client.get("/api/game/state", params={"game_id": gid}).json()
    for _ in range(4):
        pending = body.get("pending_request")
        if not pending or pending["action"] != "MULLIGAN":
            break
        r = client.post("/api/game/action", json={
            "game_id": gid, "player_id": pending["player_id"], "action": "KEEP_HAND",
        })
        assert r.status_code == 200 and r.json()["success"] is True
        body = r.json()

    # マリガンを抜けて対局フェーズ（MAIN）に入り、手番プレイヤーへ操作が渡っている
    assert body["game_state"]["turn_info"]["current_phase"] == "MAIN"

    # 手番プレイヤーがターン終了できる
    turn_pid = body["game_state"]["turn_info"]["active_player_id"]
    pname = "P1" if turn_pid == "p1" else "P2"
    r = client.post("/api/game/action", json={"game_id": gid, "player_id": pname, "action": "TURN_END"})
    assert r.status_code == 200 and r.json()["success"] is True


def test_game_action_unknown_game_returns_structured_error(client):
    r = client.post("/api/game/action", json={"game_id": "nope", "player_id": "P1", "action": "TURN_END"})
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is False
    assert body["error"] and body["error"].get("code")


def test_game_state_unknown_game(client):
    r = client.get("/api/game/state", params={"game_id": "missing"})
    assert r.status_code == 200 and r.json()["success"] is False


# --- CPU 対戦 ---------------------------------------------------------------

def test_cpu_step_contract(client):
    gid = _create_game(client, vs_cpu=True, cpu_deck="db:cpu", cpu_difficulty="hard").json()["game_id"]
    assert gid in A.CPU_GAMES and A.CPU_GAMES[gid]["difficulty"] == "hard"

    r = client.post("/api/game/cpu/step", json={"game_id": gid})
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True
    assert "cpu_acted" in body and isinstance(body["cpu_acted"], bool)
    assert body["waiting_for"] in ("cpu", "human", "human_decision", "game_over")


def test_cpu_step_on_non_cpu_game(client):
    gid = _create_game(client).json()["game_id"]  # 通常対局（CPU ではない）
    r = client.post("/api/game/cpu/step", json={"game_id": gid})
    assert r.status_code == 200 and r.json()["success"] is False


# --- サンドボックス（フリーモード） -----------------------------------------

def test_sandbox_create_and_list(client):
    r = client.post("/api/sandbox/create", json={"p1_name": "A", "p2_name": "B", "room_name": "R"})
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True
    gid = body["game_id"]
    assert gid in A.SANDBOX_GAMES

    lst = client.get("/api/sandbox/list").json()
    assert lst["success"] is True
    assert any(g["game_id"] == gid for g in lst["games"])


def test_sandbox_action_unknown_game(client):
    r = client.post("/api/sandbox/action", json={"game_id": "x", "action_type": "NOOP", "player_id": "p1"})
    assert r.status_code == 200 and r.json()["success"] is False


def test_sandbox_ws_receives_broadcast(client):
    """WS 接続中にアクションを投げると STATE_UPDATE がブロードキャストされる。"""
    gid = client.post("/api/sandbox/create", json={"p1_name": "A", "p2_name": "B"}).json()["game_id"]
    with client.websocket_connect(f"/ws/sandbox/{gid}") as ws:
        r = client.post("/api/sandbox/action", json={
            "game_id": gid, "action_type": "SET_DECK", "player_id": "p1", "deck_id": "db:z",
        })
        assert r.status_code == 200 and r.json()["success"] is True
        msg = ws.receive_json()
        assert msg["type"] == "STATE_UPDATE" and "state" in msg


# --- ルームモード・オンライン対戦ロビー -------------------------------------

def test_rule_create_list_setdeck_start(client):
    gid = client.post("/api/rule/create", json={"room_name": "Online"}).json()["game_id"]
    assert gid in A.RULE_ROOMS

    lst = client.get("/api/rule/list").json()
    assert lst["success"] is True and any(g["game_id"] == gid for g in lst["games"])

    for pid in ("p1", "p2"):
        r = client.post("/api/rule/action", json={
            "game_id": gid, "action_type": "SET_DECK", "player_id": pid, "deck_id": f"db:{pid}",
        })
        assert r.status_code == 200 and r.json()["success"] is True

    r = client.post("/api/rule/action", json={"game_id": gid, "action_type": "START"})
    assert r.status_code == 200 and r.json()["success"] is True
    assert A.RULE_ROOMS[gid]["status"] == "PLAYING"
    assert gid in A.GAMES

    # 開始後は /api/game/state がルームメッセージを返す
    s = client.get("/api/game/state", params={"game_id": gid})
    assert s.status_code == 200 and s.json()["game_id"] == gid


def test_rule_start_requires_both_ready(client):
    gid = client.post("/api/rule/create", json={}).json()["game_id"]
    r = client.post("/api/rule/action", json={"game_id": gid, "action_type": "START"})
    assert r.status_code == 200 and r.json()["success"] is False


def test_rule_action_unknown_room(client):
    r = client.post("/api/rule/action", json={"game_id": "none", "action_type": "START"})
    assert r.status_code == 200 and r.json()["success"] is False


# --- デッキ CRUD（Firestore 必須・未初期化時の振る舞い） --------------------

def test_deck_save_without_db(client):
    assert A.db is None
    r = client.post("/api/deck", json={"name": "d", "leader_id": "x", "card_uuids": []})
    assert r.status_code == 200 and r.json()["success"] is False


def test_deck_list_without_db_is_empty_ok(client):
    r = client.get("/api/deck/list")
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True and body["decks"] == []
