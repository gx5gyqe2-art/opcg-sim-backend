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
from opcg_sim.api import state
from opcg_sim.api.services import decks as deck_svc
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

    monkeypatch.setattr(deck_svc, "load_deck_mixed", _stub_load_deck_mixed)

    state.clear_all()
    with TestClient(A.app) as c:
        yield c
    state.clear_all()


# --- 基盤 -------------------------------------------------------------------

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["constants_loaded"] is True


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


# --- リプレイ種＋CPU思考トレース（Phase 2） --------------------------------------

def test_replay_disabled_by_default(client):
    """cpu_trace 未指定の CPU 対局はリプレイ記録を持たず、エンドポイントは整形エラーを返す。"""
    gid = _create_game(client, vs_cpu=True, cpu_deck="db:cpu", cpu_difficulty="hard").json()["game_id"]
    assert "cpu_trace" not in A.CPU_GAMES[gid]  # 既定では記録器を作らない＝本番無影響
    r = client.get(f"/api/game/{gid}/replay")
    assert r.status_code == 200 and r.json()["success"] is False
    assert r.json()["error"]["code"] == "REPLAY_NOT_FOUND"


def _drive_until_cpu_decides(client, gid, cpu_name, human_name, max_iter=80):
    """マリガン→ターンを進め、CPU(p2) が意思決定を 1 つ以上行うまで駆動する。"""
    for _ in range(max_iter):
        if A.GAMES[gid].winner is not None:
            break
        if len(A.CPU_GAMES[gid].get("decisions", [])) >= 1:
            break
        state = client.get("/api/game/state", params={"game_id": gid}).json()
        pending = state.get("pending_request")
        if not pending:
            break
        pid = pending["player_id"]
        if pid == cpu_name:
            client.post("/api/game/cpu/step", json={"game_id": gid})
        elif pending["action"] == "MULLIGAN":
            client.post("/api/game/action", json={"game_id": gid, "player_id": pid, "action": "KEEP_HAND"})
        elif pending["action"] == "MAIN_ACTION":
            # 人間の手番はターンを畳んで CPU に手番を渡す。
            client.post("/api/game/action", json={"game_id": gid, "player_id": pid, "action": "TURN_END"})
        else:
            break


def test_replay_capture_and_fetch(client):
    """cpu_trace=true の対局で CPU 思考トレース＋種が記録され、エンドポイントで取得できる。"""
    body = _create_game(client, vs_cpu=True, cpu_deck="db:cpu", cpu_difficulty="hard",
                        cpu_trace=True, seed=12345).json()
    gid = body["game_id"]
    meta = A.CPU_GAMES[gid]
    assert meta.get("cpu_trace") is True and meta.get("seed") == 12345
    assert meta["leaders"]["p1"] and meta["decks"]["p1"]  # 種にデッキ/リーダーが入る

    _drive_until_cpu_decides(client, gid, cpu_name="P2", human_name="P1")
    assert meta["decisions"], "CPU 思考トレースが記録されていない"

    # 思考トレースの中身（ライブは軽量＝read_ahead は省く）。
    d0 = meta["decisions"][0]
    assert d0.get("chosen") and "action_type" in d0["chosen"]
    assert "candidates" in d0 and "regret" in d0
    assert "j_components" in d0 and "total" in d0["j_components"]
    # ライブ採取は read_ahead（重い読み筋）を含まない＝CPU 思考のレイテンシを抑えるため。
    assert "read_ahead" not in d0
    # 人間の操作も card_id 基準で記録される（KEEP_HAND/TURN_END 等）。
    assert any(a.get("src") == "human" for a in meta["actions"])

    # エンドポイント取得。
    r = client.get(f"/api/game/{gid}/replay")
    assert r.status_code == 200
    rb = r.json()
    assert rb["success"] is True
    assert rb["replay"]["schema"] == A.REPLAY_SCHEMA and rb["replay"]["seed"] == 12345
    assert rb["replay"]["leaders"] and rb["replay"]["actions"]
    assert len(rb["decisions"]) == len(meta["decisions"])


def test_replay_capture_learned(client):
    """本番既定 CPU＝learned(Gen2) の cpu_trace 対局でも思考トレース＋種が記録される（R3）。

    learned のトレースは L1 の regret/j_components でなく **MCTS root 統計**（chosen/candidates=訪問%・Q・
    L1第二意見）。既定=learned の実対局リプレイ記録の担保（従来テストは hard 固定だった穴を閉じる）。
    """
    body = _create_game(client, vs_cpu=True, cpu_deck="db:cpu", cpu_difficulty="learned",
                        cpu_trace=True, seed=777).json()
    gid = body["game_id"]
    meta = A.CPU_GAMES[gid]
    assert meta.get("difficulty") == "learned" and meta.get("cpu_trace") is True

    _drive_until_cpu_decides(client, gid, cpu_name="P2", human_name="P1")
    assert meta["decisions"], "learned CPU の思考トレースが記録されていない"
    d0 = meta["decisions"][0]
    assert d0.get("difficulty") == "learned"
    assert d0.get("chosen") and "candidates" in d0 and len(d0["candidates"]) >= 1
    assert "visit_pct" in d0["candidates"][0] and "q" in d0["candidates"][0]  # MCTS root 統計
    assert "l1_move" in d0                                                    # L1 第二意見
    r = client.get(f"/api/game/{gid}/replay")
    assert r.json()["success"] is True and r.json()["replay"]["seed"] == 777


def _drive_full_or_cap(client, gid, cap=160):
    """CPU 対局を決着 or cap まで駆動（人間=受動: KEEP_HAND/TURN_END/効果は既定解決）。"""
    for _ in range(cap):
        if A.GAMES[gid].winner is not None:
            return
        state = client.get("/api/game/state", params={"game_id": gid}).json()
        pend = state.get("pending_request")
        if not pend:
            return
        pid = pend["player_id"]
        cpu_name = A.CPU_GAMES[gid]["cpu_player_id"]
        if pid == cpu_name:
            client.post("/api/game/cpu/step", json={"game_id": gid})
        elif pend["action"] == "MULLIGAN":
            client.post("/api/game/action", json={"game_id": gid, "player_id": pid, "action": "KEEP_HAND"})
        elif pend["action"] == "MAIN_ACTION":
            client.post("/api/game/action", json={"game_id": gid, "player_id": pid, "action": "TURN_END"})
        else:
            mgr = A.GAMES[gid]
            payload = mgr.default_interaction_payload(mgr.get_pending_request())
            client.post("/api/game/action", json={"game_id": gid, "player_id": pid,
                        "action": "RESOLVE_EFFECT_SELECTION", "payload": payload})


def test_replay_api_descriptor_end_to_end(client):
    """R3 実結線: API の実録画（`REPLAY_SCHEMA`＝/replay）を `replay_from_descriptor` へ食わせ、
    CPU の意思決定列が録画と一致する（coin toss=first_player='random' を seed から再現）。

    録画の CPU 思考トレース（decisions[].chosen・card_id 基準）と、再生の CPU 再 decide が一致＝
    本番の実対局を丸ごと再現できることの end-to-end 証明（人間手は注入・§R3 残の実結線）。
    """
    import replay_runner as RR
    from opcg_sim.src.core import cpu_ai

    body = _create_game(client, vs_cpu=True, cpu_deck="db:cpu", cpu_difficulty="hard",
                        cpu_trace=True, seed=4242, first_player="random").json()
    gid = body["game_id"]
    _drive_full_or_cap(client, gid, cap=160)

    rb = client.get(f"/api/game/{gid}/replay").json()
    assert rb["success"] and rb["replay"]["seed"] == 4242
    desc = rb["replay"]
    rec_chosen = [d.get("chosen") for d in rb["decisions"]]
    assert len(rec_chosen) >= 3, "CPU の意思決定が記録されていない"

    cpu_name = desc["cpu_player_id"]

    class _CpuCap:
        def __init__(self):
            self.chosen = []

        def on_decision(self, ctx, move):
            # 再生の CPU 席は run_game 名 "p2"（cpu）。録画は cpu_name。位置で判定。
            if ctx.actor.name == "p2":
                self.chosen.append(cpu_ai._describe_move(ctx.manager, move))

    cap = _CpuCap()
    rep = RR.replay_from_descriptor(A.card_db, desc, cpu_difficulty="hard",
                                    first_player="random", observers=[cap])
    # 再生した CPU の手が録画の CPU 思考トレースと（再生できた範囲で）一致する。
    n = min(len(cap.chosen), len(rec_chosen))
    assert n >= 3, f"再生の CPU 決定が少なすぎ（reproduced={rep['reproduced']} misses={rep['misses'][:1]}）"
    assert cap.chosen[:n] == rec_chosen[:n], "API 実対局の CPU 意思決定が再生で一致しない"


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
