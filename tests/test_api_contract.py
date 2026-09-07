"""API 契約テスト（ラチェット）。

- 応答が契約キーを備える（`build_game_result_hybrid` の raw フォールバックでない整形済み形）。
- `request_id` が「同一の要求なら安定」（従来は get のたびに uuid4 再生成＝フロントの新要求検知が
  毎ポーリングで誤発火する機能バグ。D-3 で決定的ハッシュ化）。

実行: OPCG_LOG_SILENT=1 python -m pytest tests/test_api_contract.py -q -s -p no:cacheprovider
"""
import conftest  # noqa: F401  (google スタブ注入 & sys.path 設定)

import pytest
from fastapi.testclient import TestClient

from opcg_sim.api import app as A
from opcg_sim.api import state
from opcg_sim.api.services import decks as deck_svc
from opcg_sim.src.models.models import CardInstance


def _load_card_db():
    db = A.card_db
    for cid in list(db.raw_db.keys()):
        db.get_card(cid)
    leader = next(c for c in db.cards.values() if c.type.name == "LEADER")
    char = next(c for c in db.cards.values() if c.type.name == "CHARACTER")
    return leader, char


@pytest.fixture
def client(monkeypatch):
    leader_master, char_master = _load_card_db()

    def _stub_load_deck_mixed(source_str, owner_id):
        return CardInstance(leader_master, owner_id), [CardInstance(char_master, owner_id) for _ in range(50)]

    monkeypatch.setattr(deck_svc, "load_deck_mixed", _stub_load_deck_mixed)
    state.clear_all()
    with TestClient(A.app) as c:
        yield c
    state.clear_all()


def _create_game(client) -> str:
    r = client.post("/api/game/create", json={"p1_deck": "db:x", "p2_deck": "db:y"})
    body = r.json()
    assert body["success"] is True, body
    return body["game_id"]


def test_result_shape_is_contract_form(client):
    """create/state 応答が契約キー（success/game_id/game_state/pending_request）を備える。"""
    gid = _create_game(client)
    body = client.get(f"/api/game/state?game_id={gid}").json()
    for k in ("success", "game_id", "game_state", "pending_request"):
        assert k in body, f"missing key: {k}"
    assert body["game_state"] is not None
    assert "turn_info" in body["game_state"] and "players" in body["game_state"]


def test_pending_request_id_is_stable(client):
    """同一の pending を複数回取得しても request_id は一致する（毎回再生成の回帰ガード）。"""
    gid = _create_game(client)
    p1 = client.get(f"/api/game/state?game_id={gid}").json().get("pending_request")
    p2 = client.get(f"/api/game/state?game_id={gid}").json().get("pending_request")
    assert p1 and p2, "pending_request should be present at game start (MULLIGAN)"
    assert p1.get("request_id"), "request_id must be present"
    assert p1["request_id"] == p2["request_id"], (
        "request_id must be stable across identical requests "
        f"(got {p1['request_id']} vs {p2['request_id']})"
    )


def test_pending_request_id_changes_when_only_unlisted_field_differs():
    """selectable_uuids が同一でも、識別に効く他フィールド（source_card_uuid / options）だけ
    異なる別要求は request_id が変わる（同名カード2枚の連続確認などの衝突回帰ガード）。

    フロントは request_id 変化を『新要求』検知に使うため、内容が違えば必ず変わる必要がある。
    selectable_uuids だけをキーにしていた旧実装ではこれらが衝突していた。

    `request_id` は Rust 化（計画 §15）後も **Python 側**（`api/engine_rs._rid`）が付ける
    （Rust は出さない＝§15.1）。要求 dict を直接与えて、ハッシュが要求の全内容を覆うことを見る
    （以前は `GameManager.active_interaction` へ偽の中断を差し込んでいた＝エンジン内部依存）。
    """
    from opcg_sim.api.engine_rs import _rid

    base = {
        "player_id": "p1",
        "action": "CONFIRM_OPTIONAL",
        "message": "「X」の効果を使用しますか？（コストを払う）",
        "selectable_uuids": [],
        "can_skip": True,
        "candidates": [],
        "constraints": None,
        "options": None,
        "source_card_uuid": "card-A",
    }
    rid_a = _rid(base, 3)
    assert rid_a == _rid(dict(base), 3), "同一要求は安定していること"
    assert rid_a == _rid(dict(base, request_id="ignored"), 3), "request_id 自身はハッシュに入れない"

    # source_card_uuid だけ異なる別要求（同名カード2枚目）→ rid が変わる。
    assert rid_a != _rid(dict(base, source_card_uuid="card-B"), 3), \
        "source_card_uuid が違えば request_id も変わること"

    # options だけ異なる CHOICE → rid が変わる。
    assert _rid(dict(base, options=["A", "B"]), 3) != _rid(dict(base, options=["A", "C"]), 3), \
        "options が違えば request_id も変わること"

    # ターンが違えば（同じ要求でも）別要求として扱われる。
    assert rid_a != _rid(base, 4), "turn_count もハッシュに含めること"


def test_player_identifier_shapes_in_the_response(client):
    """レスポンス中の「プレイヤーを指す欄」の**形**を固定する（Rust 化での取り違え防止）。

    Python 版は 2 つの形を混ぜている。フロントの契約なので揃えずにそのまま維持する:
      - `turn_info.active_player_id` … **席キー**（"p1"/"p2"。`presenters` の `PLAYER_KEYS`）
      - `players` の**キー** … 席キー（"p1"/"p2"）
      - `players.<seat>.player_id`／`.name`／カードの `owner_id`／`pending_request.player_id`／
        `action_events[].player`／`turn_info.winner` … **プレイヤー名**（リクエストの
        `p1_name`／`p2_name`）

    Rust エンジン（計画 §15）は内部を常に席（p1/p2）で回し、JSON を返す直前に表示名へ
    差し替える（`py_game::Game::rename`）。差し替える欄を 1 つ間違えるだけでフロントの
    突き合わせが壊れるため、ここでラチェットする。
    """
    gid = _create_game(client)
    body = client.get(f"/api/game/state?game_id={gid}").json()
    gs = body["game_state"]

    assert set(gs["players"]) == {"p1", "p2"}, "players のキーは席キー"
    assert gs["turn_info"]["active_player_id"] in ("p1", "p2"), \
        "active_player_id は席キー（プレイヤー名ではない）"
    assert gs["players"]["p1"]["player_id"] == "P1", "player_id はプレイヤー名"
    assert gs["players"]["p1"]["name"] == "P1"
    assert gs["players"]["p2"]["player_id"] == "P2"
    assert gs["players"]["p1"]["leader"]["owner_id"] == "P1", "カードの owner_id はプレイヤー名"
    assert gs["players"]["p2"]["zones"]["hand"][0]["owner_id"] == "P2"
    assert body["pending_request"]["player_id"] in ("P1", "P2"), \
        "pending_request.player_id はプレイヤー名"

    # action_events[].player もプレイヤー名（マリガン要求へ KEEP_HAND で答える）。
    pid = body["pending_request"]["player_id"]
    r = client.post("/api/game/action", json={
        "game_id": gid, "player_id": pid, "action": "KEEP_HAND"}).json()
    assert r["action_events"] and r["action_events"][0]["player"] == pid
