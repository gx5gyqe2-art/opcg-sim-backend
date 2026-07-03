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


def test_pending_request_id_changes_when_only_unlisted_field_differs(client):
    """selectable_uuids が同一でも、識別に効く他フィールド（source_card_uuid / options）だけ
    異なる別要求は request_id が変わる（同名カード2枚の連続確認などの衝突回帰ガード）。

    フロントは request_id 変化を『新要求』検知に使うため、内容が違えば必ず変わる必要がある。
    selectable_uuids だけをキーにしていた旧実装ではこれらが衝突していた。"""
    from opcg_sim.src.core.gamestate import Phase

    gid = _create_game(client)
    mgr = A.GAMES[gid]
    mgr.phase = Phase.MAIN  # MULLIGAN ゲートを越えて active_interaction 分岐へ

    base = {
        "action_type": "CONFIRM_OPTIONAL",
        "player_id": mgr.p1.name,
        "message": "「X」の効果を使用しますか？（コストを払う）",
        "selectable_uuids": [],
        "can_skip": True,
        "candidates": [],
        "constraints": None,
        "options": None,
        "source_card_uuid": "card-A",
    }
    mgr.active_interaction = dict(base)
    rid_a = mgr.get_pending_request()["request_id"]
    rid_a2 = mgr.get_pending_request()["request_id"]
    assert rid_a == rid_a2, "同一要求は安定していること"

    # source_card_uuid だけ異なる別要求（同名カード2枚目）→ rid が変わる。
    mgr.active_interaction = dict(base, source_card_uuid="card-B")
    rid_b = mgr.get_pending_request()["request_id"]
    assert rid_a != rid_b, "source_card_uuid が違えば request_id も変わること"

    # options だけ異なる CHOICE → rid が変わる。
    mgr.active_interaction = dict(base, options=["A", "B"])
    rid_opt1 = mgr.get_pending_request()["request_id"]
    mgr.active_interaction = dict(base, options=["A", "C"])
    rid_opt2 = mgr.get_pending_request()["request_id"]
    assert rid_opt1 != rid_opt2, "options が違えば request_id も変わること"
