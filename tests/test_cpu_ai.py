"""CPU AI（cpu_ai）と CPU 対戦エンドポイント（/api/game/cpu/step）のテスト（PR2）。

Firestore に依存しないよう load_deck_mixed をモックし、実カード DB から
リーダー + キャラ 50 枚のデッキを構築して GameManager を起動する（test_rule_online と同方式）。
"""
import random

import conftest  # noqa: F401  (google スタブ注入 & sys.path 設定)
import pytest
from fastapi.testclient import TestClient

from opcg_sim.api import app as appmod
from opcg_sim.src.models.models import CardInstance
from opcg_sim.src.core import cpu_ai
from opcg_sim.src.core.gamestate import GameManager, Player
from cpu_selfplay import build_deck, _load_db


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
    appmod.GAMES.clear()
    appmod.CPU_GAMES.clear()
    return TestClient(appmod.app)


# ---------------------------------------------------------------------------
# cpu_ai 単体
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def db():
    return _load_db()


def test_evaluate_prefers_more_life(db):
    """ライフが多いほうが高評価になる。"""
    random.seed(0)
    l1, c1 = build_deck(db, "p1")
    l2, c2 = build_deck(db, "p2")
    gm = GameManager(Player("p1", c1, l1), Player("p2", c2, l2))
    gm.start_game()
    base = cpu_ai.evaluate(gm, "p1")
    gm.p1.life.pop()  # p1 のライフを 1 枚減らす
    worse = cpu_ai.evaluate(gm, "p1")
    assert worse < base


def test_decide_returns_legal_move(db):
    """decide はその時点の合法手のいずれかを返す。"""
    random.seed(1)
    l1, c1 = build_deck(db, "p1")
    l2, c2 = build_deck(db, "p2")
    gm = GameManager(Player("p1", c1, l1), Player("p2", c2, l2))
    gm.start_game()
    pending = gm.get_pending_request()
    actor = gm.p1 if gm.p1.name == pending["player_id"] else gm.p2
    legal = gm.get_legal_actions(actor)
    move = cpu_ai.decide(gm, actor, "normal", random.Random(0))
    assert move in legal


# ---------------------------------------------------------------------------
# /api/game/cpu/step エンドポイント
# ---------------------------------------------------------------------------

def _cpu_create(client, difficulty="normal"):
    res = client.post("/api/game/create", json={
        "p1_deck": "db:a", "p2_deck": "db:b",
        "p1_name": "p1", "p2_name": "p2",
        "vs_cpu": True, "cpu_difficulty": difficulty,
    }).json()
    return res


def test_cpu_create_registers_metadata(client):
    res = _cpu_create(client, "hard")
    assert res["success"]
    gid = res["game_id"]
    assert gid in appmod.CPU_GAMES
    assert appmod.CPU_GAMES[gid]["cpu_player_id"] == "p2"
    assert appmod.CPU_GAMES[gid]["difficulty"] == "hard"


def test_cpu_step_noop_when_human_to_act(client):
    """人間(p1)のマリガン待ちでは CPU は行動しない（cpu_acted=False, waiting_for=human_decision）。"""
    gid = _cpu_create(client)["game_id"]
    step = client.post("/api/game/cpu/step", json={"game_id": gid}).json()
    assert step["success"]
    assert step["cpu_acted"] is False
    assert step["waiting_for"] == "human_decision"


def test_cpu_step_drives_cpu_after_human(client):
    """人間がマリガンを終えると、CPU step が CPU のマリガン〜ターンを進め、
    最終的に人間の手番（waiting_for in human/human_decision/game_over）へ戻る。"""
    gid = _cpu_create(client)["game_id"]
    # 人間(p1) のマリガン確定
    kept = client.post("/api/game/action", json={"game_id": gid, "action": "KEEP_HAND", "player_id": "p1", "payload": {}}).json()
    assert kept["success"]
    assert kept["pending_request"]["player_id"] == "p2"  # CPU の番へ

    # CPU が行動すべき間ポーリング（安全上限つき）
    cpu_actions = 0
    for _ in range(400):
        step = client.post("/api/game/cpu/step", json={"game_id": gid}).json()
        assert step["success"], step
        if step["cpu_acted"]:
            cpu_actions += 1
        if step["waiting_for"] != "cpu":
            break
    assert cpu_actions >= 1, "CPU が一度も行動しなかった"
    assert step["waiting_for"] in ("human", "human_decision", "game_over")


def test_cpu_full_game_progress(client):
    """人間=常にターン終了 + CPU step ポーリングで、数ターン安定して進行できる。"""
    gid = _cpu_create(client, "normal")["game_id"]
    client.post("/api/game/action", json={"game_id": gid, "action": "KEEP_HAND", "player_id": "p1", "payload": {}})

    def drain_cpu():
        last = None
        for _ in range(600):
            last = client.post("/api/game/cpu/step", json={"game_id": gid}).json()
            assert last["success"], last
            if last["waiting_for"] != "cpu":
                return last
        return last

    last = drain_cpu()
    turns_played = 0
    for _ in range(8):
        if last["waiting_for"] == "game_over":
            break
        # 人間に選択要求が出ている場合は既定解決、そうでなければターン終了。
        pend = last.get("pending_request")
        if pend and pend["player_id"] == "p1" and pend["action"] not in ("MAIN_ACTION", "MULLIGAN"):
            # 効果対話 → 既定解決
            mgr = appmod.GAMES[gid]
            payload = mgr.default_interaction_payload(mgr.get_pending_request())
            last = client.post("/api/game/action", json={"game_id": gid, "action": "RESOLVE_EFFECT_SELECTION", "player_id": "p1", "payload": payload}).json()
        elif pend and pend["player_id"] == "p1" and pend["action"] == "MAIN_ACTION":
            last = client.post("/api/game/action", json={"game_id": gid, "action": "TURN_END", "player_id": "p1", "payload": {}}).json()
            turns_played += 1
        assert last["success"], last
        last = drain_cpu()
    assert turns_played >= 1
