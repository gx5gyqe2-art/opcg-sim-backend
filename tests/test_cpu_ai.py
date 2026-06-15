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
from opcg_sim.src.core import action_api, cpu_ai
from opcg_sim.src.core.gamestate import GameManager, Player
from opcg_sim.src.core.invariants import check_invariants
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


def test_evaluate_values_hand_counter(db):
    """J値理論: 同じ手札枚数でもカウンター値の高い手札ほど高評価（防御リソース）。"""
    random.seed(0)
    l1, c1 = build_deck(db, "p1")
    l2, c2 = build_deck(db, "p2")
    gm = GameManager(Player("p1", c1, l1), Player("p2", c2, l2))
    gm.start_game()
    # p1 の手札 1 枚のカウンター値を底上げ → 評価が上がる（枚数は不変）。
    if gm.p1.hand:
        before = cpu_ai.evaluate(gm, "p1")
        gm.p1.hand[0].passive_counter += 2000
        after = cpu_ai.evaluate(gm, "p1")
        assert after > before


def test_evaluate_see_opp_hand_policy(db):
    """情報方針: see_opp_hand=False では相手手札の中身（カウンター値）を読まない。

    相手手札のカウンターを底上げしても public 評価（=False）は不変、full 評価（=True）は下がる。
    """
    random.seed(0)
    l1, c1 = build_deck(db, "p1")
    l2, c2 = build_deck(db, "p2")
    gm = GameManager(Player("p1", c1, l1), Player("p2", c2, l2))
    gm.start_game()
    if not gm.p2.hand:
        pytest.skip("相手手札が空")
    pub_before = cpu_ai.evaluate(gm, "p1", see_opp_hand=False)
    full_before = cpu_ai.evaluate(gm, "p1", see_opp_hand=True)
    gm.p2.hand[0].passive_counter += 2000  # 相手手札のカウンターを底上げ
    pub_after = cpu_ai.evaluate(gm, "p1", see_opp_hand=False)
    full_after = cpu_ai.evaluate(gm, "p1", see_opp_hand=True)
    assert pub_after == pub_before          # 公開方針は相手手札の中身を見ない
    assert full_after < full_before         # full は相手の防御力増として自分有利度が下がる


@pytest.mark.parametrize("difficulty", ["easy", "normal", "hard"])
def test_decide_returns_legal_move(db, difficulty):
    """decide はその時点の合法手のいずれかを返す（easy/normal/hard とも）。"""
    random.seed(1)
    l1, c1 = build_deck(db, "p1")
    l2, c2 = build_deck(db, "p2")
    gm = GameManager(Player("p1", c1, l1), Player("p2", c2, l2))
    gm.start_game()
    pending = gm.get_pending_request()
    actor = gm.p1 if gm.p1.name == pending["player_id"] else gm.p2
    legal = gm.get_legal_actions(actor)
    move = cpu_ai.decide(gm, actor, difficulty, random.Random(0))
    assert move in legal


def _fast_forward_to_p1_main(gm):
    """マリガン〜数ターンを既定解決で進め、turn_count>2 の p1 メインまで進める。"""
    for _ in range(80):
        pend = gm.get_pending_request()
        if pend and pend["player_id"] == "p1" and pend["action"] == "MAIN_ACTION" and gm.turn_count > 2:
            return True
        if not pend or gm.winner is not None:
            return False
        actor = gm.p1 if gm.p1.name == pend["player_id"] else gm.p2
        gm.action_events = []
        if pend["action"] == "MULLIGAN":
            action_api.apply_game_action(gm, actor, "KEEP_HAND", {})
        elif pend["action"] == "MAIN_ACTION":
            action_api.apply_game_action(gm, actor, "TURN_END", {})
        else:
            payload = gm.default_interaction_payload(pend)
            action_api.apply_game_action(gm, actor, action_api.ACT_RESOLVE_SELECTION, payload)
    return False


def test_hard_recognizes_lethal(db):
    """hard は無防備な相手（ライフ0・手札0・場0）に対し、リーダーへの止めアタックを選ぶ。

    探索木内で winner に到達する手順（リーサル）を最高評価とすることを確認する。
    """
    random.seed(0)
    l1, c1 = build_deck(db, "p1")
    l2, c2 = build_deck(db, "p2")
    gm = GameManager(Player("p1", c1, l1), Player("p2", c2, l2))
    gm.start_game()
    assert _fast_forward_to_p1_main(gm), "p1 メインへ到達できなかった"
    # 相手を無防備化（ライフ0・カウンター手札0・ブロッカー0）。
    gm.p2.life.clear()
    gm.p2.hand.clear()
    gm.p2.field.clear()
    moves = gm.get_legal_actions(gm.p1)
    scored = cpu_ai._scored_search(gm, "p1", moves, see_opp_hand=True, opp_public_only=False)
    best_score = max(s for s, _ in scored)
    assert best_score >= cpu_ai.W_WIN - cpu_ai.HARD_DEPTH, "リーサルを認識できていない"
    move = cpu_ai.decide(gm, gm.p1, "hard", random.Random(0))
    # 最短の止め＝相手リーダーへのアタックを選ぶ。
    assert move["action_type"] == "ATTACK"
    assert move["payload"]["target_ids"] == [gm.p2.leader.uuid]


def test_hard_selfplay_smoke_no_invariant_violation(db):
    """hard 方策で数十手進めてもインバリアント違反・例外が出ない（探索の実プレイ健全性）。

    フルゲームは低速なので NODE_BUDGET を小さくし、手数を区切ってスモークする。
    """
    random.seed(3)
    l1, c1 = build_deck(db, "p1")
    l2, c2 = build_deck(db, "p2")
    gm = GameManager(Player("p1", c1, l1), Player("p2", c2, l2))
    gm.start_game()
    mem = {"p1": {}, "p2": {}}
    orig_budget = cpu_ai.HARD_PER_MOVE_BUDGET
    cpu_ai.HARD_PER_MOVE_BUDGET = 12  # スモーク用に探索を浅く（高速化）
    try:
        for _ in range(60):
            if gm.winner is not None:
                break
            pend = gm.get_pending_request()
            assert pend, "勝者未確定なのに pending が無い（スタック）"
            actor = gm.p1 if gm.p1.name == pend["player_id"] else gm.p2
            move = cpu_ai.decide_guarded(gm, actor, "hard", random.Random(0), mem.setdefault(actor.name, {}))
            assert move is not None
            gm.action_events = []
            if move["kind"] == "battle":
                action_api.apply_battle_action(gm, actor, move["action_type"], move.get("card_uuid"))
            else:
                action_api.apply_game_action(gm, actor, move["action_type"], move.get("payload", {}))
            assert not check_invariants(gm), "インバリアント違反"
    finally:
        cpu_ai.HARD_PER_MOVE_BUDGET = orig_budget


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
