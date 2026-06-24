"""価値学習データの『ライブ採取』（人間ログ活用(a)/(A)）の健全性ゲート。

固定する不変条件:
  - 共有サンプラ `cpu_value_data`＝両者視点の特徴を {"f","p"} で返し manager を変更しない・プレイヤー名基準。
  - ラベル付け＝終局勝者の視点が y=1。
  - api/app.py のライブ採取＝**cpu_trace 時のみ**ターン境界で貯まり、未指定の対局には一切作用しない（本番無影響）。
  - replay エンドポイントが {"f","p"} を勝者でラベル確定して {"f","y"} で同梱（終局前は空）。
  - 采取ログ取り込み `human_log_ingest`＝エンベロープ階層を問わず value_samples を拾い不正行を除外。

実行: OPCG_LOG_SILENT=1 python -m pytest tests/test_value_capture.py -q -s -p no:cacheprovider
"""
import copy
import json
import random

import conftest  # noqa: F401
import pytest
from fastapi.testclient import TestClient

from opcg_sim.src.core.gamestate import GameManager, Player
from opcg_sim.src.core import cpu_features, cpu_value_data, journal
from opcg_sim.api import app as A
from opcg_sim.src.models.models import CardInstance
import cpu_selfplay
import human_log_ingest


# --- 共有サンプラ（純粋・高速） --------------------------------------------------

@pytest.fixture(scope="module")
def db():
    return cpu_selfplay._load_db()


def _game(db, seed=0):
    random.seed(seed)
    l1, c1 = cpu_selfplay.build_deck(db, "p1")
    l2, c2 = cpu_selfplay.build_deck(db, "p2")
    m = GameManager(Player("alice", c1, l1), Player("bob", c2, l2))
    m.start_game()
    return m


def test_turn_boundary_samples_shape_and_nonmutating(db):
    m = _game(db)
    before = copy.deepcopy(m)
    s = cpu_value_data.turn_boundary_samples(m)
    assert journal.deep_diff(before, m) is None, "採取が manager を変更した"
    assert [x["p"] for x in s] == ["alice", "bob"], "プレイヤー名基準でない"
    assert all(len(x["f"]) == cpu_features.N_FEATURES for x in s)
    # see_opp_hand=False 既定＝公平（相手手札の中身に依存しない）。視点ごとに非対称。
    assert s[0]["f"] != s[1]["f"]


def test_label_samples_by_winner_name():
    samples = [{"f": [1.0], "p": "alice"}, {"f": [2.0], "p": "bob"}]
    assert [r["y"] for r in cpu_value_data.label_samples(samples, "alice")] == [1, 0]
    assert [r["y"] for r in cpu_value_data.label_samples(samples, "bob")] == [0, 1]
    # 勝者不明（None）なら両者 0。
    assert [r["y"] for r in cpu_value_data.label_samples(samples, None)] == [0, 0]


# --- ライブ採取（api/app.py・TestClient） ----------------------------------------

@pytest.fixture
def client(monkeypatch):
    db = A.card_db
    for cid in list(db.raw_db.keys()):
        db.get_card(cid)
    leader = next(c for c in db.cards.values() if c.type.name == "LEADER")
    char = next(c for c in db.cards.values() if c.type.name == "CHARACTER")

    def _stub(source_str, owner_id):
        return CardInstance(leader, owner_id), [CardInstance(char, owner_id) for _ in range(50)]

    monkeypatch.setattr(A, "load_deck_mixed", _stub)
    for reg in (A.GAMES, A.SANDBOX_GAMES, A.RULE_ROOMS, A.CPU_GAMES):
        reg.clear()
    with TestClient(A.app) as c:
        yield c


def _create(client, **extra):
    payload = {"p1_deck": "db:x", "p2_deck": "db:y", "p1_name": "P1", "p2_name": "P2"}
    payload.update(extra)
    return client.post("/api/game/create", json=payload).json()


def _drive_until_samples(client, gid, cpu_name="P2", max_iter=60):
    """マリガン→人間 TURN_END→CPU step で進め、value_samples が貯まるまで駆動（ターン境界を跨ぐ）。"""
    for _ in range(max_iter):
        if A.GAMES[gid].winner is not None or A.CPU_GAMES[gid].get("value_samples"):
            break
        pending = client.get("/api/game/state", params={"game_id": gid}).json().get("pending_request")
        if not pending:
            break
        pid = pending["player_id"]
        if pid == cpu_name:
            client.post("/api/game/cpu/step", json={"game_id": gid})
        elif pending["action"] == "MULLIGAN":
            client.post("/api/game/action", json={"game_id": gid, "player_id": pid, "action": "KEEP_HAND"})
        elif pending["action"] == "MAIN_ACTION":
            client.post("/api/game/action", json={"game_id": gid, "player_id": pid, "action": "TURN_END"})
        else:
            break


def test_capture_disabled_without_trace(client):
    """cpu_trace 未指定＝採取器を持たず、ターンを跨いでも value_samples は作られない（本番無影響）。"""
    gid = _create(client, vs_cpu=True, cpu_deck="db:cpu", cpu_difficulty="hard")["game_id"]
    _drive_until_samples(client, gid)
    assert "value_samples" not in A.CPU_GAMES[gid]


def test_capture_accumulates_on_traced_game(client):
    """cpu_trace=true＝ターン境界で両者視点サンプルが {"f","p"} で貯まる。"""
    gid = _create(client, vs_cpu=True, cpu_deck="db:cpu", cpu_difficulty="hard",
                  cpu_trace=True, seed=777)["game_id"]
    _drive_until_samples(client, gid)
    samples = A.CPU_GAMES[gid]["value_samples"]
    assert samples, "ターンを跨いでも value_samples が貯まっていない"
    assert len(samples) % 2 == 0, "境界ごとに両者視点（2件）で貯まるはず"
    assert all(len(s["f"]) == cpu_features.N_FEATURES and s["p"] in ("P1", "P2") for s in samples)
    assert all("y" not in s for s in samples), "ラベルは終局時に確定（採取段階では未付与）"


def test_replay_endpoint_labels_value_samples(client):
    """replay 応答に value_samples キーがあり、終局前は空（ラベル付け不能）。"""
    gid = _create(client, vs_cpu=True, cpu_deck="db:cpu", cpu_difficulty="hard",
                  cpu_trace=True, seed=777)["game_id"]
    _drive_until_samples(client, gid)
    replay = client.get(f"/api/game/{gid}/replay").json()["replay"]
    assert "value_samples" in replay and isinstance(replay["value_samples"], list)
    if A.GAMES[gid].winner is None:
        assert replay["value_samples"] == [], "終局前はラベル確定できないので空"


def test_replay_labels_when_finished(client):
    """終局済みなら replay の value_samples が {"f","y"} で勝者ラベル確定される。"""
    gid = _create(client, vs_cpu=True, cpu_deck="db:cpu", cpu_difficulty="hard",
                  cpu_trace=True, seed=777)["game_id"]
    _drive_until_samples(client, gid)
    meta = A.CPU_GAMES[gid]
    # 採取済みサンプルに対し、勝者を擬似設定して replay のラベル確定経路を検証（盤面は不変）。
    if not meta["value_samples"]:
        pytest.skip("サンプル未蓄積")
    A.GAMES[gid].winner = "P1"
    replay = client.get(f"/api/game/{gid}/replay").json()["replay"]
    rows = replay["value_samples"]
    assert rows and all(set(r) == {"f", "y"} and r["y"] in (0, 1) for r in rows)
    p1_rows = [r for s, r in zip(meta["value_samples"], rows) if s["p"] == "P1"]
    assert all(r["y"] == 1 for r in p1_rows), "勝者 P1 視点が y=1 になっていない"


# --- 采取ログ取り込み ------------------------------------------------------------

def test_ingest_extracts_labeled_rows_from_envelope(tmp_path):
    n = cpu_features.N_FEATURES
    good = [{"f": [0.0] * n, "y": 1}, {"f": [1.0] * n, "y": 0}]
    bad = [{"f": [0.0, 1.0], "y": 1}, {"f": [0.0] * n, "y": 5}, {"nope": 1}]
    # フロント采取エンベロープ: envelope→replay→descriptor の入れ子。
    dump = {"capturedAt": "t", "winner": "p1",
            "replay": {"schema": "opcg-replay/v1", "value_samples": good + bad}}
    p = tmp_path / "cap.json"
    p.write_text(json.dumps(dump))
    rows = human_log_ingest.rows_from_file(str(p))
    assert rows == good, "不正行の除外/正常行の抽出が不正"


def test_ingest_handles_top_level_and_nested(tmp_path):
    n = cpu_features.N_FEATURES
    row = {"f": [0.5] * n, "y": 1}
    top = tmp_path / "top.json"; top.write_text(json.dumps({"value_samples": [row]}))
    nested = tmp_path / "nested.json"
    nested.write_text(json.dumps({"replay": {"replay": {"value_samples": [row]}}}))
    assert human_log_ingest.rows_from_file(str(top)) == [row]
    assert human_log_ingest.rows_from_file(str(nested)) == [row]
