"""リプレイ盤面フレーム（`opcg_sim/api/services/replay.py` の frames 記録＋`/replay/frames`）。

cpu_trace 対局で「アクション適用後の盤面スナップショット（コンパクト形）」が記録され、
GET /api/game/{id}/replay/frames が 種＋actions＋decisions＋frames を整合インデックス付きで
返すことを検証する（リプレイビューアのデータ供給契約）。

- frames[i].action_index == i-1（フレーム0＝初期盤面のみ None）＝ actions との明示対応
- decisions[k].action_index が指す actions は CPU のアクション
- フレームのカードは動的状態のみ（効果テキスト等のマスター情報は落ちる＝サイズ抑制）
- _FRAME_CAP 超過は記録を止め frames_truncated を立てる（暴走時のメモリ保険）

実行: OPCG_LOG_SILENT=1 python -m pytest tests/test_replay_frames.py -q -s -p no:cacheprovider
"""
import conftest  # noqa: F401  (google スタブ注入 & sys.path 設定)

import pytest
from fastapi.testclient import TestClient

from opcg_sim.api import app as A
from opcg_sim.api import state
from opcg_sim.api.services import decks as deck_svc
from opcg_sim.api.services import replay as replay_svc
from opcg_sim.src.models.models import CardInstance


# --- フィクスチャ（test_api.py と同型・1トピック=1ファイルのため複製） -------

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
        leader = CardInstance(leader_master, owner_id)
        cards = [CardInstance(char_master, owner_id) for _ in range(50)]
        return leader, cards

    monkeypatch.setattr(deck_svc, "load_deck_mixed", _stub_load_deck_mixed)

    state.clear_all()
    with TestClient(A.app) as c:
        yield c
    state.clear_all()


def _create_game(client, **extra):
    payload = {"p1_deck": "db:x", "p2_deck": "db:y", "p1_name": "P1", "p2_name": "P2"}
    payload.update(extra)
    return client.post("/api/game/create", json=payload)


def _drive_until_cpu_decides(client, gid, cpu_name, max_iter=80):
    """マリガン→ターンを進め、CPU が意思決定を 1 つ以上行うまで駆動する（test_api.py と同型）。"""
    for _ in range(max_iter):
        if A.GAMES[gid].winner is not None:
            break
        if len(A.CPU_GAMES[gid].get("decisions", [])) >= 1:
            break
        st = client.get("/api/game/state", params={"game_id": gid}).json()
        pending = st.get("pending_request")
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


# --- テスト -------------------------------------------------------------------

def test_frames_absent_without_trace(client):
    """cpu_trace 未指定はフレームを記録せず、/replay/frames は整形エラーを返す（本番無影響）。"""
    gid = _create_game(client, vs_cpu=True, cpu_deck="db:cpu", cpu_difficulty="hard").json()["game_id"]
    assert "frames" not in A.CPU_GAMES[gid]
    r = client.get(f"/api/game/{gid}/replay/frames")
    assert r.status_code == 200 and r.json()["success"] is False
    assert r.json()["error"]["code"] == "REPLAY_NOT_FOUND"


def test_frames_capture_alignment_and_fetch(client):
    """traced 対局でフレームが actions と 1:1（+初期盤面）で記録され、一括取得できる。"""
    gid = _create_game(client, vs_cpu=True, cpu_deck="db:cpu", cpu_difficulty="hard",
                       cpu_trace=True, seed=4242).json()["game_id"]
    meta = A.CPU_GAMES[gid]

    # フレーム0＝セットアップ直後（MULLIGAN・action_index=None・pending あり）。
    assert len(meta.get("frames", [])) == 1
    f0 = meta["frames"][0]
    assert f0["action_index"] is None and f0["phase"] == "MULLIGAN"
    assert f0["pending"] and f0["pending"].get("action") == "MULLIGAN"
    for pk in ("p1", "p2"):
        side = f0["players"][pk]
        assert side["leader"] and side["leader"]["card_id"]
        assert side["deck_count"] > 0 and len(side["hand"]) == 5
        # コンパクト形＝マスター情報（効果テキスト等）は持たない（card_id からフロントで引く）。
        assert "text" not in side["leader"] and "traits" not in side["leader"]

    _drive_until_cpu_decides(client, gid, cpu_name="P2")
    assert meta["decisions"], "CPU 思考トレースが記録されていない"

    # 整合: フレーム数 = アクション数 + 1、各 frames[i].action_index == i-1。
    assert len(meta["frames"]) == len(meta["actions"]) + 1
    for i, fr in enumerate(meta["frames"]):
        assert fr["action_index"] == (None if i == 0 else i - 1)

    # decisions[k].action_index は CPU のアクションを指す。
    for d in meta["decisions"]:
        a = meta["actions"][d["action_index"]]
        assert a["src"] == "cpu" and a["player"] == "P2"
        assert a["action_type"] == d["chosen"]["action_type"]

    # 一括取得（種＋actions＋decisions＋frames）。
    r = client.get(f"/api/game/{gid}/replay/frames")
    assert r.status_code == 200
    rb = r.json()
    assert rb["success"] is True and rb["replay"]["seed"] == "4242"
    assert len(rb["frames"]) == len(meta["frames"])
    assert len(rb["decisions"]) == len(meta["decisions"])
    assert rb["frames_truncated"] is False


def test_frames_cap_guards_memory(client, monkeypatch):
    """_FRAME_CAP 到達後は記録を止め frames_truncated を立てる（メモリ保険）。"""
    monkeypatch.setattr(replay_svc, "_FRAME_CAP", 1)
    gid = _create_game(client, vs_cpu=True, cpu_deck="db:cpu", cpu_difficulty="hard",
                       cpu_trace=True, seed=99).json()["game_id"]
    meta = A.CPU_GAMES[gid]
    assert len(meta["frames"]) == 1  # フレーム0 で上限到達

    st = client.get("/api/game/state", params={"game_id": gid}).json()
    pending = st["pending_request"]
    client.post("/api/game/action", json={"game_id": gid, "player_id": pending["player_id"],
                                          "action": "KEEP_HAND"})
    assert len(meta["frames"]) == 1 and meta.get("frames_truncated") is True
    assert client.get(f"/api/game/{gid}/replay/frames").json()["frames_truncated"] is True
