"""リプレイ真盤面再生（`replay_runner.state_at_action`・反実仮想レフェリーの入力）。

フレーム復元（`replay_reeval._board_from_frame`）は公開情報のみ＝パワー修正・一時効果を持たない
（実測: 実対局で 1000 にデバフされていた OP15-119 が素の 7000 で復元される）。真盤面再生は
記録された全手順を先頭から再実行するので全内部状態が正確。本テストは実 API 録画（g3 fixture）で:
  1. マーク地点（action 64）まで再生が到達する（効果対話の card_id→uuid 写像を含む）
  2. 公開情報がフレームと一致する（真盤面がフレームの上位互換であることの照合）
  3. フレームに無い内部状態（パワーデバフ）が再現されている
  4. 効果対話リゾルバの写像規則（単体・スタブ）
基盤健全性（レフェリー/検証計器の入力の健全性。ゲームプレイ自体は必須テストが別途担保）＝cpu_infra。
"""
import gzip
import json

import conftest  # noqa: F401  (google スタブ注入 & sys.path 設定)
import pytest

import replay_runner as RR
from game_driver import load_db

pytestmark = pytest.mark.cpu_infra

FIXTURE = "tests/fixtures/replays/g3_v4_replay_7943918224969915818.json.gz"


@pytest.fixture(scope="module")
def db():
    return load_db()


@pytest.fixture(scope="module")
def g3():
    import os
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), FIXTURE)
    with gzip.open(path, "rt") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def board64(db, g3):
    """action 64 直前の真盤面（module 共有＝再生1回）。"""
    m, who = RR.state_at_action(db, g3["replay"], 64)
    assert m is not None, f"真盤面再生が分岐: {who}"
    return m, who


def test_state_at_action_reaches_mark(board64):
    """効果対話（SEARCH_AND_SELECT/SELECT_RESOURCE）を跨いで action 64 まで再生できる。"""
    m, who = board64
    assert who == "p2"
    assert m.turn_count == 8
    assert m.turn_player.name == "p2"


def test_state_at_action_replays_full_game_with_frames(db, g3):
    """フレーム差分による対象特定（対象欠落 ATTACK_CONFIRM）込みで全159手を最後まで再生できる。

    フレーム無しだと「リーダー優先」推測が g3@86 のキャラ攻撃をライフ攻撃に化けさせ、
    幻のトリガー対話（菊之丞）で step 88 が分岐する（実測）。"""
    fbi = {f.get("action_index"): f for f in g3["frames"]}
    n = len(g3["replay"]["actions"])
    m, who = RR.state_at_action(db, g3["replay"], n - 1, frames=fbi)
    assert m is not None, f"全手再生が分岐: {who}"
    assert m.turn_count >= 13


def test_true_board_matches_frame_public_info(g3, board64):
    """公開情報（ライフ/手札枚数/ドン/場の構成とレスト状態）が直前フレームと一致する。"""
    m, _ = board64
    frame = next(f for f in g3["frames"] if f.get("action_index") == 63)
    for pid in ("p1", "p2"):
        fs = frame["players"][pid]
        p = m.p1 if pid == "p1" else m.p2
        assert len(p.life) == len(fs["life"]), f"{pid} life"
        assert len(p.hand) == len(fs["hand"]), f"{pid} hand"
        assert len(p.don_active) == fs["don_active"], f"{pid} don active"
        assert len(p.don_rested) == fs["don_rested"], f"{pid} don rested"
        want = sorted((e["card_id"], bool(e["is_rest"])) for e in fs["field"])
        got = sorted((c.master.card_id, bool(c.is_rest)) for c in p.field)
        assert got == want, f"{pid} field"


def test_true_board_recovers_power_modifiers(board64):
    """フレーム復元が失うパワー修正が真盤面には乗っている（OP15-119: 素7000→実効1000）。"""
    m, _ = board64
    luffy = next(c for c in m.p2.field if c.master.card_id == "OP15-119")
    assert luffy.get_power(True) == 1000


class _StubMgr:
    def __init__(self, pending):
        self._pending = pending

    def get_pending_request(self):
        return self._pending

    def default_interaction_payload(self, pending):
        return {"selected_uuids": [], "index": 0, "accepted": True,
                "position": "BOTTOM", "declared_value": 0}


class _StubActor:
    name = "p1"


def _pending(cands):
    return {"player_id": "p1",
            "selectable_uuids": [u for u, _ in cands],
            "candidates": [{"uuid": u, "card_id": c} for u, c in cands]}


def test_dialog_resolver_maps_card_ids_in_order():
    """記録の card_id 列 → 候補 uuid（列挙順・重複消費）。同名2枚は先頭から順に割り当てる。"""
    mgr = _StubMgr(_pending([("u1", "OP01-001"), ("u2", "OP01-002"), ("u3", "OP01-001")]))
    mv = RR._resolve_dialog_action(mgr, _StubActor(), {
        "action_type": "RESOLVE_EFFECT_SELECTION", "selected": ["OP01-001", "OP01-001"]})
    assert mv["payload"]["selected_uuids"] == ["u1", "u3"]


def test_dialog_resolver_homogeneous_fallback_and_miss():
    """録画側 uuid フォールバック（候補写像不能）: 残候補が同一 card_id のみなら先頭充当、
    異種混在なら None（黙って誤対応しない）。"""
    # ドン!!選択: 全候補同一 → 記録が uuid でも先頭から充当できる。
    mgr = _StubMgr(_pending([("u1", "DON"), ("u2", "DON"), ("u3", "DON")]))
    mv = RR._resolve_dialog_action(mgr, _StubActor(), {
        "action_type": "RESOLVE_EFFECT_SELECTION", "selected": ["orig-uuid-a", "orig-uuid-b"]})
    assert mv["payload"]["selected_uuids"] == ["u1", "u2"]
    # 異種混在 → 特定不能＝None（miss として検出させる）。
    mgr = _StubMgr(_pending([("u1", "OP01-001"), ("u2", "OP01-002")]))
    mv = RR._resolve_dialog_action(mgr, _StubActor(), {
        "action_type": "RESOLVE_EFFECT_SELECTION", "selected": ["orig-uuid-a"]})
    assert mv is None


def test_dialog_resolver_bare_and_overrides():
    """裸の記録（選択なし）は空選択＋既定 payload。index/accepted の記録値は上書きされる。"""
    mgr = _StubMgr(_pending([("u1", "OP01-001")]))
    mv = RR._resolve_dialog_action(mgr, _StubActor(), {"action_type": "RESOLVE_EFFECT_SELECTION"})
    assert mv["payload"]["selected_uuids"] == []
    mv = RR._resolve_dialog_action(mgr, _StubActor(), {
        "action_type": "RESOLVE_EFFECT_SELECTION", "index": 2, "accepted": False})
    assert mv["payload"]["index"] == 2 and mv["payload"]["accepted"] is False
