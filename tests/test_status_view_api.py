"""継続中の状態（凍結・アタック不可・付与キーワード・パワー増減・凍結ドン!!）の API 合流。

Rust `status_view.rs`（`Game.status_json`）の中身は Rust 側の単体テストが見る。ここは
`presenters.merge_statuses` が盤面 dict の場のカード・プレイヤーへ正しく合流し、
`build_game_result_hybrid` の応答（schema 検証後）に残ることを固定する。
無ければ「状態が画面に出ない／消えない」API 契約破壊を見逃す＝必須/標準。

実行: OPCG_LOG_SILENT=1 python -m pytest tests/test_status_view_api.py -q -s -p no:cacheprovider
"""
import json
import os
import random

import conftest  # noqa: F401  (sys.path 設定)

from opcg_sim.api.engine_rs import RsGame
from opcg_sim.api.presenters import build_game_result_hybrid, merge_statuses

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _card(uuid):
    return {"uuid": uuid, "name": uuid, "type": "キャラクター", "attribute": "-", "owner_id": "p1"}


def test_merge_fills_board_cards_and_players():
    board = {"players": {
        "p1": {"leader": _card("L1"), "stage": None,
               "zones": {"field": [_card("C1"), _card("C2")], "hand": [_card("H1")], "stage": None}},
        "p2": {"leader": _card("L2"), "stage": _card("S2"),
               "zones": {"field": [], "hand": [], "stage": _card("S2")}},
    }}
    status = {
        "cards": {
            "C1": {"statuses": [{"code": "FREEZE", "duration": "NEXT_REFRESH"}], "power_mod": -2000, "cost_mod": 0},
            "S2": {"statuses": [], "power_mod": 0, "cost_mod": 1},
        },
        "players": {"p1": {"don_frozen": 0}, "p2": {"don_frozen": 2}},
    }
    merge_statuses(board, status)
    p1, p2 = board["players"]["p1"], board["players"]["p2"]
    assert p1["zones"]["field"][0]["statuses"] == [{"code": "FREEZE", "duration": "NEXT_REFRESH"}]
    assert p1["zones"]["field"][0]["power_mod"] == -2000
    # 状態の無い場のカードも空/0 で埋まる（フロントが前回の表示を消せる）。
    assert p1["zones"]["field"][1]["statuses"] == [] and p1["leader"]["power_mod"] == 0
    # 手札は対象外。
    assert "statuses" not in p1["zones"]["hand"][0]
    assert p2["stage"]["cost_mod"] == 1 and p2["zones"]["stage"]["cost_mod"] == 1
    assert p1["don_frozen"] == 0 and p2["don_frozen"] == 2


def test_response_carries_statuses_after_schema_validation():
    with open(os.path.join(_ROOT, "opcg_sim", "data", "opcg_cards.json"), encoding="utf-8") as f:
        db = json.load(f)
    leaders = [c["number"] for c in db if c["種類"] == "リーダー"]
    chars = [c["number"] for c in db if c["種類"] == "キャラクター"]
    random.seed(0)
    m = RsGame.create_from_ids("p1", "p2", leaders[0], chars[:50], leaders[1], chars[:50], "p1")
    # マリガン→盤面が立つまで数手進める。
    for _ in range(12):
        acts = m.get_legal_actions()
        if not acts:
            break
        m.apply_move(m.get_pending_request()["player_id"], acts[-1])

    gs = build_game_result_hybrid(m, "g")["game_state"]
    for seat in ("p1", "p2"):
        p = gs["players"][seat]
        assert p["don_frozen"] == 0
        for card in [p["leader"], *p["zones"]["field"]]:
            assert isinstance(card["statuses"], list)
            assert isinstance(card["power_mod"], int) and isinstance(card["cost_mod"], int)
