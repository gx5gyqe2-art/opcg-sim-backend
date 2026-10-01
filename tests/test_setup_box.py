"""準備箱（`SETUP_BOX`・§20.7.2・WP `rs-setup-box`）が実エンジンで回ること。

`RsGame.decide(setup_box=True)` で候補に `SETUP_BOX` が並び、選ばれたときは
**原始手**（PLAY → 対話の解決）で実対局に出る（箱そのものは外へ出ない）ことを見る。
候補生成と原始手化の規則そのものは Rust の単体テスト（`search/tests_setup_box.rs`）が持つので、
ここは「API と同じ経路（`RsGame`）で通ること」の疎通＝**基盤健全性**（`cpu_infra`）。

既定（`setup_box` を渡さない）は 1 bit も変わらない——同じ seed・同じ盤面で決定が一致することも
併せて固定する。

§20.7.6（WP `rs-setup-box-2`・分析 #6 §3）で **箱ができた準備の手の素の PLAY／ACTIVATE_MAIN は
候補から落とす**（配分箱と同じ扱い）ようにした。§20.7.8（WP `rs-setup-box-3`）で箱の範囲を
「発動 → 対象選択 → 効果」までに縮め、**続きの攻撃は入れない**（そこから先は木が読む）。
"""
import os
import random

import pytest

import _bootstrap  # noqa: F401
from opcg_sim.api.engine_rs import RsGame

pytestmark = pytest.mark.cpu_infra

ENEL_LEADER = "OP15-058"       # エネル（神の裁きの条件「自分のリーダーが『エネル』」を満たす）
KAMI_NO_SABAKI = "OP15-075"    # 神の裁き（コスト0・【メイン】ドン‼-1: +1000 → 3000以下を KO）
VANILLA_LEADER = "EB01-001"
VANILLA_CHAR = "EB01-005"

EFFECTS_JSON = os.path.join(os.path.dirname(__file__), "..", "opcg_sim", "data", "opcg_effects.json")

#: 探索の原始手（`SETUP_BOX`／`DON_BOX` は箱＝実対局へは出ない）。
BOX_ACTIONS = {"SETUP_BOX", "DON_BOX"}


def _skip_if_no_effects_json():
    if not os.path.exists(EFFECTS_JSON):
        pytest.skip("効果 JSON（生成物）が無い")


def _keep_hands(game: RsGame):
    for _ in range(2):
        pending = game.get_pending_request(False)
        assert pending["action"] == "MULLIGAN", pending
        game.apply_game_action(pending["player_id"], "KEEP_HAND", {})


def _board_with_an_opponent_character(seed: str) -> RsGame:
    """p1＝エネル（手札は神の裁きだけ）・p2＝バニラ（場にキャラ 1 体）で p1 のメインに戻す。"""
    random.seed(seed)
    game = RsGame.create_from_ids(
        "P1", "P2", ENEL_LEADER, [KAMI_NO_SABAKI] * 50,
        VANILLA_LEADER, [VANILLA_CHAR] * 50, first_player="p1",
    )
    _keep_hands(game)
    game.apply_game_action("P1", "TURN_END", {})
    hand = game.board()["players"]["p2"]["zones"]["hand"]
    game.apply_game_action("P2", "PLAY", {"uuid": hand[0]["uuid"]})
    game.apply_game_action("P2", "TURN_END", {})
    pending = game.get_pending_request(False)
    assert pending["player_id"] == "P1" and pending["action"] == "MAIN_ACTION", pending
    return game


def _legal_moves(trace: dict) -> list:
    return [(s.get("move") or {}) for s in (trace.get("legal_stats") or [])]


def test_setup_box_candidates_appear_and_the_played_move_is_primitive():
    """`setup_box=True` で候補に `SETUP_BOX` が出る／実対局へ出る手は原始手のまま。"""
    _skip_if_no_effects_json()
    game = _board_with_an_opponent_character("setup-box:on")
    trace: dict = {}
    move = game.decide("P1", trace=trace, sims=32, setup_box=True)
    assert move is not None
    kinds = [m.get("action_type") for m in _legal_moves(trace)]
    assert "SETUP_BOX" in kinds, kinds
    # 神の裁きの箱は「+1000 する／しない」×「KO する／しない」で 2 段の枝が立つ。
    boxes = [m for m in _legal_moves(trace) if m.get("action_type") == "SETUP_BOX"]
    selected = [tuple(m.get("selected") or ()) for m in boxes]
    assert len(set(selected)) >= 2, boxes
    # §20.7.6: 箱ができた準備の手の**素の PLAY は候補に無い**（配分箱と同じ扱い）。
    boxed_cards = {m.get("card") for m in boxes}
    bare = [m for m in _legal_moves(trace)
            if m.get("action_type") in ("PLAY", "ACTIVATE_MAIN") and m.get("card") in boxed_cards]
    assert not bare, bare
    # §20.7.8: 箱に**続きの攻撃は入らない**＝どの枝も攻撃対象を持たない。
    assert all(not m.get("targets") for m in boxes), boxes
    # 箱は探索の中だけの手＝返る手は必ず原始手。
    assert move.get("action_type") not in BOX_ACTIONS, move
    game.apply_move("P1", move)  # 実対局に出せる（例外にならない）


def test_boxes_stats_are_reported_only_when_setup_box_is_on():
    """枝数の計測（`trace["boxes"]`）は `setup_box` を渡した decide にだけ出る。"""
    _skip_if_no_effects_json()
    game = _board_with_an_opponent_character("setup-box:stats")
    on: dict = {}
    game.decide("P1", trace=on, sims=32, setup_box=True)
    assert "boxes" in on and on["boxes"]["setup"]["boxes"] > 0, on.get("boxes")
    # 枝は `SETUP_BOX_BRANCH_CAP`（8＋「選ばない」）まで（§20.7.8＝「攻撃しない」枝は無い）。
    assert on["boxes"]["setup"]["max"] <= 9, on["boxes"]

    game2 = _board_with_an_opponent_character("setup-box:stats")
    off: dict = {}
    game2.decide("P1", trace=off, sims=32)
    assert "boxes" not in off, off.get("boxes")


def test_default_is_unchanged():
    """既定（`setup_box` を渡さない）は候補にも決定にも 1 bit も足さない。"""
    _skip_if_no_effects_json()
    chosen, kinds = [], []
    for _ in range(2):
        game = _board_with_an_opponent_character("setup-box:off")
        trace: dict = {}
        game.decide("P1", trace=trace, sims=32)
        # uuid は対局ごとに引き直されるので、card_id 基準の記述子（`chosen`）で比べる。
        chosen.append(trace.get("chosen"))
        kinds.append([m.get("action_type") for m in _legal_moves(trace)])
    assert "SETUP_BOX" not in kinds[0], kinds[0]
    assert chosen[0] == chosen[1], chosen        # 同じ seed・同じ盤面なら同じ決定
    assert kinds[0] == kinds[1], kinds


def test_moves_stay_primitive_through_a_short_self_play():
    """`setup_box=True` で両席を打たせても、実対局へ出る手は**必ず原始手**のまま。

    箱（`SETUP_BOX`／`DON_BOX`）は探索の中だけの手なので、`decide` の戻り値にも
    コミットの消化にも出てはいけない（出るとエンジンが受け取れず対局が壊れる）。
    `trace["sig"]` は原始手化の**前**の手＝箱を選んだ決定はここで分かる。
    """
    _skip_if_no_effects_json()
    game = _board_with_an_opponent_character("setup-box:selfplay")
    saw_candidate = False
    saw_box_chosen = False
    for _ in range(40):
        pending = game.get_pending_request(False)
        if not pending or game.winner is not None:
            break
        player_id = pending["player_id"]
        trace: dict = {}
        move = game.decide(player_id, trace=trace, sims=48, setup_box=True)
        if move is None:
            break
        assert move.get("action_type") not in BOX_ACTIONS, (move, trace.get("kind"))
        kinds = [m.get("action_type") for m in _legal_moves(trace)]
        saw_candidate = saw_candidate or "SETUP_BOX" in kinds
        if (trace.get("sig") or [None])[0] == "SETUP_BOX":
            saw_box_chosen = True
            assert trace.get("kind") == "main", trace.get("kind")
            # 箱を選んだ決定の先頭原始手は素の PLAY／ACTIVATE_MAIN。
            assert move["action_type"] in ("PLAY", "ACTIVATE_MAIN"), move
        game.apply_move(player_id, move)
    assert saw_candidate, "準備の手がある局面が 1 度は通るはず"
    # 箱が**選ばれる**かどうかは方策と探索数しだい（この短い自己対戦では保証しない）＝
    # 選ばれたときの契約だけ上のループで固定する。`saw_box_chosen` は読み手のための記録。
    del saw_box_chosen
