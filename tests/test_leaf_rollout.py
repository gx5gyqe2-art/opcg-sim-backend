"""葉の打ち切り（`leaf_rollout`・§20.7.9・WP `rs-leaf-rollout`）が実エンジンで回ること。

`RsGame.decide(leaf_rollout="turn_end")` で木の葉が「そのターンの終わり」まで方策で
打ち切られ、trace に実績（`rollout`）が出ることを見る。打ち切りの規則そのもの
（何手打つか・どこで止まるか・上限）は Rust の単体テスト（`search/tests_leaf_rollout.rs`）が
持つので、ここは「API と同じ経路（`RsGame`）で通ること」の疎通＝**基盤健全性**（`cpu_infra`）。

既定（`leaf_rollout` を渡さない）は 1 bit も変わらない——trace の形が増えないこと・同じ
seed・同じ盤面で決定が一致することを併せて固定する。
"""
import random

import pytest

import _bootstrap  # noqa: F401
from opcg_sim.api.engine_rs import RsGame

pytestmark = pytest.mark.cpu_infra

VANILLA_LEADER = "EB01-001"
VANILLA_CHAR = "EB01-005"

#: Rust の `search::quiesce::LEAF_ROLLOUT_MAX_PLIES`（trace にも出る）。
MAX_PLIES = 12


def _keep_hands(game: RsGame):
    for _ in range(2):
        pending = game.get_pending_request(False)
        assert pending["action"] == "MULLIGAN", pending
        game.apply_game_action(pending["player_id"], "KEEP_HAND", {})


def _board(seed: str) -> RsGame:
    """バニラ同士でターン 3 の p1 メイン（＝攻撃も登場も残っている盤面）まで進める。

    ターン 3 まで進めるのは、攻撃が合法になる（`turn_count > 2`）＝打ち切りに
    「残りの攻撃」が乗るのが本 WP の狙いだから。
    """
    random.seed(seed)
    game = RsGame.create_from_ids(
        "P1", "P2", VANILLA_LEADER, [VANILLA_CHAR] * 50,
        VANILLA_LEADER, [VANILLA_CHAR] * 50, first_player="p1",
    )
    _keep_hands(game)
    for _ in range(2):
        game.apply_game_action(game.get_pending_request(False)["player_id"], "TURN_END", {})
    pending = game.get_pending_request(False)
    assert pending["player_id"] == "P1" and pending["action"] == "MAIN_ACTION", pending
    return game


def test_turn_end_rollout_runs_and_reports_its_stats():
    """`leaf_rollout="turn_end"` が通り、trace の `rollout` に実績が入る。"""
    game = _board("leaf-rollout:on")
    trace: dict = {}
    move = game.decide("P1", trace=trace, sims=32, leaf_rollout="turn_end")
    assert move is not None
    r = trace.get("rollout")
    assert r is not None, trace.keys()
    assert r["leaves"] > 0, r                    # 葉を打ち切った
    assert r["plies"] > 0, r                     # 1 手以上打った
    assert 0 <= r["rolled"] <= r["leaves"], r
    assert 0.0 <= r["mean_plies"] <= MAX_PLIES, r
    assert 0.0 <= r["capped_frac"] <= 1.0, r
    assert r["capped"] <= r["leaves"], r
    assert r["max_plies"] == MAX_PLIES, r
    game.apply_move("P1", move)                  # 実対局に出せる（例外にならない）


def test_leaf_rollout_none_is_the_same_as_omitting_it():
    """`"none"` を明示しても既定と同じ＝trace に `rollout` は出ず、決定も変わらない。"""
    traces = []
    for kwargs in ({}, {"leaf_rollout": "none"}):
        game = _board("leaf-rollout:off")
        trace: dict = {}
        game.decide("P1", trace=trace, sims=32, **kwargs)
        traces.append(trace)
    for trace in traces:
        assert "rollout" not in trace, trace.get("rollout")
    # uuid は対局ごとに引き直されるので card_id 基準の記述子で比べる。
    assert traces[0]["chosen"] == traces[1]["chosen"], [t["chosen"] for t in traces]
    assert traces[0]["legal_stats"] == traces[1]["legal_stats"]


def test_trace_shape_is_unchanged_when_omitted():
    """省略時の trace の**欄の集合**は打ち切りを入れる前と同じ（`rollout` だけが増える欄）。"""
    game = _board("leaf-rollout:shape")
    off: dict = {}
    game.decide("P1", trace=off, sims=32)
    game2 = _board("leaf-rollout:shape")
    on: dict = {}
    game2.decide("P1", trace=on, sims=32, leaf_rollout="turn_end")
    assert set(on) - set(off) == {"rollout"}, (set(on) - set(off))
    assert set(off) - set(on) == set(), (set(off) - set(on))


def test_rollout_changes_what_the_search_sees():
    """打ち切ると葉の V が変わる＝根の Q が既定と一致しない（配線が生きている）。

    「どちらが良いか」は判定しない（それはコーディネータの仕事）。ここで見るのは
    「打ち切りが探索の値に届いている」ことだけ。
    """
    game = _board("leaf-rollout:effect")
    off: dict = {}
    game.decide("P1", trace=off, sims=48)
    game2 = _board("leaf-rollout:effect")
    on: dict = {}
    game2.decide("P1", trace=on, sims=48, leaf_rollout="turn_end")
    q_off = [s.get("q") for s in off["legal_stats"]]
    q_on = [s.get("q") for s in on["legal_stats"]]
    assert [s.get("move") for s in off["legal_stats"]] == \
        [s.get("move") for s in on["legal_stats"]], "候補は変わらない（葉の値だけの話）"
    assert q_off != q_on, (q_off, q_on)


def test_rollout_is_reproducible():
    """同じ seed・同じ盤面なら同じ決定・同じ実績（打ち切りは乱数を使わない）。"""
    out = []
    for _ in range(2):
        game = _board("leaf-rollout:repro")
        trace: dict = {}
        game.decide("P1", trace=trace, sims=32, leaf_rollout="turn_end")
        out.append((trace["chosen"], trace["rollout"]))
    assert out[0] == out[1], out


def test_moves_stay_playable_through_a_short_self_play():
    """両席を `leaf_rollout="turn_end"` で打たせても実対局が壊れない（原始手のまま進む）。"""
    game = _board("leaf-rollout:selfplay")
    saw_rollout = False
    for _ in range(40):
        pending = game.get_pending_request(False)
        if not pending or game.winner is not None:
            break
        trace: dict = {}
        move = game.decide(pending["player_id"], trace=trace, sims=24, leaf_rollout="turn_end")
        if move is None:
            break
        assert move.get("action_type") not in {"SETUP_BOX", "DON_BOX"}, move
        if trace.get("rollout", {}).get("plies", 0) > 0:
            saw_rollout = True
        game.apply_move(pending["player_id"], move)
    assert saw_rollout, "自己対戦のあいだ 1 度も葉を打ち切っていない"
