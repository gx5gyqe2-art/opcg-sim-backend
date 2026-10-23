"""ライフが少ない戦闘の応手は短い木で読む（`docs/search_efficiency_plan.md` §7.42 の 3・WP `rs-window-read`）。

blunder-audit（`docs/reports/2026-10-23_blunder_audit.md` §3 の (iii-b)）で確かめに耐えた大きな損 2 点の盤面を、
局の手の列（`tests/fixtures/window_read_points.json`）から当て直して作り、出荷の CPU（r13d・出荷の設定）に選ばせる:
  - 3301025・117（自分のライフ 0・相手のリーダーのアタック・ブロッカー 1 体）: 旧い 3 つの旗（窓は世界 1 本の静的な値）は
    素通しを選ぶ。既定（短い木）は**ブロック**を選ぶ（順位の入れ替わり・報告の深い読みと同じ向き）。
  - 3301005・107（同・【ダブルアタック】）: 旧い形は素通し。既定は短い木で読んだ窓になる（手は問わない・値が近い点）。

必須（ゲームプレイ＝対戦 API の CPU の応手そのものが変わる経路）。Rust の wheel が無い環境では skip せず fail する。

実行: OPCG_LOG_SILENT=1 python -m pytest tests/test_rs_window_read.py -q -s -p no:cacheprovider
"""
import json
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: E402,F401  (sys.path 設定＋google スタブ)

from opcg_sim.loop import decks as D  # noqa: E402
from opcg_sim.loop import driver as DR  # noqa: E402
from opcg_sim.loop import engine as E  # noqa: E402

POINTS = _os.path.join(_bootstrap.FIXTURES_DIR, "window_read_points.json")
LEGACY = {"window_carry_legacy": True, "window_worlds_legacy": True, "window_tree_legacy": True}


def _apply(game, actor, mv):
    if mv.get("kind") == "battle":
        game.apply_battle_action(actor, mv["action_type"], mv.get("card_uuid"))
    else:
        game.apply_game_action(actor, mv["action_type"], json.dumps(mv.get("payload") or {}, ensure_ascii=False))


def _board(p):
    db = D.load_db()
    p1, p2 = D.build_pair(db, p["leaders"][0], p["leaders"][1], p["seed"], "real")
    game = DR.new_game(p["seed"], p1, p2)
    for mv in p["moves"]:
        _apply(game, json.loads(game.pending_json())["player_id"], mv)
    pend = json.loads(game.pending_json())
    assert pend["player_id"] == p["seat"] and pend["action"] == "SELECT_BLOCKER"
    return game


def _decide(game, p, extra):
    spec = E.SeatSpec(None, sims=E.PLAY_SIMS, budget=E.PLAY_BUDGET, worlds=E.PLAY_WORLDS,
                      lethal_proof=E.PLAY_LETHAL_PROOF, **extra)
    opts = spec.decide_opts(p["seed"], p["turn"], p["seat"], None)
    return json.loads(game.fork().decide(p["seat"], opts))


def _points():
    E.engine()
    return {p["seed"]: p for p in json.load(open(POINTS, encoding="utf-8"))["points"]}


def test_low_life_block_beats_pass_with_the_short_tree():
    p = _points()[3301025]
    game = _board(p)
    old = _decide(game, p, LEGACY)
    new = _decide(game, p, {})
    assert old["kind"] == "window" and old["move"]["action_type"] == "PASS", old["move"]
    assert "window" not in old
    assert new["kind"] == "window" and new["window"]["tree"] is True
    assert new["move"]["action_type"] == "SELECT_BLOCKER", (new["move"], new["stats"]["Q"])


def test_tree_only_is_enough_to_flip_and_other_point_reads_a_tree():
    pts = _points()
    p = pts[3301025]
    only_tree = _decide(_board(p), p, {"window_carry_legacy": True, "window_worlds_legacy": True})
    assert only_tree["move"]["action_type"] == "SELECT_BLOCKER"
    q = pts[3301005]
    game = _board(q)
    assert _decide(game, q, LEGACY)["move"]["action_type"] == "PASS"
    assert _decide(game, q, {})["window"]["tree"] is True
