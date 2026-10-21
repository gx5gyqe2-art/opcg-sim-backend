"""迷う局面だけ世界 K 本で読み直す（`docs/search_efficiency_plan.md` §7.4・WP `play-rethink`）。

1 本 × S 回で読み、根の 2 位の訪問数 ÷ 1 位の訪問数 `r` が線 `τ` 以上なら、同じ `search_seed` で
世界 K 本 × S 回を読み直してその手に差し替える（`opcg_sim.loop.engine.decide_with_rethink`・
対戦 API の `RsGame.decide(rethink_ratio=…)` とアリーナ・生成の `driver.run_game` が同じ関数を通る）。

見ること（必須＝ゲームプレイに関わる。対戦 API の CPU の手そのものが変わりうる経路）:
  (a) off（`τ=None`・既定）で今と同一（手・trace・trace の欄）
  (b) `τ=0`（必ず発動）で、結果が世界 K 本の `decide` と同一
  (c) `τ` が十分大きい（発動しない）とき 1 本の `decide` と同一（対局を通しても同じ手順）
  (d) trace の形（`rethink: {r, fired, first_move, final_move}`）
  ＋ 読み直さない局面（窓・コミット・候補 1 つ・既に K 本以上）の判定は pure に見る。

Rust の wheel が無い環境では **skip せず fail** する（`tests/test_pimc_worlds.py` と同じ方針）。

実行: OPCG_LOG_SILENT=1 python -m pytest tests/test_play_rethink.py -q -s -p no:cacheprovider
"""
import glob
import gzip
import json
import os as _os
import random
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: E402,F401  (sys.path 設定＋google スタブ)

import pytest  # noqa: E402

from opcg_sim.api.engine_rs import RsGame, load_engine  # noqa: E402
from opcg_sim.loop import decks as D  # noqa: E402
from opcg_sim.loop import driver as DR  # noqa: E402
from opcg_sim.loop import engine as E  # noqa: E402
from opcg_sim.loop import hidden_build as hb  # noqa: E402

REPLAYS_DIR = _os.path.join(_bootstrap.FIXTURES_DIR, "replays")

SIMS = 16
WORLDS = 4
NEVER = 1.01   # r ≤ 1 なので発動しない


# --- pure（偽の decide で分岐だけ見る）------------------------------------------------

def _fake(outs):
    """呼ばれた opts を記録し、`worlds` の本数で返す out を変える偽の decide。"""
    calls = []

    def decide(opts_json):
        o = json.loads(opts_json)
        calls.append(o)
        return json.dumps(outs[int(o.get("worlds") or 1)])
    return decide, calls


def _main_out(ns, move):
    return {"kind": "main", "move": move, "groups": [{"rep": i, "n": n} for i, n in enumerate(ns)]}


def test_ratio_reads_second_over_first_of_groups():
    assert E.rethink_ratio(_main_out([40, 10, 30], {"m": 1})) == pytest.approx(0.75)
    assert E.rethink_ratio(_main_out([16], {"m": 1})) is None, "候補 1 つは読み直さない"
    assert E.rethink_ratio({"kind": "window", "groups": [{"n": 5}, {"n": 5}]}) is None
    assert E.rethink_ratio({"kind": "commit", "groups": []}) is None
    assert E.rethink_ratio(_main_out([0, 0], None)) is None


def test_off_calls_once_and_adds_nothing():
    one = _main_out([8, 8], {"m": 1})
    decide, calls = _fake({1: one, 4: _main_out([20, 1], {"m": 4})})
    out = E.decide_with_rethink(decide, json.dumps({"search_seed": 7}), None)
    assert out == one and "rethink" not in out
    assert len(calls) == 1


def test_fires_with_same_opts_but_worlds_k():
    decide, calls = _fake({1: _main_out([8, 6], {"m": 1}), 4: _main_out([20, 1], {"m": 4})})
    opts = {"search_seed": 7, "sims": 16, "commit": [], "net": "x"}
    out = E.decide_with_rethink(decide, json.dumps(opts), 0.5, 4)
    assert len(calls) == 2
    assert calls[1] == {**opts, "worlds": 4}, "読み直しは worlds だけを変える（同じ seed・持ち越し）"
    assert out["move"] == {"m": 4}
    assert out["rethink"] == {"r": 0.75, "fired": True, "first_move": {"m": 1}, "final_move": {"m": 4}}


def test_does_not_fire_below_line_or_when_worlds_already_k():
    decide, calls = _fake({1: _main_out([8, 2], {"m": 1}), 4: _main_out([20, 1], {"m": 4})})
    out = E.decide_with_rethink(decide, json.dumps({"search_seed": 7}), 0.5, 4)
    assert len(calls) == 1 and out["move"] == {"m": 1}
    assert out["rethink"] == {"r": 0.25, "fired": False, "first_move": {"m": 1}, "final_move": {"m": 1}}
    decide, calls = _fake({4: _main_out([8, 8], {"m": 4})})
    out = E.decide_with_rethink(decide, json.dumps({"search_seed": 7, "worlds": 4}), 0.0, 4)
    assert len(calls) == 1 and out["rethink"]["fired"] is False


# --- 実エンジン ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def engine():
    return load_engine()


def _hidden_with_choice() -> dict:
    """サンプルのリプレイから、木を回し候補が 2 つ以上ある決定点の `hidden` を探す。"""
    paths = sorted(glob.glob(_os.path.join(REPLAYS_DIR, "*", "*.json.gz")))
    assert paths, f"サンプルのリプレイが無い: {REPLAYS_DIR}"
    with gzip.open(paths[0], "rt", encoding="utf-8") as f:
        payload = json.load(f)
    turns = sorted({fr["turn"] for fr in (payload.get("frames") or []) if fr.get("turn")})
    for turn in turns:
        if turn < 3:
            continue
        try:
            hidden = hb.frame_to_hidden(payload, hb.turn_start_index(payload, turn), 0)
        except ValueError:
            continue
        _, tr = _decide_once(hidden)
        if tr.get("kind") == "main" and len(tr.get("candidates") or []) >= 2:
            return hidden
    raise AssertionError("候補が 2 つ以上ある決定点が無い")


def _decide_once(hidden: dict, **kw):
    random.seed(20261021)   # 探索乱数の基点（`_search_base`）を固定する
    game = RsGame.from_hidden(hidden, seed=0)
    pending = game.get_pending_request()
    assert pending, "決定点が無い盤面が復元された"
    tr: dict = {}
    move = game.decide(pending["player_id"], trace=tr, sims=SIMS, **kw)
    assert move is not None
    return move, tr


@pytest.fixture(scope="module")
def hidden(engine):
    return _hidden_with_choice()


def test_a_off_is_identical(hidden):
    base_move, base = _decide_once(hidden)
    off_move, off = _decide_once(hidden, rethink_ratio=None, rethink_worlds=WORLDS)
    assert off_move == base_move
    assert off == base, "off の trace は今までと 1 欄も変わらない"
    assert "rethink" not in off


def test_b_ratio_zero_equals_k_worlds(hidden):
    k_move, k = _decide_once(hidden, worlds=WORLDS)
    rt_move, rt = _decide_once(hidden, rethink_ratio=0.0, rethink_worlds=WORLDS)
    assert rt["rethink"]["fired"] is True
    assert rt_move == k_move
    rest = {key: v for key, v in rt.items() if key != "rethink"}
    assert rest == k, "読み直した結果は世界 K 本の decide と同一（手・統計・per_world）"


def test_c_large_ratio_equals_one_world(hidden):
    one_move, one = _decide_once(hidden)
    rt_move, rt = _decide_once(hidden, rethink_ratio=NEVER, rethink_worlds=WORLDS)
    assert rt["rethink"]["fired"] is False
    assert rt_move == one_move
    assert {key: v for key, v in rt.items() if key != "rethink"} == one


def test_d_trace_shape(hidden):
    one_move, _ = _decide_once(hidden)
    _, rt = _decide_once(hidden, rethink_ratio=0.0, rethink_worlds=WORLDS)
    info = rt["rethink"]
    assert set(info) == {"r", "fired", "first_move", "final_move"}
    assert isinstance(info["r"], float) and 0.0 <= info["r"] <= 1.0
    assert info["first_move"] == RsGame.from_hidden(hidden, seed=0).describe_move(one_move)
    assert info["final_move"] == rt["chosen"]


def test_c_driver_large_ratio_plays_the_same_game(engine):
    """アリーナ・生成の席（`run_game`）: 発動しない線では 1 本の席と同じ手順で打つ。"""
    db = D.load_db()
    seed = 2999900
    la, lb = D.leader_pair(db, seed, "random")
    p1, p2 = D.build_pair(db, la, lb, seed, "synth_roles")

    def play(spec, keep):
        seen = []

        def obs(game, name, turn, step, out, move):
            seen.append((name, json.dumps(move, sort_keys=True)))
            keep.append(out.get("rethink"))
        DR.run_game(seed, {"p1": spec, "p2": spec}, p1, p2, max_steps=40, observer=obs)
        return seen

    base_info, rt_info = [], []
    base = play(E.SeatSpec(None, sims=SIMS), base_info)
    rt = play(E.SeatSpec(None, sims=SIMS, rethink_ratio=NEVER, rethink_worlds=WORLDS), rt_info)
    assert rt == base
    assert all(i is None for i in base_info), "off の席の out に rethink は出ない"
    assert all(i is not None and i["fired"] is False for i in rt_info)
    assert "rethink_ratio" not in E.SeatSpec(None, rethink_ratio=0.5).opts, "opts には入れない"
