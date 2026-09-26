"""`two_curves_settle.py`（T137d・決着の瞬間に G(t) が Θ_opp(t) に届いているか）の算術を固める。

押さえるのは 3 つ:

1. **`declared_turns_by_seed_w`**: `settled_map` の `{(seed,w,t): bool}` から `declared=True` の
   ターンだけを `(seed,w)` ごとに昇順で集める（`False` は捨てる）。
2. **`g_at_or_before`**: `t` 以下で最後のターンの累積値を返す階段関数（`t` そのものに決定が無くても
   壊れない・`t` より前に何も無ければ 0.0）。
3. **`collect`**: 勝った席だけを測り（`dump` の `winner` と `w` が一致する行）、その席の最初の
   `declared` ターンで `G` と `Θ_opp`（`two_curves_state.state_by_turn` を再利用）を突き合わせて
   比・差を出す。**宣言が無い局**・**`state_by_turn` に無いターン**はそれぞれ数えて除外する。

**基盤健全性ではない**（器の誤りは「2 つの橋が同じ事実を指しているか」の判定を誤らせる）。必須側。
"""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import _bootstrap  # noqa: F401,E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
import two_curves_settle as TD  # noqa: E402


# ---- 1. declared_turns_by_seed_w ----------------------------------------------------------------

def test_declared_turns_keeps_only_true_and_sorts_ascending():
    settled = {(1, 0, 5): True, (1, 0, 1): True, (1, 0, 3): False, (1, 1, 2): True, (2, 0, 1): False}
    out = TD.declared_turns_by_seed_w(settled)
    assert out == {(1, 0): [1, 5], (1, 1): [2]}
    assert (2, 0) not in out                     # 全部 False の (seed,w) はキー自体が無い


def test_declared_turns_handles_empty_map():
    assert TD.declared_turns_by_seed_w({}) == {}


# ---- 2. g_at_or_before -----------------------------------------------------------------------

def test_g_at_or_before_picks_the_last_turn_not_after_t():
    turns, g = [1, 3, 5], [1.0, 2.0, 5.0]
    assert TD.g_at_or_before(turns, g, 3) == pytest.approx(2.0)     # ちょうど一致
    assert TD.g_at_or_before(turns, g, 4) == pytest.approx(2.0)     # 間＝直前の値
    assert TD.g_at_or_before(turns, g, 5) == pytest.approx(5.0)
    assert TD.g_at_or_before(turns, g, 100) == pytest.approx(5.0)   # 最後より後＝最後の値


def test_g_at_or_before_returns_zero_when_t_is_before_everything():
    assert TD.g_at_or_before([3, 5], [1.0, 2.0], 1) == pytest.approx(0.0)


def test_g_at_or_before_handles_empty_turns():
    assert TD.g_at_or_before([], [], 5) == pytest.approx(0.0)


# ---- 3. collect ----------------------------------------------------------------------------

def test_collect_measures_only_the_winning_seat_at_its_first_declared_turn(monkeypatch):
    dump = [
        {"seed": 1, "w": 0, "winner": 0, "turns": [1, 3, 5], "g": [1.0, 4.0, 10.0], "r": [0.5, 2.0, 6.0]},
        {"seed": 1, "w": 1, "winner": 0, "turns": [2, 4], "g": [0.5, 1.0], "r": [0.2, 0.4]},   # 敗者側＝測らない
    ]
    settled = {(1, 0, 1): False, (1, 0, 3): True, (1, 0, 5): True}           # 最初の宣言は t=3
    fake_state = {(0, 3): {"th_opp": 5.0, "th_me": 0.0, "a_me": 0.0, "a_opp": 0.0,
                          "life_me": 0.0, "life_opp": 0.0, "hand_me": 0.0, "hand_opp": 0.0}}
    monkeypatch.setattr(TD.KV, "_seat_decks", lambda dirs: {})
    monkeypatch.setattr(TD.LR, "settled_map", lambda dirs, limit_games: settled)
    monkeypatch.setattr(TD.TS, "state_by_turn", lambda *a, **k: fake_state)
    monkeypatch.setattr(TD.PL, "iter_games", lambda *a, **k: iter([({"seed": [1]}, {}, {}, [], [], [0])]))
    out = TD.collect(dump, ["x"])
    assert out["n_winners"] == 1                  # 席1の行は winner と w が食い違うので数えない
    assert out["n_no_declare"] == 0 and out["n_no_state"] == 0
    assert out["n_measured"] == 1
    # G(t=3) = 4.0・R(t=3) = 2.0（ちょうど一致）・Θ_opp(t=3) = 5.0 → G の比 0.8・差 -1.0／R の比 0.4・差 -3.0
    assert out["g_vs_theta"]["ratio"]["mean"] == pytest.approx(0.8)
    assert out["g_vs_theta"]["diff"]["mean"] == pytest.approx(-1.0)
    assert out["r_vs_theta"]["ratio"]["mean"] == pytest.approx(0.4)
    assert out["r_vs_theta"]["diff"]["mean"] == pytest.approx(-3.0)


def test_collect_counts_winners_with_no_declared_turn(monkeypatch):
    dump = [{"seed": 1, "w": 0, "winner": 0, "turns": [1], "g": [1.0], "r": [0.5]}]
    monkeypatch.setattr(TD.KV, "_seat_decks", lambda dirs: {})
    monkeypatch.setattr(TD.LR, "settled_map", lambda dirs, limit_games: {(1, 0, 1): False})
    monkeypatch.setattr(TD.TS, "state_by_turn", lambda *a, **k: {})
    monkeypatch.setattr(TD.PL, "iter_games", lambda *a, **k: iter([({"seed": [1]}, {}, {}, [], [], [0])]))
    out = TD.collect(dump, ["x"])
    assert out["n_winners"] == 1
    assert out["n_no_declare"] == 1
    assert out["n_measured"] == 0


def test_collect_counts_declared_turns_missing_from_state(monkeypatch):
    dump = [{"seed": 1, "w": 0, "winner": 0, "turns": [1], "g": [1.0], "r": [0.5]}]
    monkeypatch.setattr(TD.KV, "_seat_decks", lambda dirs: {})
    monkeypatch.setattr(TD.LR, "settled_map", lambda dirs, limit_games: {(1, 0, 1): True})
    monkeypatch.setattr(TD.TS, "state_by_turn", lambda *a, **k: {})        # (0,1) が無い
    monkeypatch.setattr(TD.PL, "iter_games", lambda *a, **k: iter([({"seed": [1]}, {}, {}, [], [], [0])]))
    out = TD.collect(dump, ["x"])
    assert out["n_no_declare"] == 0
    assert out["n_no_state"] == 1
    assert out["n_measured"] == 0


def test_collect_skips_seats_that_did_not_win(monkeypatch):
    dump = [{"seed": 1, "w": 1, "winner": 0, "turns": [2], "g": [1.0], "r": [0.5]}]     # w=1 は負けた席
    monkeypatch.setattr(TD.KV, "_seat_decks", lambda dirs: {})
    monkeypatch.setattr(TD.LR, "settled_map", lambda dirs, limit_games: {(1, 1, 2): True})
    monkeypatch.setattr(TD.TS, "state_by_turn", lambda *a, **k: {(1, 2): {"th_opp": 1.0}})
    monkeypatch.setattr(TD.PL, "iter_games", lambda *a, **k: iter([({"seed": [1]}, {}, {}, [], [], [0])]))
    out = TD.collect(dump, ["x"])
    assert out["n_winners"] == 0 and out["n_measured"] == 0


def test_collect_excludes_zero_or_negative_theta_opp_from_the_ratio_but_keeps_the_diff(monkeypatch):
    dump = [{"seed": 1, "w": 0, "winner": 0, "turns": [1], "g": [3.0], "r": [2.0]}]
    monkeypatch.setattr(TD.KV, "_seat_decks", lambda dirs: {})
    monkeypatch.setattr(TD.LR, "settled_map", lambda dirs, limit_games: {(1, 0, 1): True})
    monkeypatch.setattr(TD.TS, "state_by_turn", lambda *a, **k: {(0, 1): {"th_opp": 0.0}})
    monkeypatch.setattr(TD.PL, "iter_games", lambda *a, **k: iter([({"seed": [1]}, {}, {}, [], [], [0])]))
    out = TD.collect(dump, ["x"])
    assert out["g_vs_theta"]["ratio"]["n"] == 0     # 0 割りを避けて比だけ除く
    assert out["g_vs_theta"]["diff"]["n"] == 1 and out["g_vs_theta"]["diff"]["mean"] == pytest.approx(3.0)
    assert out["r_vs_theta"]["ratio"]["n"] == 0
    assert out["r_vs_theta"]["diff"]["n"] == 1 and out["r_vs_theta"]["diff"]["mean"] == pytest.approx(2.0)


# ---- 4. CLI --------------------------------------------------------------------------------------

def test_cli_reads_dump_file_and_reaches_collect(monkeypatch, tmp_path):
    p = tmp_path / "d.json"
    p.write_text('[{"seed": 1, "w": 0, "winner": 0, "turns": [], "g": []}]', encoding="utf-8")
    called = {}

    def fake_collect(dump, dirs, limit_games):
        called["dump"] = dump
        called["dirs"] = dirs
        called["limit_games"] = limit_games
        return {"n_measured": 0}

    monkeypatch.setattr(TD, "collect", fake_collect)
    TD.main(["--in", "x", "--dump", str(p), "--games", "7"])
    assert called["dirs"] == ["x"] and called["limit_games"] == 7
    assert called["dump"][0]["seed"] == 1
