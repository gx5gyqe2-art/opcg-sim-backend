"""`two_curves.py`（T137a・理論の累積 `G(t)` と棋譜の累積 `R(t)`）の算術を固める。

**フルの局面（`score_candidate`／`_state_of`／`opp_bodies_of` が全部協調して動く行）での正しさは
本器の `--verify`（`crossing_bridge` と独立に書いた実装が同じ `F_end_mean` に収束するか）で確かめる**
——実記録・合成記録の両方で完全一致を確認済み（`docs/reports/2026-09-23_two_curves.md`）。
ここでは**記録を読まずに確かめられる算術**だけを固める:

1. **`harm_of`** は `crossing_bridge.harm_of` と同じ式（独立に書いた・値が一致するのを固定）。
2. **`_price_row`** は `score_candidate` をそのまま呼び、**帳簿の規約（T58・T85）で読み直す**——
   `ATTACK` は素の値のまま・**`attach` 系の族は既定 `ATTACH_LEDGER_MODE=in_attack` でゼロになる**。
3. **`two_curves_for_game` の累積の束ね方**（`_price_row`／`AR.parts`／`AR.parts_mirror` を差し替えて、
   `g`／`r` が正しい順序・正しい席に積まれるか）。
4. **`collect` の集計**（席ごとの終点・勝った席の `R` 終点の平均）と
   **`verify_against_crossing_bridge` の差分の算術**。

**基盤健全性ではない**——器の誤りは「2 本の曲線」という理論の完成判定そのものを誤らせる（この器の
数字がそのまま T137b 以降の土台になる）ので必須側。
"""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import _bootstrap  # noqa: F401,E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
import theory_bridge as TB  # noqa: E402
import theory_order as TO  # noqa: E402
import two_curves as TC  # noqa: E402


# ---- 1. harm_of ------------------------------------------------------------------------------

def test_harm_of_matches_crossing_bridge():
    import crossing_bridge as CB
    p = {"opp_life": 0.136, "opp_hand": -0.05, "opp_body": 0.02,
         "my_life": -9, "my_hand": 9, "my_body": 9, "don": 9}
    assert TC.harm_of(p) == pytest.approx(CB.harm_of(p))
    assert TC.harm_of(p) == pytest.approx(0.106)


# ---- 2. _price_row（帳簿の規約） ------------------------------------------------------------------

def _minimal_row(my_life=3.0, opp_life=2.0, my_don=0.0, leader_power=5000.0, olp=5000.0):
    """`_price_row` が最低限読む列だけを埋めた `(sc, tok, ci)`（`tests/test_lethal_rule.py::_row` と同じ作り）。"""
    sc = np.zeros(64)
    sc[TO.SC_MY_LIFE] = my_life; sc[TO.SC_OPP_LIFE] = opp_life; sc[TO.SC_MY_DON] = my_don
    sc[TO.SC_MY_LEADER_POWER] = leader_power / 1e4; sc[TO.SC_OPP_LEADER_POWER] = olp / 1e4
    tok = np.zeros((24, 24))
    tok[0, TO.S_POWER] = leader_power / 1e4
    ci = np.zeros(24)
    return sc, tok, ci


class _Cards:
    """`_state_of`／`opp_bodies_of` が空の場でも例外を出さない最小のスタブ。"""

    def info(self, cid):
        return None


def test_price_row_attack_on_leader_matches_attack_value():
    sc, tok, ci = _minimal_row(opp_life=2.0, leader_power=7000.0, olp=5000.0)
    idx2cid = {}
    cards = _Cards()
    sig = ["ATTACK", "u", ["leader"], [], None]
    g = TC._price_row(sc, tok, ci, cards, idx2cid, sig, None, None, 0, 0, 0, TO.THETA, TO.MU)
    # cid が None（src が読めない）だと score_candidate は ATTACK でも `sp = src_power` を使うので
    # `src is None and at != "PLAY"` の分岐に落ちる——`src_power` を渡していても `src is None` で弾かれる
    # （`score_candidate` の仕様どおり）ので、リーダー狙いの値付けには実在する cid が要る。
    assert g is None


def test_price_row_zeroes_attach_family_rows_under_the_shipped_ledger_mode():
    """**T85**: `ATTACH_LEDGER_MODE=in_attack`（既定）では付与の行は帳簿で 0 になる。"""
    assert TB.ATTACH_LEDGER_MODE == "in_attack"
    sc, tok, ci = _minimal_row()
    idx2cid = {}
    cards = _Cards()
    sig = ["ATTACH_DON", "u", [], [], None]        # 対象無し＝attach 族（move_family）
    assert TB.move_family(sig) == "attach"
    g = TC._price_row(sc, tok, ci, cards, idx2cid, sig, None, None, 0, 0, 1, TO.THETA, TO.MU)
    # `src is None` で値付けできない（cid が無い）＝ None のまま——ゼロ化の分岐に入る前に None で返る。
    # ゼロ化そのものは `move_family(sig) == "attach"` の判定が正しく効くことを別途固定する。
    assert g is None


def test_price_row_turn_end_is_zero():
    sc, tok, ci = _minimal_row()
    idx2cid = {}
    cards = _Cards()
    sig = ["TURN_END", None, [], [], None]
    g = TC._price_row(sc, tok, ci, cards, idx2cid, sig, None, None, 0, 0, 0, TO.THETA, TO.MU)
    assert g == 0.0


# ---- 3. two_curves_for_game の束ね方 -------------------------------------------------------------

def _fake_rows(entries):
    """`(seed, who, turn, kind, k, ch, z)` の決定行の列から `two_curves_for_game` が読む
    `(rows, pol, ex, L, ptr, idx)` を組む。**各決定行の直後に同じターンの「閉じる行」
    （`kind=1`・決定としては読まれないが T113 のブラケットの相手になる）を 1 つ挟む**
    ——実記録では `TURN_END` などが必ずその役を果たすので、実際の並びに合わせる。
    `sig` は空文字列でよい（`_price_row`／`AR.parts` を差し替えるので中身は読まれない）。"""
    full = []
    for e in entries:
        full.append(e)
        full.append((e[0], e[1], e[2], 1, 0, -1, 0.0))     # 閉じる行（同じ席・同じターン・kind=1）
    n = len(full)
    rows = {"who": [e[1] for e in full], "turn": [e[2] for e in full],
            "kind": [e[3] for e in full], "z": [e[6] for e in full]}
    L = [e[4] for e in full]
    pol = {"pol_chosen": [e[5] for e in full], "pol_sig": ["[]"] * (n * 2),
           "pol_cid": [""] * (n * 2), "pol_tcid": [""] * (n * 2),
           "pol_si": [0] * (n * 2), "pol_ti": [0] * (n * 2), "pol_k": [0] * (n * 2)}
    rows["pol_chosen"] = pol["pol_chosen"]
    ptr = list(range(0, n * 2, 2))
    ex = {"sc": [None] * n, "tok": [None] * n, "ci": [None] * n}
    idx = list(range(n))
    return rows, pol, ex, L, ptr, idx


def test_two_curves_accumulates_in_turn_order_per_seat(monkeypatch):
    # 席 0: t=1 に 1 手（g=1.0, r=0.5）・t=3 に 1 手（g=2.0, r=1.0）→ 累積 [1.0, 3.0] / [0.5, 1.5]
    # 席 1: t=2 に 1 手（g=0.5, r=0.25）→ 累積 [0.5] / [0.25]
    entries = [
        (1, 0, 1, 0, 1, 0, 0.0),
        (1, 1, 2, 0, 1, 0, 0.0),
        (1, 0, 3, 0, 1, 0, 1.0),        # 勝者 0
    ]
    rows, pol, ex, L, ptr, idx = _fake_rows(entries)
    # `_price_row` の呼び出し引数からは (w, t) を復元しにくいので、行の順序に沿って値を返す簡易スタブに差し替える
    seq_g = iter([1.0, 0.5, 2.0])
    seq_r = iter([0.5, 0.25, 1.0])
    monkeypatch.setattr(TC, "_price_row", lambda *a, **k: next(seq_g))
    monkeypatch.setattr(TC.AR, "parts", lambda sc, tok, sc2, tok2: {"opp_life": next(seq_r), "opp_hand": 0.0, "opp_body": 0.0})
    monkeypatch.setattr(TC.AR, "parts_mirror", lambda sc, tok, sc2, tok2: {"opp_life": 0.0, "opp_hand": 0.0, "opp_body": 0.0})
    cards, idx2cid = object(), {}
    curves, winner = TC.two_curves_for_game(rows, pol, ex, L, ptr, idx, cards, idx2cid)
    assert winner == 0
    assert curves[0]["turns"] == [1, 3]
    assert curves[0]["g"] == pytest.approx([1.0, 3.0])
    assert curves[0]["r"] == pytest.approx([0.5, 1.5])
    assert curves[1]["turns"] == [2]
    assert curves[1]["g"] == pytest.approx([0.5])
    assert curves[1]["r"] == pytest.approx([0.25])


def test_two_curves_family_breakdown_sums_to_the_turn_total(monkeypatch):
    """**T137b の材料**: `g_fam`（型別の内訳・その 1 ターンぶん）の和は `g` の**増分**と一致する
    （`g_turn` を分割しただけで、新しい量は作っていない）。"""
    entries = [(1, 0, 1, 0, 1, 0, 1.0), (1, 0, 3, 0, 1, 0, 0.0)]
    rows, pol, ex, L, ptr, idx = _fake_rows(entries)
    monkeypatch.setattr(TC, "move_family", lambda sig: "attack")
    seq_g = iter([1.5, 0.5])
    monkeypatch.setattr(TC, "_price_row", lambda *a, **k: next(seq_g))
    monkeypatch.setattr(TC.AR, "parts", lambda *a, **k: {"opp_life": 0.0, "opp_hand": 0.0, "opp_body": 0.0})
    monkeypatch.setattr(TC.AR, "parts_mirror", lambda *a, **k: {"opp_life": 0.0, "opp_hand": 0.0, "opp_body": 0.0})
    curves, _winner = TC.two_curves_for_game(rows, pol, ex, L, ptr, idx, object(), {})
    assert len(curves[0]["g_fam"]) == 2
    g_incr = [curves[0]["g"][0]] + [b - a for a, b in zip(curves[0]["g"], curves[0]["g"][1:])]
    for incr, fam in zip(g_incr, curves[0]["g_fam"]):
        assert sum(fam.values()) == pytest.approx(incr)
    assert curves[0]["g_fam"][0] == {"attack": pytest.approx(1.5)}


def test_two_curves_skips_rows_with_no_valid_choice(monkeypatch):
    entries = [(1, 0, 1, 0, 0, -1, 0.0)]          # k=0（候補無し）
    rows, pol, ex, L, ptr, idx = _fake_rows(entries)
    monkeypatch.setattr(TC, "_price_row", lambda *a, **k: (_ for _ in ()).throw(AssertionError("呼ばれてはいけない")))
    cards, idx2cid = object(), {}
    curves, winner = TC.two_curves_for_game(rows, pol, ex, L, ptr, idx, cards, idx2cid)
    assert curves[0]["turns"] == [] and curves[1]["turns"] == []
    assert winner is None


def test_two_curves_treats_none_price_as_zero(monkeypatch):
    entries = [(1, 0, 1, 0, 1, 0, 1.0)]
    rows, pol, ex, L, ptr, idx = _fake_rows(entries)
    monkeypatch.setattr(TC, "_price_row", lambda *a, **k: None)
    monkeypatch.setattr(TC.AR, "parts", lambda *a, **k: {"opp_life": 0.3, "opp_hand": 0.0, "opp_body": 0.0})
    monkeypatch.setattr(TC.AR, "parts_mirror", lambda *a, **k: {"opp_life": 0.0, "opp_hand": 0.0, "opp_body": 0.0})
    curves, winner = TC.two_curves_for_game(rows, pol, ex, L, ptr, idx, object(), {})
    assert curves[0]["g"] == pytest.approx([0.0])         # None は積まない（0 のまま）
    assert curves[0]["r"] == pytest.approx([0.3])


# ---- 4. collect の集計 ------------------------------------------------------------------------

def test_collect_aggregates_ends_and_winner_r(monkeypatch):
    def fake_for_game(rows, pol, ex, L, ptr, idx, cards, idx2cid, theta, mu):
        return ({0: {"turns": [1], "g": [1.0], "r": [0.4]},
                1: {"turns": [2], "g": [2.0], "r": [0.6]}}, 0)

    monkeypatch.setattr(TC, "two_curves_for_game", fake_for_game)
    monkeypatch.setattr(TC.PL, "iter_games", lambda *a, **k: iter([({"seed": [7]}, {}, {}, [], [], [0])]))
    dump = []
    out = TC.collect(["x"], dump=dump)
    assert out["games"] == 1 and out["seats"] == 2
    assert out["end_g_mean"] == pytest.approx((1.0 + 2.0) / 2)
    assert out["end_r_mean"] == pytest.approx((0.4 + 0.6) / 2)
    assert out["winner_final_r_mean"] == pytest.approx(0.4)         # 勝者は席 0
    assert out["winner_final_r_n"] == 1
    assert len(dump) == 2 and dump[0]["seed"] == 7


def test_collect_skips_seats_with_no_turns(monkeypatch):
    def fake_for_game(*a, **k):
        return ({0: {"turns": [], "g": [], "r": []}, 1: {"turns": [1], "g": [0.5], "r": [0.2]}}, 1)

    monkeypatch.setattr(TC, "two_curves_for_game", fake_for_game)
    monkeypatch.setattr(TC.PL, "iter_games", lambda *a, **k: iter([({"seed": [1]}, {}, {}, [], [], [0])]))
    out = TC.collect(["x"])
    assert out["seats"] == 1                     # 席 0 は turns が空なので数えない


# ---- 5. verify_against_crossing_bridge の算術 -----------------------------------------------------

def test_verify_computes_the_diff_against_crossing_bridge(monkeypatch):
    monkeypatch.setattr(TC, "collect", lambda dirs, limit_games, theta, mu:
                        {"winner_final_r_mean": 1.05, "winner_final_r_n": 40})

    class _CB:
        @staticmethod
        def collect(dirs, limit_games, theta, mu):
            return ([], [], {}, [], [])

        @staticmethod
        def summarise(rows_out, ledger, turn_harm, theta_check):
            return {"ledger": {"F_end_mean": 1.0, "winners": 40}}

    monkeypatch.setitem(sys.modules, "crossing_bridge", _CB)
    out = TC.verify_against_crossing_bridge(["x"])
    assert out["two_curves_F_end_mean"] == 1.05
    assert out["crossing_bridge_F_end_mean"] == 1.0
    assert out["diff"] == pytest.approx(0.05)
    assert out["rel_diff"] == pytest.approx(0.05)


def test_cli_reaches_collect_and_verify(monkeypatch):
    called = {}

    def fake_collect(dirs, limit_games, dump=None):
        called["collect"] = (dirs, limit_games)
        return {"a": 1}

    def fake_verify(dirs, limit_games):
        called["verify"] = (dirs, limit_games)
        return {"b": 2}

    monkeypatch.setattr(TC, "collect", fake_collect)
    monkeypatch.setattr(TC, "verify_against_crossing_bridge", fake_verify)
    TC.main(["--in", "x", "--games", "5", "--verify"])
    assert called["collect"] == (["x"], 5) and called["verify"] == (["x"], 5)
