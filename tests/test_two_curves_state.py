"""`two_curves_state.py`（T137c・曲線のズレ×状態量の表）の算術を固める。

押さえるのは 3 つ:

1. **`state_by_turn`**: `(w, t)` の**最初の `kind=0` 行**から状態を組む・**相手の `A`／手札価格は
   相手の直近の自席ターン**から読む（`kappa_vector._opp_at` と同じ規約）・**相手がまだ 1 ターンも
   打っていない席は state が無い**（先手の最初のターン）・ライフ／手札の枚数は `sc` から直接読む。
2. **`collect`**: `dump`（`two_curves.py --dump` の中身）と `state_by_turn` の出力を `(seed, w, t)` で
   突き合わせ、**残差 `resid = g − r`** を作り、**生の相関**と**`j`（ターン内の位置）の影を抜いた相関**
   （`rate_tracking.demean_by` の再利用）を状態量ごとに出す。**当てはめはしない**——係数は 1 つも作らない。
3. **CLI**（`--dump` を読み `collect` に届く）。

**基盤健全性ではない**——器の誤りは「ズレが盤面から読めるか」という理論の完成判定そのものを誤らせる
（T137a/b と同じ役割）ので必須側。
"""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import _bootstrap  # noqa: F401,E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
import theory_order as TO  # noqa: E402
import two_curves_state as TS  # noqa: E402


def _sc(life_me=3.0, life_opp=2.0, hand_me=4.0, hand_opp=5.0):
    sc = np.zeros(64)
    sc[TO.SC_MY_LIFE] = life_me
    sc[TO.SC_OPP_LIFE] = life_opp
    sc[TO.SC_MY_HAND] = hand_me
    sc[TO.SC_OPP_HAND] = hand_opp
    return sc


def _rows_ex(entries):
    """`(who, turn, kind)` の列から `state_by_turn` が読む `(rows, ex, idx)` を組む。
    `sc` は行ごとに別オブジェクト（`id(sc)` で行を見分けるスタブに使う）。"""
    rows = {"who": [e[0] for e in entries], "turn": [e[1] for e in entries], "kind": [e[2] for e in entries]}
    ex = {"sc": [_sc() for _ in entries], "tok": [object() for _ in entries], "ci": [object() for _ in entries]}
    idx = list(range(len(entries)))
    return rows, ex, idx


# ---- 1. state_by_turn -------------------------------------------------------------------------

def test_state_by_turn_reads_from_the_first_row_of_each_own_turn(monkeypatch):
    # t=1 席0（先手・相手にまだ手番が無い＝除外）／t=2 席1（相手＝席0が t=1 に打っている＝採用）
    entries = [(0, 1, 0), (1, 2, 0)]
    rows, ex, idx = _rows_ex(entries)
    rate_by_id = {id(ex["sc"][0]): 1.0, id(ex["sc"][1]): 2.0}
    monkeypatch.setattr(TS.KV, "_deck_of", lambda *a, **k: None)
    monkeypatch.setattr(TS.KV, "rate_of_row", lambda sc, tok, ci, idx2cid, cards, theta, mu, deck_ids=None, j=None:
                        rate_by_id[id(sc)])
    monkeypatch.setattr(TS.KV, "g_of_row", lambda *a, **k: None)
    monkeypatch.setattr(TS.KV, "state_of_row", lambda sc, tok, a_me, a_opp, j, g_me=None, g_opp=None:
                        (10.0, 20.0, a_me, a_opp, j))
    out = TS.state_by_turn(rows, ex, idx, object(), {}, {}, seed_g=1)
    assert (0, 1) not in out                     # 先手の最初のターンは相手情報が無い
    assert (1, 2) in out
    st = out[(1, 2)]
    assert st["a_me"] == pytest.approx(2.0)       # 席1自身のレート
    assert st["a_opp"] == pytest.approx(1.0)      # 相手（席0）の直近の自席ターン（t=1）のレート
    assert st["th_me"] == pytest.approx(10.0) and st["th_opp"] == pytest.approx(20.0)


def test_state_by_turn_reads_life_and_hand_from_sc(monkeypatch):
    entries = [(0, 1, 0), (1, 2, 0)]
    rows, ex, idx = _rows_ex(entries)
    ex["sc"][0] = _sc(life_me=3.0, life_opp=2.0, hand_me=4.0, hand_opp=5.0)
    ex["sc"][1] = _sc(life_me=1.0, life_opp=6.0, hand_me=2.0, hand_opp=7.0)
    monkeypatch.setattr(TS.KV, "_deck_of", lambda *a, **k: None)
    monkeypatch.setattr(TS.KV, "rate_of_row", lambda sc, tok, ci, idx2cid, cards, theta, mu, deck_ids=None, j=None: 0.0)
    monkeypatch.setattr(TS.KV, "g_of_row", lambda *a, **k: None)
    monkeypatch.setattr(TS.KV, "state_of_row", lambda sc, tok, a_me, a_opp, j, g_me=None, g_opp=None:
                        (0.0, 0.0, a_me, a_opp, j))
    out = TS.state_by_turn(rows, ex, idx, object(), {}, {}, seed_g=1)
    st = out[(1, 2)]
    assert st["life_me"] == pytest.approx(1.0) and st["life_opp"] == pytest.approx(6.0)
    assert st["hand_me"] == pytest.approx(2.0) and st["hand_opp"] == pytest.approx(7.0)


def test_state_by_turn_uses_only_the_first_row_of_a_turn(monkeypatch):
    # 席0の t=1 に 2 行（1 手目・2 手目）→ `rate_of_row` は 1 回しか呼ばれない（最初の行だけ）
    entries = [(0, 1, 0), (0, 1, 0), (1, 2, 0)]
    rows, ex, idx = _rows_ex(entries)
    calls = []
    monkeypatch.setattr(TS.KV, "_deck_of", lambda *a, **k: None)

    def fake_rate(sc, tok, ci, idx2cid, cards, theta, mu, deck_ids=None, j=None):
        calls.append(id(sc))
        return 5.0

    monkeypatch.setattr(TS.KV, "rate_of_row", fake_rate)
    monkeypatch.setattr(TS.KV, "g_of_row", lambda *a, **k: None)
    monkeypatch.setattr(TS.KV, "state_of_row", lambda sc, tok, a_me, a_opp, j, g_me=None, g_opp=None:
                        (0.0, 0.0, a_me, a_opp, j))
    TS.state_by_turn(rows, ex, idx, object(), {}, {}, seed_g=1)
    # 席0の t=1 は 1 回だけ（2 行目は無視）・席1の t=2 は別ターンなので別途 1 回＝合計 2 回
    assert calls == [id(ex["sc"][0]), id(ex["sc"][2])]


def test_state_by_turn_ignores_rows_with_kind_not_zero(monkeypatch):
    entries = [(0, 1, 0), (0, 1, 1), (1, 2, 0)]   # 2 行目は kind=1（閉じる行）
    rows, ex, idx = _rows_ex(entries)
    monkeypatch.setattr(TS.KV, "_deck_of", lambda *a, **k: None)
    monkeypatch.setattr(TS.KV, "rate_of_row", lambda *a, **k: 1.0)
    monkeypatch.setattr(TS.KV, "g_of_row", lambda *a, **k: None)
    monkeypatch.setattr(TS.KV, "state_of_row", lambda sc, tok, a_me, a_opp, j, g_me=None, g_opp=None:
                        (0.0, 0.0, a_me, a_opp, j))
    out = TS.state_by_turn(rows, ex, idx, object(), {}, {}, seed_g=1)
    assert set(out.keys()) == {(1, 2)}


def test_state_by_turn_adds_no_parts_by_default(monkeypatch):
    """**T143**: `with_parts` の既定は `False`＝従来の 8 量だけ（T137c の表は 1 ビットも動かない）。"""
    entries = [(0, 1, 0), (1, 2, 0)]
    rows, ex, idx = _rows_ex(entries)
    monkeypatch.setattr(TS.KV, "_deck_of", lambda *a, **k: None)
    monkeypatch.setattr(TS.KV, "rate_of_row", lambda *a, **k: 1.0)
    monkeypatch.setattr(TS.KV, "g_of_row", lambda *a, **k: None)
    monkeypatch.setattr(TS.KV, "state_of_row", lambda sc, tok, a_me, a_opp, j, g_me=None, g_opp=None:
                        (0.0, 0.0, a_me, a_opp, j))
    monkeypatch.setattr(TS.CB, "threshold_parts",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("呼ばれてはいけない")))
    out = TS.state_by_turn(rows, ex, idx, object(), {}, {}, seed_g=1)
    assert set(out[(1, 2)].keys()) == set(TS.STATE_KEYS)


def test_state_by_turn_parts_use_the_same_row_and_the_opponents_hand_price(monkeypatch):
    """**T143**: 3 項は**同じ行**の `sc`／`tok` と、`th_opp` に渡したのと**同じ相手の手札価格**で割る。"""
    entries = [(0, 1, 0), (1, 2, 0)]
    rows, ex, idx = _rows_ex(entries)
    g_by_id = {id(ex["sc"][0]): 0.031, id(ex["sc"][1]): 0.047}
    calls = []
    monkeypatch.setattr(TS.KV, "_deck_of", lambda *a, **k: None)
    monkeypatch.setattr(TS.KV, "rate_of_row", lambda *a, **k: 1.0)
    monkeypatch.setattr(TS.KV, "g_of_row", lambda sc, *a, **k: g_by_id[id(sc)])
    monkeypatch.setattr(TS.KV, "state_of_row", lambda sc, tok, a_me, a_opp, j, g_me=None, g_opp=None:
                        (0.0, 0.3, a_me, a_opp, j))

    def fake_parts(sc, tok, g_hand=None):
        calls.append((id(sc), id(tok), g_hand))
        return (0.1362, 0.11, 0.0538)

    monkeypatch.setattr(TS.CB, "threshold_parts", fake_parts)
    out = TS.state_by_turn(rows, ex, idx, object(), {}, {}, seed_g=1, with_parts=True)
    st = out[(1, 2)]
    # 席1の t=2 の行（ex の 1 番）と、相手（席0）の直近の自席ターン（t=1・ex の 0 番）の手札価格
    assert calls == [(id(ex["sc"][1]), id(ex["tok"][1]), pytest.approx(0.031))]
    assert (st["th_opp_life"], st["th_opp_hand"], st["th_opp_body"]) == pytest.approx((0.1362, 0.11, 0.0538))


def test_state_by_turn_parts_sum_back_to_th_opp_on_a_real_board(monkeypatch):
    """**T143 の恒等式**: 本物の `state_of_row`／`threshold_parts` で、**3 項の和 = `th_opp`**（ビット一致）。
    盤面は相手のライフ 2・手札 4・場のブロッカー 1 体（`THETA_HAND_MODE` などは出荷既定のまま）。"""
    entries = [(0, 1, 0), (1, 2, 0)]
    rows, ex, idx = _rows_ex(entries)
    sc = _sc(life_me=3.0, life_opp=2.0, hand_me=5.0, hand_opp=4.0)
    sc[TO.SC_MY_LEADER_POWER] = 0.5; sc[TO.SC_OPP_LEADER_POWER] = 0.5
    tok = np.zeros((22, 24), np.float32)
    tok[0, TO.S_POWER] = 0.5; tok[1, TO.S_POWER] = 0.5
    s_opp = TO.SLOT_OPP_FIELD.start
    tok[s_opp, TO.S_POWER], tok[s_opp, TO.S_IS_CHAR], tok[s_opp, TO.S_IS_BLOCKER] = 0.4, 1.0, 1.0
    s_own = TO.SLOT_OWN_FIELD.start
    tok[s_own, TO.S_POWER], tok[s_own, TO.S_IS_CHAR], tok[s_own, TO.S_CAN_ATTACK] = 0.7, 1.0, 1.0
    ex["sc"][1], ex["tok"][1] = sc, tok
    monkeypatch.setattr(TS.KV, "_deck_of", lambda *a, **k: None)
    monkeypatch.setattr(TS.KV, "rate_of_row", lambda *a, **k: 0.1)
    monkeypatch.setattr(TS.KV, "g_of_row", lambda *a, **k: 0.04)
    out = TS.state_by_turn(rows, ex, idx, object(), {}, {}, seed_g=1, with_parts=True)
    st = out[(1, 2)]
    parts = st["th_opp_life"] + st["th_opp_hand"] + st["th_opp_body"]
    assert parts == st["th_opp"]                                   # 同じ 3 つの浮動小数を同じ順に足す
    assert st["th_opp_life"] == pytest.approx(TO.LAM * 2.0)       # ライフの項は λ × 相手のライフ
    assert st["th_opp_body"] > 0.0                                 # アクティブなブロッカー 1 体


# ---- 2. collect ---------------------------------------------------------------------------------

def test_collect_matches_dump_and_state_by_seed_and_turn(monkeypatch):
    dump = [
        {"seed": 1, "w": 0, "turns": [1, 3], "g": [2.0, 5.0], "r": [1.0, 1.0]},   # resid = [1.0, 4.0]
        {"seed": 1, "w": 1, "turns": [2, 4], "g": [3.0, 2.0], "r": [1.0, 3.0]},   # resid = [2.0, -1.0]
    ]
    fake_state = {
        (0, 1): {"th_me": 1.0, "th_opp": 0.0, "a_me": 0.0, "a_opp": 0.0,
                 "life_me": 0.0, "life_opp": 0.0, "hand_me": 0.0, "hand_opp": 0.0},
        (0, 3): {"th_me": 4.0, "th_opp": 0.0, "a_me": 0.0, "a_opp": 0.0,
                 "life_me": 10.0, "life_opp": 0.0, "hand_me": 0.0, "hand_opp": 0.0},
        (1, 2): {"th_me": 2.0, "th_opp": 0.0, "a_me": 0.0, "a_opp": 0.0,
                 "life_me": 0.0, "life_opp": 0.0, "hand_me": 0.0, "hand_opp": 0.0},
        (1, 4): {"th_me": -1.0, "th_opp": 0.0, "a_me": 0.0, "a_opp": 0.0,
                 "life_me": 10.0, "life_opp": 0.0, "hand_me": 0.0, "hand_opp": 0.0},
    }
    monkeypatch.setattr(TS.KV, "_seat_decks", lambda dirs: {})
    monkeypatch.setattr(TS, "state_by_turn", lambda rows, ex, idx, cards, idx2cid, seat_decks, seed_g, theta, mu:
                        fake_state)
    monkeypatch.setattr(TS.PL, "iter_games", lambda *a, **k: iter([({"seed": [1]}, {}, {}, [], [], [0])]))
    out = TS.collect(dump, ["x"])
    assert out["n"] == 4
    # `th_me` は残差そのものにしてある → 生の相関も j を抜いた相関も 1.0（完全な線形一致）
    assert out["by_state"]["th_me"]["n"] == 4
    assert out["by_state"]["th_me"]["corr_raw"] == pytest.approx(1.0)
    assert out["by_state"]["th_me"]["corr_demeaned"] == pytest.approx(1.0)
    # `life_me` は `j`（ターン内の位置）だけの関数（同じ位置なら常に同じ値）→ j を抜くと定数＝相関 None
    assert out["by_state"]["life_me"]["corr_demeaned"] is None


def test_collect_skips_turns_with_no_matching_state(monkeypatch):
    dump = [{"seed": 1, "w": 0, "turns": [1, 3], "g": [1.0, 2.0], "r": [0.0, 0.0]}]
    fake_state = {(0, 1): {k: 0.0 for k in TS.STATE_KEYS}}          # t=3 の state が無い
    monkeypatch.setattr(TS.KV, "_seat_decks", lambda dirs: {})
    monkeypatch.setattr(TS, "state_by_turn", lambda *a, **k: fake_state)
    monkeypatch.setattr(TS.PL, "iter_games", lambda *a, **k: iter([({"seed": [1]}, {}, {}, [], [], [0])]))
    out = TS.collect(dump, ["x"])
    assert out["n"] == 1


def test_collect_skips_dump_entries_with_no_matching_game(monkeypatch):
    dump = [{"seed": 99, "w": 0, "turns": [1], "g": [1.0], "r": [0.0]}]     # 読み直した記録には無い seed
    monkeypatch.setattr(TS.KV, "_seat_decks", lambda dirs: {})
    monkeypatch.setattr(TS, "state_by_turn", lambda *a, **k: {})
    monkeypatch.setattr(TS.PL, "iter_games", lambda *a, **k: iter([({"seed": [1]}, {}, {}, [], [], [0])]))
    out = TS.collect(dump, ["x"])
    assert out["n"] == 0
    assert out["by_state"]["th_me"]["corr_raw"] is None


def test_collect_respects_games_limit(monkeypatch):
    calls = {"n": 0}

    def fake_iter(*a, **k):
        for seed in (1, 2, 3):
            calls["n"] += 1
            yield ({"seed": [seed]}, {}, {}, [], [], [0])

    monkeypatch.setattr(TS.KV, "_seat_decks", lambda dirs: {})
    monkeypatch.setattr(TS, "state_by_turn", lambda *a, **k: {})
    monkeypatch.setattr(TS.PL, "iter_games", fake_iter)
    TS.collect([], ["x"], limit_games=1)
    assert calls["n"] == 2                # 1 本を超えたところで打ち切る（break の位置は two_curves.py と同じ）


# ---- 3. CLI --------------------------------------------------------------------------------------

def test_cli_reads_dump_file_and_reaches_collect(monkeypatch, tmp_path):
    p = tmp_path / "d.json"
    p.write_text('[{"seed": 1, "w": 0, "turns": [], "g": [], "r": []}]', encoding="utf-8")
    called = {}

    def fake_collect(dump, dirs, limit_games):
        called["dump"] = dump
        called["dirs"] = dirs
        return {"n": 0, "by_state": {}}

    monkeypatch.setattr(TS, "collect", fake_collect)
    TS.main(["--in", "x", "--dump", str(p), "--games", "3"])
    assert called["dirs"] == ["x"]
    assert called["dump"][0]["seed"] == 1
