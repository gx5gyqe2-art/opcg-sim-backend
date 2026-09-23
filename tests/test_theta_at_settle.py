"""`theta_at_settle.py`（T143・決着の瞬間の `Θ_opp` をライフ／手札／体に分解する）の算術を固める。

押さえるのは 5 つ:

1. **`need_parts`**: `要 = 累積(終局) − 累積(宣言ターンの直前)` を総額と 3 部品で・宣言より前に何も無ければ
   0 から・**`r_life` 等の無い古い dump は黙って 0 にせず落ちる**。
2. **`join_row`**: 1 行の器・橋・要の 3 つを取り違えずに 1 行へ並べる。
3. **`invariants`**: 予告 1 の恒等式（3 項の和・同じ行のライフ・手札のブロッカーを引いた体・2 つの実装の要・
   窓は切るだけ）が**破れた行を数える**（黙って通さない）。
4. **`block`／`summarise`**: 平均どうしの比・部品ごとの超過・**1 行の器 − 橋 の 5 つの内訳の和が差の総額に一致**・
   `t_left` での分割。
5. **`collect`** の突き合わせ（勝った席だけ・宣言が無い／1 行の器に無い／橋に無い を別々に数える）と CLI。

**基盤健全性ではない**——器の誤りは「`Θ` のどの項が、決着の瞬間に要った損害を超えているか」という
次に直す項の選択を誤らせる（T137d の読みを訂正する根拠になる数字）。必須側。
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import _bootstrap  # noqa: F401,E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
import theta_at_settle as TA  # noqa: E402


def _seat(turns, r, r_life, r_hand, r_body, winner=0, w=0, seed=1):
    return {"seed": seed, "w": w, "winner": winner, "turns": turns, "r": r,
            "r_life": r_life, "r_hand": r_hand, "r_body": r_body}


# ---- 1. need_parts ---------------------------------------------------------------------------

def test_need_parts_is_end_minus_the_cumulative_before_the_declare_turn():
    seat = _seat([1, 3, 5], [0.10, 0.30, 0.55], [0.1362, 0.2724, 0.4086], [-0.05, -0.02, 0.06], [0.0138, 0.0476, 0.0814])
    nd = TA.need_parts(seat, 5)                  # 宣言は最後のターン＝直前は t=3
    assert nd["total"] == pytest.approx(0.55 - 0.30)
    assert nd["life"] == pytest.approx(0.4086 - 0.2724)
    assert nd["hand"] == pytest.approx(0.06 - (-0.02))
    assert nd["body"] == pytest.approx(0.0814 - 0.0476)


def test_need_parts_counts_from_zero_when_nothing_precedes_the_declare_turn():
    seat = _seat([3, 5], [0.2, 0.5], [0.1, 0.3], [0.05, 0.1], [0.05, 0.1])
    nd = TA.need_parts(seat, 3)
    assert nd["total"] == pytest.approx(0.5) and nd["life"] == pytest.approx(0.3)


def test_need_parts_reads_the_declare_turn_even_if_it_has_no_row_in_the_dump():
    """宣言ターンに値付けできた決定行が無くても、直前は「`t` より前の最後のターン」（階段関数）。"""
    seat = _seat([1, 5], [0.1, 0.6], [0.1, 0.4], [0.0, 0.1], [0.0, 0.1])
    assert TA.need_parts(seat, 3)["total"] == pytest.approx(0.6 - 0.1)


def test_need_parts_refuses_an_old_dump_without_the_parts():
    seat = {"seed": 1, "w": 0, "winner": 0, "turns": [1], "r": [0.1]}
    with pytest.raises(ValueError):
        TA.need_parts(seat, 1)


# ---- 2. join_row -----------------------------------------------------------------------------

def test_join_row_keeps_the_three_sources_apart():
    a = {"th_opp_life": 0.2724, "th_opp_hand": 0.11, "th_opp_body": 0.02, "th_opp": 0.4024,
         "life_opp": 2.0, "hand_opp": 4.0}
    b = {"t_left": 1, "th_life": 0.2724, "th_hand": 0.03, "th_hand_raw": 0.09, "th_body": 0.05, "th_hb": 0.03,
         "theta": 0.3524, "need": 0.2}
    nd = {"total": 0.2, "life": 0.2724, "hand": -0.0981, "body": 0.0257}
    row = TA.join_row(a, b, nd)
    assert (row["a_hand"], row["b_hand"], row["b_hand_raw"], row["b_hb"]) == (0.11, 0.03, 0.09, 0.03)
    assert (row["need_hand"], row["need_bridge"], row["t_left"]) == (-0.0981, 0.2, 1)


# ---- 3. invariants ---------------------------------------------------------------------------

def _good_row(**over):
    row = {"t_left": 1,
           "a_life": 0.2724, "a_hand": 0.11, "a_body": 0.02, "a_total": 0.4024,
           "b_life": 0.2724, "b_hand": 0.03, "b_hand_raw": 0.09, "b_body": 0.05, "b_hb": 0.03, "b_total": 0.3524,
           "need_total": 0.2, "need_life": 0.2724, "need_hand": -0.0981, "need_body": 0.0257, "need_bridge": 0.2,
           "life_opp": 2.0, "hand_opp": 4.0}
    row.update(over)
    return row


def test_invariants_are_clean_on_rows_that_satisfy_the_identities():
    inv = TA.invariants([_good_row(), _good_row()])
    assert all(v["n_bad"] == 0 for v in inv.values()), inv


def test_invariants_count_each_broken_identity_separately():
    rows = [_good_row(a_total=0.5),                          # 1 行の器の和が合わない
            _good_row(b_life=0.1362, b_total=0.2162),         # 同じ行のはずがライフが違う（和は合わせてある）
            _good_row(need_bridge=0.25),                      # 2 つの実装の要が違う
            _good_row(b_hand=0.12, b_total=0.4424)]           # 窓が手札を増やしている（和は合わせてある）
    inv = TA.invariants(rows)
    assert inv["sum_one_row"]["n_bad"] == 1
    assert inv["life_same_row"]["n_bad"] == 1
    assert inv["need_two_impls"]["n_bad"] == 1
    assert inv["window_only_cuts"]["n_bad"] == 1
    assert inv["sum_bridge"]["n_bad"] == 0 and inv["need_parts_sum"]["n_bad"] == 0


# ---- 4. block / summarise ------------------------------------------------------------------

def test_block_reports_ratios_of_means_and_part_by_part_excess():
    blk = TA.block([_good_row(), _good_row(a_hand=0.13, a_total=0.4224, need_total=0.22, need_bridge=0.22,
                                           need_body=0.0457)])
    assert blk["n"] == 2
    assert blk["one_row"]["hand"] == pytest.approx(0.12)
    assert blk["one_row_over_need"] == pytest.approx(round(((0.4024 + 0.4224) / 2) / 0.21, 4))
    assert blk["excess_one_row"]["hand"] == pytest.approx(0.12 - (-0.0981), abs=1e-4)
    assert blk["share_need_hand_negative"] == pytest.approx(1.0)


def test_the_five_differences_add_up_to_the_gap_between_the_two_thetas():
    """**1 行の器 − 橋** = `g` の読み ＋ 窓 ＋ 手札のブロッカー ＋ ライフ ＋ 体（ブロッカー）——代数で恒等。"""
    rows = [_good_row(), _good_row(a_hand=0.2, a_total=0.4924, b_hand_raw=0.15, b_hand=0.05, b_total=0.3724)]
    d = TA.block(rows)["one_row_minus_bridge"]
    parts = d["g_reading"] + d["window"] + d["hand_blocker"] + d["life"] + d["body_blockers"]
    assert parts == pytest.approx(d["total"], abs=1e-3)       # 各項を小数 4 桁に丸めた誤差まで
    assert d["window"] > 0.0 and d["hand_blocker"] < 0.0


def test_block_is_empty_for_no_rows():
    assert TA.block([]) == {"n": 0}


def test_need_identity_compares_after_the_bridges_zero_floor():
    """橋の `need` は `max(0, ·)`——dump の要が負の行は、切った 0 と橋の 0 が一致すれば検算は通る。"""
    row = _good_row(need_total=-0.03, need_life=0.1362, need_hand=-0.1919, need_body=0.0257, need_bridge=0.0)
    inv = TA.invariants([row])
    assert inv["need_two_impls"]["n_bad"] == 0 and inv["need_parts_sum"]["n_bad"] == 0
    assert TA.summarise([row, _good_row()])["n_need_total_negative"] == 1


def test_summarise_splits_the_last_turn_from_earlier_declarations():
    out = TA.summarise([_good_row(), _good_row(t_left=3)])
    assert out["all"]["n"] == 2
    assert out["by_t_left"]["1"]["n"] == 1 and out["by_t_left"]["2+"]["n"] == 1
    assert out["invariants"]["sum_one_row"]["n_bad"] == 0
    assert out["n_need_total_negative"] == 0


# ---- 5. collect / CLI ----------------------------------------------------------------------

def _stub_passes(monkeypatch, settled, one_row_state, theta_check):
    monkeypatch.setattr(TA.KV, "_seat_decks", lambda dirs: {})
    monkeypatch.setattr(TA.LR, "settled_map", lambda dirs, limit_games: settled)
    monkeypatch.setattr(TA.PL, "iter_games", lambda *a, **k: iter([({"seed": [1]}, {}, {}, [], [], [0]),
                                                                   ({"seed": [2]}, {}, {}, [], [], [0])]))
    monkeypatch.setattr(TA.TS, "state_by_turn",
                        lambda rows, ex, idx, cards, idx2cid, seat_decks, seed_g, theta, mu, with_parts=False:
                        one_row_state.get(seed_g, {}))
    monkeypatch.setattr(TA.CB, "collect", lambda dirs, limit_games, theta, mu, mode: ([], [], {}, [], theta_check))


def _state():
    return {"th_opp_life": 0.2724, "th_opp_hand": 0.11, "th_opp_body": 0.02, "th_opp": 0.4024,
            "life_opp": 2.0, "hand_opp": 4.0}


def _bridge_row(seed, t):
    return {"seed": seed, "who": 0, "t": t, "t_left": 1, "th_life": 0.2724, "th_hand": 0.03, "th_hand_raw": 0.09,
            "th_body": 0.05, "th_hb": 0.03, "theta": 0.3524, "need": 0.2}


def test_collect_joins_the_winner_at_its_first_declared_turn(monkeypatch):
    dump = [_seat([1, 3], [0.1, 0.3], [0.1362, 0.2724], [0.0, -0.05], [0.0, 0.0], seed=1),
            _seat([2], [0.1], [0.1], [0.0], [0.0], winner=0, w=1, seed=1)]          # 負けた席＝測らない
    settled = {(1, 0, 1): False, (1, 0, 3): True}
    _stub_passes(monkeypatch, settled, {1: {(0, 3): _state()}}, [_bridge_row(1, 3)])
    out = TA.collect(dump, ["x"])
    assert out["n_winners"] == 1 and out["n_measured"] == 1
    assert out["all"]["need"]["total"] == pytest.approx(0.2)            # 0.3 − 0.1
    assert out["all"]["need"]["life"] == pytest.approx(0.1362)


def test_collect_counts_each_missing_join_separately(monkeypatch):
    dump = [_seat([1, 3], [0.1, 0.3], [0.1, 0.2], [0.0, 0.0], [0.0, 0.1], seed=1),        # 宣言なし
            _seat([1, 3], [0.1, 0.3], [0.1, 0.2], [0.0, 0.0], [0.0, 0.1], seed=2),        # 1 行の器に無い
            _seat([1, 3], [0.1, 0.3], [0.1, 0.2], [0.0, 0.0], [0.0, 0.1], seed=3)]        # 橋に無い
    settled = {(2, 0, 3): True, (3, 0, 3): True}
    _stub_passes(monkeypatch, settled, {3: {(0, 3): _state()}}, [])
    monkeypatch.setattr(TA.PL, "iter_games", lambda *a, **k: iter([({"seed": [s]}, {}, {}, [], [], [0])
                                                                   for s in (1, 2, 3)]))
    out = TA.collect(dump, ["x"])
    assert (out["n_winners"], out["n_no_declare"], out["n_no_one_row"], out["n_no_bridge"]) == (3, 1, 1, 1)
    assert out["n_measured"] == 0


def test_cli_reads_the_dump_and_reaches_collect(monkeypatch, tmp_path):
    p = tmp_path / "d.json"
    p.write_text(json.dumps([_seat([1], [0.1], [0.1], [0.0], [0.0])]), encoding="utf-8")
    called = {}

    def fake_collect(dump, dirs, limit_games):
        called.update(dump=dump, dirs=dirs, limit_games=limit_games)
        return {"n_measured": 0}

    monkeypatch.setattr(TA, "collect", fake_collect)
    TA.main(["--in", "x", "--dump", str(p), "--games", "4"])
    assert called["dirs"] == ["x"] and called["limit_games"] == 4 and called["dump"][0]["seed"] == 1
