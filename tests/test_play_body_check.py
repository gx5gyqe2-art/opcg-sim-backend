"""`play_body_check.py`（T157・出す手の値段は両端で外れているか）の算術を固める。

登場した枠の見つけ方・体を追う在庫の足し方（`nu_stock` と同じ作法）・帯の切り方・束ねの比を値で押さえる。
"""
import os
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.cpu_infra

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bootstrap  # noqa: F401,E402

_SCRIPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import play_body_check as PB  # noqa: E402
from theory_order import MU, S_IS_CHAR, SLOT_OWN_FIELD  # noqa: E402

N_SLOTS = SLOT_OWN_FIELD.stop + 4
N_COLS = S_IS_CHAR + 2


def _board(chars):
    """`chars` = {枠: card_idx} の盤面（tok・ci）。"""
    tok = np.zeros((N_SLOTS, N_COLS), dtype=np.float32)
    ci = np.zeros(N_SLOTS, dtype=np.int64)
    for s, c in chars.items():
        tok[s, S_IS_CHAR] = 1.0
        ci[s] = c
    return tok, ci


def test_cost_and_nu_bands_follow_the_printed_cost_and_card_units():
    assert PB.cost_band(1) == "c1_2" and PB.cost_band(2) == "c1_2"
    assert PB.cost_band(4) == "c3_4" and PB.cost_band(6) == "c5_6" and PB.cost_band(10) == "c7_10"
    assert PB.cost_band(0) == "c0"
    assert PB.nu_band(1.0 * MU) == "nu_lt_1_5"
    assert PB.nu_band(2.0 * MU) == "nu_1_5_2_5"
    assert PB.nu_band(3.4 * MU) == "nu_ge_2_5"


def test_landing_slot_is_the_slot_where_the_card_newly_appears():
    """前の盤面に同じ札が居た枠は飛ばし、新しく現れた枠を返す。現れなければ None。"""
    s0, s1 = SLOT_OWN_FIELD.start, SLOT_OWN_FIELD.start + 1
    tb, cb = _board({s0: 7})
    ta, ca = _board({s0: 7, s1: 7})
    assert PB.landing_slot(tb, cb, ta, ca, 7) == s1
    assert PB.landing_slot(tb, cb, tb, cb, 7) is None
    assert PB.landing_slot(tb, cb, ta, ca, 9) is None


def test_follow_body_sums_attacks_while_alive_and_marks_removal():
    """登場ターンから、枠に同じ札が居る間の攻撃の実現を足す。消えたら止めて died。終局まで居れば died=False。
    `turn_start` の値は 3 要素（`(sc, tok, ci)`・4 要素目〔行 index〕を足しても崩れないことも見る）。"""
    s = SLOT_OWN_FIELD.start
    alive = _board({s: 5})
    gone = _board({})
    ts = [1, 3, 5, 7]
    turn_start = {1: (None, *alive), 3: (None, *alive), 5: (None, *gone), 7: (None, *alive, 99)}
    atk = {(3, s): [0.05, 0.02], (5, s): [0.9], (7, s): [0.9]}
    stock, n_atk, turns, died = PB.follow_body(turn_start, ts, 0, s, 5, atk)
    assert stock == pytest.approx(0.07) and n_atk == 2 and turns == 2 and died is True
    stock, n_atk, turns, died = PB.follow_body(turn_start, ts, 3, s, 5, atk)
    assert stock == pytest.approx(0.9) and n_atk == 1 and turns == 1 and died is False


# --- P8-7(a): aux_def から場を離れた先を読む（effect_fate） -----------------
def test_effect_fate_reads_left_dest_at_the_last_alive_turn_start():
    """`turns_alive`（`follow_body` が確認した最後の生存ターン数）から `ts` の該当行を引き、
    その行 index で `aux_def` の左端の枠（`a_idx = slot - SLOT_OWN_FIELD.start + 1`）を読む。"""
    s = SLOT_OWN_FIELD.start
    alive = _board({s: 5})
    ts = [1, 3, 5]
    aux_def = np.zeros((3, 6, 4), dtype=np.float32)
    aux_def[1, 1, 3] = PB.LEFT_DEST_TRASH_EFFECT       # 行 index 1（ts[1]=3）の枠 1（=field slot s）
    turn_start = {1: (None, *alive, 0), 3: (None, *alive, 1), 5: (None, *alive, 2)}
    # turns_alive=2 → 最後の生存ターンは ts[0+2-1]=ts[1]=3 → 行 index 1
    assert PB.effect_fate(turn_start, ts, 0, 2, s, aux_def) == PB.LEFT_DEST_TRASH_EFFECT


def test_effect_fate_is_none_without_aux_def_or_out_of_range():
    s = SLOT_OWN_FIELD.start
    alive = _board({s: 5})
    ts = [1, 3]
    turn_start = {1: (None, *alive, 0), 3: (None, *alive, 1)}
    assert PB.effect_fate(turn_start, ts, 0, 1, s, None) is None                      # aux_def が無い波
    assert PB.effect_fate(turn_start, ts, 0, 5, s, np.zeros((2, 6, 4))) is None        # ts の外


def _row(cost_band="c1_2", nu=0.06, price=-0.01, stock=0.06, dg=0.0, slot=10, died=True, fate=None):
    return {"cost_band": cost_band, "nu_band": PB.nu_band(nu), "price": price, "nu": nu, "mu": MU,
            "opportunity": 0.01, "effect": 0.0, "dg": dg, "dh": 0.05, "counter": 1000.0, "counter_card": False,
            "slot": slot, "stock": stock, "n_atk": 2, "turns_alive": 3, "turns_left": 4, "died": died,
            "fate": fate}


def test_block_reports_fate_shares_when_present():
    rs = [_row(fate=PB.LEFT_DEST_TRASH_BATTLE), _row(fate=PB.LEFT_DEST_TRASH_EFFECT),
          _row(fate=PB.LEFT_DEST_HAND), _row(fate=None)]
    o = PB.block(rs)
    assert o["fate_n"] == 3                                    # fate=None の行は数えない
    assert o["battle_share"] == pytest.approx(1 / 3, abs=1e-3)
    assert o["effect_trash_share"] == pytest.approx(1 / 3, abs=1e-3)
    assert o["effect_hand_share"] == pytest.approx(1 / 3, abs=1e-3)
    assert o["effect_deck_share"] == pytest.approx(0.0, abs=1e-3)
    assert o["effect_removed_share"] == pytest.approx(2 / 3, abs=1e-3)


def test_block_has_no_fate_keys_when_no_fate_is_known():
    o = PB.block([_row(), _row(fate=None)])
    assert "fate_n" not in o and "effect_removed_share" not in o


def test_block_reports_card_units_and_stock_over_nu_on_tracked_rows_only():
    rs = [_row(nu=0.06, stock=0.03, dg=0.0), _row(nu=0.06, stock=0.09, dg=MU),
          dict(_row(), slot=None, stock=None)]
    o = PB.block(rs)
    assert o["n"] == 3 and o["n_tracked"] == 2
    assert o["stock_over_nu"] == pytest.approx(0.06 / 0.06)
    assert o["stock_cards"] == pytest.approx(0.06 / MU, abs=1e-3)
    assert o["guard_alt_cards"] == pytest.approx((0.0 + MU + 0.0) / 3 / MU, abs=1e-3)
    assert o["guard_zero_share"] == pytest.approx(2 / 3, abs=1e-3)
    assert o["neg_price_share"] == pytest.approx(1.0)
    assert o["real_per_attack_cards"] == pytest.approx(0.12 / 4 / MU, abs=1e-3)


def test_summarise_splits_by_cost_and_by_nu():
    rs = [_row(cost_band="c1_2", nu=0.06), _row(cost_band="c7_10", nu=0.2)]
    out = PB.summarise(rs)
    assert set(out["by_cost"]) == {"c1_2", "c7_10"}
    assert set(out["by_nu"]) == {"nu_lt_1_5", "nu_ge_2_5"}
    assert out["all"]["n"] == 2
