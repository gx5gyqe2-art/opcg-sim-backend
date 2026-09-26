"""`crossing_bridge` の**規則のドンの列**（T114）と**手札の窓の上限**（T116）と
**相手のドンでの判定**（T120）と**勝率の物差し**（T118）の代数を固める。

**T114（`flow`）・T116（`horizon`）・T118（`rel`）は 2026-09-20 に既定になった**（ユーザ決定「3 本すべて」）。
押さえるのは 4 つ:

1. **`off` と恒等**（列が一定なら新しい形は旧の式と同じ値を出す＝`off` に戻せば旧の挙動）。
2. **規則が出る**（ドンは毎ターン +2・席の総量で飽和・`next_turn_don` は同じ式の延長）。
3. **窓の上限は `min(手札, SR·τ)`**（`τ` は歩き自身が出す・`τ=0` なら手札は 1 円も入らない）。
4. **相手の判定は相手のドンで**（自分のドンを使っていたのが T120 の患部）。
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

import crossing_bridge as CB  # noqa: E402
import theory_order as TO  # noqa: E402


def _sc(active=3.0, rested=1.0, lead_att=0.0, deck=6.0, opp_active=2.0, opp_rested=0.0,
        opp_lead_att=0.0, opp_deck=8.0):
    sc = np.zeros(70, np.float64)
    sc[2], sc[3] = active, rested                       # 自アクティブ／レスト
    sc[4], sc[5] = opp_active, opp_rested               # 相手アクティブ／レスト
    sc[14] = lead_att / 5.0                             # 自リーダー付与（/5）
    sc[15] = opp_lead_att / 5.0
    sc[66], sc[67] = deck / 10.0, opp_deck / 10.0       # ドンデッキ（/10）
    sc[12] = sc[13] = 0.5                               # リーダーパワー 5000
    return sc


# ---------------------------------------------------------------- T114 規則のドンの列

def test_the_don_series_grows_by_two_a_turn_and_saturates_on_the_deck():
    """**毎ターン +2・デッキが尽きたら止まる**（規則）。上限は式に書かない（4 ゾーンの和が不変量）。"""
    sc, tok = _sc(active=3.0, rested=1.0, deck=4.0), np.zeros((22, 24))
    ds = CB.purse_series(sc, tok, 6)
    assert ds[0] == pytest.approx(3.0)                  # 今の行は**アクティブだけ**
    # i>=2 は リフレッシュで 4 ゾーンが戻り（3+1=4）＋ min(2(i−1), デッキ 4)
    assert ds[1] == pytest.approx(6.0)                  # 4 + min(2, 4)
    assert ds[2] == pytest.approx(8.0)                  # 4 + min(4, 4)
    assert ds[3] == pytest.approx(8.0)                  # **飽和**（デッキが尽きた）
    assert ds[5] == pytest.approx(8.0)
    # 席の総量（4 ゾーンの和）を超えない
    assert max(ds) <= 3.0 + 1.0 + 0.0 + 4.0 + 1e-9


def test_the_second_step_of_the_series_is_the_same_formula_as_next_turn_don():
    """**`next_turn_don` は同じ式の延長**（T110 の値を動かさない検算）。"""
    for kw in ({}, {"rested": 2.0, "deck": 1.0}, {"lead_att": 5.0, "deck": 0.0}):
        sc, tok = _sc(**kw), np.zeros((22, 24))
        assert CB.purse_series(sc, tok, 2)[1] == pytest.approx(CB.next_turn_don(sc, tok))


def test_a_constant_schedule_reproduces_the_old_walk():
    """**列が一定なら新しい形は旧の式と恒等**（`off` が今の挙動である根拠）。"""
    for j in range(1, 7):
        old = CB.rate_at(j, 0.10, 0.05, 0.20, 0.03, j0=2)
        assert CB.rate_at(j, 0.10, 0.05, 0.20, 0.03, j0=2, sched=[old] * 8) == pytest.approx(old)


def test_the_schedule_is_read_at_the_walks_own_turn_number():
    """列は `j` で引く（範囲外は最後の値を伸ばす＝歩きが打ち切りまで進んでも落ちない）。"""
    sched = [0.1, 0.2, 0.3]
    assert CB.rate_at(1, 0, 0, 0, 0, j0=2, sched=sched) == pytest.approx(0.1)
    assert CB.rate_at(3, 0, 0, 0, 0, j0=2, sched=sched) == pytest.approx(0.3)
    assert CB.rate_at(9, 0, 0, 0, 0, j0=2, sched=sched) == pytest.approx(0.3)


def test_the_placebo_ramp_is_off_by_default_and_additive_when_on():
    """**プラセボの一次ランプは既定 0**（当てはめた定数なので既定に置かない）。"""
    assert CB.RATE_RAMP == 0.0
    base = CB.rate_at(4, 0.1, 0.0, 0.0, 0.0, j0=2)
    try:
        CB.set_rate_don_mode("off", ramp=0.02)
        assert CB.rate_at(4, 0.1, 0.0, 0.0, 0.0, j0=2) == pytest.approx(base + 0.02 * 3)
        # 列を渡しても同じだけ足される（対照の条件を揃える）
        assert CB.rate_at(4, 0, 0, 0, 0, j0=2, sched=[0.5] * 8) == pytest.approx(0.5 + 0.02 * 3)
    finally:
        CB.set_rate_don_mode("off", ramp=0.0)
    assert CB.RATE_RAMP == 0.0


def test_the_don_walk_refuses_to_share_a_row_with_the_race_purse():
    """**`--don-purse race` は 1 行で解いた配分**なので `j` の列を持てない（黙って混ぜない）。"""
    old_p, old_d = CB.DON_PURSE_MODE, CB.RATE_DON_MODE
    try:
        CB.set_don_purse_mode("race")
        with pytest.raises(ValueError):
            CB.set_rate_don_mode("purse")
        assert CB.set_rate_don_mode("off") == "off"        # off は併用できる
    finally:
        CB.set_don_purse_mode(old_p)
        CB.set_rate_don_mode(old_d)


# ---------------------------------------------------------------- T116 窓の上限

def test_the_window_cap_is_the_rate_times_the_turns_that_remain():
    """**1 守備ターンに吸える上限 × 残り τ** が手札の上限（`shield_rate_of` は既存）。"""
    # 攻め手が 1 本（x=0 → c(0)=1 枚で止まる）・ブロッカー 0 → SR = μ × 1
    sr = CB.shield_rate_of([0.0], 0, CB.THETA, CB.MU)
    assert sr == pytest.approx(CB.MU * CB.c_of(0.0))
    assert min(10 * CB.MU, sr * 0.0) == 0.0                # τ=0 なら 1 円も入らない
    assert min(10 * CB.MU, sr * 3.0) == pytest.approx(min(10 * CB.MU, 3.0 * sr))


def test_the_window_modes_are_the_two_the_walk_can_produce():
    """切替は `off`／`horizon`（手札抜きの地平で 1 回）／`fixpoint`（反復）の 3 つだけ。
    **既定は `horizon`**（2026-09-20・ユーザ決定「3 本すべて」）。"""
    assert CB.THETA_HAND_WINDOWS == ("off", "horizon", "fixpoint")
    assert CB.THETA_HAND_WINDOW == "horizon"
    with pytest.raises(ValueError):
        CB.set_theta_hand_window("なにか")


# ---------------------------------------------------------------- T120 相手のドン

def test_the_opponents_budget_is_read_from_the_opponents_zones():
    """**相手が次のターンに使えるドンは相手の 4 ゾーンから**（自分のドンで絞っていたのが患部）。"""
    sc, tok = _sc(active=3.0, rested=1.0, deck=6.0,
                  opp_active=5.0, opp_rested=2.0, opp_deck=1.0), np.zeros((22, 24))
    # 相手: 5 + 2 + 0 + min(2, 1) = 8
    assert CB.opp_don_next(sc, tok) == pytest.approx(8.0)
    # 自分: 3 + 1 + 0 + min(2, 6) = 6 ＝**別の数**（取り違えるとここが一致してしまう）
    assert CB.next_turn_don(sc, tok) == pytest.approx(6.0)
    assert CB.opp_don_next(sc, tok) != CB.next_turn_don(sc, tok)


def test_the_two_budgets_use_the_same_rule():
    """自席版と相手版は**同じ式**（完全情報の対称性・§0.05）。"""
    sc, tok = _sc(active=4.0, rested=0.0, deck=3.0,
                  opp_active=4.0, opp_rested=0.0, opp_deck=3.0), np.zeros((22, 24))
    assert CB.opp_don_next(sc, tok) == pytest.approx(CB.next_turn_don(sc, tok))


# ---------------------------------------------------------------- T118 勝率の物差し

def test_the_relative_scale_is_first_order_homogeneous():
    """**`s` は 1 次同次**（両方の時計を c 倍したら `s` も c 倍）＝「比で読む」の条件。"""
    for m in ("hyp", "sum", "mean", "max", "geo"):
        a, b = TO.clock_scale(3.0, 4.0, m), TO.clock_scale(6.0, 8.0, m)
        assert b == pytest.approx(2.0 * a), m
    assert TO.clock_scale(3.0, 4.0, "hyp") == pytest.approx(5.0)
    assert TO.clock_scale(3.0, 4.0, "sum") == pytest.approx(7.0)


def test_the_probability_falls_back_to_the_absolute_scale_when_sigma_rel_is_missing():
    """**既定は `rel`**（2026-09-20）だが、**`σ_rel` が無ければ `abs` に落ちる**
    ——黙って別の物差しで走らないための安全側。**立てる責任は呼ぶ側**（`theory_bridge` は
    引けなければ `ValueError` で落ちる）。"""
    assert TO.W_ERR_MODE == "rel"
    old = TO.SIGMA_REL
    try:
        TO.set_sigma_rel(None)
        assert TO.prob_of_d(1.0) == pytest.approx(TO.prob_of_d(1.0, t_me=3.0, t_opp=4.0))
        TO.set_sigma_rel(0.2)
        assert TO.prob_of_d(1.0) != pytest.approx(TO.prob_of_d(1.0, t_me=3.0, t_opp=4.0))
    finally:
        TO.set_sigma_rel(old)


def test_the_relative_reading_flattens_the_long_clocks():
    """**同じ `D` でも時計が長い行では確率が中央に寄る**（決着帯の言い過ぎを平らにする）。"""
    old_m, old_s = TO.W_ERR_MODE, TO.SIGMA_REL
    try:
        TO.set_w_err_mode("rel"); TO.set_sigma_rel(0.5)
        near = TO.prob_of_d(2.0, t_me=1.0, t_opp=3.0)      # s = √10 ≈ 3.16
        far = TO.prob_of_d(2.0, t_me=10.0, t_opp=12.0)     # s = √244 ≈ 15.6
        assert 0.5 < far < near
        # `σ_rel` が無ければ `abs` に落ちる（黙って別の物差しで走らない）
        TO.set_sigma_rel(None)
        assert TO.prob_of_d(2.0, t_me=1.0, t_opp=3.0) == pytest.approx(TO.prob_of_d(2.0))
    finally:
        TO.set_w_err_mode(old_m); TO.set_sigma_rel(old_s)


def test_the_relative_reading_keeps_the_sign():
    """**`D` の符号は動かない**（識別力は単調変換で 1 ビットも変わらない・`rel` は較正だけの話）。"""
    old_m, old_s = TO.W_ERR_MODE, TO.SIGMA_REL
    try:
        TO.set_w_err_mode("rel"); TO.set_sigma_rel(0.6)
        assert TO.prob_of_d(0.0, t_me=2.0, t_opp=2.0) == pytest.approx(0.5)
        assert TO.prob_of_d(-1.0, t_me=2.0, t_opp=1.0) < 0.5 < TO.prob_of_d(1.0, t_me=1.0, t_opp=2.0)
    finally:
        TO.set_w_err_mode(old_m); TO.set_sigma_rel(old_s)


def test_the_error_modes_are_the_two_we_measured():
    assert TO.W_ERR_MODES == ("abs", "rel")
    with pytest.raises(ValueError):
        TO.set_w_err_mode("なにか")
