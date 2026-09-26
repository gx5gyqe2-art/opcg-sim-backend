"""**K-1／K-2**（2026-09-26）: 決着前の条件と揃えた時計（`SETTLE_COND_MODE`）と整数ターンの床（`SIGMA_FLOOR_MODE`）。

* **K-1**: 決着前フィルタが残す行では持ち主の今のターンは決着の段ではない（規則上の詰みが無い）。交点の橋の `p` を
  「私が届く段 ≤ 相手が届く段」（整数の自席ターン・同じ段なら手番の私が先）の確率として、`k_me ≥ 2` の条件の下で読む
  （`theory_order.whole_turn_race_prob`）。`whole` は条件を掛けない対照。
* **K-2**: 局は整数ターンで終わる＝差に掛かる一様な核（分散 1/12）を幅に二乗和で足す。`σ_rel` は同じ正規の最尤で
  床を入れて測り直す（`crossing_bridge.sigma_rel_mle`・床 0 なら従来の `std(r/s)` そのもの）。
* 報告のみ: 整数ターンの残差 `max(1, ⌈τ⌉) − t_act`（`τ − t_act` は完璧な予測でも平均 ≈ −0.5）。

既定はどちらも `off`＝従来と 1 ビットも変わらない（ここで固める）。

**基盤健全性**（`cpu_infra`）——交点の橋の較正の器（`tests/scripts/`）の算術と配線だけを見る。
"""
import math
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
import pre_settle_asymmetry as PSA  # noqa: E402
import theory_order as T  # noqa: E402
import win_calib as WC  # noqa: E402


_SHIPPED_SETTLE_COND = T.SETTLE_COND_MODE


def test_whole_is_the_shipped_default():
    """K-4（ユーザ決定 2026-09-26「aでお願いします」）: 整数ターンの式と合わせた幅が既定。"""
    assert _SHIPPED_SETTLE_COND == "whole"


@pytest.fixture(autouse=True)
def _restore_modes():
    old = (T.SETTLE_COND_MODE, T.SIGMA_FLOOR_MODE, T.W_ERR_MODE, T.SIGMA_REL, T.W_MOVER_MODE, CB.PRE_SETTLE_MODE)
    T.set_settle_cond_mode("off")      # 以下の代数は旧の Φ の形（off）を基準に書いてある
    yield
    T.set_settle_cond_mode(old[0]); T.set_sigma_floor_mode(old[1]); T.set_w_err_mode(old[2])
    T.set_sigma_rel(old[3]); T.set_w_mover_mode(old[4]); CB.set_pre_settle_mode(old[5])


def _phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


_GRID = [(d, tm, to) for tm in (0.0, 0.4, 1.0, 1.7, 3.2, 9.0) for to in (0.0, 0.6, 1.3, 2.5, 7.0)
         for d in (to - tm,)]


# ---- 既定・切替 ---------------------------------------------------------------------------------

def test_both_switches_default_off_and_reject_unknown_names():
    assert T.SETTLE_COND_MODE == "off" and T.SIGMA_FLOOR_MODE == "off"
    assert T.SETTLE_COND_MODES == ("off", "on", "whole") and T.SIGMA_FLOOR_MODES == ("off", "on")
    with pytest.raises(ValueError):
        T.set_settle_cond_mode("なにか")
    with pytest.raises(ValueError):
        T.set_sigma_floor_mode("なにか")
    assert T.SETTLE_COND_MODE == "off" and T.SIGMA_FLOOR_MODE == "off"
    assert T.TURN_ROUND_VAR == pytest.approx(1.0 / 12.0)


def test_off_is_bit_identical_to_the_t154_formula_and_ignores_first_open():
    """`off`／`off` は `Φ((D + 1/2)/(σ_rel·s))` そのもの（T151／T154 の式）で、`first_open` を見ない。"""
    T.set_w_err_mode("rel"); T.set_sigma_rel(0.3078); T.set_w_mover_mode("half")
    for d, tm, to in _GRID:
        s = math.sqrt(tm * tm + to * to)
        want = 0.5 if s == 0 else 0.5 * (1.0 + math.erf((float(d) + 0.5) / ((0.3078 * s) * math.sqrt(2.0))))
        got = T.prob_of_d(d, t_me=tm, t_opp=to, mover=True)
        assert got == want                                                         # ビット一致
        assert T.prob_of_d(d, t_me=tm, t_opp=to, mover=True, first_open=False) == want


def test_switches_do_not_touch_the_one_row_instruments():
    """1 行の器（`mover=False`）には K-1／K-2 とも掛からない（手番の半ターンと同じ範囲）。"""
    T.set_w_err_mode("rel"); T.set_sigma_rel(0.3)
    base = [T.prob_of_d(d, t_me=tm, t_opp=to) for d, tm, to in _GRID]
    T.set_settle_cond_mode("on"); T.set_sigma_floor_mode("on")
    assert [T.prob_of_d(d, t_me=tm, t_opp=to, first_open=False) for d, tm, to in _GRID] == base


# ---- K-2: 整数ターンの床 -----------------------------------------------------------------------

def test_floor_adds_one_twelfth_in_quadrature_and_keeps_the_pair_antisymmetric():
    T.set_w_err_mode("rel"); T.set_sigma_rel(0.3); T.set_sigma_floor_mode("on")
    for d, tm, to in _GRID:
        s = math.sqrt(tm * tm + to * to)
        want = _phi((d + 0.5) / math.sqrt((0.3 * s) ** 2 + 1.0 / 12.0))
        assert T.prob_of_d(d, t_me=tm, t_opp=to, mover=True) == pytest.approx(want, abs=1e-15)
    # 尺度 0 の行も定義できる（従来は 0.5 に落としていた）
    assert T.prob_of_d(0.0, t_me=0.0, t_opp=0.0, mover=True) == pytest.approx(_phi(0.5 * math.sqrt(12.0)))
    for d in (-3.0, -1.0, -0.5, 0.0, 0.7, 2.0):                                  # 鏡の対は反対称のまま
        assert (T.prob_of_d(d, t_me=1.0, t_opp=1.0, mover=True)
                + T.prob_of_d(-1.0 - d, t_me=1.0, t_opp=1.0, mover=True)) == pytest.approx(1.0)


def test_floor_widens_most_where_the_clocks_are_short():
    T.set_w_err_mode("rel"); T.set_sigma_rel(0.3)
    near_off = T.prob_of_d(0.0, t_me=1.0, t_opp=1.0, mover=True)
    far_off = T.prob_of_d(0.0, t_me=10.0, t_opp=10.0, mover=True)
    T.set_sigma_floor_mode("on")
    near_on = T.prob_of_d(0.0, t_me=1.0, t_opp=1.0, mover=True)
    far_on = T.prob_of_d(0.0, t_me=10.0, t_opp=10.0, mover=True)
    assert near_on < near_off and far_on < far_off
    assert (near_off - near_on) > 10 * (far_off - far_on)


def test_sigma_rel_mle_without_floor_is_the_old_std_and_recovers_sigma_with_the_floor():
    rng = np.random.default_rng(7)
    s = rng.uniform(0.3, 8.0, 20000)
    r = 0.25 * s * rng.standard_normal(len(s)) - 0.1 * s
    assert CB.sigma_rel_mle(r, s, 0.0) == float((r / s).std())                   # 従来の手順そのもの
    r2 = r - rng.uniform(0.0, 1.0, len(s))                                       # 整数ターンの丸め（一様・分散 1/12）
    got = CB.sigma_rel_mle(r2, s, T.TURN_ROUND_VAR, T.TURN_ROUND_MEAN)
    assert got == pytest.approx(0.25, abs=0.01)
    assert CB.sigma_rel_mle(r2, s, 0.0) > got                                     # 床を抜かないと丸めが比例の幅に混ざる
    # 尺度 0 の行は外す・全部 0 なら None・床だけで説明できれば境界解 0
    assert CB.sigma_rel_mle(np.r_[r2, 5.0], np.r_[s, 0.0], T.TURN_ROUND_VAR, T.TURN_ROUND_MEAN) == pytest.approx(got)
    assert CB.sigma_rel_mle([1.0], [0.0], 1.0 / 12.0) is None
    tiny = rng.uniform(-0.01, 0.01, 50)
    assert CB.sigma_rel_mle(tiny, np.full(50, 2.0), 1.0 / 12.0) == 0.0


def test_sigma_rel_for_reads_the_floor_table_only_when_the_floor_is_on(monkeypatch, tmp_path):
    prof = {"sigma_rel": {"blockers": {"theory": {"real": 0.31, "syn": 0.33}}},
            "sigma_rel_floor": {"blockers": {"theory": {"real": 0.21, "syn": 0.23}}}}
    monkeypatch.setattr(CB, "load_harm_profiles", lambda path=None: prof)
    d_real = tmp_path / "w_real"; d_real.mkdir()
    (d_real / "meta_n_record.json").write_text('{"decks": "user"}', encoding="utf-8")
    assert CB.sigma_rel_for([str(d_real)]) == pytest.approx(0.33)                 # 別のセット（cross）
    T.set_sigma_floor_mode("on")
    assert CB.sigma_rel_for([str(d_real)]) == pytest.approx(0.23)
    assert CB.sigma_rel_for([str(d_real)], "real") == pytest.approx(0.21)


def test_the_fixture_carries_the_floor_table_next_to_the_old_one():
    prof = CB.load_harm_profiles()
    old = prof["sigma_rel"]["blockers"]["theory"]
    new = prof["sigma_rel_floor"]["blockers"]["theory"]
    assert set(new) == {"real", "syn"} and all(0.0 < new[k] < old[k] for k in ("real", "syn"))


# ---- K-1: 整数ターンの競争・決着前の条件 ------------------------------------------------------------

def _mc(tm, to, sm, so, k0, n=400000, seed=0):
    rng = np.random.default_rng(seed)
    x = tm + sm * rng.standard_normal(n); y = to + so * rng.standard_normal(n)
    km = np.maximum(1, np.ceil(x)); ko = np.maximum(1, np.ceil(y))
    m = km >= k0
    return float((km[m] <= ko[m]).mean()), int(m.sum())


@pytest.mark.parametrize("tm,to,sm,so", [(1.5, 2.0, 0.4, 0.5), (0.6, 1.4, 0.2, 0.4), (3.2, 2.9, 1.0, 0.9),
                                         (2.2, 1.1, 0.7, 0.3)])
def test_whole_turn_race_matches_a_simulation_of_the_rule(tm, to, sm, so):
    for k0 in (1, 2):
        mc, n = _mc(tm, to, sm, so, k0)
        tol = 5.0 * math.sqrt(max(mc * (1.0 - mc), 1e-4) / n)                    # 5 標準誤差（シミュレーションの揺れ）
        assert T.whole_turn_race_prob(tm, to, sm, so, k0) == pytest.approx(mc, abs=tol)


def test_removing_the_first_step_is_bayes_with_the_certain_win():
    """規則: 私が第 1 段で届けば相手に手番は回らない＝必ず勝つ。だから `p(k≥1) = q + (1 − q)·p(k≥2)`
    （`q = P(X_me ≤ 1)`）が恒等式で成り立つ。"""
    for tm, to, sm, so in [(1.5, 2.0, 0.4, 0.5), (0.6, 1.4, 0.2, 0.4), (3.2, 2.9, 1.0, 0.9), (0.9, 0.3, 0.3, 0.1)]:
        q = 1.0 - T._upper(1.0, tm, sm)
        p1 = T.whole_turn_race_prob(tm, to, sm, so, 1)
        p2 = T.whole_turn_race_prob(tm, to, sm, so, 2)
        assert p1 == pytest.approx(q + (1.0 - q) * p2, abs=1e-12)
        assert p2 <= p1 + 1e-12                                                    # 条件は私の勝ちを減らすだけ


def test_pinned_clock_loses_the_mover_half_turn():
    """`τ_me ≤ 1` で幅 0 なら条件の下で私は第 2 段の頭に届く＝相手の第 1 段が先。p → P(X_opp > 1)（半ターンは消える）。"""
    assert T.whole_turn_race_prob(0.5, 0.5, 0.0, 0.3, 2) == pytest.approx(T._upper(1.0, 0.5, 0.3))
    assert T.whole_turn_race_prob(0.5, 1.5, 0.0, 0.0, 2) == 1.0                 # 相手は第 2 段・同じ段なら私が先
    assert T.whole_turn_race_prob(0.5, 0.9, 0.0, 0.0, 2) == 0.0                 # 相手は第 1 段で届く
    assert T.whole_turn_race_prob(0.0, 0.0, 0.0, 0.0, 1) == 1.0                 # 両方いま届く＝手番の私


def test_whole_turn_race_tends_to_the_floor_formula_when_the_clocks_are_wide():
    """幅が 1 ターンより十分広いと、割合は一様に近づき、整数ターンの式は `Φ((D + 1/2)/√(σ² + 1/12))` に近づく
    （K-2 の床が同じ規則の近似であることの検算）。"""
    for tm, to in [(12.0, 13.0), (15.0, 14.0), (20.0, 20.0)]:
        sm, so = 0.3 * tm, 0.3 * to
        want = _phi((to - tm + 0.5) / math.sqrt(sm * sm + so * so + 1.0 / 12.0))
        assert T.whole_turn_race_prob(tm, to, sm, so, 1) == pytest.approx(want, abs=2e-3)


def test_settle_cond_routes_prob_of_d_through_the_race_only_on_turn_start_rows():
    T.set_w_err_mode("rel"); T.set_sigma_rel(0.3); T.set_w_mover_mode("half")
    tm, to = 0.8, 1.6
    T.set_settle_cond_mode("on")
    p_closed = T.prob_of_d(to - tm, t_me=tm, t_opp=to, mover=True, first_open=False)
    p_open = T.prob_of_d(to - tm, t_me=tm, t_opp=to, mover=True, first_open=True)
    assert p_closed == pytest.approx(T.whole_turn_race_prob(tm, to, 0.3 * tm, 0.3 * to, 2))
    assert p_open == pytest.approx(T.whole_turn_race_prob(tm, to, 0.3 * tm, 0.3 * to, 1))
    assert p_closed < p_open
    T.set_settle_cond_mode("whole")                                              # 条件を掛けない・幅はまだ打つターン数（K-5）
    assert T.prob_of_d(to - tm, t_me=tm, t_opp=to, mover=True, first_open=False) == pytest.approx(
        T.whole_turn_race_prob(tm, to, 0.3 * 1.0, 0.3 * to, 1))
    tm = 1.4                                                                      # τ ≥ 1 なら幅は `on` と同じ σ·τ
    assert T.prob_of_d(to - tm, t_me=tm, t_opp=to, mover=True, first_open=False) == pytest.approx(
        T.whole_turn_race_prob(tm, to, 0.3 * tm, 0.3 * to, 1))
    T.set_settle_cond_mode("on"); T.set_w_mover_mode("off")                       # 半ターンを使わない構成には掛けない
    assert T.prob_of_d(to - tm, t_me=tm, t_opp=to, mover=True, first_open=False) == pytest.approx(
        _phi((to - tm) / (0.3 * math.hypot(tm, to))))


# ---- K-5: `whole` の幅（まだ打つターン数）と、それに揃えた測り方 -------------------------------------------

def test_whole_width_is_the_number_of_turns_still_to_play():
    """時計 1 本の幅の尺度は `max(1, τ)`＝今のターンは τ がいくら小さくても丸ごと 1 ターン打たれる。"""
    assert [T.whole_clock_scale(x) for x in (0.0, 1e-9, 0.4, 1.0, 3.2)] == [1.0, 1.0, 1.0, 1.0, 3.2]
    T.set_w_err_mode("rel"); T.set_sigma_rel(0.4); T.set_w_mover_mode("half"); T.set_settle_cond_mode("whole")
    for d, tm, to in _GRID:
        want = T.whole_turn_race_prob(tm, to, 0.4 * max(1.0, tm), 0.4 * max(1.0, to), 1)
        assert T.prob_of_d(d, t_me=tm, t_opp=to, mover=True) == want
        assert T.prob_of_d(d, t_me=tm, t_opp=to, mover=True, first_open=False) == want   # 条件は掛けない


@pytest.mark.parametrize("tm,to", [(0.0, 0.0), (0.0, 1.3), (0.3, 2.5), (2.2, 0.0), (0.9, 0.2)])
def test_whole_with_the_turn_width_matches_a_simulation_of_the_rule(tm, to):
    sm, so = 0.4 * T.whole_clock_scale(tm), 0.4 * T.whole_clock_scale(to)
    mc, n = _mc(tm, to, sm, so, 1)
    tol = 5.0 * math.sqrt(max(mc * (1.0 - mc), 1e-4) / n)
    assert T.whole_turn_race_prob(tm, to, sm, so, 1) == pytest.approx(mc, abs=tol)


def test_whole_never_says_exactly_0_or_1_at_tiny_clocks():
    """従来の幅 `σ·τ` は `τ = 0` で幅 0＝「いま届く」を確率 1 と言っていた（p がちょうど 1／相手なら 0）。"""
    assert T.whole_turn_race_prob(0.0, 2.0, 0.0, 0.4 * 2.0, 1) == 1.0                # 弱点（従来の幅）
    T.set_w_err_mode("rel"); T.set_sigma_rel(0.4); T.set_w_mover_mode("half"); T.set_settle_cond_mode("whole")
    tiny = (0.0, 1e-9, 0.01, 0.3, 0.9)
    for tm in tiny:
        for to in tiny + (1.5, 3.0):
            p = T.prob_of_d(to - tm, t_me=tm, t_opp=to, mover=True)
            q = T.prob_of_d(tm - to, t_me=to, t_opp=tm, mover=True)
            assert 0.0 < p < 1.0 and 0.0 < q < 1.0
            if tm == to:
                assert p > 0.5                                                   # 同じ段なら手番の私が先


def test_log_interval_is_exact_in_the_body_and_finite_in_the_far_tails():
    for lo, hi in [(-0.5, 0.7), (2.0, 3.0), (-3.0, -2.0), (-math.inf, 1.0), (-math.inf, -2.5)]:
        direct = _phi(hi) - (0.0 if lo == -math.inf else _phi(lo))
        assert CB._log_interval(lo, hi) == pytest.approx(math.log(direct), rel=1e-9)
    far = CB._log_interval(40.0, 41.0)                                           # 直接の差は 0（桁落ち）
    assert math.isfinite(far) and far == pytest.approx(-0.5 * 1600 - math.log(40.0) - 0.5 * math.log(2 * math.pi), abs=1e-3)
    assert CB._log_interval(-41.0, -40.0) == pytest.approx(far)
    assert CB._log_interval(-math.inf, -40.0) == pytest.approx(far, abs=1e-3)


def _whole_sim(sigma, n, seed):
    rng = np.random.default_rng(seed)
    tau = rng.uniform(0.0, 8.0, n)
    tau[: n // 10] = 0.0                                                          # 「いま届く」と言う行も混ぜる
    x = tau + sigma * np.maximum(1.0, tau) * rng.standard_normal(n)
    return tau, np.maximum(1, np.ceil(x)).astype(int)


def test_sigma_rel_whole_mle_recovers_a_known_width_from_simulated_records():
    for sigma, seed in ((0.25, 1), (0.4, 2), (0.6, 3)):
        tau, act = _whole_sim(sigma, 20000, seed)
        assert CB.sigma_rel_whole_mle(tau, act) == pytest.approx(sigma, rel=0.03)
    # 従来の手順の 1 本版（残差の sd ÷ 尺度）は丸めを混ぜる＝同じ記録で別の数になる
    tau, act = _whole_sim(0.4, 20000, 2)
    naive = float(((tau - act) / np.maximum(1.0, tau)).std())
    assert abs(naive - 0.4) > 0.02


def _race_sim(sigma, n, seed):
    """2 本の時計の競争（`whole_turn_race_prob` と同じ規則）: 行の持ち主が手番・`k_me ≤ k_opp` なら持ち主の勝ち。
    記録に残るのは勝った席が届いた段と、両席の予測の時計だけ（負けた席がいつ届いたかは残らない＝選び方がある）。"""
    rng = np.random.default_rng(seed)
    tm = rng.uniform(0.0, 8.0, n); to = rng.uniform(0.0, 8.0, n)
    tm[: n // 10] = 0.0
    xm = tm + sigma * np.maximum(1.0, tm) * rng.standard_normal(n)
    xo = to + sigma * np.maximum(1.0, to) * rng.standard_normal(n)
    km = np.maximum(1, np.ceil(xm)).astype(int); ko = np.maximum(1, np.ceil(xo)).astype(int)
    won = km <= ko
    return np.where(won, tm, to), np.where(won, km, ko), np.where(won, to, tm), won


def test_sigma_rel_whole_mle_with_the_censored_loser_recovers_sigma_from_a_race():
    """勝った席だけの尤度は「早く届いた方が勝つ」選び方を幅に混ぜる（模擬で約 +5%）。負けた席の打ち切り
    （手番の私が段 a で勝った→相手は `X > a − 1`・相手が段 a で勝った→私は `X > a`）を入れると取り戻せる。"""
    for sigma, seed in ((0.25, 1), (0.4, 2), (0.6, 3)):
        tw, aw, tl, wm = _race_sim(sigma, 20000, seed)
        got = CB.sigma_rel_whole_mle(tw, aw, tl, wm)
        assert got == pytest.approx(sigma, rel=0.02)
        winner_only = CB.sigma_rel_whole_mle(tw, aw)
        assert abs(winner_only - sigma) > abs(got - sigma)


def test_the_censored_loser_uses_the_race_step_convention():
    """1 行で因子を書き下して照合: 私（手番）が段 2 で勝った→相手 `P(X > 1)`・相手が段 2 で勝った→私 `P(X > 2)`・
    段 1 で勝った手番の私→相手の因子は 1（打ち切りなし）。"""
    def loglik(sigma, rows):
        out = 0.0
        for tw, a, tl, first in rows:
            cw, cl = sigma * max(1.0, tw), sigma * max(1.0, tl)
            pw = _phi((a - tw) / cw) - (_phi((a - 1 - tw) / cw) if a >= 2 else 0.0)
            thr = a - 1 if first else a
            pl = (1.0 - _phi((thr - tl) / cl)) if thr >= 1 else 1.0
            if pw <= 0.0 or pl <= 0.0:
                return -math.inf                                                  # 素朴な式が桁落ちする狭すぎる幅
            out += math.log(pw) + math.log(pl)
        return out
    rows = [(1.2, 2, 3.0, True), (2.5, 2, 0.4, False), (0.3, 1, 0.2, True), (3.0, 5, 6.0, True), (0.0, 2, 1.5, False)]
    got = CB.sigma_rel_whole_mle(*zip(*[(r[0], r[1], r[2], r[3]) for r in rows]))
    grid = [x / 1000.0 for x in range(50, 3000)]
    best = max(grid, key=lambda s: loglik(s, rows))
    assert got == pytest.approx(best, abs=1e-3)


def test_sigma_rel_whole_mle_edges():
    assert CB.sigma_rel_whole_mle([], []) is None
    tau = np.array([0.0, 0.5, 1.3, 2.7, 4.0]); act = np.array([1, 1, 2, 3, 4])   # 全行が予測どおりの段
    assert CB.sigma_rel_whole_mle(tau, act) == 0.0                                # 境界解（狭いほど尤度が上がる）
    one_off = CB.sigma_rel_whole_mle(np.r_[tau, 1.3], np.r_[act, 4])              # 1 行外れると幅が要る
    assert 0.0 < one_off < math.inf


def test_sigma_rel_for_reads_the_whole_table_when_whole_is_on(monkeypatch, tmp_path):
    prof = {"sigma_rel": {"blockers": {"theory": {"real": 0.31, "syn": 0.33}}},
            "sigma_rel_floor": {"blockers": {"theory": {"real": 0.21, "syn": 0.23}}},
            "sigma_rel_whole": {"blockers": {"theory": {"real": 0.41, "syn": 0.43}}}}
    monkeypatch.setattr(CB, "load_harm_profiles", lambda path=None: prof)
    d_real = tmp_path / "w_real"; d_real.mkdir()
    (d_real / "meta_n_record.json").write_text('{"decks": "user"}', encoding="utf-8")
    T.set_settle_cond_mode("whole")
    assert CB.sigma_rel_for([str(d_real)]) == pytest.approx(0.43)                 # 別のセット（cross）
    T.set_sigma_floor_mode("on")                                                  # `whole` は床を足さない＝表も `whole`
    assert CB.sigma_rel_for([str(d_real)]) == pytest.approx(0.43)
    T.set_settle_cond_mode("on")                                                  # `on` は従来の表のまま
    assert CB.sigma_rel_for([str(d_real)]) == pytest.approx(0.23)
    T.set_settle_cond_mode("off"); T.set_sigma_floor_mode("off")
    assert CB.sigma_rel_for([str(d_real)]) == pytest.approx(0.33)


def test_the_fixture_carries_the_whole_table():
    prof = CB.load_harm_profiles()
    tbl = prof["sigma_rel_whole"]["blockers"]
    for slope in ("theory", "curve"):
        assert set(tbl[slope]) == {"real", "syn"} and all(0.0 < v < 1.0 for v in tbl[slope].values())
    old = prof["sigma_rel"]["blockers"]["theory"]
    assert all(tbl["theory"][k] > old[k] for k in ("real", "syn"))              # 従来の手順は幅を小さく測っていた


def test_off_stays_bit_identical_after_whole_was_used():
    T.set_w_err_mode("rel"); T.set_sigma_rel(0.3078); T.set_w_mover_mode("half")
    base = [T.prob_of_d(d, t_me=tm, t_opp=to, mover=True) for d, tm, to in _GRID]
    T.set_settle_cond_mode("whole")
    [T.prob_of_d(d, t_me=tm, t_opp=to, mover=True) for d, tm, to in _GRID]
    T.set_settle_cond_mode("off")
    assert [T.prob_of_d(d, t_me=tm, t_opp=to, mover=True) for d, tm, to in _GRID] == base


# ---- 配線（win_calib／pre_settle_asymmetry／crossing_bridge） ----------------------------------------

def test_first_open_of_reads_the_settled_flag_and_defaults_to_open():
    rows = [{"settled_me": False}, {"settled_me": True}, {}]
    assert WC.first_open_of(rows) == [False, True, True]


def test_probs_of_passes_first_open_per_row():
    T.set_w_err_mode("rel"); T.set_settle_cond_mode("on")
    rs = [(0.8, 0.8, 1.6, 1.0)] * 2
    p = WC.probs_of(rs, 0.3, first_open=[False, True])
    assert p[0] < p[1]
    T.set_settle_cond_mode("off")
    q = WC.probs_of(rs, 0.3, first_open=[False, True])
    assert q[0] == q[1] == WC.probs_of(rs, 0.3)[0]


def test_win_calib_cli_flags_reach_the_modules(monkeypatch):
    monkeypatch.setattr(WC, "collect_calib", lambda *a, **k: {"x": 1})
    monkeypatch.setattr(CB, "sigma_rel_for", lambda dirs: 1.0)
    WC.main(["--in", "x", "--settle-cond", "on", "--sigma-floor", "on"])
    assert T.SETTLE_COND_MODE == "on" and T.SIGMA_FLOOR_MODE == "on"
    with pytest.raises(SystemExit):
        WC.main(["--in", "x", "--settle-cond", "なにか"])
    with pytest.raises(SystemExit):
        WC.main(["--in", "x", "--sigma-floor", "なにか"])


def test_pre_settle_asymmetry_cli_flags_are_scoped_to_the_call(monkeypatch):
    seen = {}

    def fake_collect(dirs, limit, *a, **k):
        seen["modes"] = (T.SETTLE_COND_MODE, T.SIGMA_FLOOR_MODE)
        return [], [], {"games": 0}, [], []
    monkeypatch.setattr(CB, "collect", fake_collect)
    monkeypatch.setattr(CB, "sigma_rel_for", lambda dirs: 0.3)
    monkeypatch.setattr(PSA, "add_volatility", lambda rows, dirs: rows)
    PSA.collect(["x"], settle_cond="on", sigma_floor="on")
    assert seen["modes"] == ("on", "on")
    assert T.SETTLE_COND_MODE == "off" and T.SIGMA_FLOOR_MODE == "off"             # 呼び出しの外へ漏れない


def test_collect_reads_the_settled_map_when_settle_cond_is_on_even_without_pre_settle(monkeypatch):
    import lethal_rule as LR
    calls = []
    monkeypatch.setattr(LR, "settled_map", lambda dirs, limit_games: calls.append(1) or {})
    monkeypatch.setattr(CB.PL, "iter_games", lambda *a, **k: iter([]))
    CB.collect(["x"])
    assert calls == []                                                            # 既定（off）は読まない
    T.set_settle_cond_mode("on")
    CB.collect(["x"])
    assert calls == [1]


# ---- 報告のみ: 整数ターンの残差 --------------------------------------------------------------------

def _row(won, tau_me, tau_opp, t_me, t_opp):
    return {"who": 0, "won": won, "t_me_act": t_me, "t_opp_act": t_opp, "theta_me": 0.5, "theta_opp": 0.5,
            "tau_me_hist": tau_me, "tau_opp_hist": tau_opp, "pred_hist": tau_me <= tau_opp,
            "tau_me_theory": tau_me, "tau_opp_theory": tau_opp, "pred_theory": tau_me <= tau_opp}


def test_summarise_reports_the_integer_residual_next_to_the_old_one_without_changing_it():
    """完璧な予測（`τ ∈ (t_act − 1, t_act]`）は `τ − t_act` が負でも整数の残差は 0。"""
    rows = ([_row(True, 1.4, 3.0, 2, 3)] * 10 + [_row(True, 0.3, 2.0, 1, 2)] * 10
            + [_row(False, 3.0, 2.7, 4, 2)] * 10 + [_row(False, 2.0, 0.0, 3, 1)] * 10)
    o = CB.summarise(rows, [])["by_slope"]["theory"]
    assert o["bias"] == pytest.approx((10 * -0.6 + 10 * -0.7 + 10 * 0.7 + 10 * -1.0) / 40, abs=1e-3)   # 従来のまま
    # 整数: ⌈1.4⌉−2=0・⌈0.3⌉−1=0・⌈2.7⌉−2=+1・max(1,⌈0⌉)−1=0
    assert o["bias_int"] == pytest.approx(0.25) and o["exact_int"] == pytest.approx(0.75)
    assert o["late_int"] == pytest.approx(0.25) and o["early_int"] == 0.0 and o["mae_int"] == pytest.approx(0.25)
    assert o["sigma_rel_floor"] is not None
    # K-5: 勝った席の届いた段 2・1・2・1（τ 1.4・0.3・2.7・0）——⌈2.7⌉ = 3 ≠ 2 の外れがあるので幅は正
    # 負けた席の打ち切り: 負けた席の τ 3.0・2.0・3.0・2.0／勝ったのは手番の席か True・True・False・False
    assert o["sigma_rel_whole"] == pytest.approx(round(CB.sigma_rel_whole_mle(
        [1.4] * 10 + [0.3] * 10 + [2.7] * 10 + [0.0] * 10, [2] * 10 + [1] * 10 + [2] * 10 + [1] * 10,
        [3.0] * 10 + [2.0] * 10 + [3.0] * 10 + [2.0] * 10, [True] * 20 + [False] * 20), 4))
    assert o["sigma_rel_whole"] > 0.0


def test_tau_resid_by_bin_reports_integer_residuals():
    rows = [{"tau_me_theory": 1.4, "tau_opp_theory": 9.0, "t_me_act": 2, "t_opp_act": 4, "won": True},
            {"tau_me_theory": 6.0, "tau_opp_theory": 2.2, "t_me_act": 5, "t_opp_act": 2, "won": False}]
    out = WC.tau_resid_by_bin(rows, [0.8, 0.3], nbin=2)
    assert out["won_resid_me"] == pytest.approx(-0.6)
    assert out["won_resid_me_int"] == pytest.approx(0.0) and out["lost_resid_opp_int"] == pytest.approx(1.0)
