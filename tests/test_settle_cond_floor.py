"""**K-1／K-2／K-5**（2026-09-26）: 交点の橋の行の勝率 `p` を整数ターンの競争で読む式と、その幅 `σ_rel` の測り方。

* **整数ターンの競争**（`whole`・既定・定数）: 私が届く段 `k_me = max(1, ⌈X_me⌉)`・相手が届く段 `k_opp`（同じ段なら手番の私が先）・
  `X ~ N(τ, (σ_rel·max(1, τ))²)`。`p = P(k_me ≤ k_opp)`。式は Rust（`leaves_to::prob_of_d`・`theory_rs.prob_of_d` 経由）。
* **1 行の器**（`mover=False`）と尺度 `hyp` 以外の行は `Φ((D + 手番の半ターン)/(σ_rel·s))`。
* **幅の測り方**（Python に残る集計・`crossing_bridge.sigma_rel_mle`／`sigma_rel_whole_mle`）: 正規の最尤。

段 7（2026-10-07）: Python の式の写しを消し、規則・定義から手で導ける値（模擬・恒等式）との突き合わせだけを残した。

**基盤健全性**（`cpu_infra`）——交点の橋の較正の器（`tests/scripts/`）の算術だけを見る。
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
import theory_rs as TR  # noqa: E402


@pytest.fixture(autouse=True)
def _restore_clock():
    old = dict(TR.CLOCK)
    yield
    TR.CLOCK.clear()
    TR.CLOCK.update(old)


def _phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _p(d, tm, to, scale_mode="hyp", mover=True):
    return TR.prob_of_d([(d, tm, to)], scale_mode=scale_mode, mover=mover)[0]


_GRID = [(d, tm, to) for tm in (0.0, 0.4, 1.0, 1.7, 3.2, 9.0) for to in (0.0, 0.6, 1.3, 2.5, 7.0)
         for d in (to - tm,)]


def test_non_hyp_scales_keep_the_t154_formula_bit_for_bit():
    """尺度が `hyp` 以外（`win_calib` の `scale_auc`）の行は `Φ((D + 1/2)/(σ_rel·s))` そのもの（T151／T154 の式）。"""
    TR.set_w_err_mode("rel"); TR.set_sigma_rel(0.3078)
    for d, tm, to in _GRID:
        s = tm + to                                                                # `sum` の尺度
        want = 0.5 if s == 0 else 0.5 * (1.0 + math.erf((float(d) + 0.5) / ((0.3078 * s) * math.sqrt(2.0))))
        assert _p(d, tm, to, scale_mode="sum") == want                             # ビット一致


def test_the_one_row_instruments_read_phi_without_the_half_turn():
    """1 行の器（`mover=False`）には整数ターンの式は掛からない＝`Φ(D/(σ_rel·s))`・`s = √(τ_me² + τ_opp²)`。"""
    TR.set_w_err_mode("rel"); TR.set_sigma_rel(0.3)
    for d, tm, to in _GRID:
        s = math.sqrt(tm * tm + to * to)
        want = 0.5 if s == 0 else 0.5 * (1.0 + math.erf(float(d) / ((0.3 * s) * math.sqrt(2.0))))
        assert _p(d, tm, to, mover=False) == want


def test_whole_width_is_the_number_of_turns_still_to_play():
    """時計 1 本の幅の尺度は `max(1, τ)`＝今のターンは τ がいくら小さくても丸ごと 1 ターン打たれる。"""
    assert TR.whole_clock_scale([0.0, 1e-9, 0.4, 1.0, 3.2]) == [1.0, 1.0, 1.0, 1.0, 3.2]


@pytest.mark.parametrize("tm,to", [(0.0, 0.0), (0.0, 1.3), (0.3, 2.5), (2.2, 0.0), (0.9, 0.2), (1.5, 2.0),
                                   (3.2, 2.9)])
def test_the_turn_start_row_is_the_integer_turn_race_of_the_rule(tm, to):
    """交点の橋の行（`mover=True`・`rel`・`hyp`）の `p` は規則の競争の模擬と一致する（幅 `σ_rel·max(1, τ)`）。"""
    TR.set_w_err_mode("rel"); TR.set_sigma_rel(0.4)
    sm, so = 0.4 * max(1.0, tm), 0.4 * max(1.0, to)
    mc, n = _mc(tm, to, sm, so, 1)
    tol = 5.0 * math.sqrt(max(mc * (1.0 - mc), 1e-4) / n)                         # 5 標準誤差（模擬の揺れ）
    assert _p(to - tm, tm, to) == pytest.approx(mc, abs=tol)


def test_the_race_tends_to_the_floor_formula_when_the_clocks_are_wide():
    """幅が 1 ターンより十分広いと、整数ターンの式は `Φ((D + 1/2)/√(σ² + 1/12))` に近づく（K-2 の床の検算）。"""
    TR.set_w_err_mode("rel"); TR.set_sigma_rel(0.3)
    for tm, to in [(12.0, 13.0), (15.0, 14.0), (20.0, 20.0)]:
        sm, so = 0.3 * tm, 0.3 * to
        want = _phi((to - tm + 0.5) / math.sqrt(sm * sm + so * so + 1.0 / 12.0))
        assert _p(to - tm, tm, to) == pytest.approx(want, abs=2e-3)


def test_the_race_never_says_exactly_0_or_1_at_tiny_clocks():
    """幅は 1 ターンを下回らない＝「いま届く」行でも p はちょうど 0／1 にならない・同じ段なら手番の私が先。"""
    TR.set_w_err_mode("rel"); TR.set_sigma_rel(0.4)
    tiny = (0.0, 1e-9, 0.01, 0.3, 0.9)
    for tm in tiny:
        for to in tiny + (1.5, 3.0):
            p = _p(to - tm, tm, to)
            q = _p(tm - to, to, tm)
            assert 0.0 < p < 1.0 and 0.0 < q < 1.0
            if tm == to:
                assert p > 0.5


def _mc(tm, to, sm, so, k0, n=400000, seed=0):
    rng = np.random.default_rng(seed)
    x = tm + sm * rng.standard_normal(n); y = to + so * rng.standard_normal(n)
    km = np.maximum(1, np.ceil(x)); ko = np.maximum(1, np.ceil(y))
    m = km >= k0
    return float((km[m] <= ko[m]).mean()), int(m.sum())


def test_sigma_rel_mle_without_floor_is_the_old_std_and_recovers_sigma_with_the_floor():
    rng = np.random.default_rng(7)
    s = rng.uniform(0.3, 8.0, 20000)
    r = 0.25 * s * rng.standard_normal(len(s)) - 0.1 * s
    assert CB.sigma_rel_mle(r, s, 0.0) == float((r / s).std())                   # 従来の手順そのもの
    r2 = r - rng.uniform(0.0, 1.0, len(s))                                       # 整数ターンの丸め（一様・分散 1/12）
    got = CB.sigma_rel_mle(r2, s, TR.TURN_ROUND_VAR, TR.TURN_ROUND_MEAN)
    assert got == pytest.approx(0.25, abs=0.01)
    assert CB.sigma_rel_mle(r2, s, 0.0) > got                                     # 床を抜かないと丸めが比例の幅に混ざる
    # 尺度 0 の行は外す・全部 0 なら None・床だけで説明できれば境界解 0
    assert CB.sigma_rel_mle(np.r_[r2, 5.0], np.r_[s, 0.0], TR.TURN_ROUND_VAR, TR.TURN_ROUND_MEAN) == pytest.approx(got)
    assert CB.sigma_rel_mle([1.0], [0.0], 1.0 / 12.0) is None
    tiny = rng.uniform(-0.01, 0.01, 50)
    assert CB.sigma_rel_mle(tiny, np.full(50, 2.0), 1.0 / 12.0) == 0.0


# ---- K-1: 整数ターンの競争・決着前の条件 ------------------------------------------------------------


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
