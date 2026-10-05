"""**K-1／K-2**（2026-09-26）: 決着前の条件と揃えた時計（`SETTLE_COND_MODE`）と整数ターンの床（`SIGMA_FLOOR_MODE`）。

* **K-1**: 決着前フィルタが残す行では持ち主の今のターンは決着の段ではない（規則上の詰みが無い）。交点の橋の `p` を
  「私が届く段 ≤ 相手が届く段」（整数の自席ターン・同じ段なら手番の私が先）の確率として、`k_me ≥ 2` の条件の下で読む
  （`theory_order.whole_turn_race_prob`）。`whole` は条件を掛けない対照。
* **K-2**: 局は整数ターンで終わる＝差に掛かる一様な核（分散 1/12）を幅に二乗和で足す。`σ_rel` は同じ正規の最尤で
  床を入れて測り直す（`crossing_bridge.sigma_rel_mle`・床 0 なら従来の `std(r/s)` そのもの）。
* 報告のみ: 整数ターンの残差 `max(1, ⌈τ⌉) − t_act`（`τ − t_act` は完璧な予測でも平均 ≈ −0.5）。

既定は `whole`（K-4・K-5）・床なし。**波C（2026-10-05）**: `SETTLE_COND_MODE` の `off`／`on` と `SIGMA_FLOOR_MODE=on` は
削除し、どちらも定数になった（凍結ブランチ `claude/theory-switches-final` で再現）。従来の `Φ` の形は `mover=False`（1 行の器）
と尺度 `hyp` 以外の行でだけ生きる。

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
import theory_order as T  # noqa: E402
import win_calib as WC  # noqa: E402


_SHIPPED_SETTLE_COND = T.SETTLE_COND_MODE


def test_whole_is_the_shipped_default():
    """K-4（ユーザ決定 2026-09-26「aでお願いします」）: 整数ターンの式と合わせた幅が既定。"""
    assert _SHIPPED_SETTLE_COND == "whole"


@pytest.fixture(autouse=True)
def _restore_modes():
    old = (T.W_ERR_MODE, T.SIGMA_REL, CB.PRE_SETTLE_MODE)
    yield
    T.set_w_err_mode(old[0]); T.set_sigma_rel(old[1]); CB.set_pre_settle_mode(old[2])


def _phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


_GRID = [(d, tm, to) for tm in (0.0, 0.4, 1.0, 1.7, 3.2, 9.0) for to in (0.0, 0.6, 1.3, 2.5, 7.0)
         for d in (to - tm,)]


# ---- 既定・切替 ---------------------------------------------------------------------------------

def test_both_switches_are_constants_now():
    """波C: `whole`・床なしの定数（出力の欄 `settle_cond`／`sigma_floor` のため名前は残る）。"""
    assert T.SETTLE_COND_MODE == "whole" and T.SIGMA_FLOOR_MODE == "off"
    for gone in ("set_settle_cond_mode", "set_sigma_floor_mode", "SETTLE_COND_MODES", "SIGMA_FLOOR_MODES"):
        assert not hasattr(T, gone), gone
    assert T.TURN_ROUND_VAR == pytest.approx(1.0 / 12.0)


def test_non_hyp_scales_keep_the_t154_formula_bit_for_bit():
    """尺度が `hyp` 以外（`win_calib` の `scale_auc`）の行は `Φ((D + 1/2)/(σ_rel·s))` そのもの（T151／T154 の式）。
    （波C までは旧 `off` で `hyp` の行もこの式だった・`off` は削除。）"""
    T.set_w_err_mode("rel"); T.set_sigma_rel(0.3078)
    for d, tm, to in _GRID:
        s = T.clock_scale(tm, to, "sum")
        want = 0.5 if s == 0 else 0.5 * (1.0 + math.erf((float(d) + 0.5) / ((0.3078 * s) * math.sqrt(2.0))))
        got = T.prob_of_d(d, t_me=tm, t_opp=to, scale_mode="sum", mover=True)
        assert got == want                                                         # ビット一致


def test_switches_do_not_touch_the_one_row_instruments():
    """1 行の器（`mover=False`）には整数ターンの式は掛からない（手番の半ターンと同じ範囲）＝`Φ(D/(σ_rel·s))`。"""
    T.set_w_err_mode("rel"); T.set_sigma_rel(0.3)
    for d, tm, to in _GRID:
        s = math.sqrt(tm * tm + to * to)
        want = 0.5 if s == 0 else 0.5 * (1.0 + math.erf(float(d) / ((0.3 * s) * math.sqrt(2.0))))
        assert T.prob_of_d(d, t_me=tm, t_opp=to) == want


# ---- K-2: 整数ターンの床 -----------------------------------------------------------------------

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
    T.set_w_err_mode("rel"); T.set_sigma_rel(0.3)
    tm, to = 0.8, 1.6
    # 条件を掛けない・幅はまだ打つターン数（K-5）
    assert T.prob_of_d(to - tm, t_me=tm, t_opp=to, mover=True) == pytest.approx(
        T.whole_turn_race_prob(tm, to, 0.3 * 1.0, 0.3 * to, 1))
    tm = 1.4                                                                      # τ ≥ 1 なら幅は σ·τ
    assert T.prob_of_d(to - tm, t_me=tm, t_opp=to, mover=True) == pytest.approx(
        T.whole_turn_race_prob(tm, to, 0.3 * tm, 0.3 * to, 1))
    # 半ターンを使わない行（1 行の器）には掛けない
    assert T.prob_of_d(to - tm, t_me=tm, t_opp=to, mover=False) == pytest.approx(
        _phi((to - tm) / (0.3 * math.hypot(tm, to))))


# ---- K-5: `whole` の幅（まだ打つターン数）と、それに揃えた測り方 -------------------------------------------

def test_whole_width_is_the_number_of_turns_still_to_play():
    """時計 1 本の幅の尺度は `max(1, τ)`＝今のターンは τ がいくら小さくても丸ごと 1 ターン打たれる。"""
    assert [T.whole_clock_scale(x) for x in (0.0, 1e-9, 0.4, 1.0, 3.2)] == [1.0, 1.0, 1.0, 1.0, 3.2]
    T.set_w_err_mode("rel"); T.set_sigma_rel(0.4)
    for d, tm, to in _GRID:
        want = T.whole_turn_race_prob(tm, to, 0.4 * max(1.0, tm), 0.4 * max(1.0, to), 1)
        assert T.prob_of_d(d, t_me=tm, t_opp=to, mover=True) == want


@pytest.mark.parametrize("tm,to", [(0.0, 0.0), (0.0, 1.3), (0.3, 2.5), (2.2, 0.0), (0.9, 0.2)])
def test_whole_with_the_turn_width_matches_a_simulation_of_the_rule(tm, to):
    sm, so = 0.4 * T.whole_clock_scale(tm), 0.4 * T.whole_clock_scale(to)
    mc, n = _mc(tm, to, sm, so, 1)
    tol = 5.0 * math.sqrt(max(mc * (1.0 - mc), 1e-4) / n)
    assert T.whole_turn_race_prob(tm, to, sm, so, 1) == pytest.approx(mc, abs=tol)


def test_whole_never_says_exactly_0_or_1_at_tiny_clocks():
    """従来の幅 `σ·τ` は `τ = 0` で幅 0＝「いま届く」を確率 1 と言っていた（p がちょうど 1／相手なら 0）。"""
    assert T.whole_turn_race_prob(0.0, 2.0, 0.0, 0.4 * 2.0, 1) == 1.0                # 弱点（従来の幅）
    T.set_w_err_mode("rel"); T.set_sigma_rel(0.4)
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
    """`sigma_rel_for` は `whole` の表だけを引く（従来の表・床の表を引く切替は波C で削除）。"""
    prof = {"sigma_rel": {"blockers": {"theory": {"real": 0.31, "syn": 0.33}}},
            "sigma_rel_floor": {"blockers": {"theory": {"real": 0.21, "syn": 0.23}}},
            "sigma_rel_whole": {"blockers": {"theory": {"real": 0.41, "syn": 0.43}}}}
    monkeypatch.setattr(CB, "load_harm_profiles", lambda path=None: prof)
    d_real = tmp_path / "w_real"; d_real.mkdir()
    (d_real / "meta_n_record.json").write_text('{"decks": "user"}', encoding="utf-8")
    assert CB.sigma_rel_for([str(d_real)]) == pytest.approx(0.43)                 # 別のセット（cross）
    assert CB.sigma_rel_for([str(d_real)], "real") == pytest.approx(0.41)


def test_the_fixture_carries_the_whole_table():
    prof = CB.load_harm_profiles()
    tbl = prof["sigma_rel_whole"]["blockers"]
    for slope in ("theory", "curve"):
        assert set(tbl[slope]) == {"real", "syn"} and all(0.0 < v < 1.0 for v in tbl[slope].values())


# ---- 配線（win_calib／pre_settle_asymmetry／crossing_bridge） ----------------------------------------

def test_collect_reads_the_settled_map_only_for_the_pre_settle_filter(monkeypatch):
    import lethal_rule as LR
    calls = []
    monkeypatch.setattr(LR, "settled_map", lambda dirs, limit_games: calls.append(1) or {})
    monkeypatch.setattr(CB.PL, "iter_games", lambda *a, **k: iter([]))
    CB.collect(["x"])
    assert calls == []                                                            # 決着前フィルタが `off` なら読まない
    CB.set_pre_settle_mode("on")
    CB.collect(["x"])
    assert calls == [1]                                                           # `on`／`game` だけが読む


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
