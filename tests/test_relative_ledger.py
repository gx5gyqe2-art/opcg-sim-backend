"""**紐付けの法則**の代数を固める（T122・2026-09-20・正本は `docs/game_theory.md` §17.9）。

出荷の `W = Φ(D/(σ_rel·s))` は `D` も `s` も時計について 1 次同次なので、**`z` は 0 次同次＝
`W` は 2 本の時計の比 `r = T_me/T_opp` だけの関数**。そこから 1 行で出る:

    ΔW = K(r) · Δlog(T_opp/T_me)        K(r) = φ(z)·r(1+r)/(σ_rel·(1+r²)^{3/2})

押さえるのは 6 つ:

1. **0 次同次**——2 本の時計を同時に c 倍しても `z`・`W`・`K` が 1 ビットも動かない。
2. **`K` は `dW/dlog(T_opp/T_me)` そのもの**（数値微分と一致・両側から同じ値）。
3. **通貨の付け替え（`Θ` と価格を同時に c 倍）で相対の腕は不変・絶対額の腕は c 倍**（**P5**）。
4. **両席の `A` に共通の掛け算誤差を入れても相対の腕は不変**（**P7**・`clock` の読みで厳密）。
5. **`curve` の読みには速さの軸が無い**ので、速さを動かす手の `Δlog` は厳密に 0（T121 の 45% の裏側）。
6. **当てはめゼロの較正**（`sep`＝勝ち負けの平均差・理想 1.0／`bias`＝理想 0）が定義どおり動く。

**`κ` の物差しの齟齬**（§17.9.6-1・`w_of_d` は `σ_D`・`prob_of_d` は `σ_rel·s`）を切り替える
`KAPPA_SIGMA_MODE` も、**既定は現状のまま**であることを含めてここでラチェットする。
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

import kappa_vector as KV  # noqa: E402
import relative_ledger as RL  # noqa: E402
import theory_order as TO  # noqa: E402

SIG = 0.2418                                   # 合成の `σ_rel`（実の記録に当てる交差の値）
_FLAT_PROF = [0.2] * 14                        # 一定の輪郭（`τ = Θ/0.2` と手で解ける）
_RAMP_PROF = [0.05, 0.1, 0.2, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]


@pytest.fixture(autouse=True)
def _restore_modes():
    d, k, s = KV.D_MODE, TO.KAPPA_SIGMA_MODE, TO.SIGMA_REL
    yield
    KV.set_d_mode(d); TO.set_kappa_sigma_mode(k); TO.set_sigma_rel(s)


# --------------------------------------------------------------------------- 1. 0 次同次
def test_z_and_w_and_k_are_invariant_to_scaling_both_clocks():
    """**`W` は 2 本の時計の「比」だけの関数**——これが §17.9 の全部の土台。"""
    for c in (0.25, 1.0, 9.0):
        assert RL.z_of(7.0 * c, 9.0 * c, SIG) == pytest.approx(RL.z_of(7.0, 9.0, SIG))
        assert RL.w_of(7.0 * c, 9.0 * c, SIG) == pytest.approx(RL.w_of(7.0, 9.0, SIG))
        assert RL.k_of(7.0 * c, 9.0 * c, SIG) == pytest.approx(RL.k_of(7.0, 9.0, SIG))


def test_z_is_strictly_decreasing_in_the_ratio():
    """`z = (1−r)/(σ√(1+r²))` は `r` について厳密に単調減少（`dz/dr = −(1+r)(1+r²)^{-3/2}/σ`）。"""
    prev = None
    for r in (0.1, 0.3, 0.7, 1.0, 1.5, 3.0, 9.0):
        cur = RL.z_of(r, 1.0, SIG)               # T_me = r・T_opp = 1
        if prev is not None:
            assert cur < prev
        prev = cur
    assert RL.z_of(1.0, 1.0, SIG) == pytest.approx(0.0)


def test_w_is_a_half_when_the_clocks_are_equal():
    assert RL.w_of(5.0, 5.0, SIG) == pytest.approx(0.5)
    assert RL.w_of(3.0, 9.0, SIG) > 0.5 > RL.w_of(9.0, 3.0, SIG)


def test_any_first_order_homogeneous_scale_keeps_the_ordering():
    """**T118 の「1 次同次ならどの尺度でも AUC が一致」の説明**——どれも `r` の単調関数だから。"""
    pairs = [(2.0, 9.0), (5.0, 6.0), (7.0, 4.0), (9.0, 2.0)]
    ranks = {}
    for mode in ("hyp", "sum", "mean", "max", "geo"):
        zs = [(RL.z_of(a, b, SIG) if mode == "hyp"
               else (b - a) / (SIG * TO.clock_scale(a, b, mode))) for a, b in pairs]
        ranks[mode] = [i for i, _ in sorted(enumerate(zs), key=lambda kv: kv[1])]
    assert len(set(tuple(v) for v in ranks.values())) == 1


# --------------------------------------------------------------------------- 2. K は本当に導関数か
def test_k_is_the_derivative_of_w_with_respect_to_the_log_ratio():
    """**`K(r) = dW/dlog(T_opp/T_me)`**——`game_theory.md` §17.9.9 の検算そのもの。"""
    a, b, eps = 7.0, 9.0, 1e-6
    k = RL.k_of(a, b, SIG)
    up = (RL.w_of(a, b * math.exp(eps), SIG) - RL.w_of(a, b, SIG)) / eps
    dn = (RL.w_of(a * math.exp(-eps), b, SIG) - RL.w_of(a, b, SIG)) / eps
    assert up == pytest.approx(k, rel=1e-4)
    assert dn == pytest.approx(k, rel=1e-4)      # **両側から同じ値**＝4 軸が 1 つの `K` を共有する理由


def test_k_is_positive_and_peaks_near_even_clocks():
    ks = {r: RL.k_of(r, 1.0, SIG) for r in (0.05, 0.25, 1.0, 4.0, 20.0)}
    assert all(v > 0.0 for v in ks.values())
    assert ks[1.0] == max(ks.values())
    assert ks[0.05] < ks[0.25] and ks[20.0] < ks[4.0]


def test_k_vanishes_in_a_decided_game():
    """**決着した局では `K → 0`**（`r → 0` で `K ∝ r`）＝**`ΔW` として正しい**。

    ここが「AUC を単独で判定に使ってはいけない」理由——決着帯を増幅する腕ほど AUC で勝つので、
    正しい `K` はその指標では罰される（器の docstring・報告の §判定）。"""
    assert RL.k_of(1e-4, 1.0, SIG) < 1e-3
    assert RL.k_of(1.0, 1e-4, SIG) < 1e-3


# --------------------------------------------------------------------------- 3〜4. 不変量（P5・P7）
def _st(th_me=0.9, th_opp=0.6, a_me=0.12, a_opp=0.10, j=2):
    return (th_me, th_opp, a_me, a_opp, j)


def test_p5_currency_rescale_leaves_the_relative_reading_untouched():
    """**P5**: 耐久と価格を同時に c 倍しても、時計は c 倍されるだけなので**比は動かない**。"""
    KV.set_d_mode("clock")
    st0 = _st(); dx = {"th_opp": -0.05}
    base = RL.dlog_of(st0, KV.apply_dx(st0, dx))
    for c in (0.5, 3.0):
        st0c = (st0[0] * c, st0[1] * c, st0[2], st0[3], st0[4])
        dxc = {"th_opp": -0.05 * c}
        assert RL.dlog_of(st0c, KV.apply_dx(st0c, dxc)) == pytest.approx(base)
        a0, b0 = RL.clocks_of(st0c)
        assert RL.k_of(a0, b0, SIG) == pytest.approx(RL.k_of(*RL.clocks_of(st0), sigma_rel=SIG))


def test_p5_the_absolute_reading_does_move_with_the_currency():
    """**対照**: 絶対額の腕は c 倍でそのまま c 倍になる（不変ではない＝そこが患部）。"""
    assert (0.05 * 3.0) == pytest.approx(3.0 * 0.05)   # `abs_flat` は価格そのもの


def test_p7_a_common_multiplicative_error_in_both_rates_cancels():
    """**P7**: 両席の `A` に共通の掛け算誤差が在っても `r = T_me/T_opp` は厳密に不変。

    T112 は「偏りの 74.7%／73.7% は `A` の軌跡の誤り」と示した＝**両席にほぼ共通**なので、
    比で読む形はその誤りに強いはず、という予告の代数側。"""
    KV.set_d_mode("clock")
    st0 = _st(); dx = {"th_opp": -0.05}
    base = RL.dlog_of(st0, KV.apply_dx(st0, dx))
    a0, b0 = RL.clocks_of(st0)
    for c in (0.7, 1.3, 4.0):
        stc = (st0[0], st0[1], st0[2] * c, st0[3] * c, st0[4])
        assert RL.dlog_of(stc, KV.apply_dx(stc, dx)) == pytest.approx(base)
        assert RL.k_of(*RL.clocks_of(stc), sigma_rel=SIG) == pytest.approx(RL.k_of(a0, b0, SIG))


def test_p7_a_one_sided_rate_error_does_not_cancel():
    """**打ち消えるのは「共通の掛け算」だけ**——片側だけずれたら当然動く（過大な主張をしない）。"""
    KV.set_d_mode("clock")
    st0 = _st()
    one = (st0[0], st0[1], st0[2] * 1.3, st0[3], st0[4])
    assert RL.k_of(*RL.clocks_of(one), sigma_rel=SIG) != pytest.approx(
        RL.k_of(*RL.clocks_of(st0), sigma_rel=SIG))


# --------------------------------------------------------------------------- 5. curve に速さの軸は無い
def test_curve_reading_gives_zero_log_change_for_rate_moves():
    """**輪郭は両席共通の固定の表**なので、速さを動かす手は `Δlog` が厳密に 0（T121 の 45%）。"""
    KV.set_d_mode("curve")
    st0 = _st()
    assert RL.dlog_of(st0, KV.apply_dx(st0, {"a_me": 0.5}), _RAMP_PROF) == pytest.approx(0.0)
    assert RL.dlog_of(st0, KV.apply_dx(st0, {"a_opp": -0.05}), _RAMP_PROF) == pytest.approx(0.0)
    assert RL.dlog_of(st0, KV.apply_dx(st0, {"th_opp": -0.05}), _RAMP_PROF) != pytest.approx(0.0)


def test_clock_reading_does_price_rate_moves():
    KV.set_d_mode("clock")
    st0 = _st()
    assert RL.dlog_of(st0, KV.apply_dx(st0, {"a_me": 0.05})) > 0.0
    assert RL.dlog_of(st0, KV.apply_dx(st0, {"a_opp": -0.02})) > 0.0


def test_clocks_of_agrees_with_the_bridge_reading_of_d():
    """**`D = T_opp − T_me` は `kappa_vector.d_of` と同じ値**（同じ関数を呼んでいる＝読みが 1 本）。"""
    st0 = _st()
    for mode, prof in (("clock", None), ("curve", _FLAT_PROF)):
        KV.set_d_mode(mode)
        a, b = RL.clocks_of(st0, prof)
        assert (b - a) == pytest.approx(KV.d_of(st0, prof))


def test_an_attack_always_moves_the_log_ratio_the_right_way():
    """相手の耐久を削れば `T_me` が縮む＝`log(T_opp/T_me)` は**増える**（自分に有利）。"""
    for mode, prof in (("clock", None), ("curve", _FLAT_PROF)):
        KV.set_d_mode(mode)
        st0 = _st()
        assert RL.dlog_of(st0, KV.apply_dx(st0, {"th_opp": -0.05}), prof) > 0.0
        assert RL.dlog_of(st0, KV.apply_dx(st0, {"th_me": +0.05}), prof) > 0.0


def test_the_killing_blow_diverges_in_the_log_reading():
    """**§17.9.4**: `Δlog Θ_opp = log(Θ_opp/(Θ_opp − p))` は `p → Θ_opp` で発散する
    ＝**絶対額の帳簿はとどめの一撃を原理的に軽く見る**（T80／T119 の予言）。"""
    KV.set_d_mode("clock")
    st0 = _st(th_opp=0.60)
    prev = 0.0
    for p in (0.30, 0.55, 0.59, 0.5999):
        cur = RL.dlog_of(st0, KV.apply_dx(st0, {"th_opp": -p}))
        assert cur > prev
        prev = cur
    assert prev > 5.0                       # 有限の価格では届かない大きさになる
    # **厳密形（`Φ(z')−Φ(z)`）は発散しない**＝終盤はこちらを使う
    fin = RL.w_of(*RL.clocks_of(KV.apply_dx(st0, {"th_opp": -0.5999})), sigma_rel=SIG) \
        - RL.w_of(*RL.clocks_of(st0), sigma_rel=SIG)
    assert 0.0 < fin <= 1.0


# --------------------------------------------------------------------------- 6. 較正と κ の物差し
def test_calibration_is_one_and_zero_for_a_perfect_ledger():
    """`Σ = z − W₀` を厳密に満たす帳簿なら `sep = 1`・`bias = 0`（定義の検算）。

    **`W₀` が勝ち負けで釣り合っていること**が `sep = 1` の条件なので、そう作る。"""
    zs = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    w0 = [0.6, 0.6, 0.45, 0.45, 0.5, 0.5]
    xs = [z - w for z, w in zip(zs, w0)]
    c = RL.calib_of(xs, zs, w0)
    assert c["w0_gap"] == pytest.approx(0.0)
    assert c["sep"] == pytest.approx(1.0)
    assert c["bias"] == pytest.approx(0.0)


def test_calibration_sep_shifts_exactly_by_the_w0_gap():
    """**`sep` の理想は `1 − w0_gap`**——釣り合いが崩れた分だけ動く（読むときに必ず `w0_gap` を見る）。"""
    zs = [1.0, 0.0, 1.0, 0.0]
    w0 = [0.7, 0.3, 0.7, 0.3]                      # 勝った側の `W₀` が +0.4 高い
    xs = [z - w for z, w in zip(zs, w0)]
    c = RL.calib_of(xs, zs, w0)
    assert c["w0_gap"] == pytest.approx(0.4)
    assert c["sep"] == pytest.approx(1.0 - 0.4)


def test_calibration_sep_scales_with_the_ledger():
    """**帳簿を c 倍すれば `sep` も c 倍**（尺度に騙されない＝傾きと違って当てはめが要らない）。"""
    zs = [1.0, 0.0, 1.0, 0.0]; w0 = [0.5] * 4
    xs = [z - w for z, w in zip(zs, w0)]
    assert RL.calib_of([3.0 * x for x in xs], zs, w0)["sep"] == pytest.approx(3.0)


def test_calibration_is_none_without_both_labels():
    assert RL.calib_of([1.0, 2.0], [1.0, 1.0], [0.5, 0.5])["sep"] is None


def test_kappa_sigma_mode_defaults_to_the_shipped_behaviour():
    """**既定は動かさない**（`abs`）——直すかどうかはユーザ判断（§17.9.6-1）。"""
    assert TO.KAPPA_SIGMA_MODE == "abs"
    with pytest.raises(ValueError):
        TO.set_kappa_sigma_mode("rel")


def test_kappa_sigma_match_uses_the_same_yardstick_as_the_win_probability():
    """**`match` なら `κ` は本当に `W` の微分になる**——`w` の `σ` が `prob_of_d` と同じ `σ_rel·s`。"""
    TO.set_sigma_rel(SIG)
    a, b = 7.0, 9.0
    d = b - a
    sd = SIG * TO.clock_scale(a, b)
    TO.set_kappa_sigma_mode("abs")
    k_abs = TO.state_factor(d, "curve", t_me=a, t_opp=b)
    TO.set_kappa_sigma_mode("match")
    k_match = TO.state_factor(d, "curve", t_me=a, t_opp=b)
    assert k_abs == pytest.approx(TO.w_of_d(d) / TO.W_BAR)
    assert k_match == pytest.approx(TO.w_of_d(d, sd) / TO.W_BAR)
    assert sd > TO.SIGMA_D                     # 出荷の物差しの方が広い（κ が鋭すぎた）
    assert k_match != pytest.approx(k_abs)


def test_kappa_sigma_match_falls_back_when_the_clocks_are_missing():
    """時計が渡されなければ `match` でも `abs` と同じ（黙って別の値にしない）。"""
    TO.set_sigma_rel(SIG)
    TO.set_kappa_sigma_mode("match")
    assert TO.state_factor(1.0, "curve") == pytest.approx(TO.w_of_d(1.0) / TO.W_BAR)


def test_flat_mode_is_still_exactly_one():
    TO.set_kappa_sigma_mode("match")
    assert TO.state_factor(1.0, "flat", t_me=3.0, t_opp=9.0) == 1.0


def test_the_ledger_exposes_both_seat_switches_and_they_reach_the_module(monkeypatch):
    """**T134**: **切替が器の側に無いと「動かなかった」を誤って読む**——本 T で実際に踏んだ
    （`--theta-side` だけ渡した測定を「①＋③」と名付けていた）。**帳簿にも両方在ることを固定する。**"""
    import crossing_bridge as CB                       # noqa: PLC0415
    a = RL.build_parser().parse_args(["--in", "x", "--theta-side", "symmetric",
                                      "--slope-take", "life"])
    assert (a.theta_side, a.slope_take) == ("symmetric", "life")
    monkeypatch.setattr(RL, "collect", lambda *args, **kw: {})
    try:
        RL.main(["--in", "x", "--theta-side", "symmetric", "--slope-take", "life"])
        assert (CB.THETA_SIDE_MODE, CB.SLOPE_TAKE_MODE) == ("symmetric", "life")
    finally:
        CB.set_theta_side_mode("legacy")
        CB.set_slope_take_mode("const")


def test_pre_settle_defaults_to_off_and_the_cli_flag_reaches_collect(monkeypatch):
    """**T138b**: `--pre-settle` は `collect(..., pre_settle=True/False)` にそのまま届く
    （既定は `off`＝`False`・省略した呼び出しでも壊れない）。"""
    called = {}
    monkeypatch.setattr(RL, "collect", lambda *a, **kw: called.update(kw) or {})
    RL.main(["--in", "x"])
    assert called["pre_settle"] is False
    called.clear()
    RL.main(["--in", "x", "--pre-settle", "on"])
    assert called["pre_settle"] is True


def test_pre_settle_only_reads_the_settled_map_when_asked(monkeypatch):
    """**T138b**: `pre_settle=True` のときだけ `lethal_rule.settled_map(dirs, limit_games)` を
    1 回読む（`off` は記録をもう 1 度読み直すコストを払わない）。"""
    import lethal_rule as LR
    calls = []
    monkeypatch.setattr(LR, "settled_map", lambda dirs, limit_games: calls.append((dirs, limit_games)) or {})
    monkeypatch.setattr(RL.PL, "iter_games", lambda *a, **k: iter([]))
    monkeypatch.setattr(RL.CB, "profile_for", lambda dirs: [0.2] * 12)
    monkeypatch.setattr(RL.CB, "sigma_rel_for", lambda dirs, slope="curve": 1.0)
    monkeypatch.setattr(RL.KV, "_seat_decks", lambda dirs: {})
    out = RL.collect(["x"], 7, pre_settle=False)
    assert calls == [] and out["pre_settle"] is False
    out2 = RL.collect(["x"], 7, pre_settle=True)
    assert calls == [(["x"], 7)] and out2["pre_settle"] is True


def test_the_curve_reading_cannot_measure_a_switch_that_only_moves_the_rate():
    """**T134 の教訓**: 既定の `D_MODE=curve` は**速さの軸を持たない**（`d_of` は `a_me`／`a_opp` を
    一度も読まない）＝**`A` だけを動かす切替は帳簿の既定では厳密に 0**。**「動かなかった」を
    「効かなかった」と読まないための固定**（`--d-mode theory` で測ること）。"""
    KV.set_d_mode("curve")
    prof = {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4, 5: 0.5}
    st_slow = (1.0, 1.0, 0.05, 0.05, 2)
    st_fast = (1.0, 1.0, 5.00, 5.00, 2)
    assert KV.d_of(st_slow, prof) == KV.d_of(st_fast, prof)
    KV.set_d_mode("theory")
    KV.set_rate_shape(None, None)
    assert KV.d_of((1.0, 2.0, 0.05, 0.05, 2)) != KV.d_of((1.0, 2.0, 5.00, 0.05, 2))
    KV.set_d_mode("curve")


def test_parts_split_the_game_sum_by_winner_and_loser_last_turns():
    """**T145**: `parts_of` は局ごとの `rel_K` の和を勝者／敗者の最後のターン・宣言した行に割る。
    **勝者の視点に直す**（席 1 が勝った局は符号を返す）・腕は全体の和から部分を引いた値。"""
    rows = []
    for g in range(12):
        win = g % 2                                  # 勝者の席を交互に
        sg = 1.0 if win == 0 else -1.0               # 勝者視点 → 席 0 視点
        # 勝者の最後のターン（宣言した行）+3・敗者の最後のターン +2（敗者視点）・それ以外は勝者に +0.5
        part = {(win, True, True): 3.0 * sg, (1 - win, True, False): -2.0 * sg,
                (win, False, False): 0.5 * sg}
        tot = sum(part.values())
        rows.append((1.0 if win == 0 else 0.0, tot, part))
    out = RL.parts_of(rows)
    m = out["mean_winner_view"]
    assert out["games"] == 12
    assert m["winner_last"] == pytest.approx(3.0) and m["winner_declared"] == pytest.approx(3.0)
    assert m["loser_last"] == pytest.approx(-2.0) and m["loser_declared"] == 0.0
    assert m["rest"] == pytest.approx(0.5)
    # 全体は勝者に +1.5・宣言（勝者の最後）を引くと −1.5＝**符号が反転**・両方の最後を引くと +0.5
    assert out["auc"]["all"] == 1.0
    assert out["auc"]["minus_declared"] == 0.0
    assert out["auc"]["minus_winner_last"] == 0.0
    assert out["auc"]["minus_both_last"] == 1.0
    assert out["auc"]["minus_loser_last"] == 1.0


def test_parts_flag_reaches_collect_and_reads_flags_without_dropping_rows(monkeypatch):
    """**T145**: `--parts` は `collect(..., parts=True)` に届き、`parts=True` は旗を読むだけで
    行を落とさない（`pre_settle` は `False` のまま）。"""
    called = {}
    monkeypatch.setattr(RL, "collect", lambda *a, **kw: called.update(kw) or {})
    RL.main(["--in", "x", "--parts"])
    assert called["parts"] is True and called["pre_settle"] is False
    called.clear()
    RL.main(["--in", "x"])
    assert called["parts"] is False
