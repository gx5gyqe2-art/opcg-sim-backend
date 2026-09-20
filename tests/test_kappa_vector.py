"""`kappa_vector` の**勾配の代数**を固める（T121・2026-09-20）。

帳簿の `κ` は**スカラー 1 個**だった＝「軸が全部同じ重み」という特別な場合。
本器は `D`（2 つの到達時刻の差）を**軸ごとに偏微分**する。押さえるのは 6 つ:

1. **`clock` の 4 つの偏微分が式どおり**（`+1/A_opp`・`−1/A_me`・`+Θ_opp/A_me²`・`−Θ_me/A_opp²`）と**符号**。
2. **次数**（`D` は時間の単位＝`Θ` について 1 次・`A` について −1 次。勾配はその 1 つ下）。
3. **`curve`（帳簿の正本）には速さの軸が無い**——輪郭は固定の表なので `A` を動かしても `D` は 1 秒も動かない。
   **これが T121 の一番大事な観測**なので、テストで固定する（壊れたら黙って通ってはいけない）。
4. **`curve` の中でも 2 つの耐久の重みは別の数**（`1/prof[j+τ_me]` と `1/prof[j+τ_opp]`）。
5. **手→軸の表**（型ごとにどの軸へ載るか・攻撃と守りは価格そのまま・出す/付与は 1 ターンあたり）。
6. **プラセボが情報を持たない置換である**（大きさは保存・軸だけ回る）。

**`exact` は「同じ式の近似を深くした腕」**なので、`Δx → 0` で `vector` に一致することも見る。
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
import kappa_vector as KV  # noqa: E402
import theory_order as TO  # noqa: E402


@pytest.fixture(autouse=True)
def _restore_d_mode():
    """`D_MODE` はモジュール変数なので、テストごとに既定（`curve`＝帳簿の正本）へ戻す。"""
    old = KV.D_MODE
    yield
    KV.set_d_mode(old)


#: 損害の輪郭のダミー（**一定 0.2／ターン**）。一定なら `τ = Θ/0.2` と手で解けるので検算に使える。
_FLAT_PROF = [0.2] * 14
#: 段のある輪郭（序盤が遅い＝実測の形）。`τ` の位置で勾配が変わることを見るため。
_RAMP_PROF = [0.05, 0.1, 0.2, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]


# --------------------------------------------------------------------------- 1. clock の偏微分
def test_clock_grad_matches_the_four_partials():
    th_me, th_opp, a_me, a_opp = 7.0, 5.0, 1.25, 0.8
    g = KV.grad_clock(th_me, th_opp, a_me, a_opp)
    assert g["th_me"] == pytest.approx(1.0 / a_opp)
    assert g["th_opp"] == pytest.approx(-1.0 / a_me)
    assert g["a_me"] == pytest.approx(th_opp / (a_me * a_me))
    assert g["a_opp"] == pytest.approx(-th_me / (a_opp * a_opp))


def test_clock_grad_agrees_with_a_numeric_derivative():
    """**勾配は `d_of` の数値微分と一致する**（式と実装が別物になっていないこと）。"""
    KV.set_d_mode("clock")
    st = (6.0, 4.5, 1.1, 0.9, 3)
    g = KV.grad_clock(*st[:4])
    h = 1e-6
    for k, i in (("th_me", 0), ("th_opp", 1), ("a_me", 2), ("a_opp", 3)):
        up = list(st); up[i] += h
        dn = list(st); dn[i] -= h
        num = (KV.d_of(tuple(up)) - KV.d_of(tuple(dn))) / (2 * h)
        assert g[k] == pytest.approx(num, rel=1e-5, abs=1e-7)


def test_grad_keys_are_exactly_the_axes_in_both_readings():
    KV.set_d_mode("clock")
    assert set(KV.grad_of((3.0, 3.0, 1.0, 1.0, 2))) == set(KV.AXES)
    KV.set_d_mode("curve")
    assert set(KV.grad_of((3.0, 3.0, 1.0, 1.0, 2), _FLAT_PROF)) == set(KV.AXES)


def test_signs_say_which_way_each_axis_moves_d():
    g = KV.grad_clock(6.0, 5.0, 1.2, 0.9)
    assert g["th_me"] > 0.0      # 自分の耐久が増えれば自分の時計が伸びる＝D は増える
    assert g["th_opp"] < 0.0     # 相手の耐久が増えれば相手の時計が伸びる＝D は減る
    assert g["a_me"] > 0.0       # 自分の速さが上がれば相手の時計が縮む＝D は増える
    assert g["a_opp"] < 0.0      # 相手の速さが上がれば自分の時計が縮む＝D は減る


def test_a_me_partial_is_zero_when_the_opponent_has_no_endurance_left():
    """**相手の耐久が 0 なら速さを上げても `D` は動かない**（もう削るものが無い＝規則どおり）。"""
    assert KV.grad_clock(6.0, 0.0, 1.2, 0.9)["a_me"] == pytest.approx(0.0)


def test_clock_grad_is_zero_on_the_side_that_hit_the_cap():
    """**打ち切りに当たった側の軸は 0**（`τ` がもう動かない＝触っても `D` は変わらない）。"""
    g = KV.grad_clock(100.0, 5.0, 1.0, 0.01)      # Θ_me/A_opp = 10000 ≫ CAP
    assert g["th_me"] == 0.0 and g["a_opp"] == 0.0
    assert g["th_opp"] != 0.0 and g["a_me"] != 0.0


def test_floor_keeps_the_gradient_finite_at_zero_rate():
    g = KV.grad_clock(0.0, 0.0, 0.0, 0.0)
    assert all(math.isfinite(v) for v in g.values())
    assert g["th_me"] == pytest.approx(1.0 / KV.A_FLOOR)


# --------------------------------------------------------------------------- 2. 次数（単位）
def test_d_is_first_order_in_theta_and_minus_first_in_rate():
    """`Θ → cΘ` で `D` は `c` 倍・`A → cA` で `1/c` 倍（`D` は時間の単位）。"""
    KV.set_d_mode("clock")
    st = (6.0, 5.0, 1.2, 0.9, 3)
    d = KV.d_of(st)
    c = 3.0
    assert KV.d_of((c * st[0], c * st[1], st[2], st[3], 3)) == pytest.approx(c * d)
    assert KV.d_of((st[0], st[1], c * st[2], c * st[3], 3)) == pytest.approx(d / c)


def test_gradient_is_one_order_below_d_on_each_axis():
    """勾配は `D` の 1 つ下の次数＝耐久の軸は `Θ` に 0 次・速さの軸は `Θ` に 1 次。"""
    st = (6.0, 5.0, 1.2, 0.9)
    g0 = KV.grad_clock(*st)
    c = 2.5
    g1 = KV.grad_clock(c * st[0], c * st[1], st[2], st[3])
    assert g1["th_me"] == pytest.approx(g0["th_me"])          # Θ に依らない
    assert g1["th_opp"] == pytest.approx(g0["th_opp"])
    assert g1["a_me"] == pytest.approx(c * g0["a_me"])        # Θ に比例
    assert g1["a_opp"] == pytest.approx(c * g0["a_opp"])


def test_the_body_to_attack_weight_ratio_is_the_opponents_clock():
    """**出す手と攻める手の重みの比は `τ_opp`**——これが T111 の「率は `1/τ`」を導いた形。

    `∂D/∂A_me ÷ |∂D/∂Θ_opp| = (Θ_opp/A_me²)·A_me = Θ_opp/A_me = τ_opp`。
    **新しい定数はどこにも要らない**（比が局面から出る）。"""
    th_me, th_opp, a_me, a_opp = 6.0, 5.0, 1.25, 0.9
    g = KV.grad_clock(th_me, th_opp, a_me, a_opp)
    assert g["a_me"] / abs(g["th_opp"]) == pytest.approx(th_opp / a_me)


# --------------------------------------------------------------------------- 3. curve に速さの軸は無い
def test_curve_reading_has_no_rate_axes_at_all():
    """**帳簿の `D` は `A` を 1 つも見ていない**（輪郭は両席共通の固定の表）。T121 の骨。"""
    KV.set_d_mode("curve")
    assert KV.LIVE_AXES["curve"] == ("th_me", "th_opp")
    g = KV.grad_of((6.0, 5.0, 1.2, 0.9, 2), _FLAT_PROF)
    assert g["a_me"] == 0.0 and g["a_opp"] == 0.0
    # `A` を 100 倍しても `D` は 1 秒も動かない
    d0 = KV.d_of((6.0, 5.0, 1.2, 0.9, 2), _FLAT_PROF)
    d1 = KV.d_of((6.0, 5.0, 120.0, 90.0, 2), _FLAT_PROF)
    assert d0 == pytest.approx(d1)


def test_curve_reading_needs_a_profile():
    KV.set_d_mode("curve")
    with pytest.raises(ValueError):
        KV.d_of((6.0, 5.0, 1.2, 0.9, 2))


def test_curve_tau_is_theta_over_the_flat_profile_height():
    """一定の輪郭なら `τ = Θ/h`（手で解ける）＝`D = (Θ_me − Θ_opp)/h`。"""
    KV.set_d_mode("curve")
    d = KV.d_of((1.0, 0.6, 9.9, 9.9, 0), _FLAT_PROF)
    assert d == pytest.approx((1.0 - 0.6) / 0.2)


def test_curve_gradient_is_one_over_the_profile_height_at_each_clock():
    """**2 つの耐久の重みは別の数**——それぞれの `τ` における輪郭の高さの逆数。

    段のある輪郭で `Θ` を変えると、片方は遅い帯・もう片方は速い帯に着地するので
    `|∂D/∂Θ_me| ≠ |∂D/∂Θ_opp|`。**スカラー `κ` はこの 2 つを 1 つに潰している**。"""
    KV.set_d_mode("curve")
    g = KV.grad_of((0.9, 0.03, 1.0, 1.0, 0), _RAMP_PROF)
    assert g["th_me"] > 0.0 and g["th_opp"] < 0.0
    assert abs(g["th_me"]) == pytest.approx(1.0 / 0.4, rel=0.05)    # 遠い側は速い帯（0.4）で解決
    assert abs(g["th_opp"]) == pytest.approx(1.0 / 0.05, rel=0.05)  # 近い側は序盤の遅い帯（0.05）
    assert abs(g["th_me"]) != pytest.approx(abs(g["th_opp"]), rel=0.5)


def test_curve_gradient_equals_one_over_h_on_a_flat_profile():
    KV.set_d_mode("curve")
    g = KV.grad_of((1.0, 0.6, 1.0, 1.0, 0), _FLAT_PROF)
    assert g["th_me"] == pytest.approx(1.0 / 0.2, rel=1e-3)
    assert g["th_opp"] == pytest.approx(-1.0 / 0.2, rel=1e-3)


# --------------------------------------------------------------------------- 4. 手 → 軸の表
def _ctx():
    sc = np.zeros(70, float)
    sc[12] = sc[13] = 0.5                     # 双方リーダー 5000
    tok = np.zeros((22, 24), float)
    tok[0, 0] = 0.5                           # 自リーダーのパワー 5000（枠 0・`S_POWER` は /1e4）
    return sc, tok


def test_attack_moves_the_opponents_endurance_down_at_face_value():
    sc, tok = _ctx()
    dx = KV.axis_of_move("attack", 0.37, ["ATTACK", None, "L"], None, None, sc, tok, 5000.0, 4.0)
    assert dx == {"th_opp": -0.37}


def test_guard_moves_my_own_endurance_up_at_face_value():
    sc, tok = _ctx()
    dx = KV.axis_of_move("guard", 0.21, None, None, None, sc, tok, 5000.0, 4.0)
    assert dx == {"th_me": 0.21}


def test_unknown_family_falls_back_to_the_endurance_axis():
    sc, tok = _ctx()
    for fam in ("other", "end"):
        dx = KV.axis_of_move(fam, 0.11, None, None, None, sc, tok, 5000.0, 4.0)
        assert dx == {"th_opp": -0.11}


def test_play_of_a_body_lands_on_my_rate_in_per_turn_units():
    """**出した体は毎ターンの攻撃の価値で `A_me` に載る**（価格そのままではない）。"""
    class _C:
        def info(self, cid):
            return {"power": 5000.0}

    sc, tok = _ctx()
    dx = KV.axis_of_move("play", 0.99, ["PLAY"], "X", _C(), sc, tok, 5000.0, 4.0)
    assert set(dx) == {"a_me"}
    assert dx["a_me"] == pytest.approx(TO.attack_value(5000.0, 5000.0, True, KV.THETA, KV.MU))
    assert dx["a_me"] != pytest.approx(0.99)          # 価格をそのまま使っていない


def test_play_of_a_bodyless_card_is_spread_over_the_remaining_turns():
    class _C:
        def info(self, cid):
            return {"power": 0.0}

    sc, tok = _ctx()
    dx = KV.axis_of_move("play", 0.8, ["PLAY"], "X", _C(), sc, tok, 5000.0, 4.0)
    assert dx == {"a_me": pytest.approx(0.2)}


def test_attach_is_the_difference_of_two_attack_values():
    sc, tok = _ctx()
    dx = KV.axis_of_move("attach", 0.5, ["DON_BOX"], None, None, sc, tok, 5000.0, 4.0, don_k=2)
    want = (TO.attack_value(5000.0 + 2000.0, 5000.0, True, KV.THETA, KV.MU)
            - TO.attack_value(5000.0, 5000.0, True, KV.THETA, KV.MU))
    assert dx == {"a_me": pytest.approx(want)}


def test_attach_of_zero_don_moves_nothing():
    sc, tok = _ctx()
    dx = KV.axis_of_move("attach", 0.5, ["DON_BOX"], None, None, sc, tok, 5000.0, 4.0, don_k=0)
    assert dx["a_me"] == pytest.approx(0.0)


def test_effect_without_removal_harm_lands_on_my_rate_only():
    sc, tok = _ctx()
    dx = KV.axis_of_move("effect", 0.6, ["ACTIVATE_MAIN"], None, None, sc, tok, 5000.0, 3.0)
    assert set(dx) == {"a_me"}


def test_removal_hits_both_the_opponents_endurance_and_rate():
    """**除去は 2 つの軸を同時に動かす**（体は耐久にも速さにも数えられている＝二重計上ではない）。"""
    import deck_refill as DR
    sc, tok = _ctx()
    orig = DR.card_effect_harm
    try:
        DR.card_effect_harm = lambda cid, mlp=5000.0, r=3, boards=None: 0.9
        dx = KV.axis_of_move("effect", 0.6, ["ACTIVATE_MAIN"], "X", None, sc, tok, 5000.0, 3.0)
    finally:
        DR.card_effect_harm = orig
    assert dx["th_opp"] == pytest.approx(-0.9)
    assert dx["a_opp"] == pytest.approx(-0.3)
    assert "a_me" not in dx


def test_the_rate_moving_families_are_exactly_the_ones_curve_cannot_price():
    """**出す・付与・（除去でない）効果は `curve` では 1 円も動かせない**——帳簿の穴の正体。"""
    KV.set_d_mode("curve")
    sc, tok = _ctx()

    class _C:
        def info(self, cid):
            return {"power": 5000.0}

    st = (0.9, 0.7, 1.0, 1.0, 1)
    g = KV.grad_of(st, _RAMP_PROF)
    for fam, cid, cs in (("play", "X", _C()), ("attach", None, None), ("effect", None, None)):
        dx = KV.axis_of_move(fam, 0.5, None, cid, cs, sc, tok, 5000.0, 3.0, don_k=1)
        assert set(dx) <= {"a_me", "a_opp"}
        assert KV.dot(g, dx) == pytest.approx(0.0)
        assert KV.d_of(KV.apply_dx(st, dx), _RAMP_PROF) == pytest.approx(KV.d_of(st, _RAMP_PROF))


# --------------------------------------------------------------------------- 5. プラセボと台帳
def test_placebo_a_keeps_the_magnitudes_and_only_rotates_the_axes():
    d = {"th_opp": -0.4, "a_me": 0.25}
    p = KV._perm_axes(d)
    assert sorted(abs(v) for v in p.values()) == sorted(abs(v) for v in d.values())
    assert set(p) != set(d)
    assert p == {"a_me": -0.4, "a_opp": 0.25}


def test_placebo_a_is_a_permutation_of_the_axis_list():
    full = {k: float(i + 1) for i, k in enumerate(KV.AXES)}
    p = KV._perm_axes(full)
    assert set(p) == set(KV.AXES)
    assert sorted(p.values()) == sorted(full.values())


def test_placebo_a_returns_to_the_original_after_four_rotations():
    d = {"th_me": 1.0, "a_opp": -2.0}
    cur = dict(d)
    for _ in range(len(KV.AXES)):
        cur = KV._perm_axes(cur)
    assert cur == d


def test_dot_ignores_axes_the_move_does_not_touch():
    g = KV.grad_clock(6.0, 5.0, 1.2, 0.9)
    assert KV.dot(g, {"th_opp": -1.0}) == pytest.approx(-g["th_opp"])
    assert KV.dot(g, {}) == pytest.approx(0.0)


def test_apply_dx_clamps_endurance_at_zero_and_rate_at_the_floor():
    st = KV.apply_dx((1.0, 0.5, 0.3, 0.2, 4), {"th_me": -5.0, "th_opp": -5.0,
                                               "a_me": -5.0, "a_opp": -5.0})
    assert st == (0.0, 0.0, KV.A_FLOOR, KV.A_FLOOR, 4)


def test_apply_dx_keeps_the_turn_index():
    assert KV.apply_dx((1.0, 0.5, 0.3, 0.2, 7), {"th_opp": -0.1})[4] == 7


def test_exact_difference_converges_to_the_gradient_for_small_moves():
    """**`exact` は `Δx → 0` で `vector` に一致する**（腕の違いは近似の深さだけ）。"""
    KV.set_d_mode("clock")
    st = (6.0, 5.0, 1.2, 0.9, 3)
    g = KV.grad_clock(*st[:4])
    for eps in (1e-3, 1e-5):
        dx = {"th_opp": -eps, "a_me": 0.5 * eps}
        exact = KV.d_of(KV.apply_dx(st, dx)) - KV.d_of(st)
        assert exact == pytest.approx(KV.dot(g, dx), rel=2e-2)


def test_exact_and_gradient_part_ways_when_the_move_is_large():
    """**序盤の「体を 1 つ出す」は小さくない**——一次近似が実際にずれることを固定する
    （ずれるから `exact` を別の腕として並べている）。"""
    KV.set_d_mode("clock")
    st = (6.0, 5.0, 0.3, 0.9, 1)           # A_me が小さい＝体 1 つで倍になる帯
    g = KV.grad_clock(*st[:4])
    dx = {"a_me": 0.3}
    exact = KV.d_of(KV.apply_dx(st, dx)) - KV.d_of(st)
    assert KV.dot(g, dx) > 1.5 * exact  # 一次は行き過ぎる（凸性）


def test_auc_is_the_rank_statistic_with_ties_at_one_half():
    assert KV.auc_of([1.0, 2.0, 3.0, 4.0], [0, 0, 1, 1]) == pytest.approx(1.0)
    assert KV.auc_of([4.0, 3.0, 2.0, 1.0], [0, 0, 1, 1]) == pytest.approx(0.0)
    assert KV.auc_of([1.0, 1.0, 1.0, 1.0], [0, 0, 1, 1]) == pytest.approx(0.5)


def test_auc_is_none_without_both_labels():
    assert KV.auc_of([1.0, 2.0], [1, 1]) is None


def test_score_is_invariant_to_a_positive_rescale():
    """**判定に使う量は尺度に依らない**（傾きだけが正規化で動く＝T119 の教訓）。"""
    xs = [0.1, -0.4, 0.7, 0.2, -0.9, 0.5, 0.3, -0.2, 0.8, -0.6, 0.4]
    zs = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1]
    a = KV._score(xs, zs)
    b = KV._score([5.0 * x for x in xs], zs)
    assert a["auc"] == b["auc"]
    assert a["corr"] == b["corr"]
    assert a["slope"] != b["slope"]


def test_set_d_mode_rejects_unknown_names():
    with pytest.raises(ValueError):
        KV.set_d_mode("walk")


# --------------------------------------------------------------------------- 6. 帳簿側の退行
def test_the_ledger_can_take_a_row_without_two_clocks():
    """**`κ` の形を比べる腕（`W_MODE=flat`／`clock`）が帳簿で回ること**。

    T118 で `rel` の物差しに 2 本の時計を渡すようにしたとき、**`curve` 以外の読みでは
    `tau_me`／`tau_opp` が存在しない**のに無条件で読んでいて `KeyError` で落ちていた
    （T121 で `--w-mode flat` を回そうとして踏んだ）。時計が無ければ `abs` の物差しへ落ちる。"""
    import theory_bridge as TB
    assert TB._W_of({"d0": 0.4, "t_me0": None, "t_opp0": None}) is not None
    assert TB._W_of({"d0": 0.4}) is not None            # 欄そのものが無くても落ちない
    # 時計を渡した側と渡さない側で**別の値**になる（物差しが局面で変わる＝T118 が効いている）
    TO.set_sigma_rel(0.22)
    try:
        a = TB._W_of({"d0": 0.4, "t_me0": 9.0, "t_opp0": 7.0})
        b = TB._W_of({"d0": 0.4, "t_me0": None, "t_opp0": None})
    finally:
        TO.set_sigma_rel(None)
    assert a != pytest.approx(b)


# --------------------------------------------------------------------------- 7. T126: 輪郭を速さで伸縮する読み
@pytest.fixture
def _prof_th():
    old = KV.PROFILE_TH
    KV.set_profile_th([0.0, 0.08, 0.12, 0.19, 0.24, 0.30])
    yield
    KV.set_profile_th(old)


def test_curve_scaled_is_a_third_reading_with_all_four_axes_live():
    assert "curve_scaled" in KV.D_MODES
    assert KV.LIVE_AXES["curve_scaled"] == KV.AXES
    assert KV.LIVE_AXES["curve"] == ("th_me", "th_opp")


def test_profile_scale_needs_the_theory_rate_curve():
    old = KV.PROFILE_TH
    KV.set_profile_th(None)
    try:
        with pytest.raises(ValueError):
            KV.profile_scale(0.1, 2)
    finally:
        KV.set_profile_th(old)


def test_profile_scale_is_the_rate_over_the_typical_rate(_prof_th):
    assert KV.profile_scale(0.24, 3) == pytest.approx(0.24 / 0.19)
    assert KV.profile_scale(0.19, 3) == pytest.approx(1.0)      # 典型どおりなら伸縮しない


def test_profile_scale_skips_the_first_turn_where_no_rate_exists(_prof_th):
    """**`prof_th[0] = 0` を分母にしない**（`RATE_T1_MODE=on`＝最初の自席ターンは打てない規則）。

    床で割ると倍率が 55 倍に飛び、**5.0% の行が打ち切りに貼り付いた**（T126 で踏んだ）。
    **速さが定義される最初のターンまで進めて割る**。"""
    assert KV.profile_scale(0.08, 0) == pytest.approx(0.08 / 0.08)   # j=0 → j=1 の値で割る
    assert KV.profile_scale(0.08, 0) != pytest.approx(0.08 / CB.SLOPE_FLOOR)


def test_profile_scale_clamps_past_the_end_of_the_curve(_prof_th):
    assert KV.profile_scale(0.3, 99) == pytest.approx(0.3 / 0.30)


def test_curve_scaled_lets_the_rate_move_the_clock(_prof_th):
    """**これが T126 の狙い**——`curve` では体を出しても `D` が 1 ビットも動かない。"""
    prof = [0.05, 0.1, 0.2, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]
    st = (0.9, 0.6, 0.12, 0.10, 2)
    fast = (0.9, 0.6, 0.24, 0.10, 2)
    KV.set_d_mode("curve")
    assert KV.d_of(fast, prof) == pytest.approx(KV.d_of(st, prof))     # 動かない（患部）
    KV.set_d_mode("curve_scaled")
    assert KV.d_of(fast, prof) > KV.d_of(st, prof)                     # 速くなれば有利になる


def test_curve_scaled_gives_the_rate_axes_a_gradient(_prof_th):
    prof = [0.05, 0.1, 0.2, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]
    KV.set_d_mode("curve_scaled")
    g = KV.grad_of((0.9, 0.6, 0.12, 0.10, 2), prof)
    assert g["a_me"] > 0.0 and g["a_opp"] < 0.0
    KV.set_d_mode("curve")
    g2 = KV.grad_of((0.9, 0.6, 0.12, 0.10, 2), prof)
    assert g2["a_me"] == 0.0 and g2["a_opp"] == 0.0


def test_curve_scaled_still_needs_the_harm_curve(_prof_th):
    KV.set_d_mode("curve_scaled")
    with pytest.raises(ValueError):
        KV.d_of((0.9, 0.6, 0.12, 0.10, 2))


def test_profile_th_for_follows_the_cross_convention():
    """**分母も `profile_for` と同じ規約**（`cross`＝測る記録と別のセット・§0.1 条件 1）。"""
    assert CB.profile_th_for(None, "real")[1] == pytest.approx(0.0773)
    assert CB.profile_th_for(None, "syn")[1] == pytest.approx(0.0815)
    assert CB.profile_th_for(None, "real")[0] == 0.0          # 最初の自席ターンは打てない規則


def test_scale_clamp_is_a_diagnostic_and_off_by_default(_prof_th):
    """**倍率の締めは診断用**（定数を 2 つ置くので理論には入れられない）。既定は無し。"""
    assert KV.SCALE_CLAMP is None
    try:
        KV.set_scale_clamp((0.5, 2.0))
        assert KV.profile_scale(9.9, 3) == pytest.approx(2.0)      # 上で止まる
        assert KV.profile_scale(0.001, 3) == pytest.approx(0.5)    # 下で止まる
        assert KV.profile_scale(0.19, 3) == pytest.approx(1.0)     # 帯の中は素通し
    finally:
        KV.set_scale_clamp(None)
    assert KV.profile_scale(9.9, 3) > 2.0                          # 外せば元に戻る


# --------------------------------------------------------------------------- 8. T127: 積み上がる歩きの読み
@pytest.fixture
def _shape():
    old = dict(KV.RATE_SHAPE)
    yield
    KV.RATE_SHAPE.update(old)


def test_theory_is_a_fourth_reading_with_all_four_axes_live():
    assert "theory" in KV.D_MODES
    assert KV.LIVE_AXES["theory"] == KV.AXES


def test_rate_shape_is_normalised_to_one(_shape):
    KV.set_rate_shape((1.0, 2.0, 1.0, 0.0), (0.0, 0.0, 0.0, 0.0))
    assert sum(KV.RATE_SHAPE["me"]) == pytest.approx(1.0)
    assert KV.RATE_SHAPE["me"] == pytest.approx((0.25, 0.5, 0.25, 0.0))
    assert KV.RATE_SHAPE["opp"] == (0.0, 1.0, 0.0, 0.0)      # 全部 0 なら盤面に倒す


def test_theory_uses_no_table_at_all(_shape):
    """**表を一切使わない**（加速は状態から出る）＝輪郭を渡さなくても解ける。"""
    KV.set_d_mode("theory")
    KV.set_rate_shape((0.3, 0.4, 0.2, 0.1), (0.3, 0.4, 0.2, 0.1))
    assert math.isfinite(KV.d_of((0.9, 0.6, 0.12, 0.10, 2)))   # prof を渡さない


def test_theory_lets_the_rate_move_the_clock(_shape):
    KV.set_d_mode("theory")
    KV.set_rate_shape((0.3, 0.4, 0.2, 0.1), (0.3, 0.4, 0.2, 0.1))
    st = (0.9, 0.6, 0.12, 0.10, 2)
    assert KV.d_of((0.9, 0.6, 0.24, 0.10, 2)) > KV.d_of(st)


def test_theory_is_exactly_invariant_to_a_currency_rescale(_shape):
    """**P5（単位の付け替え）は設計から出る**——`Θ` と `A` を同じだけ倍にすれば `τ` は不変。

    **表を使わないので単位は `A` だけが持つ**。これが `curve` にできないこと（T126 の患部）。"""
    KV.set_d_mode("theory")
    KV.set_rate_shape((0.3, 0.4, 0.2, 0.1), (0.3, 0.4, 0.2, 0.1))
    st = (0.9, 0.6, 0.12, 0.10, 2)
    for c in (0.5, 3.0, 7.0):
        assert KV.d_of((st[0] * c, st[1] * c, st[2] * c, st[3] * c, st[4])) == pytest.approx(KV.d_of(st))


def test_acceleration_and_p7_cannot_both_hold(_shape):
    """**加速する時計は `A` について 1 次同次になれない**（T127 で踏んだ・P7 は要求として過剰だった）。

    `R_k = A·f(k)` で `f` が増えるなら `Σ_{k≤τ} f` を `1/c` にする `τ` は `τ/c` より大きい。
    **一定の形なら厳密に通り、加速する形なら通らない**——これを両方固定する。"""
    KV.set_d_mode("theory")
    st = (0.9, 0.6, 0.12, 0.10, 2)
    fast = (0.9, 0.6, 0.12 * 1.3, 0.10 * 1.3, 2)
    KV.set_rate_shape((0.0, 1.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0))     # 一定（加速なし）
    assert KV.d_of(fast) == pytest.approx(KV.d_of(st) / 1.3)
    KV.set_rate_shape((0.3, 0.4, 0.2, 0.1), (0.3, 0.4, 0.2, 0.1))     # 加速あり
    assert KV.d_of(fast) > KV.d_of(st) / 1.3


def test_a_constant_shape_degenerates_to_the_clock_reading(_shape):
    """形が全部盤面なら `tau_grow` は `Θ/A` に退化する（既定がそれ＝読みの連続性）。"""
    KV.set_rate_shape((0.0, 1.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0))
    KV.set_d_mode("theory")
    a = KV.d_of((0.9, 0.6, 0.12, 0.10, 4))
    KV.set_d_mode("clock")
    assert a == pytest.approx(KV.d_of((0.9, 0.6, 0.12, 0.10, 4)), rel=1e-9)


def test_theory_gives_the_rate_axes_a_gradient(_shape):
    KV.set_d_mode("theory")
    KV.set_rate_shape((0.3, 0.4, 0.2, 0.1), (0.3, 0.4, 0.2, 0.1))
    g = KV.grad_of((0.9, 0.6, 0.12, 0.10, 2))
    assert g["a_me"] > 0.0 and g["a_opp"] < 0.0
    assert g["th_me"] > 0.0 and g["th_opp"] < 0.0
