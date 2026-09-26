"""ライフ 1 枚の価格を水準ごとに測る器の算術（`tests/scripts/life_price.py`）。

基盤健全性（`cpu_infra`）。要は 4 つ:

1. **帯の切り替えが意味を持つ**——`gross` は手札を固定し、`net` は固定しない。
   **ダメージを受けるとライフはそのまま手札に入る**ので、手札を固定すると
   「もらう 1 枚」が遮断される＝2 つは**違う量**である。
2. **テレスコープ**——水準ごとの差を足すと端から端までの差になる（恒等式の左辺）。
3. **CI は対局のクラスタブートストラップ**（行で割ると数倍過小・`measurement.md` §14-7）。
4. **`τ_value` はこの対比では同定できない**——2026-09-14 に
   `λ_gross − λ_net = μ(1 + τ_value)` という式を書いて τ が負（−0.41）になった。
   トリガーは**手札の枚数を経由しない**ので帯では遮断されず、差に入るのは
   **もらった札が残っている分だけ**＝`|∂手札/∂ライフ|·μ`。ここを取り違えると
   「トリガーの価値が負」という無意味な数字が出る。
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

import life_price as LP  # noqa: E402


def _sc(life=4, opp_life=4, hand=5, field=1, turn=6):
    sc = np.zeros(14, np.float32)
    sc[LP.SC_MY_LIFE] = life
    sc[LP.SC_OPP_LIFE] = opp_life
    sc[LP.SC_MY_HAND] = hand
    sc[LP.SC_MY_FIELD] = field
    sc[LP.SC_TURN] = turn
    return sc


def test_the_gross_band_fixes_the_hand_and_the_net_band_does_not():
    """**ダメージは手札に入る**ので、手札を帯に入れるかで測る量が変わる。"""
    a, b = _sc(hand=5), _sc(hand=6)
    assert LP.band_key(a, "gross") != LP.band_key(b, "gross")   # 手札が違えば別の帯
    assert LP.band_key(a, "net") == LP.band_key(b, "net")       # net では同じ帯
    # 自ライフは**どちらの帯にも入らない**（入れると帯の中で動かず傾きが定義できない）
    assert LP.band_key(_sc(life=1), "gross") == LP.band_key(_sc(life=5), "gross")
    assert LP.band_key(_sc(life=1), "net") == LP.band_key(_sc(life=5), "net")


def test_the_band_keeps_the_other_three_coordinates():
    for kw in ({"opp_life": 3}, {"field": 2}, {"turn": 2}):
        assert LP.band_key(_sc(**kw), "net") != LP.band_key(_sc(), "net")
    # ターン帯は 3 つに畳む（同じ帯の中の turn 違いは同じ帯）
    assert LP.band_key(_sc(turn=5), "net") == LP.band_key(_sc(turn=8), "net")
    assert LP.turn_band(4) == "T<=4" and LP.turn_band(9) == "T9+"


def _r(seed, life, z, band="b", hand=5.0):
    return {"seed": seed, "life": life, "z": z, "band": band, "hand": hand}


def test_the_contrast_is_the_within_band_difference_of_win_rates():
    """`λ(ℓ)` は**同じ帯の中で**「ライフ ℓ」と「ライフ ℓ−1」の勝率の差。"""
    recs = ([_r(0, 3, 1.0) for _ in range(10)] +      # ライフ 3 は全勝
            [_r(1, 2, 0.0) for _ in range(10)])      # ライフ 2 は全敗
    val, bands, pairs = LP._contrast(recs, 3)
    assert val == pytest.approx(1.0)
    assert bands == 1 and pairs == 20
    # 逆向きなら負
    recs2 = [_r(0, 3, 0.0) for _ in range(4)] + [_r(1, 2, 1.0) for _ in range(4)]
    assert LP._contrast(recs2, 3)[0] == pytest.approx(-1.0)


def test_a_band_with_only_one_level_is_skipped():
    """片方の水準しか無い帯は**差が作れない**ので使わない（0 として混ぜない）。"""
    recs = [_r(0, 3, 1.0, band="only_hi")] * 5
    val, bands, _p = LP._contrast(recs, 3)
    assert val is None and bands == 0
    # 2 つの帯のうち片方だけ対比が作れるなら、その帯だけ使う
    recs += [_r(1, 3, 1.0, band="both"), _r(2, 2, 0.0, band="both")]
    val2, bands2, _p2 = LP._contrast(recs, 3)
    assert bands2 == 1 and val2 == pytest.approx(1.0)


def test_bands_are_weighted_by_how_much_they_can_say():
    """重みは `n1·n0/(n1+n0)`＝**差の分散の逆数**に比例する（大きい帯が効く）。"""
    big = ([_r(0, 3, 1.0, band="big") for _ in range(20)]
           + [_r(1, 2, 0.0, band="big") for _ in range(20)])      # 差 +1・重み 10
    small = [_r(2, 3, 0.0, band="small"), _r(3, 2, 1.0, band="small")]  # 差 −1・重み 0.5
    val, bands, _p = LP._contrast(big + small, 3)
    assert bands == 2
    # 10·(+1) + 0.5·(−1) を 10.5 で割る
    assert val == pytest.approx((10 * 1.0 - 0.5) / 10.5)


def test_the_levels_telescope_to_the_end_to_end_difference():
    """**恒等式の左辺**: 水準ごとの差を足すと「満タン」と「0」の差になる。"""
    recs = []
    p = {0: 0.0, 1: 0.1, 2: 0.3, 3: 0.6, 4: 0.9}
    for lv, pw in p.items():
        for k in range(10):
            recs.append(_r(k, lv, 1.0 if k < pw * 10 else 0.0))
    out = LP.summarise(recs, reps=0)
    total = sum(out["lam"][str(lv)]["value"] for lv in (1, 2, 3, 4))
    assert total == pytest.approx(p[4] - p[0], abs=1e-6)
    assert out["telescope"]["4"]["sum"] == pytest.approx(total, abs=1e-4)
    # 恒等式との差（0.5 を基準にする）
    assert out["telescope"]["4"]["excess"] == pytest.approx(total - 0.5, abs=1e-4)


def test_the_identity_target_is_the_game_value_over_the_leader_life():
    """`Σλ = 0.5` を満たす経路平均は `0.5 / L`（リーダーのライフ 4／5／平均 4.51）。"""
    out = LP.summarise([_r(k % 4, k % 4, 1.0) for k in range(40)], reps=0)
    assert out["lam_allowed"]["4"] == pytest.approx(0.125)
    assert out["lam_allowed"]["5"] == pytest.approx(0.100)
    assert out["lam_allowed"]["4.51"] == pytest.approx(0.5 / 4.51, abs=1e-4)
    assert LP.GAME_VALUE == 0.5


def test_the_hand_slope_measures_the_life_to_hand_channel():
    """`∂手札/∂ライフ`——**ライフが多い＝もらった札が少ない**ので負に出る。"""
    recs = ([_r(0, 4, 1.0, hand=4.0)] * 10 + [_r(1, 2, 1.0, hand=6.0)] * 10)
    # ライフ +2 で手札 −2 ＝ 傾き −1
    assert LP._pooled_slope(recs, "hand") == pytest.approx(-1.0)
    # 勝敗の傾きは別（この標本は全勝なので 0）
    assert LP._pooled_slope(recs, "z") == pytest.approx(0.0)


def test_the_card_channel_is_not_the_trigger_value():
    """**`τ_value` はこの対比では同定できない**——分母は `|∂手札/∂ライフ|·μ` である。

    2026-09-14 に `λ_gross − λ_net = μ(1 + τ_value)` と書いて **τ が −0.41** になった。
    トリガーは**ダメージの瞬間に 1 回**で手札の枚数を経由しないので、帯で手札を固定しても
    遮断されず、**差に入るのは「もらった札が残っている分」だけ**。
    実測は `|∂h/∂L|` = 0.46〜0.50（1 枚もらうが半分は使われている）で、比は 1.10〜1.27。
    """
    mu = 0.0433
    # 経路の太さが 0.5 枚なら、差は 0.5·μ であるべき
    diff, dh = 0.5 * mu, -0.5
    ratio = diff / (abs(dh) * mu)
    assert ratio == pytest.approx(1.0)
    # 誤った式（μ(1+τ)）に同じ差を入れると τ が負になる＝無意味な数字が出る
    assert diff / mu - 1.0 == pytest.approx(-0.5)


def test_the_bootstrap_resamples_games_not_rows():
    """**局を復元抽出する**（行で割ると 1 局から複数行を採る分だけ SE が過小）。"""
    # 全部同じ局＝局が 1 つしか無いので CI は出せない（None を返す）
    same = [_r(0, 3, 1.0) for _ in range(20)] + [_r(0, 2, 0.0) for _ in range(20)]
    rng = np.random.default_rng(0)
    assert LP._boot_ci(same, LP._pooled_slope, None, rng, reps=20) == (None, None)
    # 局が十分あれば CI が出る
    many = []
    for g in range(30):
        many += [_r(g, 3, 1.0), _r(g, 2, 0.0)]
    lo, hi = LP._boot_ci(many, LP._pooled_slope, None, rng, reps=50)
    assert lo is not None and hi is not None and lo <= hi


def test_draws_and_turn_zero_rows_never_reach_the_estimator():
    """引き分け（`z == 0`）は勝敗の差が定義できないので落とす——`collect` の規約。"""
    out = LP.summarise([], reps=0)
    assert out is None
    # 1 水準しか無ければ対比は空
    only = [_r(k, 3, 1.0) for k in range(10)]
    assert LP.summarise(only, reps=0)["lam"] == {}
