"""無差別点 `Θ` を盤面から出す器の算術（`tests/scripts/theta_price.py`）。

基盤健全性（`cpu_infra`）。要は 4 つ:

1. **`Θ` は順序統計量である**——「守る G 本のうち最も高いものの費用」＝
   `c(x)` を安い順に並べた **G 番目**。**平均で代用すると必ず過小に出る**
   （`game_theory.md` §8 が `c̄_mean < Θ` を明記している）。
2. **`G = 0` なら `Θ` は無い**（領域 1＝全部受けても死なない＝制約が無い＝シャドー価格も無い）。
   0 を入れて平均に混ぜると `Θ` が沈む。
3. **飽和点は `c(x) ≥ Θ` になる最小の x**。`Θ < 1` なら `c(0) = 1` が既に超えるので **`x* = 0`**
   ＝超過パワーを積む価値が無い。
4. **`λ` を通らない**のが本器の存在理由（`Θ = λ/μ − 1 − τ_value` の右辺を使わずに左辺を出す）。
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

import theta_price as T  # noqa: E402
import theory_order as TO  # noqa: E402


def _tk(my_leader=5000, opp_leader=5000, opp_chars=(), my_blockers=0):
    """枠 0 自L・1 相L・2〜6 自場・7〜11 相場（列は**生のトークン**＝`theory_order` の規約）。

    2026-09-14 に `race_state` の 5 列版から生のトークンへ移した——**`Θ` の材料は
    `theory_order` に 1 つだけ置く**ことにしたので、器の側も同じ列を読む。
    """
    tk = np.zeros((22, 24), np.float32)
    tk[0, TO.S_POWER] = my_leader / 1e4
    tk[1, TO.S_POWER] = opp_leader / 1e4
    for j, p in enumerate(opp_chars):
        tk[7 + j, TO.S_POWER] = p / 1e4
        tk[7 + j, TO.S_IS_CHAR] = 1.0
    for j in range(my_blockers):
        tk[2 + j, TO.S_IS_CHAR] = 1.0
        tk[2 + j, TO.S_IS_BLOCKER] = 1.0
    return tk


def _sc(life=4, don=6, turn=6):
    sc = np.zeros(14, np.float32)
    sc[T.SC_MY_LIFE] = life
    sc[T.SC_MY_DON] = don
    sc[T.SC_TURN] = turn
    return sc


def test_theta_is_the_g_th_cheapest_not_the_mean():
    """**`Θ` は順序統計量**。平均で代用すると過小に出る（§8 の `c̄_mean < Θ`）。"""
    # 相手 3 体＋リーダー＝4 攻撃・ライフ 1・ブロッカー 0 → G = 3
    tk = _tk(my_leader=5000, opp_leader=5000, opp_chars=(6000, 8000, 10000))
    r = T.window(_sc(life=1), tk)
    assert r["n_attacks"] == 4 and r["G"] == 3
    # x = 0, 1000, 3000, 5000 → c = 1.00, 1.00, 2.25, 3.63（安い順）
    # G=3 番目は 2.25
    assert r["theta"] == pytest.approx(2.25)
    # 守る 3 本の平均は (1.00+1.00+2.25)/3 = 1.4167 ＜ Θ
    assert r["c_mean_blocked"] == pytest.approx(1.4167, abs=1e-3)
    assert r["c_mean_blocked"] < r["theta"]
    # 全部の平均は 4 本の平均
    assert r["c_mean_all"] == pytest.approx((1.0 + 1.0 + 2.25 + 3.63) / 4, abs=1e-3)


def test_region_one_has_no_shadow_price():
    """**`G = 0` なら `Θ` は無い**（全部受けても死なない＝制約が無い）。0 を混ぜると沈む。"""
    tk = _tk(opp_chars=(6000,))                   # 2 攻撃
    r = T.window(_sc(life=5), tk)                 # ライフ 5 ＞ 2 攻撃
    assert r["G"] == 0 and r["theta"] is None
    assert "c_mean_blocked" not in r and "x_star" not in r
    # 集計でも外れる（region1_share に出る）
    recs = [dict(r, seed=g) for g in range(4)]
    out = T.summarise(recs)
    assert out["constrained"] == 0 and out["region1_share"] == 1.0
    assert "theta" not in out


def test_blockers_reduce_the_number_you_must_counter():
    """ブロッカーは**手札を使わずに 1 回止める**ので `G` から引く。"""
    tk0 = _tk(opp_chars=(6000, 8000), my_blockers=0)
    tk1 = _tk(opp_chars=(6000, 8000), my_blockers=1)
    assert T.window(_sc(life=1), tk0)["G"] == 2
    assert T.window(_sc(life=1), tk1)["G"] == 1


def test_the_implied_tau_is_the_identity_residual():
    """`τ_value = λ/μ − 1 − Θ`——**`Θ` を測れば残差で `τ_value` が出る**。"""
    tk = _tk(my_leader=5000, opp_leader=5000, opp_chars=(6000,))
    recs = [dict(T.window(_sc(life=1), tk), seed=g) for g in range(6)]
    out = T.summarise(recs, side="opp")
    th = out["theta"]["mean"]
    assert out["tau_implied"]["from_measured_lam"] == pytest.approx(
        T.LAM_OVER_MU_MINUS_1["opp"] - th, abs=1e-3)
    # 自席の価格を使うと `λ/μ − 1` が大きいので τ も大きく出る
    out2 = T.summarise(recs, side="own")
    assert (out2["tau_implied"]["from_measured_lam"]
            > out["tau_implied"]["from_measured_lam"])


def test_theta_does_not_use_lambda_at_all():
    """**本器の存在理由**——`Θ` は盤面だけから出る（`λ`・`μ`・`τ` を通らない）。

    2026-09-14 に `λ` の補正から `Θ ≈ 0.67` を導いたが、`Θ` は**比 `λ/μ`** に依るので
    `λ` の水準だけ直すのは誤りだった。本器は右辺を使わずに左辺を出すので、その種の誤りが起きない。
    """
    tk = _tk(my_leader=5000, opp_leader=5000, opp_chars=(6000, 8000))
    r = T.window(_sc(life=1), tk)
    # 同じ盤面なら、どんな λ／μ を仮定しても `theta` は変わらない
    assert r["theta"] == pytest.approx(T.window(_sc(life=1), tk)["theta"])
    # `window` は価格の定数を 1 つも参照しない（引数に無い）
    import inspect
    src = inspect.getsource(T.window)
    for name in ("LAM", "MU", "tau", "TAU"):
        assert name not in src


def test_the_measured_don_share_is_the_accounting_one():
    """付与に回るドンの割合は**実測**（2.94 / 6.29）。全部回す想定は現実より強い。"""
    assert T.DON_SHARE == pytest.approx(2.94 / 6.29, abs=0.01)
