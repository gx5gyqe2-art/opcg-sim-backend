"""`bias_budget.py`（T112・終局の偏りの加法分解）の算術と不変量を固める。

本器の値は**当てはめではなく恒等式**なので、押さえるのは 3 つ:

1. `res == res_rate + res_target` が**行ごとに厳密**（浮動小数の誤差だけ）。
2. `turn_harm`／`theta_check`／`rows_out` の**突き合わせが鍵（`g`・`who`・`j`）で行われ、
   食い違えば黙らずに落ちる**（並び順に頼っていない）。
3. 判断行の母数では**負けた席の行が勝った席の行より厳密に +1** ずれる
   （同じ `τ` を 1 ターン短い物差しで測るため）＝台帳の偏りに乗っている物差しのぶん。
"""
import os
import sys

import pytest

pytestmark = pytest.mark.cpu_infra

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bootstrap  # noqa: F401,E402

_SCRIPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import bias_budget as BB  # noqa: E402
import crossing_bridge as CB  # noqa: E402


def _harm(g, who, j, t_left, harm, won=True, slope=0.1):
    return {"g": g, "who": who, "j": j, "t_left": t_left, "harm": harm,
            "won": won, "slope_theory": slope, "priced": harm}


def _check(g, who, j, t_left, theta, tau, need):
    return {"g": g, "who": who, "j": j, "t_left": t_left, "theta": theta,
            "tau": tau, "need": need, "th_life": theta, "th_hand": 0.0, "th_body": 0.0}


def _one_game(harms, theta=1.0, taus=None, g=1, who=0):
    """1 局ぶんの `turn_harm`／`theta_check` を作る（`harms` は勝った席の自席ターンの損害）。"""
    n = len(harms)
    taus = taus if taus is not None else [5.0] * n
    th = [_harm(g, who, j, n - j, h) for j, h in enumerate(harms)]
    tc = [_check(g, who, j, n - j, theta, taus[j], sum(harms[j:])) for j in range(n)]
    return th, tc


# ---------------------------------------------------------------- tau_of_sequence

def test_the_walk_reaches_the_target_with_the_same_fraction_rule():
    """**端数はそのターンの損害で按分する**（`tau_grow` と同じ規則）。"""
    # 0.4 ずつ 3 回: 2 ターンで 0.8、3 ターン目の 0.5 ぶんで 1.0 に届く
    assert BB.tau_of_sequence(1.0, [0.4, 0.4, 0.4], 0.4) == pytest.approx(2.5)
    # 1 ターン目で越えるなら端数だけ
    assert BB.tau_of_sequence(0.5, [1.0], 1.0) == pytest.approx(0.5)
    # ちょうど届くターンは整数
    assert BB.tau_of_sequence(0.8, [0.4, 0.4], 0.4) == pytest.approx(2.0)


def test_the_recorded_sequence_runs_out_and_the_tail_takes_over():
    """**記録は勝った席の最後のターンで終わる**ので、足りないぶんは平均で延長する。"""
    # 0.5（実測）→ 0.25（平均）で埋める: 2 ターンで 0.75、3 ターン目でちょうど 1.0
    assert BB.tau_of_sequence(1.0, [0.5], 0.25) == pytest.approx(3.0)
    # 延長の材料が 0 なら永遠に届かない＝打ち切り
    assert BB.tau_of_sequence(1.0, [0.1], 0.0) == pytest.approx(BB.CAP)
    # 的が 0 なら 0 ターン（もう届いている）
    assert BB.tau_of_sequence(0.0, [0.5], 0.5) == 0.0


def test_the_cap_is_the_same_one_the_bridge_uses():
    """打ち切りは**橋と同じ値**（別に持たない）。"""
    assert BB.CAP == CB.RACE_CAP


# ---------------------------------------------------------------- 恒等式

def test_the_split_is_an_identity_row_by_row():
    """**偏り = A の軌跡の誤り + 的の差**が行ごとに厳密に成り立つ。"""
    th, tc = _one_game([0.3, 0.5, 0.4], theta=1.0, taus=[6.0, 4.0, 2.0])
    rows = BB.decompose_turns(BB.winner_blocks(th, tc))
    assert len(rows) == 3
    for r in rows:
        assert r["res"] == pytest.approx(r["res_rate"] + r["res_target"], abs=1e-12)
    out = BB.summarise_budget(rows)
    assert out["identity_max_abs_error"] < 1e-9
    assert out["n"] == 3


def test_the_target_term_is_the_theta_over_need_gap_in_turns():
    """**的の項は「`Θ` が実際の総量と同じなら 0」**——同じにした局で消えることで確かめる。"""
    # 損害 0.5×4、`Θ` = 2.0 = 要（j=0 の行）。実際の残りターンは 4。
    th, tc = _one_game([0.5, 0.5, 0.5, 0.5], theta=2.0, taus=[4.0] * 4)
    rows = BB.decompose_turns(BB.winner_blocks(th, tc))
    first = rows[0]
    assert first["need"] == pytest.approx(2.0)
    assert first["tau_harm"] == pytest.approx(4.0)       # 実際の損害の列は 4 ターンで届く
    assert first["res_target"] == pytest.approx(0.0)     # 的と要が同じなら 0
    assert first["res_rate"] == pytest.approx(0.0)       # τ も 4 なので軌跡の誤りも 0


def test_a_theta_bigger_than_the_need_shows_up_in_the_target_term():
    """**`Θ` が要より大きければ的の項が正**（時間の単位で）。"""
    th, tc = _one_game([0.5, 0.5], theta=1.5, taus=[2.0, 2.0])
    rows = BB.decompose_turns(BB.winner_blocks(th, tc))
    # 要（j=0）は 1.0 なのに `Θ` は 1.5 ＝ 実際の列では 3 ターン必要（延長 0.5）
    assert rows[0]["need"] == pytest.approx(1.0)
    assert rows[0]["tau_harm"] == pytest.approx(3.0)
    assert rows[0]["res_target"] == pytest.approx(1.0)   # 3 − 2
    assert rows[0]["res_rate"] == pytest.approx(-1.0)    # τ=2 − 3


# ---------------------------------------------------------------- 突き合わせの不変量

def test_the_pairing_is_by_key_not_by_order():
    """**並べ替えても同じ組**（`g`・`who`・`j` で突き合わせている）。"""
    th, tc = _one_game([0.3, 0.5, 0.4], theta=1.0)
    a = BB.decompose_turns(BB.winner_blocks(th, tc))
    b = BB.decompose_turns(BB.winner_blocks(list(reversed(th)), list(reversed(tc))))
    assert [r["res"] for r in a] == [r["res"] for r in b]


def test_a_mismatched_pairing_fails_loudly():
    """**残りターンが食い違ったら落ちる**（黙って別の行を組にしない）。"""
    th, tc = _one_game([0.3, 0.5], theta=1.0)
    tc[1]["t_left"] = 7
    with pytest.raises(AssertionError):
        BB.winner_blocks(th, tc)


def test_a_missing_row_fails_loudly():
    """**本数が違えば落ちる**（勝った席の自席ターンと検算の行は 1 対 1）。"""
    th, tc = _one_game([0.3, 0.5], theta=1.0)
    with pytest.raises(AssertionError):
        BB.winner_blocks(th[:1], tc)


def test_the_losing_seats_row_is_exactly_one_turn_shorter():
    """**同じターンを両側から見た行の差は厳密に 1**——`τ` も `Θ` も同じで、違うのは物差しだけ。

    `crossing_bridge.summarise` は負けた席の行で `t_opp_act`（そのターンを除いた残り）を読む。
    台帳の偏りにはこの +1 が半分の行に乗っている。"""
    th, tc = _one_game([0.5, 0.5, 0.5], theta=1.0, taus=[3.0, 2.0, 1.0], g=1, who=0)
    # 勝った席（who=0）の j=1 を、自分の行と相手（負けた席 who=1）の行の両方から見る
    rows_out = [
        {"g": 1, "who": 0, "won": True, "j_me": 1, "j_opp": 0,
         "t_me_act": 2.0, "t_opp_act": 0.0, "theta_me": 1.0, "theta_opp": 9.0,
         "tau_me_theory": 2.0, "tau_opp_theory": 9.0},
        {"g": 1, "who": 1, "won": False, "j_me": 0, "j_opp": 1,
         "t_me_act": 0.0, "t_opp_act": 1.0, "theta_me": 9.0, "theta_opp": 1.0,
         "tau_me_theory": 9.0, "tau_opp_theory": 2.0},
    ]
    rows = BB.decompose_rows(rows_out, BB.winner_blocks(th, tc))
    assert len(rows) == 2
    own = [r for r in rows if r["own"]][0]
    opp = [r for r in rows if not r["own"]][0]
    assert (own["g"], own["who"], own["j"]) == (opp["g"], opp["who"], opp["j"])
    assert opp["res"] - own["res"] == pytest.approx(1.0)
    # 分解のうち**動くのは的の項だけ**（`τ` も `τ_harm` も同じ行を見ているので軌跡の誤りは同じ）
    assert opp["res_rate"] == pytest.approx(own["res_rate"])
    assert opp["res_target"] - own["res_target"] == pytest.approx(1.0)
    out = BB.summarise_budget(rows)
    assert out["by_side"]["paired_n"] == 1
    assert out["by_side"]["paired_offset_mean"] == pytest.approx(1.0)
    assert out["by_side"]["paired_offset_max_abs_error"] < 1e-9


def test_two_paths_that_disagree_about_tau_fail_loudly():
    """**`τ` は 1 本の式から来る**（T101）＝2 つの出口で違えば落ちる。"""
    th, tc = _one_game([0.5, 0.5], theta=1.0, taus=[2.0, 1.0])
    rows_out = [{"g": 1, "who": 0, "won": True, "j_me": 0, "j_opp": 0,
                 "t_me_act": 2.0, "t_opp_act": 1.0, "theta_me": 1.0, "theta_opp": 1.0,
                 "tau_me_theory": 7.0, "tau_opp_theory": 7.0}]      # 検算の側は 2.0
    with pytest.raises(AssertionError):
        BB.decompose_rows(rows_out, BB.winner_blocks(th, tc))


# ---------------------------------------------------------------- 成り立つ条件

def test_the_decomposition_only_claims_to_be_exact_on_a_static_target():
    """**的が動く構成では降りる**（第 2 項の意味が変わるので恒等式として売らない）。
    **C-5c（2026-09-25）以降、出荷既定は的が動く**（`THETA_RETURN_MODE=untap`＝レスト中のブロッカーが
    2 段目から戻る）ので、既定のままでは降りる。恒等式として読むなら `--theta-return off` で回す。"""
    assert BB.static_target() is False       # 出荷既定（untap）
    old, old_ret = CB.RACE_MODE, CB.THETA_RETURN_MODE
    try:
        CB.set_theta_return_mode("off")
        assert BB.static_target() is True
        CB.set_race_mode("net")
        assert BB.static_target() is False
        with pytest.raises(SystemExit):
            BB.collect_budget(["/nonexistent"], 1)
    finally:
        CB.set_race_mode(old)
        CB.set_theta_return_mode(old_ret)


def test_every_mode_that_moves_the_target_is_named():
    """**的を動かす 4 つの切替**が条件に全部入っている（1 つ足したら落ちる）。"""
    base = {n: getattr(CB, n) for n in ("RACE_MODE", "THETA_HAND_PLACE", "THETA_RETURN_MODE", "RATE_DECAY_MODE")}
    CB.THETA_RETURN_MODE = "off"                 # 他の 3 つを測るために的を止めておく（既定は untap）
    for name, off, on in (("RACE_MODE", "static", "net"),
                          ("THETA_HAND_PLACE", "stock", "shield"),
                          ("THETA_RETURN_MODE", "off", "untap"),
                          ("RATE_DECAY_MODE", "off", "ko")):
        old = getattr(CB, name)
        try:
            setattr(CB, name, on)
            assert BB.static_target() is False, name
            setattr(CB, name, off)
            assert BB.static_target() is True, name
        finally:
            setattr(CB, name, old)
    for n, v in base.items():
        setattr(CB, n, v)
