"""`theory_bridge.py`（理論と勝敗を繋ぐ橋・T28）の符号と「できない」の扱いを固める。

**符号が 1 つ狂うと結論が反転する器**なので、向きを値で押さえる。
守り側の**「守れなかった行を誤りと数えない」**は `measurement.md` §1 そのもの。
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

import theory_bridge as B  # noqa: E402
import theory_order as T  # noqa: E402


def _tok(opp_lead=5000, my_lead=5000, blocker=False):
    tok = np.zeros((22, 24), np.float32)
    tok[0, T.S_POWER] = my_lead / 1e4
    tok[1, T.S_POWER] = opp_lead / 1e4
    if blocker:
        tok[2, T.S_POWER] = 0.3
        tok[2, T.S_IS_CHAR] = 1.0
        tok[2, B.GA.S_BLOCKER] = 1.0
    return tok


def _sc(don=5):
    sc = np.zeros(127, np.float32)
    sc[T.SC_MY_DON] = don
    return sc


def test_a_step_is_never_positive():
    """`s_t` は「最善からの取りこぼし」なので **0 以下**。正が出たら向きが逆。"""
    tok, sc = _tok(opp_lead=8000), _sc()
    for played in ("take", "guard"):
        got = B.guard_step(tok, sc, played, free=9000.0, paid=[])
        assert got["s"] <= 0.0


def test_doing_what_the_theory_advises_costs_nothing():
    """助言どおりに打った行は **0**——減点は「違う手を打った」ときだけ。"""
    tok, sc = _tok(opp_lead=6000), _sc()          # 超過 1000＝c(1000)=1.00 < Θ=1.15 → 守る
    got = B.guard_step(tok, sc, "guard", free=9000.0, paid=[])
    assert got["theory_says"] == "guard"
    assert got["s"] == pytest.approx(0.0)


def test_taking_when_the_theory_says_guard_is_penalised():
    """安く守れたのに受けた＝取りこぼし。**`Θ·μ − c(x)·μ` ちょうど**になる。"""
    tok, sc = _tok(opp_lead=6000), _sc()
    got = B.guard_step(tok, sc, "take", free=9000.0, paid=[])
    expect = -(T.THETA * T.MU - T.c_of(1000.0) * T.MU)
    assert got["s"] == pytest.approx(expect)
    assert got["s"] < 0.0


def test_a_guard_you_could_not_afford_is_not_a_mistake():
    """**「〜しない」ではなく「〜できない」**（`measurement.md` §1）。

    カウンターを持っていなければ受けるしかない。そこを減点すると
    **能力の限界を選択の誤りとして数える**ことになる。
    """
    tok, sc = _tok(opp_lead=6000), _sc(don=0)
    poor = B.guard_step(tok, sc, "take", free=0.0, paid=[])     # 守る手段が無い
    rich = B.guard_step(tok, sc, "take", free=9000.0, paid=[])  # 守れたのに受けた
    assert poor["can_guard"] is False and poor["s"] == pytest.approx(0.0)
    assert rich["can_guard"] is True and rich["s"] < 0.0


def test_a_blocker_makes_the_guard_free_of_don():
    """ブロッカーはレストするだけ＝**ドンを使わない**ので予算に関係しない。"""
    tok, sc = _tok(opp_lead=9000, blocker=True), _sc(don=0)
    got = B.guard_step(tok, sc, "take", free=0.0, paid=[])
    assert got["can_guard"] is True


def test_a_huge_attack_is_taken_not_guarded():
    """**`c(x) > Θ` なら受けるのが正しい**——高すぎる攻撃に払うのは損。"""
    tok, sc = _tok(opp_lead=14000), _sc()
    got = B.guard_step(tok, sc, "take", free=20000.0, paid=[])
    assert got["theory_says"] == "take"
    assert got["s"] == pytest.approx(0.0)


def test_no_incoming_attack_is_not_a_decision():
    assert B.guard_step(_tok(opp_lead=1000), _sc(), "take", free=0.0, paid=[]) is None


def _seat(s_atk=0.0, s_grd=0.0, n_atk=5, n_grd=5, z=1.0, silent=0):
    return {"z": z, "s_atk": s_atk, "s_grd": s_grd, "n_atk": n_atk, "n_grd": n_grd,
            "n_silent": silent, "v0": [0.1]}


def test_the_pairing_is_my_loss_minus_the_opponents():
    """`ΔS > 0` は**自席の方が取りこぼしが少ない**＝理論に近い打ち方をした、の向き。"""
    per = {(1, 0): _seat(s_atk=-0.1, z=1.0), (1, 1): _seat(s_atk=-0.5, z=0.0)}
    got = B.pair_games(per)
    assert len(got) == 1 and got[0]["dS"] == pytest.approx(0.4)
    assert got[0]["z"] == 1.0


def test_the_per_row_version_divides_each_seat_by_its_own_count():
    """**手数で正規化した版**——先手は手番が 1 つ多いので生の和は席順を拾いうる。"""
    per = {(1, 0): _seat(s_atk=-1.0, n_atk=10, n_grd=0),
           (1, 1): _seat(s_atk=-1.0, n_atk=5, n_grd=0, z=0.0)}
    got = B.pair_games(per)[0]
    assert got["dS"] == pytest.approx(0.0)              # 生の和では差が無い
    assert got["dS_per_row"] == pytest.approx(-0.1 + 0.2)   # 1 手あたりでは差が出る
    assert got["dn"] == 5


def test_the_verdict_reads_the_normalised_slope():
    """**判定は正規化した版**で出す（生の和は手数の差に汚染されうる）。"""
    rng = np.random.default_rng(0)
    pairs = []
    for i in range(200):
        d = float(rng.normal())
        pairs.append({"seed": i, "z": 1.0 if d > 0 else 0.0, "dS": d, "dS_per_row": d,
                      "dS_atk": d, "dS_grd": d, "n": 10, "dn": 0, "silent": 0, "v0": 0.1})
    assert B.summarise(pairs, reps=50)["verdict"] == "bridge_holds"
    for p in pairs:                                     # 向きだけ反転させる
        p["dS_per_row"] = -p["dS_per_row"]
    assert B.summarise(pairs, reps=50)["verdict"] == "bridge_inverted"


def _rec():
    return {"seed": 1, "who": 0, "z": 1.0, "s_atk": 0.0, "s_grd": 0.0,
            "n_atk": 0, "n_grd": 0, "n_silent": 0, "v0": [], "band": {}}


def test_a_row_is_added_to_its_own_band():
    """**T28-b の要点**——行ごとに帯を決めてから足す。

    局ごとに足してから帯で切ると、**決着後の雑さが接戦帯に混ざる**（T28 で踏んだ）。
    """
    r = _rec()
    B._add(r, "close", -0.1, "atk")
    B._add(r, "decided", -0.9, "atk")
    assert r["band"]["close"]["s"] == pytest.approx(-0.1)
    assert r["band"]["decided"]["s"] == pytest.approx(-0.9)
    assert r["band"]["close"]["n"] == 1 and r["band"]["decided"]["n"] == 1
    assert r["s_atk"] == pytest.approx(-1.0)          # 全帯の合計も保つ


def test_the_two_sides_are_kept_apart_inside_a_band():
    """攻めと守りは帯の中でも分ける——**どちらの半分が効いているか**を見るため。"""
    r = _rec()
    B._add(r, "close", -0.2, "atk")
    B._add(r, "close", -0.5, "grd")
    b = r["band"]["close"]
    assert b["s_atk"] == pytest.approx(-0.2) and b["s_grd"] == pytest.approx(-0.5)
    assert b["n"] == 2


def test_pairing_within_a_band_needs_both_seats_present_there():
    """**片方の席にその帯の行が無い局は使えない**（差が定義できない）。"""
    a, b = _rec(), _rec()
    b["who"] = 1; b["z"] = 0.0
    B._add(a, "close", -0.1, "atk")
    per = {(1, 0): a, (1, 1): b}
    assert B.pair_by_band(per, "close") == []       # 席 1 に close の行が無い
    B._add(b, "close", -0.5, "atk")
    got = B.pair_by_band(per, "close")
    assert len(got) == 1 and got[0]["dS_per_row"] == pytest.approx(0.4)


def test_the_two_silent_modes_differ_in_the_denominator():
    """**P2 の暫定値**——無言の行を母数に入れるかどうかで 1 手あたりの値が変わる。

    `zero` は「取りこぼし 0」として母数に入れる（薄まる）・`exclude` は入れない。
    **実測でこの選択は結論を動かした**ので、テストで違いを固定する。
    """
    z = _rec()
    B._add(z, "close", -0.4, "atk")
    B._add(z, "close", 0.0, "atk")                  # zero が足す無言の行
    e = _rec()
    B._add(e, "close", -0.4, "atk")                 # exclude は足さない
    assert z["band"]["close"]["s"] / z["band"]["close"]["n"] == pytest.approx(-0.2)
    assert e["band"]["close"]["s"] / e["band"]["close"]["n"] == pytest.approx(-0.4)


def test_a_blocker_row_is_always_comfortably_affordable():
    """**T28-c**——ブロッカーはレストするだけでドンを使わないので、余裕は無限。"""
    got = B.guard_step(_tok(opp_lead=14000, blocker=True), _sc(don=0), "take",
                       free=0.0, paid=[])
    assert got["margin"] == float("inf") and got["comfortable"] is True


def test_the_margin_is_how_much_the_guard_overshoots_the_attack():
    """余裕＝**守る力 − 来る攻撃**。ここが小さい行を減点すると**貧しい席を罰する**。"""
    tok, sc = _tok(opp_lead=8000), _sc()          # 超過 3000
    got = B.guard_step(tok, sc, "take", free=5000.0, paid=[])
    assert got["margin"] == pytest.approx(2000.0)
    assert got["comfortable"] is True             # 既定の閾値 2000 にちょうど届く
    tight = B.guard_step(tok, sc, "take", free=3500.0, paid=[])
    assert tight["can_guard"] is True and tight["comfortable"] is False


def test_the_comfort_threshold_is_a_provisional_value_you_can_sweep():
    """**閾値は暫定値**なので振れる（§0.4 の規則 2＝感度を付ける）。"""
    tok, sc = _tok(opp_lead=8000), _sc()
    got = B.guard_step(tok, sc, "take", free=5000.0, paid=[], margin_comfort=6000.0)
    assert got["can_guard"] is True and got["comfortable"] is False


def test_a_row_that_could_not_be_guarded_is_never_comfortable():
    got = B.guard_step(_tok(opp_lead=9000), _sc(don=0), "take", free=0.0, paid=[])
    assert got["can_guard"] is False and got["comfortable"] is False
