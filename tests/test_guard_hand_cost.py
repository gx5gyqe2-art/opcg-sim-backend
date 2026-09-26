"""**G-2（2026-09-25・ユーザ決定「これで行きましょう」）**: 守りの判断（`theory_bridge.guard_step`）の 2 つの直し。

1. **守れたかを規則どおりに判定する**（`GUARD_AFFORD_MODE`・既定 `rule`）——同じパワーは命中するので、超過 `x` を止めるには
   カウンター合計が `x + 1000` 以上要る。旧 `lenient`（合計 ≥ `x`）は超過 0 ならカウンター 0 枚でも「守れた」にしていた。
2. **守る費用をこの手札で実際に失う価値で測る切替**（`GUARD_S_COST_MODE`・既定 `curve`＝従来の `c(x)·μ`・`hand` が新形）——
   止める札の組 `S` のうち `V(手札) − V(手札 − S)` が一番小さい組の減り。`V` は出す計画（T66）＋ **これから来る**
   相手ターンの守る備え（T67 と同じ目的を**厳密な最大**で解いたもの・今の窓から **1 ラウンド割り引く**）を次の自席ターンの時点で読んだもの。

**G-2 の修正（2026-09-26・レビュー D1〜D5）**: 守る備えを貪欲な割り当てから厳密な最大へ（V が手札について単調になる）・
最初の相手ターンを `s = 1 − ko_p` で割り引く（「止める組がそれしか無い」手札が必ず受けると同点になっていた）・
V の差は T67 の札ごとの価値 `max(ΔH, ΔG)` とは違う量（`ΔH + ΔG`・割引前）であることを値で固定・実行全体の記録に実際に効いた閾値を刻む。

**符号と循環が 1 つ狂うと判断が反転する器**なので、向き・循環・規則の境目を値で押さえる。**基盤健全性**（`cpu_infra`）。
"""
import argparse
import inspect
import itertools
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

import guard_afford as GA  # noqa: E402
import hand_guard as HG  # noqa: E402
import theory_bridge as B  # noqa: E402
import theory_order as T  # noqa: E402

MU = T.MU
TAKE = T.THETA * T.MU
S_DISC = 1.0 - T.KO_P
LIFE0_TAKE = T.theta_take(0.0) * T.MU                           # ライフ 0 で受ける損（致死）


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


def _slot(counter, v, cost=0.0, event=False, paid=None):
    """手で作る手札の 1 枚（`guard_hand_reading` の枠と同じ形・無料の印字は event でない札だけ）。"""
    free = 0.0 if (event or counter <= 0.0) else float(counter)
    return {"cid": None, "cost": float(cost), "counter": float(counter), "event": event, "free": free,
            "paid": paid, "v": v, "item": None}


def _hand(slots, xs_future=(1000.0,), caps=(7, 9, 10, 10), take=TAKE):
    return {"slots": list(slots), "caps": list(caps), "xs_future": list(xs_future), "take": float(take),
            "mu": MU, "inflow": None}


def _old_guard_step(tok, sc, played, free, paid, theta=T.THETA, mu=T.MU, margin_comfort=None, guard_g=None):
    """**G-2 より前の `guard_step` の写し**（HEAD f86e7d30 のまま）——`lenient` が旧を 1 ビットも違わず再現するかの基準。"""
    xs = [x for x in B.incoming_x(tok) if x >= -B.PWR_EPS]
    if not xs:
        return None
    x = max(xs)
    blocker = bool((np.asarray(tok)[B.GA.SLOT_OWN_FIELD][:, B.GA.S_BLOCKER] > 0.5).any())
    budget = int(round(float(sc[B.SC_MY_DON])))
    afford_pw = free + B.GA.knapsack(paid, budget)
    can_guard = bool(blocker or afford_pw >= x)
    cost_take = float(theta) * float(mu)
    cost_guard = float(B.c_of(x)) * float(mu)
    best = min(cost_take, cost_guard) if can_guard else cost_take
    actual = cost_guard if played == "guard" else cost_take
    g_paid = -float(actual)
    price = min(cost_take, cost_guard)
    g_delta = float(price) - float(actual)
    _gm = B.GUARD_G_MODE if guard_g is None else guard_g
    g = 0.0 if _gm == "zero" else (g_delta if _gm == "delta" else g_paid)
    if played == "guard" and not can_guard:
        actual = best
    margin = (float("inf") if blocker else float(afford_pw) - float(x))
    return {"s": -max(0.0, actual - best), "g": g, "g_paid": g_paid, "g_delta": g_delta, "price": float(price),
            "x": x, "can_guard": can_guard, "margin": margin,
            "comfortable": bool(can_guard and margin >= (B.MARGIN_COMFORT
                                                        if margin_comfort is None
                                                        else float(margin_comfort))),
            "played": played, "theory_says": ("guard" if (can_guard and cost_guard < cost_take)
                                              else "take")}


def _grid():
    for opp in (4000, 5000, 5500, 6000, 7000, 8000, 10000, 14000):
        for free in (0.0, 500.0, 1000.0, 2000.0, 2500.0, 3000.0, 4000.0, 6000.0, 9000.0):
            for paid in ([], [(1, 2000.0)], [(3, 1000.0)]):
                for don in (0, 1, 3):
                    for blocker in (False, True):
                        for played in ("take", "guard"):
                            for theta in (T.THETA, 1.15, 8.18):
                                for gg in (None, "paid", "zero"):
                                    yield opp, free, paid, don, blocker, played, theta, gg


# ---- 1. 既定と切替 ----

def test_the_defaults_are_the_rule_and_the_old_cost():
    assert B.GUARD_AFFORD_MODE == "rule"          # 既定は規則どおり（ユーザ決定 2026-09-25）
    assert B.GUARD_S_COST_MODE == "curve"         # 判断の守る費用は従来のまま（`hand` は測ってから採否）
    assert B.GUARD_AFFORD_MODES == ("rule", "lenient") and B.GUARD_S_COST_MODES == ("curve", "hand")


def test_unknown_modes_are_refused_everywhere():
    for bad in ("", "strict", "なにか"):
        with pytest.raises(ValueError):
            B.set_guard_afford_mode(bad)
        with pytest.raises(ValueError):
            B.set_guard_s_cost_mode(bad)
        with pytest.raises(ValueError):
            B.afford_need(0.0, bad)
        with pytest.raises(ValueError):
            B.guard_step(_tok(6000), _sc(), "take", 9000.0, [], afford=bad)
        with pytest.raises(ValueError):
            B.guard_step(_tok(6000), _sc(), "take", 9000.0, [], s_cost=bad)
    assert B.GUARD_AFFORD_MODE == "rule" and B.GUARD_S_COST_MODE == "curve"


def test_the_cli_flags_reach_the_modes():
    ap = argparse.ArgumentParser()
    B.add_guard_afford_arg(ap)
    B.add_guard_s_cost_arg(ap)
    try:
        a = ap.parse_args([])
        assert B.apply_guard_afford(a) == "rule" and B.apply_guard_s_cost(a) == "curve"      # 省略時は不動
        a = ap.parse_args(["--guard-afford", "lenient", "--guard-s-cost", "hand"])
        assert B.apply_guard_afford(a) == "lenient" and B.GUARD_AFFORD_MODE == "lenient"
        assert B.apply_guard_s_cost(a) == "hand" and B.GUARD_S_COST_MODE == "hand"
        with pytest.raises(SystemExit):
            ap.parse_args(["--guard-afford", "strict"])                   # 知らない値は argparse が弾く
    finally:
        B.set_guard_afford_mode("rule")
        B.set_guard_s_cost_mode("curve")
    # 器の `main` が両方を組み込み、実行前に反映していること（片方だけ付け忘れると CLI の値が黙って捨てられる）
    src = inspect.getsource(B.main)
    for piece in ("add_guard_afford_arg(ap)", "add_guard_s_cost_arg(ap)", "apply_guard_afford(a)", "apply_guard_s_cost(a)"):
        assert piece in src, piece
    assert src.index("apply_guard_s_cost(a)") < src.index("collect(a.src")


def test_the_module_switch_is_what_the_step_reads_when_no_override_is_given():
    tok, sc = _tok(opp_lead=5000), _sc(don=0)                  # 超過 0・カウンター 0 枚
    try:
        B.set_guard_afford_mode("lenient")
        assert B.guard_step(tok, sc, "take", 0.0, [])["can_guard"] is True
        B.set_guard_afford_mode("rule")
        assert B.guard_step(tok, sc, "take", 0.0, [])["can_guard"] is False
    finally:
        B.set_guard_afford_mode("rule")


# ---- 2. 規則どおりの「守れたか」 ----

def test_an_equal_power_attack_is_not_stopped_by_no_counters():
    """超過 0（同じパワー）は命中する＝カウンター 0 枚では止まらない。旧は「守れた」にして受けた行を罰していた。"""
    tok, sc = _tok(opp_lead=5000), _sc(don=0)
    rule = B.guard_step(tok, sc, "take", 0.0, [], afford="rule")
    old = B.guard_step(tok, sc, "take", 0.0, [], afford="lenient")
    assert rule["x"] == pytest.approx(0.0)
    assert rule["can_guard"] is False and rule["s"] == 0.0            # 守れない＝受けても誤りでない
    assert old["can_guard"] is True and old["s"] < 0.0                # 旧は受けたことを罰していた


def test_a_counter_total_of_exactly_the_excess_still_loses_the_battle():
    tok, sc = _tok(opp_lead=7000), _sc(don=0)                   # 超過 2000
    assert B.guard_step(tok, sc, "take", 2000.0, [], afford="rule")["can_guard"] is False
    assert B.guard_step(tok, sc, "take", 2000.0, [], afford="lenient")["can_guard"] is True
    assert B.guard_step(tok, sc, "take", 2500.0, [], afford="rule")["can_guard"] is False    # 許容は PWR_EPS（丸め）だけ
    assert B.guard_step(tok, sc, "take", 3000.0, [], afford="rule")["can_guard"] is True     # x + 1000 ちょうどで勝つ
    assert B.afford_need(2000.0, "rule") == pytest.approx(3000.0 - T.PWR_EPS)
    assert B.afford_need(2000.0, "lenient") == pytest.approx(2000.0)


def test_paid_event_counters_still_need_the_don_under_the_rule():
    tok = _tok(opp_lead=6000)                                   # 超過 1000 → 2000 要る
    assert B.guard_step(tok, _sc(don=1), "take", 0.0, [(1, 2000.0)])["can_guard"] is True
    assert B.guard_step(tok, _sc(don=0), "take", 0.0, [(1, 2000.0)])["can_guard"] is False
    assert B.guard_step(tok, _sc(don=1), "take", 0.0, [(1, 1000.0)])["can_guard"] is False   # 払えても 1000 では足りない
    assert B.guard_step(_tok(opp_lead=9000, blocker=True), _sc(don=0), "take", 0.0, [])["can_guard"] is True   # ブロッカーは不変


def test_lenient_reproduces_the_old_step_bit_for_bit():
    """`lenient` ＋ `curve` は旧の `guard_step` と**全部の欄が完全に一致**（手で組んだ 20,412 行・攻撃の無い行は両方 `None`）。"""
    n = 0
    for opp, free, paid, don, blocker, played, theta, gg in _grid():
        tok, sc = _tok(opp_lead=opp, blocker=blocker), _sc(don=don)
        old = _old_guard_step(tok, sc, played, free, paid, theta=theta, guard_g=gg)
        new = B.guard_step(tok, sc, played, free, paid, theta=theta, guard_g=gg, afford="lenient")
        if old is None:
            assert new is None
            continue
        assert {k: new[k] for k in old} == old, (opp, free, paid, don, blocker, played, theta, gg)
        n += 1
    assert n > 10000


def test_under_the_rule_only_the_rows_between_x_and_x_plus_1000_change():
    """`rule` ＋ `curve` で旧と違うのは**`x ≤ 守る力 < x + 1000` の行だけ**で、変わるのは「守れたか」とそれに従う欄だけ
    （帳簿の欄・余裕・価格は 1 ビットも動かない）。向きは必ず「守れた → 守れない」。"""
    derived = {"can_guard", "s", "theory_says", "comfortable"}
    changed = 0
    for opp, free, paid, don, blocker, played, theta, gg in _grid():
        tok, sc = _tok(opp_lead=opp, blocker=blocker), _sc(don=don)
        old = _old_guard_step(tok, sc, played, free, paid, theta=theta, guard_g=gg)
        new = B.guard_step(tok, sc, played, free, paid, theta=theta, guard_g=gg)
        if old is None:
            continue
        diff = {k for k in old if new[k] != old[k]}
        assert diff <= derived, diff
        pw = free + GA.knapsack(paid, don)
        between = (not blocker) and (old["x"] <= pw < old["x"] + 1000.0 - T.PWR_EPS)
        if diff:
            changed += 1
            assert between and old["can_guard"] is True and new["can_guard"] is False
            assert new["s"] == 0.0                                  # 守れない行は受けても守っても誤りでない
        if between:
            assert new["can_guard"] is False
    assert changed > 0


def test_curve_mode_with_a_hand_reading_changes_no_old_field():
    """既定の `curve` で手札の読み（枚数の監査）を渡しても従来の欄は同じ——足すのは監査の欄だけ。"""
    rd = {"slots": [_slot(2000.0, None), _slot(0.0, None, cost=3), _slot(2000.0, None, cost=1, event=True, paid=(1, 2000.0))],
          "caps": None, "xs_future": None, "take": TAKE, "mu": MU, "inflow": None}
    for opp, free, paid, don, blocker, played, theta, gg in itertools.islice(_grid(), 0, None, 7):
        tok, sc = _tok(opp_lead=opp, blocker=blocker), _sc(don=don)
        a = B.guard_step(tok, sc, played, free, paid, theta=theta, guard_g=gg)
        b = B.guard_step(tok, sc, played, free, paid, theta=theta, guard_g=gg, hand=rd)
        if a is None:
            assert b is None
            continue
        old_keys = set(_old_guard_step(tok, sc, played, free, paid, theta=theta, guard_g=gg))
        assert {k: a[k] for k in old_keys} == {k: b[k] for k in old_keys}
        assert b["cost_guard_hand"] is None and b["cost_guard_source"] == "curve" and b["s_cost_mode"] == "curve"
        assert b["cost_guard_curve"] == pytest.approx(B.c_of(b["x"]) * MU) and b["cost_guard_s"] == b["cost_guard_curve"]
        assert b["n_hand_cards"] == 3
        assert b["n_counter_cards"] == (2 if don >= 1 else 1)         # イベントはドンが払えるときだけ数える
        assert a["n_hand_cards"] is None and a["cost_guard_hand"] is None


# ---- 3. 手札で測る守る費用（`hand`） ----

def test_one_cheap_counter_costs_less_than_two_big_characters():
    """2000 カウンター 1 枚で止まる手札は、1000 カウンターの大型を 2 枚切る手札より安い（一律の `c(x)·μ` は同じ値を付ける）。"""
    one = B.guard_hand_cost(_hand([_slot(2000.0, 0.0), _slot(0.0, 0.12, cost=8)]), 1000.0, 5)
    two = B.guard_hand_cost(_hand([_slot(1000.0, 0.12, cost=8), _slot(1000.0, 0.10, cost=7)]), 1000.0, 5)
    assert one["set"] == (0,) and two["set"] == (0, 1)
    assert one["cost"] < two["cost"]
    assert two["cost"] > TAKE                                  # 大型 2 枚は受けるより高い＝受けるのが正しい
    tok, sc = _tok(opp_lead=6000), _sc(don=5)
    got = B.guard_step(tok, sc, "guard", 2000.0, [], s_cost="hand",
                       hand=_hand([_slot(1000.0, 0.12, cost=8), _slot(1000.0, 0.10, cost=7)]))
    assert got["theory_says"] == "take" and got["s"] == pytest.approx(-(two["cost"] - TAKE))


def test_a_counter_only_card_is_not_free():
    """出す価値 0 のカウンター専用札も**次の相手ターンの備えを失う**のでタダではない。
    `hand_guard.guard_cost_min_v`（使ったときの価値だけ）はこの札を 0 と読む——そこが直した点。"""
    h = _hand([_slot(2000.0, 0.0)], xs_future=(1000.0,))
    got = B.guard_hand_cost(h, 1000.0, 5)
    assert HG.guard_cost_min_v([(2000.0, 0.0)], 1000.0)[0] == 0.0
    assert got["cost"] == pytest.approx(S_DISC * TAKE)          # 次の相手ターン（1 ラウンド後）に同じ攻撃を受けることになる
    assert got["cost"] > 0.0


def test_a_spare_counter_is_cheaper_than_the_last_one():
    """最初に来る相手ターンは今の窓の 1 ラウンド後（`s = 1 − ko_p`）・2 枚目はその次（`s²`）にしか要らない＝安い。"""
    one = B.guard_hand_cost(_hand([_slot(2000.0, 0.0)]), 1000.0, 5)["cost"]
    two = B.guard_hand_cost(_hand([_slot(2000.0, 0.0), _slot(2000.0, 0.0)]), 1000.0, 5)["cost"]
    three = B.guard_hand_cost(_hand([_slot(2000.0, 0.0)] * 3), 1000.0, 5)["cost"]
    assert one == pytest.approx(S_DISC * TAKE)
    assert two == pytest.approx(S_DISC ** 2 * TAKE)
    assert three == pytest.approx(0.0)                          # 地平（相手ターン 2 回）の外＝要らない札
    # 判断: 止める札が 1 枚しか無くても守るのが最善（**同点にならない**）——受けた行は (1 − s)·Θμ だけ罰する
    tok, sc = _tok(opp_lead=6000), _sc(don=5)
    g1t = B.guard_step(tok, sc, "take", 2000.0, [], s_cost="hand", hand=_hand([_slot(2000.0, 0.0)]))
    g1g = B.guard_step(tok, sc, "guard", 2000.0, [], s_cost="hand", hand=_hand([_slot(2000.0, 0.0)]))
    assert g1t["theory_says"] == "guard" and g1t["cost_guard_s"] < g1t["cost_take"]
    assert g1t["s"] == pytest.approx(-(1.0 - S_DISC) * TAKE) and g1t["s"] < 0.0
    assert g1g["s"] == pytest.approx(0.0)
    g2t = B.guard_step(tok, sc, "take", 4000.0, [], s_cost="hand", hand=_hand([_slot(2000.0, 0.0)] * 2))
    g2g = B.guard_step(tok, sc, "guard", 4000.0, [], s_cost="hand", hand=_hand([_slot(2000.0, 0.0)] * 2))
    assert g2t["theory_says"] == "guard" and g2t["s"] == pytest.approx(-(1.0 - S_DISC ** 2) * TAKE)
    assert g2g["s"] == pytest.approx(0.0)


def test_the_only_stopper_is_used_against_a_lethal_attack_at_life_0():
    """**D2 の再現**: ライフ 0 で超過 1000 の攻撃（受ければ負け）を、それを止める 2000 カウンターを持ったまま受けた。
    割り引く前は守る費用 = 受ける損ちょうどの同点で「受けろ」・罰点 0 になっていた。今は守れと言い、受けた行を罰する。"""
    tok, sc = _tok(opp_lead=6000), _sc(don=5)
    th0 = T.theta_take(0.0)
    h = _hand([_slot(2000.0, 0.0)], xs_future=(1000.0,), take=LIFE0_TAKE)
    took = B.guard_step(tok, sc, "take", 2000.0, [], theta=th0, s_cost="hand", hand=h)
    assert took["cost_take"] == pytest.approx(LIFE0_TAKE)
    assert took["cost_guard_s"] == pytest.approx(S_DISC * LIFE0_TAKE)
    assert took["theory_says"] == "guard"
    assert took["s"] == pytest.approx(-(1.0 - S_DISC) * LIFE0_TAKE) and took["s"] < -0.1
    guarded = B.guard_step(tok, sc, "guard", 2000.0, [], theta=th0, s_cost="hand", hand=h)
    assert guarded["s"] == pytest.approx(0.0)


def test_the_current_attack_is_not_counted_inside_its_own_cost():
    """**循環を切る**: V はこれから来る相手ターンだけを数える。今の窓の攻撃 `x` は V に入らない。
    (a) 次に来る攻撃が無ければ、カウンター専用札で今の攻撃を止める費用は 0（今の攻撃を止められることが費用に入れば `Θ·μ`）。
    (b) 次に来る攻撃をこの札では止められないなら、費用は今の `x` に依らず出す価値だけ。"""
    assert B.guard_hand_cost(_hand([_slot(2000.0, 0.0)], xs_future=()), 1000.0, 5)["cost"] == pytest.approx(0.0)
    assert B.guard_hand_cost(_hand([_slot(2000.0, 0.0)], xs_future=(1000.0,)), 1000.0, 5)["cost"] == pytest.approx(S_DISC * TAKE)
    h = _hand([_slot(2000.0, 0.01, cost=1)], xs_future=(3000.0,))            # 次の攻撃は 4000 要る＝この札では止まらない
    for x in (0.0, 1000.0):
        assert B.guard_hand_cost(h, x, 5)["cost"] == pytest.approx(0.01)
    # 判断の側でも: 次の攻撃が無い手札なら受けた行は `Θ·μ` まるごと罰される
    tok, sc = _tok(opp_lead=6000), _sc(don=5)
    got = B.guard_step(tok, sc, "take", 2000.0, [], s_cost="hand", hand=_hand([_slot(2000.0, 0.0)], xs_future=()))
    assert got["cost_guard_hand"] == pytest.approx(0.0) and got["s"] == pytest.approx(-TAKE)


def test_an_unreadable_card_counts_at_the_market_price():
    """値の読めない札は `μ`（`spent` の分岐と同じ規約）——使ったときの価値がちょうど `μ` の札と区別がつかない。"""
    for xs in ((), (1000.0,), (3000.0,)):
        u = B.guard_hand_cost(_hand([_slot(2000.0, None, cost=2), _slot(1000.0, 0.02, cost=1)], xs_future=xs), 1000.0, 5)
        r = B.guard_hand_cost(_hand([_slot(2000.0, MU, cost=2), _slot(1000.0, 0.02, cost=1)], xs_future=xs), 1000.0, 5)
        assert u == r
    lone = B.guard_hand_cost(_hand([_slot(2000.0, None, cost=2)], xs_future=()), 1000.0, 5)
    assert lone["cost"] == pytest.approx(MU)                    # 次のターンに出せる＝ちょうど μ を失う


def _mixed():
    return _hand([_slot(1000.0, 0.02, cost=2), _slot(1000.0, 0.03, cost=3), _slot(2000.0, 0.05, cost=4),
                  _slot(0.0, 0.10, cost=5), _slot(2000.0, 0.0, cost=1, event=True, paid=(1, 2000.0))])


def test_spending_more_cards_than_needed_is_never_cheaper():
    """余計な札を足した組は、足りている組より安くならない（V は札が多いほど下がらない）。
    選ばれた組は過不足が無い（どの 1 枚を抜いても足りない）。来る攻撃が 1 本の手札と、何本も来る手札（D1 の形）の両方で。"""
    for xs in ((1000.0,), (2000.0, 0.0), (1000.0, 2000.0, 3000.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1000.0)):
        _check_supersets_and_minimality(dict(_mixed(), xs_future=list(xs)))


def _check_supersets_and_minimality(h):
    cache = {}
    n = len(h["slots"])
    for r in range(1, n):
        for S in itertools.combinations(range(n), r):
            base = B.guard_set_loss(h, S, cache)
            assert base >= 0.0
            for j in range(n):
                if j not in S:
                    assert B.guard_set_loss(h, S + (j,), cache) >= base - 1e-12, (S, j)
    for x in (0.0, 1000.0, 2000.0, 3000.0):
        got = B.guard_hand_cost(h, x, 1)
        need = B.afford_need(x, "rule")
        tot = lambda S: sum(h["slots"][i]["free"] for i in S) + GA.knapsack(
            [h["slots"][i]["paid"] for i in S if h["slots"][i]["paid"]], 1)
        assert tot(got["set"]) >= need
        for k in got["set"]:
            assert tot(tuple(i for i in got["set"] if i != k)) < need      # 1 枚でも抜けば足りない
        # 最小性: 足りる組はどれも選ばれた組より安くない
        for r in range(1, n + 1):
            for S in itertools.combinations(range(n), r):
                if tot(S) >= need:
                    assert B.guard_set_loss(h, S, cache) >= got["cost"] - 1e-12


def test_a_bigger_attack_never_costs_less_to_stop():
    for xs in ((1000.0,), (2000.0, 0.0), (1000.0, 2000.0, 3000.0, 0.0, 0.0, 0.0)):
        _check_bigger_attack(dict(_mixed(), xs_future=list(xs)))


def _check_bigger_attack(h):
    costs = [B.guard_hand_cost(h, x, 1)["cost"] for x in (0.0, 1000.0, 2000.0, 3000.0, 4000.0, 5000.0)]
    finite = [c for c in costs if c is not None]
    assert finite == sorted(finite) and len(finite) >= 4
    assert costs[-1] is None or costs[-1] >= finite[-1]


def test_an_event_counter_is_usable_only_if_its_don_is_payable():
    h = _hand([_slot(2000.0, 0.0, cost=2, event=True, paid=(2, 2000.0))])
    assert B.guard_hand_cost(h, 1000.0, 1)["cost"] is None
    assert B.guard_hand_cost(h, 1000.0, 2)["cost"] is not None
    tok = _tok(opp_lead=6000)
    poor = B.guard_step(tok, _sc(don=1), "take", 0.0, [(2, 2000.0)], s_cost="hand", hand=h)
    rich = B.guard_step(tok, _sc(don=2), "take", 0.0, [(2, 2000.0)], s_cost="hand", hand=h)
    assert poor["can_guard"] is False and poor["s"] == 0.0 and poor["n_counter_cards"] == 0
    assert rich["can_guard"] is True and rich["n_counter_cards"] == 1


def test_hand_mode_uses_the_rule_whatever_the_afford_switch_says():
    """`hand` は止める組が在るかで判定する＝`Σ無料 + knapsack(有料, ドン) ≥ x + 1000` と同値（手で組んだ全組で一致）。
    `lenient` を指定しても規則どおり（止める組が無ければ守る費用が定義できない）。"""
    rng = np.random.default_rng(7)
    for _ in range(300):
        slots = []
        for _k in range(int(rng.integers(0, 6))):
            if rng.random() < 0.3:
                c = int(rng.integers(0, 4))
                slots.append(_slot(float(rng.choice([1000.0, 2000.0, 3000.0])), 0.01, cost=c, event=True,
                                   paid=(c, float(rng.choice([1000.0, 2000.0, 3000.0])))))
            else:
                slots.append(_slot(float(rng.choice([0.0, 1000.0, 2000.0])), float(rng.random() * 0.1), cost=int(rng.integers(0, 8))))
        don = int(rng.integers(0, 4))
        opp = int(rng.choice([5000, 6000, 7000, 8000, 9000]))
        free = sum(s["free"] for s in slots)
        paid = [s["paid"] for s in slots if s["paid"]]
        tok, sc = _tok(opp_lead=opp), _sc(don=don)
        got = B.guard_step(tok, sc, "take", free, paid, s_cost="hand", afford="lenient", hand=_hand(slots))
        rule = B.guard_step(tok, sc, "take", free, paid, afford="rule")
        assert got["can_guard"] == rule["can_guard"]
        assert got["afford_mode"] == "rule"
        assert (got["cost_guard_hand"] is not None) == rule["can_guard"]


def test_blocker_rows_keep_the_old_price_and_the_ledger_never_moves():
    """限界の固定: ブロッカーが居る行は従来の `c(x)·μ` で判断する。`hand` は判断だけの切替＝帳簿の欄は `curve` と同じ。"""
    tok, sc = _tok(opp_lead=8000, blocker=True), _sc(don=5)
    h = _hand([_slot(2000.0, 0.0), _slot(2000.0, 0.0)], xs_future=())
    for played in ("take", "guard"):
        hb = B.guard_step(tok, sc, played, 4000.0, [], s_cost="hand", hand=h)
        cb = B.guard_step(tok, sc, played, 4000.0, [])
        assert hb["cost_guard_source"] == "blocker" and hb["cost_guard_hand"] is not None
        assert hb["s"] == cb["s"] and hb["theory_says"] == cb["theory_says"]
    tok2 = _tok(opp_lead=8000)                                   # ブロッカー無し
    for played in ("take", "guard"):
        for gg in ("paid", "delta", "zero"):
            hh = B.guard_step(tok2, sc, played, 4000.0, [], s_cost="hand", hand=h, guard_g=gg)
            cc = B.guard_step(tok2, sc, played, 4000.0, [], guard_g=gg)
            assert hh["cost_guard_source"] == "hand"
            for k in ("g", "g_paid", "g_delta", "price", "margin", "x", "can_guard"):
                assert hh[k] == cc[k], k


def test_hand_mode_without_a_hand_reading_refuses_to_guess():
    with pytest.raises(ValueError):
        B.guard_step(_tok(6000), _sc(), "take", 9000.0, [], s_cost="hand")
    with pytest.raises(ValueError):
        B.guard_step(_tok(6000), _sc(), "take", 9000.0, [], s_cost="hand",
                     hand={"slots": [], "caps": None, "xs_future": None, "take": TAKE, "mu": MU, "inflow": None})


def test_the_audit_fields_reach_the_window_bookkeeping():
    """監査の欄は `_finish_guard`（G-1 が行ごとの記録に monkeypatch した経路）まで届き、`hand` の行だけ集計される。"""
    tok, sc = _tok(opp_lead=6000), _sc(don=5)
    got = B.guard_step(tok, sc, "take", 4000.0, [], s_cost="hand", hand=_hand([_slot(2000.0, 0.0)] * 2))
    for k in ("cost_guard_curve", "cost_guard_hand", "n_hand_cards", "n_counter_cards", "cost_guard_source"):
        assert k in got
    seen = []
    stats = {"grd_rows": 0, "grd_by_life": {}, "grd_comfortable": 0}
    B._finish_guard(dict(got, g=0.0, g_delta=0.0), "take", 3.0, 1.0, "close", 1.0, 0, 4, {}, {}, stats,
                    lambda *a, **k: seen.append(a))
    assert stats["grd_hand_rows"] == 1 and stats["grd_hand_priced"] == 1
    assert stats["grd_hand_cost_sum"] == pytest.approx(got["cost_guard_hand"])
    assert stats["grd_hand_curve_sum"] == pytest.approx(got["cost_guard_curve"])
    assert seen and seen[0][2] == pytest.approx(got["s"])            # 帯に入る `s` は手札で測った判断のもの
    stats2 = {"grd_rows": 0, "grd_by_life": {}, "grd_comfortable": 0}
    B._finish_guard(B.guard_step(tok, sc, "take", 4000.0, []), "take", 3.0, 1.0, "close", 1.0, 0, 4, {}, {}, stats2,
                    lambda *a, **k: None)
    assert "grd_hand_rows" not in stats2                            # `curve` の行は集計しない


# ---- 4. 記録の行からの読み（本物の札） ----

def _real_row():
    from opcg_sim.learned.train import plan_labels as PL
    cards = PL.Cards()
    vocab = GA._vocab()
    idx2cid = {i: c for c, i in vocab.items()}
    # 2000 カウンターのキャラ 2・1000 カウンターのキャラ 2・【カウンター】イベント（上げ幅はトークンから・コスト 1）・カウンター無しのイベント
    hand = ["EB01-007", "EB01-014", "EB01-004", "EB01-005", "EB01-009", "EB01-010"]
    tok = np.zeros((22, 24), np.float32)
    ci = np.zeros(24, np.int64)
    tok[0, T.S_POWER], tok[1, T.S_POWER] = 0.5, 0.6              # 超過 1000
    for j, cid in enumerate(hand):
        s = GA.SLOT_HAND.start + j
        inf = cards.info(cid)
        assert inf is not None, cid
        ci[s] = vocab[cid]
        cv = float(inf.get("counter") or 0.0)
        if inf.get("event"):
            cv = 2000.0 if cid == "EB01-009" else 0.0
            tok[s, GA.S_IS_EVENT] = 1.0
        tok[s, GA.S_COUNTER] = cv / GA.COUNTER_SCALE
        tok[s, T.S_POWER] = float(inf.get("power") or 0.0) / 1e4
        tok[s, 23] = 0.1
    sc = np.zeros(127, np.float32)
    sc[T.SC_MY_DON], sc[T.SC_OPP_LEADER_POWER], sc[T.SC_OPP_LIFE], sc[T.SC_MY_LIFE] = 1, 0.5, 3, 3
    return tok, sc, ci, idx2cid, cards


def test_the_reading_splits_counters_exactly_like_guard_afford():
    tok, sc, ci, idx2cid, cards = _real_row()
    free, paid, nslots = GA.hand_counters(tok, ci, idx2cid, cards)
    for values in (False, True):
        rd = B.guard_hand_reading(tok, sc, ci, idx2cid, cards, take=TAKE, values=values)
        assert sum(s["free"] for s in rd["slots"]) == pytest.approx(free)
        assert [s["paid"] for s in rd["slots"] if s["paid"] is not None] == paid
        assert len(rd["slots"]) == nslots == 6
    assert free == pytest.approx(6000.0) and paid == [(1.0, 2000.0)]


def test_the_reading_values_come_from_hand_plan_on_the_next_turn():
    import hand_plan as HP
    from price_realised import don_stock
    tok, sc, ci, idx2cid, cards = _real_row()
    rd = B.guard_hand_reading(tok, sc, ci, idx2cid, cards, take=TAKE, values=True)
    items = HP.hand_items(tok, ci, idx2cid, cards, 5000.0, 3.0)
    assert [s["item"] for s in rd["slots"] if s["item"] is not None] == items          # 札ごとの価値は hand_plan のもの
    # 出す計画は次の自席ターンから（**式を写さずに数で**）: この行のドンはアクティブ 1 枚だけ＝総在庫 1
    # → 次の自席ターンは 1 + 2 = 3 枚（全部アクティブ）・その後 5, 7・「それより後」は残り 3 ターンなので 10 × 1
    assert don_stock(sc, tok, "me") == pytest.approx(1.0)
    assert rd["caps"] == [3, 5, 7, 10]
    sc2 = sc.copy()
    sc2[T.SC_MY_DON] = 9                                                               # 総在庫 9 → 次は 10 で頭打ち
    rd2 = B.guard_hand_reading(tok, sc2, ci, idx2cid, cards, take=TAKE, values=True)
    assert rd2["caps"] == [10, 10, 10, 10]
    sc3 = sc.copy()
    sc3[T.SC_OPP_LIFE] = 5                                                             # 残り 5 ターン → 「それより後」は 10 × 2
    assert B.guard_hand_reading(tok, sc3, ci, idx2cid, cards, take=TAKE, values=True)["caps"] == [3, 5, 7, 20]
    assert rd["xs_future"] == HG.incoming(tok) == [pytest.approx(1000.0)]
    lite = B.guard_hand_reading(tok, sc, ci, idx2cid, cards, take=TAKE, values=False)
    assert lite["caps"] is None and all(s["v"] is None and s["item"] is None for s in lite["slots"])
    free, paid, _n = GA.hand_counters(tok, ci, idx2cid, cards)
    got = B.guard_step(tok, sc, "take", free, paid, s_cost="hand", hand=rd)
    assert got["cost_guard_source"] == "hand" and got["cost_guard_hand"] is not None
    assert got["cost_guard_hand"] >= 0.0 and got["hand_set_n"] >= 1
    assert got["n_hand_cards"] == 6 and got["n_counter_cards"] == 5                   # イベントはドン 1 で払える


# ---- 5. G-2 の修正（2026-09-26・レビュー D1〜D5） ----

def _brute_guard(items, xs, take, s=S_DISC, turns=HG.GUARD_TURNS, start=0):
    """**独立の総当たり**: 札ごとに「使わない／どの (相手ターン, 攻撃) に切るか」を全部試す（`guard_value_exact` の再帰とは別の数え方）。
    足りない組を割り当てた割り当ては捨てる。受ける方が安い組を割り当てた割り当ても数えるが、割り当てない方が必ず良いので最大は変わらない。"""
    pairs = [(t, x) for t in range(turns) for x in xs if x >= -T.PWR_EPS]
    n = len(items)
    best = 0.0
    for assign in itertools.product(range(len(pairs) + 1), repeat=n):
        tot, ok = 0.0, True
        for j, (t, x) in enumerate(pairs, start=1):
            S = [i for i in range(n) if assign[i] == j]
            if not S:
                continue
            if sum(items[i][0] for i in S) < x + 1000.0 - T.PWR_EPS:
                ok = False
                break
            tot += s ** (start + t) * (take - sum(0.0 if items[i][1] is None else items[i][1] for i in S))
        if ok and tot > best:
            best = tot
    return best


def _rand_items(rng, n):
    return [(float(rng.choice([0.0, 1000.0, 2000.0])), float(rng.choice([0.0, 0.01, 0.03, 0.05, 0.09]))) for _ in range(n)]


def test_the_exact_guard_readiness_is_the_true_maximum_and_never_below_the_greedy():
    """**D1**: 守る備えは全部の割り当ての最大——独立の総当たりと一致し、貪欲（`guard_value`・同じ目的の 1 つの割り当て）以上。
    `start=1` は全体をちょうど `s` 倍する。"""
    rng = np.random.default_rng(11)
    n_gap = 0
    for _ in range(150):
        items = _rand_items(rng, int(rng.integers(0, 5)))
        xs = [float(x) for x in rng.choice([0.0, 1000.0, 2000.0, 3000.0], size=int(rng.integers(0, 3)))]
        ex = HG.guard_value_exact(items, xs, TAKE)
        assert ex == pytest.approx(_brute_guard(items, xs, TAKE), abs=1e-12), (items, xs)
        greedy = HG.guard_value(items, xs, TAKE)
        assert ex >= greedy - 1e-12
        n_gap += int(ex > greedy + 1e-9)
        assert HG.guard_value_exact(items, xs, TAKE, start=1) == pytest.approx(S_DISC * ex, abs=1e-12)
    assert n_gap > 0                                             # 貪欲が取りこぼす手札は実在する
    # 負の v が在っても定義どおり（足りる組を全部調べる経路）
    items = [(1000.0, -0.01), (1000.0, 0.02), (2000.0, 0.0)]
    assert HG.guard_value_exact(items, [0.0, 1000.0], TAKE) == pytest.approx(_brute_guard(items, [0.0, 1000.0], TAKE))


def test_the_value_of_a_hand_never_drops_when_a_card_is_added():
    """**D1 の単調性**: 相方待ちを読み直さない手札では `V(手札) ≥ V(手札 − 1 枚)`（出す計画は DP の最大・守る備えは厳密な最大）。
    過不足の無い組に絞ってよい根拠。何本も攻撃が来る形で乱択 200 手札。"""
    rng = np.random.default_rng(5)
    for _ in range(200):
        n = int(rng.integers(1, 6))
        slots = [_slot(float(rng.choice([0.0, 1000.0, 2000.0])), float(rng.choice([0.0, 0.02, 0.05, 0.1])),
                       cost=int(rng.integers(0, 8))) for _k in range(n)]
        xs = [float(x) for x in rng.choice([0.0, 1000.0, 2000.0, 3000.0], size=int(rng.integers(0, 4)))]
        h = _hand(slots, xs_future=xs)
        assert B.hand_value_monotone(h)
        cache = {}
        v_all = B._hand_v(h, range(n), cache)
        for j in range(n):
            assert v_all >= B._hand_v(h, [i for i in range(n) if i != j], cache) - 1e-12, (slots, xs, j)


def test_several_future_attacks_are_priced_by_the_best_assignment():
    """**D1 のレビューの再現**（貪欲では損が 0 に潰れた 2 例）。
    (a) 1000 と 2000 のカウンター専用札・来る攻撃が超過 2000 と 0: 貪欲は 2 枚とも 2000 の攻撃に当てるので 1 枚の損が 0 に見えた。
        最善は 0 の攻撃に 1 枚ずつ（1 ラウンド後と 2 ラウンド後）＝V = (s + s²)Θμ。今 1 枚切ると残りは s·Θμ ＝ 損 s²·Θμ。
    (b) 1000 カウンター 5 枚（v = 0.01〜0.05）・来る攻撃 3000, 2000, 1000, 0, 0, 0: 貪欲では 1 枚目を抜くと V が上がり、損 0 だった。"""
    h = _hand([_slot(1000.0, 0.0), _slot(2000.0, 0.0)], xs_future=(2000.0, 0.0))
    assert HG.guard_value([(1000.0, 0.0), (2000.0, 0.0)], [2000.0, 0.0], TAKE) == pytest.approx(TAKE)   # 貪欲の読み
    got = B.guard_hand_cost(h, 0.0, 5)
    assert got["cost"] == pytest.approx(S_DISC ** 2 * TAKE) and got["cost"] > 0.0
    five = [(1000.0, 0.01 * (i + 1)) for i in range(5)]
    xs = (1000.0, 2000.0, 3000.0, 0.0, 0.0, 0.0)
    h5 = _hand([_slot(c, v, cost=i + 1) for i, (c, v) in enumerate(five)], xs_future=xs)
    cache = {}
    v_all = B._hand_v(h5, range(5), cache)
    for j in range(5):
        rest = [it for i, it in enumerate(five) if i != j]
        assert B._hand_v(h5, [i for i in range(5) if i != j], cache) <= v_all
        # V の守る備えは独立の総当たりと一致（1 ラウンド後から数える）
        assert HG.guard_value_exact(rest, list(xs), TAKE, start=1) == pytest.approx(_brute_guard(rest, list(xs), TAKE, start=1))
    c5 = B.guard_hand_cost(h5, 0.0, 5)
    assert c5["cost"] > 0.01 and c5["set"] == (0,)
    loss0 = v_all - B._hand_v(h5, [1, 2, 3, 4], cache)
    assert c5["cost"] == pytest.approx(loss0)


def test_the_value_lost_is_not_t67s_per_card_value():
    """**D3**: 手札の差で読む失う価値は T67 の札ごとの価値 `max(ΔH, ΔG)` とは**違う量**。
    2000 カウンター（v = 0.03・コスト 2）1 枚・次の攻撃 1000: 割引前の差は `ΔH + ΔG`＝`max(v, Θμ)`＝0.0872、
    T67 は `max(ΔH, ΔG)`＝0.0572。差はちょうど `min(ΔH, ΔG)`。割り引くと `ΔH + s·ΔG`。"""
    import hand_plan as HP
    card = {"cid": None, "cost": 2.0, "v": 0.03, "counter": 2000.0, "event": False}
    d = HP.card_deltas([], card, [7, 9, 10, 10], [1000.0], TAKE)
    assert d["dh"] == pytest.approx(0.03) and d["dg"] == pytest.approx(TAKE - 0.03)
    assert d["dtotal"] == pytest.approx(0.0572, abs=1e-4)
    item = {"cost": 2.0, "v": 0.03, "counter": 2000.0}
    v0 = (B.hand_value_next([item], [7, 9, 10, 10], [1000.0], TAKE, guard_start=0)
          - B.hand_value_next([], [7, 9, 10, 10], [1000.0], TAKE, guard_start=0))
    assert v0 == pytest.approx(0.0872, abs=1e-4) and v0 == pytest.approx(TAKE)
    assert v0 == pytest.approx(d["dh"] + d["dg"])
    assert v0 == pytest.approx(d["dtotal"] + min(d["dh"], d["dg"]))
    got = B.guard_hand_cost(_hand([_slot(2000.0, 0.03, cost=2)], xs_future=(1000.0,)), 1000.0, 5)
    assert got["cost"] == pytest.approx(d["dh"] + S_DISC * d["dg"])          # 判断が使う値（1 ラウンド割り引く）


def test_the_run_records_the_afford_threshold_that_was_actually_used():
    """**D4**: `--guard-afford lenient --guard-s-cost hand` でも行ごとの判定は規則どおり。実行全体の記録の `guard_afford` は
    実際に効いた `rule`、指定された値は `guard_afford_requested` に別に刻む。"""
    assert B.effective_guard_afford("lenient", "hand") == "rule"
    assert B.effective_guard_afford("lenient", "curve") == "lenient"
    assert B.effective_guard_afford("rule", "curve") == "rule"
    try:
        B.set_guard_afford_mode("lenient")
        B.set_guard_s_cost_mode("hand")
        assert B.effective_guard_afford() == "rule"
        got = B.guard_step(_tok(opp_lead=6000), _sc(don=5), "take", 2000.0, [], hand=_hand([_slot(2000.0, 0.0)]))
        assert got["afford_mode"] == B.effective_guard_afford()
    finally:
        B.set_guard_afford_mode("rule")
        B.set_guard_s_cost_mode("curve")
    for src in (inspect.getsource(B.collect), inspect.getsource(B.main)):
        assert '"guard_afford": effective_guard_afford()' in src
        assert '"guard_afford_requested": GUARD_AFFORD_MODE' in src


def test_cutting_a_partner_lowers_the_partner_waiting_card(monkeypatch):
    """**D5**: 相方を切ると、手札に残った相方待ちの札の価値も落ちる（`apply_inflow` を残した手札で読み直す）。
    その手札では V の単調性が保証できないので、足りる組を全部調べる。"""
    import hand_plan as HP
    import search_price as SP

    class _Cards:
        def info(self, cid):
            return {"E": {"cost": 4}, "P": {"cost": 2}, "Z": {"cost": 5}}.get(cid)
    monkeypatch.setattr(SP, "enabler_target", lambda cid, cards_json=None: {"cost_max": 2} if cid == "E" else None)
    monkeypatch.setattr(SP, "eligible_hand_cards",
                        lambda target, items, cards, skip_cid=None: [it["cid"] for it in items if it["cid"] == "P"])
    monkeypatch.setattr(SP, "eligible_deck_cards", lambda target, deck, cards: [c for c in deck if c == "P"])
    monkeypatch.setattr(HP, "_base_value", lambda cid, info, cards, olp, r, field=(), st_base=None: 0.10)
    monkeypatch.setattr(HP, "_value_with_partner",
                        lambda cid, info, cards, olp, r, partner, field=(), st_base=None: 0.10 + 0.12)
    monkeypatch.setattr(HP, "inflow_per_turn", lambda items, xs, take, deck, cards: 1.0)
    monkeypatch.setattr(HP, "INFLOW_MODE", "on")
    e = {"cid": "E", "cost": 4.0, "v": 0.10, "counter": 0.0, "event": False}
    p = {"cid": "P", "cost": 2.0, "v": 0.02, "counter": 1000.0, "event": False}
    slots = [dict(_slot(0.0, 0.10, cost=4), cid="E", item=e), dict(_slot(1000.0, 0.02, cost=2), cid="P", item=p)]
    h = dict(_hand(slots, xs_future=()), inflow={"deck": ["Z", "Z"], "cards": _Cards(), "olp": 5000.0, "r": 3.0,
                                                 "field": [], "st_base": None})
    assert not B.hand_value_monotone(h)
    got = B.guard_hand_cost(h, 0.0, 5)                          # 今の攻撃（超過 0）を止められるのは P だけ
    assert got["set"] == (1,)
    # 読み直し無し（E は 0.10 のまま）なら損は P 自身の 0.02 だけ。読み直すと E の相方の取り分 0.12 も失う
    static = B.guard_hand_cost(dict(h, inflow=None), 0.0, 5)
    assert static["cost"] == pytest.approx(0.02)
    assert got["cost"] == pytest.approx(0.02 + 0.12)
    # 相方の居ない山（["Z", "Z"]）では E は base のまま＝V(手札 − P) は E = 0.10
    assert B._hand_v(h, [0], {}) == pytest.approx(0.10)
    monkeypatch.setattr(HP, "INFLOW_MODE", "off")               # 読み直さない切替なら単調＝過不足の無い組に絞る
    assert B.hand_value_monotone(h)
