"""`price_realised.py`（T41・型ごとの「価格」対「実現した価値」）の算術を固める。

**実現の側は A 層の実測だけで作る**（式 `nu_of` を通さない）のが器の値打ちなので、
その約束と、ドンを**総在庫**で数えること（付与で動かない）を値で押さえる。
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

import price_realised as PR  # noqa: E402
import theory_order as T  # noqa: E402


def _sc(my_life=4, opp_life=4, my_hand=5, opp_hand=5, my_act=3, my_rest=0, opp_act=3, opp_rest=0,
        my_ldon=0, opp_ldon=0):
    sc = np.zeros(70, np.float32)
    sc[0], sc[1] = my_life, opp_life
    sc[2], sc[3], sc[4], sc[5] = my_act, my_rest, opp_act, opp_rest
    sc[6], sc[7] = my_hand, opp_hand
    sc[12], sc[13] = 0.5, 0.5                     # リーダー 5000/5000
    sc[14], sc[15] = my_ldon / 5.0, opp_ldon / 5.0
    return sc


def _tok(own=(), opp=()):
    """`own`/`opp` は (パワー, 付与ドン) の列。"""
    tok = np.zeros((22, 24), np.float32)
    for base, bodies in ((2, own), (7, opp)):
        for k, (pw, don) in enumerate(bodies):
            tok[base + k, PR.S_POWER] = pw / 1e4
            tok[base + k, PR.S_ATTACHED_DON] = don / 5.0
            tok[base + k, PR.S_IS_CHAR] = 1.0
    return tok


def test_the_realised_nu_is_the_measured_band_value_not_the_formula():
    """実現の側の `ν` は**帯ごとの実測値**——式 `nu_of` とは別物（式の欠陥が実現に混ざらない）。"""
    assert PR.nu_meas_of(3000.0, 5000.0) == PR.NU_MEAS["lt_leader"]
    assert PR.nu_meas_of(5000.0, 5000.0) == PR.NU_MEAS["leader_to_sat"]
    assert PR.nu_meas_of(7000.0, 5000.0) == PR.NU_MEAS["leader_to_sat"]     # 飽和点まで
    assert PR.nu_meas_of(7000.0 + 2 * PR.PWR_EPS, 5000.0) == PR.NU_MEAS["over_sat"]
    # 式の側（`base`）はリーダー未満を 0 と言う——実測は 0.069。ここが混ざっていないことの証拠
    assert T.nu_of(3000.0, 5000.0, 4.128, is_blocker=False, mode="base") == 0.0
    assert PR.nu_meas_of(3000.0, 5000.0) > 0.0


def test_the_don_stock_counts_active_rested_and_attached_alike():
    """**ドンは総在庫**——付与しても動かず、`RAMP_DON` だけが増やす。"""
    tok = _tok(own=((5000, 2),), opp=())
    sc = _sc(my_act=3, my_rest=1, my_ldon=1)
    assert PR.don_stock(sc, tok, "me") == pytest.approx(3 + 1 + 1 + 2)
    assert PR.don_stock(sc, tok, "opp") == pytest.approx(3)
    # 付与＝アクティブ 1 減・キャラ付与 1 増 → 総在庫は不変
    tok2 = _tok(own=((5000, 3),), opp=())
    sc2 = _sc(my_act=2, my_rest=1, my_ldon=1)
    assert PR.don_stock(sc2, tok2, "me") == pytest.approx(PR.don_stock(sc, tok, "me"))
    assert PR.state_meas(sc2, tok2) == pytest.approx(PR.state_meas(sc, tok))


def test_the_state_is_priced_with_measured_prices_and_is_antisymmetric():
    """`S_meas` は 4 通貨の実測価格の差の和。等しい盤面は 0・差は席を入れ替えると符号が返る。"""
    assert PR.state_meas(_sc(), _tok()) == pytest.approx(0.0)
    assert PR.state_meas(_sc(my_life=5), _tok()) == pytest.approx(PR.LAM)
    assert PR.state_meas(_sc(my_hand=6), _tok()) == pytest.approx(PR.MU)
    assert PR.state_meas(_sc(my_act=4), _tok()) == pytest.approx(PR.DELTA)
    assert PR.state_meas(_sc(), _tok(own=((3000, 0),))) == pytest.approx(PR.NU_MEAS["lt_leader"])
    assert PR.state_meas(_sc(), _tok(opp=((3000, 0),))) == pytest.approx(-PR.NU_MEAS["lt_leader"])
    a = PR.state_meas(_sc(my_life=3, opp_life=5, my_hand=7, opp_hand=4), _tok(own=((9000, 0),)))
    b = PR.state_meas(_sc(my_life=5, opp_life=3, my_hand=4, opp_hand=7), _tok(opp=((9000, 0),)))
    assert a == pytest.approx(-b)


def test_the_effect_breakdown_reads_the_first_action_of_the_activated_ability():
    """効果の型の内訳は**最初の能力の最初の動作の型**で切る（読めなければ `?`）。"""
    assert PR.primary_action("OP15-058") == "RAMP_DON"        # エネル: ドン 1＋4 を追加し 4 を付与
    assert PR.primary_action("__no_such_card__") == "?"
    assert PR.primary_action(None) == "?"


def test_the_gross_price_adds_back_the_don_opportunity_cost_the_theory_charges():
    """登場・イベントの価格が引く `費用·δ` は**機会費用**で在庫の差分に現れない——`gross` で足し戻す。"""
    assert PR.DON_COST == pytest.approx(0.66 * PR.MU)
    pv = T.play_value(5000.0, 4, 5000.0, 4.128)
    assert pv + 4 * PR.DON_COST == pytest.approx(T.play_value(5000.0, 0, 5000.0, 4.128))


def test_attached_don_does_not_lift_the_band_of_the_realised_body_value():
    """**付けたドンは実現に暗黙に加算されない**（ユーザ指摘 2026-09-16）——`power_now` は所有者のターンに
    付与ドンを載せるので、帯は素のパワー（`power_now − 1000·付与ドン`）で決める。"""
    bare = _tok(own=((4000, 0),))
    with_don = _tok(own=((6000, 2),))        # 4000 に 2 枚付けて 6000 に見えている
    assert PR.side_nu_meas(bare, PR.SLOT_OWN_FIELD, 5000.0) == pytest.approx(PR.NU_MEAS["lt_leader"])
    assert PR.side_nu_meas(with_don, PR.SLOT_OWN_FIELD, 5000.0) == pytest.approx(PR.NU_MEAS["lt_leader"])
    # 付与＝アクティブ 2 減・キャラ付与 2 増（総在庫は同じ）→ 盤面の評価も同じ
    assert PR.state_meas(_sc(my_act=1), with_don) == pytest.approx(PR.state_meas(_sc(my_act=3), bare))


def test_the_turn_end_column_and_the_flow_split_do_not_touch_the_family_sums():
    """**T53**: 行 → ターン末の実現は参考値（後の行と重なる）で、型の和は次の判断点の実現のまま。
    ターン単位の恒等式は「後で効く効果が在るターン」と無いターンで分けて出す。登場の内訳は部品の平均。"""
    def rec(w, rows_, turns):
        return {"seed": 1, "who": w, "z": 1.0 if w == 0 else 0.0,
                "price": {f: sum(r["price"] for r in rows_ if r["fam"] == f) for f in PR.MOVE_FAMILIES},
                "real": {f: sum(r["real"] for r in rows_ if r["fam"] == f) for f in PR.MOVE_FAMILIES},
                "n": {f: sum(1 for r in rows_ if r["fam"] == f) for f in PR.MOVE_FAMILIES},
                "rows": rows_, "turns": turns}
    pp = {"nu_minus_mu": 0.1, "effect": 0.02, "opportunity": 0.003, "opp_life": 0.0, "opp_hand": 0.0,
          "opp_body": 0.0, "my_life": 0.0, "my_hand": -0.055, "my_body": 0.15, "don": 0.0}
    rows_a = [{"fam": "effect", "price": 0.02, "real": 0.0, "real_te": 0.08, "gross": 0.02, "act": "ACTIVE_DON",
               "cid": "x", "turn": 1, "play_parts": None}] * 25 + \
             [{"fam": "play", "price": 0.12, "real": 0.09, "real_te": 0.09, "gross": 0.123, "act": None,
               "cid": "y", "turn": 3, "play_parts": pp}] * 25
    turns_a = {1: {"price": 0.05, "first": 0.0, "last": 0.10, "acts": {"ACTIVE_DON"}},
               3: {"price": 0.12, "first": 0.0, "last": 0.09, "acts": set()}}
    turns_b = {2: {"price": 0.04, "first": 0.0, "last": 0.02, "acts": set()}}
    per = {(1, 0): rec(0, rows_a, turns_a), (1, 1): rec(1, [], turns_b)}
    per_many = {}
    for s in range(25):                      # ターンの群は 20 以上で出す
        a = rec(0, rows_a, turns_a); a["seed"] = s
        b = rec(1, [], turns_b); b["seed"] = s
        per_many[(s, 0)] = a; per_many[(s, 1)] = b
    out = PR.summarise(per_many, reps=10)
    eff = out["by_family"]["effect"]
    assert eff["real_mean"] == pytest.approx(0.0) and eff["real_turn_end_mean"] == pytest.approx(0.08)
    assert out["effect_by_action"]["ACTIVE_DON"]["real_turn_end_mean"] == pytest.approx(0.08)
    g = out["turns_by_flow_effect"]
    assert g["with_flow_effect"]["turns"] == 25 and g["with_flow_effect"]["ratio_real_over_price"] == pytest.approx(2.0)
    assert g["without"]["turns"] == 50 and g["without"]["price_mean"] == pytest.approx(0.08)
    pb = out["play_breakdown"]
    assert pb["n"] == 25 * 25 and pb["nu_minus_mu"] == pytest.approx(0.1) and pb["my_body"] == pytest.approx(0.15)
    assert PR.FLOW_ACTS >= {"ACTIVE_DON", "GRANT_KEYWORD", "BUFF", "REST"}


def test_the_hand_quality_yardstick_counts_cards_that_enter_the_hand_at_their_gain():
    """**T69**: 物差しは入った札を μ ではなく `max(ΔH, ΔG)` で数える＝補正は Σ(gain − μ)・`count` なら 0・不正な mode は弾く。"""
    before = PR.HAND_MEAS_MODE
    try:
        assert PR.quality_correction([0.08, 0.0], mu=0.05) == pytest.approx(0.08 - 0.05 + 0.0 - 0.05)
        assert PR.quality_correction([], mu=0.05) == 0.0
        PR.set_hand_meas_mode("count")
        assert PR.hand_quality_delta(None, None, None, None, None, None) == (0.0, [])      # 旧規約は何も読まない
        with pytest.raises(ValueError):
            PR.set_hand_meas_mode("guess")
        assert PR.set_hand_meas_mode("quality") == "quality"
    finally:
        PR.set_hand_meas_mode(before)
