"""理論の順序（4 通貨で候補手を並べる）の算術（`tests/scripts/theory_order.py`）。

基盤健全性（`cpu_infra`）。記録もエンジンも要らない純関数だけを固める。要は 4 つ:

1. **攻撃の価値は `min`**（相手が「守る／受ける」の安い方を選ぶ）＝`game_theory.md` §14.1。
2. **`x < 0` の攻撃は価値 0**（通らない）。
3. **飽和点を超えたドン付与は価値 0**（相手はもう受けるので段を買えない）。
4. **理論と方策を同じペア集合で比べる**（値付けできない候補を外した後の集合）。
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

import theory_order as T  # noqa: E402


def test_cost_curve_is_a_staircase_and_tolerates_f16_rounding():
    assert T.c_of(-3000) == 0.0                         # 通らない攻撃は守る必要が無い
    assert T.c_of(0) == pytest.approx(1.00)             # **x=0 でも命中するので 1 枚要る**
    assert T.c_of(1000) == pytest.approx(1.00)
    assert T.c_of(1000.0002) == pytest.approx(1.00)     # f16 の丸めで段が上がらない
    assert T.c_of(1500) == pytest.approx(1.28)          # 段の途中は上の段の値
    assert T.c_of(3000) == pytest.approx(2.25)
    assert T.c_of(6000) == pytest.approx(3.63 + 0.66)   # 5000 超は平均の傾きで伸ばす


def test_saturation_point_moves_with_theta():
    """`x* = min{x : c(x) ≥ Θ}`。**Θ が上がると飽和点も上がる**（相手のライフが薄い帯）。"""
    assert T.saturation_x(0.8) == 1000.0                # c(1000)=1.00 ≥ 0.8
    assert T.saturation_x(1.15) == 2000.0               # c(2000)=1.28 ≥ 1.15
    assert T.saturation_x(2.0) == 3000.0
    assert T.saturation_x(10.0) == 5000.0               # 曲線の端で止める


def test_attack_on_leader_is_the_min_of_guarding_and_taking():
    """守る費用が受ける費用を超えたら、**それ以上は価値が増えない**（飽和）。"""
    mu, theta = 0.05, 1.15
    take = theta * mu
    # x=1000 → c=1.00 枚 < Θ なので守る方が安い＝守る費用が価値
    assert T.attack_value(6000, 5000, True, theta, mu) == pytest.approx(1.00 * mu)
    # x=3000 → c=2.25 枚 > Θ なので相手は受ける＝価値は take で止まる
    assert T.attack_value(8000, 5000, True, theta, mu) == pytest.approx(take)
    # さらに積んでも増えない
    assert T.attack_value(12000, 5000, True, theta, mu) == pytest.approx(take)


def test_attack_that_cannot_connect_is_worth_zero():
    """自分のパワーが対象以下なら通らない＝価値 0（実測で打った攻撃の 38.5% がここ）。"""
    assert T.attack_value(4000, 5000, True) == 0.0
    # 同値は通る（攻撃側 ≥ 対象）＝守るには 1 枚要るので、その分の価値が在る
    assert T.attack_value(5000, 5000, True, 1.15, 0.05) == pytest.approx(1.00 * 0.05)


def test_attack_on_character_compares_against_that_character():
    """キャラ狙いは「守る費用」と「そのキャラを失う損」の min。"""
    mu, theta = 0.05, 1.15
    cheap_body = 0.001
    v = T.attack_value(7000, 5000, False, theta, mu, nu_target=cheap_body)
    assert v == pytest.approx(cheap_body)               # 体が安いならそれが上限
    big_body = 10.0
    v2 = T.attack_value(7000, 5000, False, theta, mu, nu_target=big_body)
    assert v2 == pytest.approx(T.c_of(2000) * mu)       # 体が高いなら守る費用が上限


def test_attach_don_is_the_increment_of_the_attack_value():
    """付与の価値は**攻撃の価値の増分**＝平らな段では 0・飽和点より上でも 0。"""
    mu, theta = 0.05, 1.15                              # 飽和点 x* = 2000
    # **x 0→1000 は 0**（実測の曲線は c(0)=c(1000)=1.00＝1 枚でどちらも止まる）
    assert T.attach_value(5000, 5000, 1, theta, mu) == 0.0
    # x 1000→2000 は Θ で潰れた分だけ（1.15 − 1.00）
    assert T.attach_value(6000, 5000, 1, theta, mu) == pytest.approx((1.15 - 1.00) * mu)
    # 既に飽和点に居るなら足しても 0
    assert T.attach_value(7000, 5000, 1, theta, mu) == 0.0
    assert T.attach_value(9000, 5000, 3, theta, mu) == 0.0
    # 通らない攻撃に付与しても 0（x = −2000 → +1000 でも届かない）
    assert T.attach_value(3000, 5000, 1, theta, mu) == 0.0
    # **`attack_value` の差と厳密に一致する**
    for k in (1, 2, 3):
        inc = (T.attack_value(6000 + 1000 * k, 5000, True, theta, mu)
               - T.attack_value(6000, 5000, True, theta, mu))
        assert T.attach_value(6000, 5000, k, theta, mu) == pytest.approx(inc)


def test_play_pays_the_card_and_the_don():
    """登場は `ν − μ − cost·δ`＝**高コストで弱い体は負の値**になる。"""
    mu = 0.05
    weak_expensive = T.play_value(1000, 9, 5000, 3, 1.15, mu)
    strong_cheap = T.play_value(8000, 2, 5000, 3, 1.15, mu)
    assert weak_expensive < 0 < strong_cheap


def test_score_candidate_marks_unscorable_as_none():
    class _Cards:
        def info(self, cid):
            return {"C_ATK": {"power": 7000, "cost": 3, "leader": False, "event": False},
                    "C_LEAD": {"power": 5000, "cost": 0, "leader": True, "event": False},
                    "C_EV": {"power": 0, "cost": 1, "leader": False, "event": True}}.get(cid)

    ctx = {"theta": 1.15, "mu": 0.05, "opp_leader_power": 5000.0,
           "my_leader_power": 5000.0, "r_turns": 3.0, "don_k": 1}
    cards = _Cards()
    assert T.score_candidate(["TURN_END", None, [], [], None], None, None, ctx, cards) == 0.0
    assert T.score_candidate(["ACTIVATE_MAIN", "u", [], [], None], "C_ATK", None, ctx,
                             cards) is None            # 効果の中身が要る
    assert T.score_candidate(["PLAY", "u", [], [], None], "C_EV", None, ctx,
                             cards) is None            # イベントは値付けできない
    assert T.score_candidate(["PLAY", "u", [], [], None], "C_ATK", None, ctx,
                             cards) is not None
    atk = T.score_candidate(["ATTACK", "u", ["t"], [], None], "C_ATK", "C_LEAD", ctx, cards)
    assert atk == pytest.approx(T.attack_value(7000, 5000, True, 1.15, 0.05))


def _cards():
    class _Cards:
        def info(self, cid):
            return {"C_ATK": {"power": 7000, "cost": 3, "leader": False, "event": False},
                    "C_LEAD": {"power": 5000, "cost": 0, "leader": True, "event": False},
                    "C_CHAR": {"power": 4000, "cost": 2, "leader": False, "event": False}}.get(cid)
    return _Cards()


CTX = {"theta": 1.15, "mu": 0.05, "opp_leader_power": 5000.0,
       "my_leader_power": 5000.0, "r_turns": 3.0, "don_k": 1}


def test_don_box_with_a_target_is_an_attack_not_an_attach():
    """**記録に `ATTACK` は出てこない**——攻撃は `DON_BOX`（対象付き）の形で来る。

    `search/decide.rs::don_box_first_primitive` は `don_k<=0` かつ対象付きの `DON_BOX` を
    素の `ATTACK` に落とす＝**対象付きの `DON_BOX` は「ドンを k 枚付けてから殴る」**。
    2026-09-13 にここを付与として値付けしていた（全候補の 44% を誤った式で測っていた）。
    """
    cards = _cards()
    box = T.score_candidate(["DON_BOX", "u", ["t"], [], None], "C_ATK", "C_LEAD", CTX, cards)
    # ドン 1 枚を付けてから殴る＝8000 対 5000（超過 3000）の攻撃
    assert box == pytest.approx(T.attack_value(8000, 5000, True, 1.15, 0.05))
    # 付与として測ると別の値になる（＝取り違えは数字に出る）
    assert box != pytest.approx(T.attach_value(7000, 5000, 1, 1.15, 0.05))


def test_don_box_without_a_target_is_still_an_attach():
    cards = _cards()
    bare = T.score_candidate(["DON_BOX", "u", [], [], None], "C_ATK", None, CTX, cards)
    assert bare == pytest.approx(T.attach_value(7000, 5000, 1, 1.15, 0.05))


def test_a_character_target_is_priced_against_that_character():
    """キャラ狙いは `min(c(x)·μ, ν(対象))`＝リーダー狙いと別の式になる。"""
    cards = _cards()
    ch = T.score_candidate(["DON_BOX", "u", ["t"], [], None], "C_ATK", "C_CHAR", CTX, cards)
    nu_t = T.nu_of(4000, 5000.0, 3.0, 1.15, 0.05)
    assert ch == pytest.approx(T.attack_value(8000, 4000, False, 1.15, 0.05, nu_target=nu_t))


def _box(src=None, tgt=None):
    return T.score_candidate(["DON_BOX", "u", ["t"], [], None], "C_ATK", "C_LEAD", CTX, _cards(),
                             src_power=src, tgt_power=tgt)


def test_current_power_overrides_the_printed_power():
    """ドンが付いたキャラや強化されたキャラは**印字では測れない**（枠の現在値を渡す）。

    印字 7000 の札でも枠が 4000 なら弱い攻撃として値付けされる（逆も同じ）。
    """
    assert _box(src=4000.0) == pytest.approx(T.attack_value(5000, 5000, True, 1.15, 0.05))
    # 対象側の現在パワーも効く（強くなった対象は殴りにくい＝通らないので 0）
    assert _box(tgt=20000.0) == 0.0


def test_piling_power_past_the_saturation_point_buys_nothing():
    """**飽和点を超えたら価値は増えない**（`min(c(x), Θ)` の天井＝理論の中心の予言）。

    Θ=1.15 なので飽和点は 2000（`c(2000) = 1.28 ≥ Θ`）。5000 のリーダー相手なら
    7000 で天井に当たり、そこから積んでも同じ値になる。
    """
    assert T.saturation_x(1.15) == 2000.0
    cap = 1.15 * 0.05
    assert _box(src=6000.0) == pytest.approx(cap)      # +1 ドンで 7000＝超過 2000
    assert _box(src=9000.0) == pytest.approx(cap)      # 10000 でも同じ
    # 飽和より下では**ちゃんと増える**（天井が効いているだけで式が死んでいるのではない）
    assert _box(src=4000.0) < cap


def test_row_order_uses_the_same_pairs_for_theory_and_prior():
    """**値付けできない候補を外した後の同じ集合**で両方を測る（比較が公平になる）。"""
    n = [100.0, 90.0, 80.0, 1.0]        # 4 本目は訪問が足りない
    q = [0.5, 0.3, 0.1, 0.9]
    p = [0.1, 0.2, 0.3, 0.4]            # 方策は Q と逆順
    theory = [3.0, 2.0, 1.0, None]      # 理論は Q と同順・4 本目は値付け不能
    out = T.row_order(n, q, p, theory, n_min=5, q_eps=0.0, n_min_frac=0.0)
    assert out["k"] == 4 and out["k_scored"] == 3 and out["k_kept"] == 3
    assert out["th_pairs"] == out["p_pairs"] == 3       # 同じペア集合
    assert out["th_agree"] == 3                        # 理論は全ペア一致
    assert out["p_agree"] == 0                         # 方策は全ペア逆
    # 値付けできる候補が 1 本以下ならペアは作れない
    thin = T.row_order(n, q, p, [1.0, None, None, None], n_min=5, q_eps=0.0, n_min_frac=0.0)
    assert thin["th_pairs"] == 0 and thin["p_pairs"] == 0


def test_block_and_verdict():
    recs = [{"th_agree": 6, "th_pairs": 10, "p_agree": 5, "p_pairs": 10,
             "k": 5, "k_scored": 4, "k_kept": 4, "seed": s} for s in (1, 2, 3)]
    b = T.block(recs)
    assert b["order_acc_theory"] == pytest.approx(0.6)
    assert b["order_acc_prior"] == pytest.approx(0.5)
    assert b["gain"] == pytest.approx(0.1)
    assert b["games"] == 3 and b["theory_se"] == 0.0
    assert T.verdict(b) == "price_teaches"
    low = T.block([dict(r, th_agree=5) for r in recs])
    assert T.verdict(low) == "price_does_not_teach"
    mid = T.block([dict(r, th_agree=53, th_pairs=100, p_pairs=100) for r in recs])
    assert T.verdict(mid) == "partly"
    assert T.verdict(T.block([])) is None
