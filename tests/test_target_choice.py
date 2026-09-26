"""`target_choice.py`（攻撃の対象選びは正しかったか）の算術を固める。

この器は **`shield_rate` の妥当性を点検する**ためのものなので、判定の向きが逆に出ると
「身代わりは水増しか過小か」の結論がそのまま反転する。**向きを決める箇所を全部固定する**。
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

import shield_rate as S  # noqa: E402
import target_choice as TC  # noqa: E402
import theory_order as T  # noqa: E402


def _rec(x_lead, hit_char, v_actual, v_best_char, foe_life=3, seed=1):
    return {"x_lead": x_lead, "hit_char": hit_char, "v_actual": v_actual,
            "v_best_char": v_best_char, "foe_life": foe_life, "seed": seed}


def test_a_leader_attack_that_cannot_connect_is_worth_nothing():
    """`x_lead < 0` は 0（`shield_rate.absorb_value` と同じ規約）。"""
    assert TC.leader_value(-2000.0) == 0.0
    assert TC.leader_value(0.0) > 0.0
    assert TC.leader_value(20000.0) == pytest.approx(T.THETA * T.MU)   # Θμ で頭打ち


def test_a_character_attack_is_capped_by_the_measured_nu_not_the_formula():
    """**`ν` は実測の帯別を使う**——式の `ν` はこの帯で 0 なので判定が自明になる。"""
    v = TC.char_value(12000.0, 3000.0, "lt_leader")
    assert v == pytest.approx(S.NU_MEASURED["lt_leader"])     # 大きく上回れば ν で頭打ち
    # 上回れない攻撃は通らない＝0
    assert TC.char_value(2000.0, 3000.0, "lt_leader") == 0.0
    # わずかに上回るだけなら c(x) 側が効く（ν より安い）
    assert 0.0 < TC.char_value(3000.0, 3000.0, "lt_leader") < S.NU_MEASURED["lt_leader"]


def test_hitting_a_character_when_the_leader_was_worth_more_is_counted_wrong():
    """**ユーザ指摘の形**——殴る必要が無かったのに殴った＝身代わり率の水増し。"""
    # リーダー狙い 0.0634（Θμ）対 キャラ狙い 0.01 → キャラを殴ったのは誤り
    out = TC.judge([_rec(5000.0, True, 0.01, 0.01)], theta=1.15)
    assert out["char_wrong"] == 1 and out["char_wrong_share"] == 1.0
    assert out["char_wrong_loss"] > 0.0
    # 逆にキャラの方が高ければ誤りではない
    ok = TC.judge([_rec(5000.0, True, 0.20, 0.20)], theta=1.15)
    assert ok["char_wrong"] == 0


def test_hitting_the_leader_when_a_character_was_worth_more_is_counted_wrong():
    """**逆向きも数える**——これを落とすと「水増し」しか見えなくなる。"""
    out = TC.judge([_rec(5000.0, False, 0.0, 0.20)], theta=1.15)
    assert out["face_wrong"] == 1 and out["face_wrong_share"] == 1.0
    assert out["face_wrong_missed"] > 0.0


def test_the_headline_ratio_counts_both_directions():
    """`最適な吸収 / 観測` = （正しかったキャラ狙い ＋ 殴るべきだったリーダー狙い）／観測。

    **1 を下回れば身代わり率は水増し・上回れば過小**（事前登録の読み方）。
    """
    # キャラ狙い 2 本（1 本は誤り）・リーダー狙い 2 本（1 本はキャラを殴るべきだった）
    recs = [_rec(5000.0, True, 0.20, 0.20), _rec(5000.0, True, 0.01, 0.01),
            _rec(5000.0, False, 0.0, 0.20), _rec(5000.0, False, 0.0, 0.01)]
    out = TC.judge(recs, theta=1.15)
    assert out["char_attacks"] == 2 and out["face_attacks"] == 2
    assert out["optimal_absorbs_over_observed"] == pytest.approx((1 + 1) / 2)


def test_the_verdict_needs_every_theta_to_agree():
    """**向きが全ての `Θ` で揃って初めて主張になる**——揃わなければ「判定できない」。"""
    up = [{"optimal_absorbs_over_observed": v} for v in (1.2, 3.1)]
    down = [{"optimal_absorbs_over_observed": v} for v in (0.4, 0.9)]
    mixed = [{"optimal_absorbs_over_observed": v} for v in (0.8, 1.3)]
    assert TC.verdict(up) == "cpu_under_attacks_characters"
    assert TC.verdict(down) == "cpu_over_attacks_characters"
    assert TC.verdict(mixed) == "depends_on_theta"
    assert TC.verdict([]) is None


def test_raising_theta_makes_leader_attacks_look_better():
    """**判定は `Θ` に強く依る**——だから掃引する。

    `Θ` を上げるとリーダー狙いの価値が上がり、「キャラを殴るべきだった」が減る。
    この単調性が崩れたら、掃引の読み方そのものが変わる。
    """
    recs = [_rec(5000.0, False, 0.0, 0.10) for _ in range(10)]
    low = TC.judge(recs, theta=1.15)["face_wrong_share"]
    high = TC.judge(recs, theta=5.00)["face_wrong_share"]
    assert low == 1.0 and high == 0.0


def test_by_life_drops_thin_cells():
    """ライフ別は薄いセルを出さない（**定数 `Θ` の壊れ方を見る**のが目的なので）。"""
    recs = ([_rec(5000.0, False, 0.0, 0.20, foe_life=0) for _ in range(40)]
            + [_rec(5000.0, False, 0.0, 0.20, foe_life=5) for _ in range(5)])
    out = TC.by_life(recs, theta=1.15, min_n=30)
    assert set(out) == {"0"}
    assert out["0"]["face_wrong_share"] == 1.0
