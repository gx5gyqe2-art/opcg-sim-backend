"""`effect_value.py`（効果をコストと実行内容で値付け・P2-1）の写像を固める。

**この器の値打ちは「自由なつまみが無い」こと**——写像表だけで係数を選ぶ余地が無い。
符号（誰に効くか）と「値付けできないものを 0 にしない」が壊れると、
T16 が止まった「勘定が複数の仕方で閉じられる」状態に戻る。
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

import effect_value as E  # noqa: E402
import theory_order as T  # noqa: E402


def _act(kind, player="SELF", count=1, base=0, zone=None, up_to=False):
    return {"type": kind, "value": {"base": base, "multiplier": 1, "divisor": 1},
            "target": {"player": player, "count": count, "zone": zone,
                       "is_up_to": up_to}}


def test_who_the_action_hits_decides_the_sign():
    """**零和**——相手の資源が動けば自分にとっては逆符号。"""
    assert E.action_value(_act("KO", "OPPONENT")) > 0      # 相手の体を消す＝得
    assert E.action_value(_act("KO", "SELF")) < 0          # 自分の体が消える＝損
    assert E.action_value(_act("DRAW")) > 0                # 自分が引く＝得
    assert E.action_value(_act("DISCARD", "OPPONENT")) > 0  # 相手が落とす＝得
    assert E.action_value(_act("DISCARD", "SELF")) < 0


def test_removing_a_body_and_making_one_have_opposite_signs():
    """**`KO` は体が消え、`PLAY_CARD` は増える**——同じ `ν` でも向きが逆。"""
    assert E.action_value(_act("PLAY_CARD", "SELF")) > 0
    assert E.action_value(_act("KO", "SELF")) < 0
    assert E.action_value(_act("PLAY_CARD", "SELF")) == pytest.approx(
        -E.action_value(_act("KO", "SELF")))


def test_a_power_buff_is_converted_through_don_not_a_new_constant():
    """**`+1000` パワー ≈ ドン 1 個**——ここで新しい係数を作らないのが肝。"""
    got = E.action_value(_act("BUFF", "SELF", base=2000))
    assert got == pytest.approx(2.0 * E.DELTA)
    # 相手のパワーを下げる（負の base を相手に）のは自分の得
    assert E.action_value(_act("BUFF", "OPPONENT", base=-3000)) > 0


def test_life_and_don_use_the_measured_prices():
    assert E.action_value(_act("HEAL", "SELF", count=1)) == pytest.approx(T.LAM)
    assert E.action_value(_act("RAMP_DON", "SELF", count=1)) == pytest.approx(E.DELTA)


def test_returning_don_is_a_loss_but_resting_it_is_tempo():
    """**戻ってくるものは `δ` の損ではない**——`REST_DON` はテンポの系統へ送る。"""
    assert E.action_value(_act("RETURN_DON", "OPPONENT", count=2)) > 0
    assert E.action_value(_act("REST_DON", "OPPONENT")) is None
    assert E.family_of("REST_DON") == "tempo"
    assert E.family_of("RETURN_DON") == "other"      # 写せるので系統分けは要らない
    # **減るドンは持ち主の損**——相手に撃てば自分の得（`DRAW` 対 `DISCARD` と同じ）
    assert E.action_value(_act("RETURN_DON", "SELF", count=2)) < 0


def test_trash_depends_on_the_zone_it_hits():
    """`TRASH` は場なら体・手札なら札——**ゾーンで意味が変わる**。"""
    assert E.action_value(_act("TRASH", "OPPONENT", zone="FIELD")) == pytest.approx(
        E.NU_AVG)
    assert E.action_value(_act("TRASH", "OPPONENT", zone="HAND")) == pytest.approx(T.MU)
    assert E.action_value(_act("TRASH", "OPPONENT", zone=None)) is None


def test_an_action_with_no_price_returns_none_not_zero():
    """**0 にしない**——「効果が無い」と「値付けできていない」を混ぜない。

    `exit_ledger` の除去で踏んだ形（0 を返して「起きていない」と誤読させた）。
    """
    for kind in ("LOOK", "REST", "GRANT_KEYWORD", "ARRANGE"):
        assert E.action_value(_act(kind, "OPPONENT")) is None


def test_the_families_name_what_is_still_missing():
    """値付けできない動作は**系統に分類して出す**＝次に価格を付ける場所が判る。"""
    assert E.family_of("LOOK") == "info"
    assert E.family_of("DECK_BOTTOM") == "info"
    assert E.family_of("ATTACK_DISABLE") == "tempo"
    assert E.family_of("REPLACE_EFFECT") == "grant"


def test_all_targets_are_capped_at_the_field_limit():
    """`count = -1`（すべて）は**場の上限 5 体**で打ち切る（無限に足さない）。"""
    assert E._count({"count": -1}) == E.ALL_COUNT
    assert E._count({"count": 3}) == 3
    assert E._count(None) == 1


def test_a_cost_is_always_subtracted():
    """**コストは必ず損**——写像表の符号ではなく「役割」で向きが決まる。

    コストで自分のキャラをリリースしても、表の上では `KO(SELF)` で既に負。
    そこを更に符号で扱うと二重になるので、**絶対値を引く**と決めてある。
    """
    ab = {"effect": _act("DRAW", "SELF", count=2),
          "cost": {"actions": [_act("KO", "SELF", count=1)]}}
    v, unp = E.ability_value(ab)
    assert unp == []
    assert v == pytest.approx(2 * T.MU - E.NU_AVG)
    # **コストを引かなければ 2μ のまま**——引いていることを値で押さえる
    assert v < 2 * T.MU


def test_one_unpriced_action_makes_the_whole_ability_unpriced():
    """**部分的に足して 0 扱いにしない**——読めない項がある能力は `None` を返す。"""
    ab = {"effect": {"type": "DRAW", "value": {"base": 1}, "target": {"player": "SELF"},
                     "sub_effect": _act("LOOK", "SELF")}}
    v, unp = E.ability_value(ab)
    assert v is None
    assert ("LOOK", "info") in unp


def test_sub_effects_are_walked():
    """効果は木なので `sub_effect` を辿る（辿らないと項が落ちる）。"""
    ab = {"effect": {"type": "DRAW", "value": {"base": 1}, "target": {"player": "SELF"},
                     "sub_effect": _act("DRAW", "SELF", count=1)}}
    v, unp = E.ability_value(ab)
    assert unp == [] and v == pytest.approx(2 * T.MU)


def test_the_shipped_card_pool_is_priced_about_half_and_the_rest_is_classified():
    """**同梱のカードで回ることを押さえる**（写像表が壊れたら被覆率が落ちる）。

    値は動いてよいが、**系統分けの合計が未値付けの数と一致する**ことは常に成り立つ。
    """
    cards = E.load_cards()
    out = E.summarise(cards)
    assert out["abilities"] > 3000
    assert 0.3 < out["coverage"] < 0.9
    assert set(out["unpriced_by_family"]) <= {"info", "tempo", "grant", "other"}
