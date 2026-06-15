"""CPU 自デッキ勝ち筋プラン（cpu_self_plan）＋ evaluate へのプラン補正のテスト（docs/SPEC.md §2.5.5）。

方針: プラン未指定（plan=None）では現行挙動と完全同値（回帰ガード）。プラン供給時のみ、自分側の
評価重み（置物の存在価値・カウンター温存）と逆算項（リーサル誘導）がデッキ依存で作動することを検証。
"""
import random
import types

import conftest  # noqa: F401  (google スタブ注入 & sys.path 設定)
import pytest

from opcg_sim.src.core import cpu_ai, cpu_self_plan
from opcg_sim.src.core.gamestate import GameManager, Player
from cpu_selfplay import build_deck, _load_db


# ---------------------------------------------------------------------------
# build_plan（構成からの自動分類・純関数）
# ---------------------------------------------------------------------------

def _master(counter=0, cost=3, keywords=None, text=""):
    return types.SimpleNamespace(counter=counter, cost=cost,
                                 keywords=set(keywords or []), effect_text=text)


def test_build_plan_empty_is_neutral():
    p = cpu_self_plan.build_plan([])
    assert p is cpu_self_plan.NEUTRAL
    assert p.archetype == "midrange"
    # 中立は全乗数 1.0（≒現行挙動）。
    assert (p.vanilla_body_mult, p.attacker_mult, p.life_mult, p.counter_mult) == (1.0, 1.0, 1.0, 1.0)


def test_build_plan_low_cost_no_counter_is_aggro():
    cards = [_master(counter=0, cost=1)] * 10 + [_master(counter=0, cost=2)] * 10
    p = cpu_self_plan.build_plan(cards)
    assert p.archetype == "aggro"
    assert p.aggro_lean > 0.6
    assert p.vanilla_body_mult >= 1.0 and p.attacker_mult > 1.0 and p.clock_rate > 1.0


def test_build_plan_counter_heavy_is_control():
    cards = ([_master(counter=2000, cost=5, keywords=["ブロッカー"], text="このキャラをKOする")] * 8
             + [_master(counter=1000, cost=4)] * 12)
    p = cpu_self_plan.build_plan(cards)
    assert p.archetype == "control"
    assert p.aggro_lean < 0.4
    # コントロールは置物を割り引き・カウンター温存を重視。
    assert p.vanilla_body_mult < 1.0 and p.counter_mult > 1.0 and p.life_mult >= 1.0


# ---------------------------------------------------------------------------
# evaluate へのプラン補正（実カード使用）
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def db():
    return _load_db()


def _new_gm(db):
    random.seed(0)
    l1, c1 = build_deck(db, "p1")
    l2, c2 = build_deck(db, "p2")
    gm = GameManager(Player("p1", c1, l1), Player("p2", c2, l2))
    gm.start_game()
    return gm


def _low_impact_char(gm):
    """効果なし・素パワー<5000・関連キーワード無しのキャラ（置物）を 1 枚見つける。"""
    for c in list(gm.p1.deck):
        if c.master.type.name == "CHARACTER" and cpu_ai._is_low_impact(c):
            return c
    return None


def _plan(archetype):
    return cpu_self_plan.PlanProfile(
        n_cards=50, archetype=archetype,
        aggro_lean=0.8 if archetype == "aggro" else 0.2,
        avg_cost=2.0, **cpu_self_plan._PRESETS[archetype])


def test_plan_none_is_regression_identical(db):
    """plan=None は現行挙動（プラン引数の無い呼び出し）と完全同値。"""
    gm = _new_gm(db)
    assert cpu_ai.evaluate(gm, "p1") == cpu_ai.evaluate(gm, "p1", plan=None)


def test_is_low_impact_detection(db):
    """効果/関連キーワードを持つ体は置物扱いしない（割引対象外）。"""
    gm = _new_gm(db)
    low = _low_impact_char(gm)
    assert low is not None, "テスト用の置物（効果なし低パワー）が見つからない"
    # 効果テキストを持つキャラは置物ではない。
    eff = next((c for c in gm.p1.deck if c.master.type.name == "CHARACTER"
                and (c.master.effect_text or "").strip()), None)
    if eff is not None:
        assert not cpu_ai._is_low_impact(eff)


def test_control_discounts_vanilla_body_aggro_boosts(db):
    """同じ置物キャラを場に足したときの評価上昇は control < none < aggro（デッキ依存の置物許容度）。"""
    gm = _new_gm(db)
    c = _low_impact_char(gm)
    assert c is not None
    gm.p1.deck.remove(c)

    def delta(plan):
        before = cpu_ai.evaluate(gm, "p1", plan=plan)
        gm.p1.field.append(c)
        c.is_rest = False
        c.is_newly_played = False  # 確立済み（攻め圧が立つ）
        after = cpu_ai.evaluate(gm, "p1", plan=plan)
        gm.p1.field.remove(c)
        return after - before

    d_none = delta(None)
    d_aggro = delta(_plan("aggro"))
    d_control = delta(_plan("control"))
    assert d_control < d_none < d_aggro


def test_control_values_retained_counter_more(db):
    """control は自分の手札カウンター価値を高く見る（防御札の温存＝出し渋り）。"""
    gm = _new_gm(db)
    if not gm.p1.hand:
        pytest.skip("手札が空")

    def delta(plan):
        before = cpu_ai.evaluate(gm, "p1", plan=plan)
        gm.p1.hand[0].passive_counter += 2000
        after = cpu_ai.evaluate(gm, "p1", plan=plan)
        gm.p1.hand[0].passive_counter -= 2000
        return after - before

    assert delta(_plan("control")) > delta(None) > 0


def test_plan_progress_rewards_lethal_board(db):
    """逆算リーサル: 相手ライフを削り切れる本数のアクティブ体を持つ盤面が加点される。"""
    gm = _new_gm(db)
    gm.turn_player = gm.p1
    # 相手ライフを 1 枚に。
    gm.p2.life[:] = gm.p2.life[:1]
    me, opp = gm.p1, gm.p2
    base = cpu_ai._plan_progress(gm, me, opp, True, _plan("aggro"))
    # リーダーに届く確立済みアクティブ体を 1 体用意（素パワーをリーダー以上へ）。
    c = next((x for x in list(gm.p1.deck) if x.master.type.name == "CHARACTER"), None)
    assert c is not None
    gm.p1.deck.remove(c)
    gm.p1.field.append(c)
    c.is_rest = False
    c.is_newly_played = False
    c.passive_power_override = int(cpu_ai._power_cap(gm.p2)) + 1000
    with_reach = cpu_ai._plan_progress(gm, me, opp, True, _plan("aggro"))
    assert with_reach > base
