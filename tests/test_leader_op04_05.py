"""リーダー効果テスト — OP04 / OP05（仕様書 docs/leader_specs/OP04-05.md 由来）。

テキスト準拠の「正しい挙動」をアサートする。バグ(🐛)は xfail(strict=True)、
要確認(⚠️)で不安定なものは xfail(strict=False)。

実行:
    OPCG_LOG_SILENT=1 python -m pytest tests/test_leader_op04_05.py -q -s -p no:cacheprovider
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest

from leader_test_helpers import (
    build, get_ability, auto_resolve, select_uuids,
    add_char, make_char, clear_field, set_life, zone_counts,
)
from engine_helpers import action
from opcg_sim.src.models.enums import ActionType, TriggerType
from opcg_sim.src.models.effect_types import Ability, GameAction, ValueSource


# ---------------------------------------------------------------------------
# 補助
# ---------------------------------------------------------------------------

def _attach_don(player, card, n=1):
    """コストエリアのアクティブドンを n 枚、card に付与する（【ドン!!×N】条件用）。"""
    for _ in range(n):
        d = player.don_active.pop()
        d.is_rest = False
        d.attached_to = card.uuid
        player.don_attached_cards.append(d)
        card.attached_don = getattr(card, "attached_don", 0) + 1


def _make_rested_don(player, n):
    """コストエリアのアクティブドンを n 枚レストにする。"""
    for _ in range(n):
        d = player.don_active.pop()
        d.is_rest = True
        player.don_rested.append(d)


# ===========================================================================
# OP04-001 ネフェルタリ・ビビ
#   【起動メイン】【ターン1回】コスト：1ドロー → 自キャラ1枚まで【速攻】付与(THIS_TURN)
# ===========================================================================

def test_op04_001_draw_and_grant_haste():
    """OP04-001 起動メイン: 1ドローし、選んだ自キャラ1枚にTHIS_TURN【速攻】付与。"""
    gm, p1, p2, L = build("OP04-001")
    clear_field(p1)
    c = add_char(p1)
    before = zone_counts(p1)
    ab = get_ability(L.master, "ACTIVATE_MAIN")
    gm.resolve_ability(p1, ab, L)
    # 速攻付与対象としてキャラを明示選択
    auto_resolve(gm, p1, plan=[select_uuids([c.uuid])])
    after = zone_counts(p1)
    assert after["hand"] == before["hand"] + 1   # 1ドロー
    assert after["deck"] == before["deck"] - 1
    assert "速攻" in (c.current_keywords | c.timed_keywords)


def test_op04_001_cost_rests_two_don():
    """OP04-001 起動メイン: コスト記号➁＝ドン!!2枚レストのはず（実装は1枚レスト疑い）。"""
    gm, p1, p2, L = build("OP04-001")
    clear_field(p1)
    add_char(p1)
    before = zone_counts(p1)
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    auto_resolve(gm, p1)
    after = zone_counts(p1)
    assert after["don_rested"] == before["don_rested"] + 2   # ➁=2枚レスト


def test_op04_001_no_character_still_draws():
    """OP04-001 起動メイン: 自キャラ0体でもドローとドン消費は発生（付与対象は最大0で許容）。"""
    gm, p1, p2, L = build("OP04-001")
    clear_field(p1)
    before = zone_counts(p1)
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    auto_resolve(gm, p1)
    after = zone_counts(p1)
    assert after["hand"] == before["hand"] + 1
    assert after["don_rested"] >= before["don_rested"] + 1   # コスト消費は発生


def test_op04_001_turn_limit_blocks_second():
    """OP04-001 起動メイン: 【ターン1回】制限で同一ターン2回目は不発。"""
    gm, p1, p2, L = build("OP04-001")
    clear_field(p1)
    add_char(p1)
    ab = get_ability(L.master, "ACTIVATE_MAIN")
    gm.resolve_ability(p1, ab, L)
    auto_resolve(gm, p1)
    h1 = len(p1.hand)
    gm.resolve_ability(p1, ab, L)
    auto_resolve(gm, p1)
    assert len(p1.hand) == h1   # 2回目はドローしない


# ===========================================================================
# OP04-019 ドンキホーテ・ドフラミンゴ
#   【自分のターン終了時】自分のドン!!2枚までをアクティブにする
# ===========================================================================

def test_op04_019_active_two_rested_don():
    """OP04-019 ターン終了時: レストドン2枚以上あれば2枚をアクティブにする。"""
    gm, p1, p2, L = build("OP04-019")
    _make_rested_don(p1, 3)
    rested_before = len(p1.don_rested)
    gm.resolve_ability(p1, get_ability(L.master, "TURN_END"), L)
    auto_resolve(gm, p1)
    assert len(p1.don_rested) == rested_before - 2   # 2枚アクティブ化


def test_op04_019_only_one_rested_activates_one():
    """OP04-019 ターン終了時: レストドン1枚なら1枚のみ（「まで」=最大2）。"""
    gm, p1, p2, L = build("OP04-019")
    _make_rested_don(p1, 1)
    gm.resolve_ability(p1, get_ability(L.master, "TURN_END"), L)
    auto_resolve(gm, p1)
    assert len(p1.don_rested) == 0


def test_op04_019_no_rested_no_change():
    """OP04-019 ターン終了時: レストドン0枚なら変化なし。"""
    gm, p1, p2, L = build("OP04-019")
    a0, r0 = len(p1.don_active), len(p1.don_rested)
    gm.resolve_ability(p1, get_ability(L.master, "TURN_END"), L)
    auto_resolve(gm, p1)
    assert (len(p1.don_active), len(p1.don_rested)) == (a0, r0)


# ===========================================================================
# OP04-020 イッショウ
#   能力0【ドン!!×1/自分のターン中】相手キャラ全体 コスト-1
#   能力1【自分のターン終了時】コスト：自分のコスト5以下キャラ1枚までをアクティブ化
# ===========================================================================

def test_op04_020_opponent_chars_cost_minus_one():
    """OP04-020 自ターン中: 相手キャラすべてのコストを-1する。"""
    gm, p1, p2, L = build("OP04-020")
    p2.field = []
    e1 = add_char(p2, cost=3)
    e2 = add_char(p2, cost=4)
    _attach_don(p1, L, 1)
    gm.resolve_ability(p1, get_ability(L.master, "YOUR_TURN"), L)
    auto_resolve(gm, p1)
    assert e1.current_cost == 2 and e2.current_cost == 3


def test_op04_020_turn_end_activates_low_cost_char():
    """OP04-020 ターン終了時: コスト5以下の自レストキャラ1枚をアクティブにする。"""
    gm, p1, p2, L = build("OP04-020")
    clear_field(p1)
    c = add_char(p1, cost=3, rest=True)
    gm.resolve_ability(p1, get_ability(L.master, "TURN_END"), L)
    auto_resolve(gm, p1, plan=[select_uuids([c.uuid])])
    assert c.is_rest is False   # アクティブ化


def test_op04_020_turn_end_no_eligible_char():
    """OP04-020 ターン終了時: コスト6の自キャラのみなら対象不在で対象選択は空。"""
    gm, p1, p2, L = build("OP04-020")
    clear_field(p1)
    c = add_char(p1, cost=6, rest=True)
    gm.resolve_ability(p1, get_ability(L.master, "TURN_END"), L)
    auto_resolve(gm, p1)
    assert c.is_rest is True   # コスト6は対象外


# ===========================================================================
# OP04-039 レベッカ
#   【起動メイン】【ターン1回】コスト：手札6枚以下なら上2枚を見て
#   《ドレスローザ》1枚までを手札、残りをトラッシュ
# ===========================================================================

def test_op04_039_look_grab_dresrosa():
    """OP04-039 起動メイン: 手札6枚以下、上2枚のドレスローザ1枚を手札、残りトラッシュ。"""
    gm, p1, p2, L = build("OP04-039")
    p1.hand = p1.hand[:5]
    d1 = make_char(p1, name="ドレ", traits=["ドレスローザ"])
    d2 = make_char(p1, name="他", traits=["その他"])
    p1.deck.insert(0, d1)
    p1.deck.insert(1, d2)
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    auto_resolve(gm, p1, plan=[select_uuids([d1.uuid])])
    assert d1 in p1.hand
    assert d2 in p1.trash


def test_op04_039_no_dresrosa_both_trashed():
    """OP04-039 起動メイン: 上2枚ともドレスローザでなければ手札追加0、2枚ともトラッシュ。"""
    gm, p1, p2, L = build("OP04-039")
    p1.hand = p1.hand[:5]
    d1 = make_char(p1, name="他1", traits=["その他"])
    d2 = make_char(p1, name="他2", traits=["その他"])
    p1.deck.insert(0, d1)
    p1.deck.insert(1, d2)
    hand_before = len(p1.hand)
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    auto_resolve(gm, p1)
    assert d1 in p1.trash and d2 in p1.trash
    assert len(p1.hand) == hand_before


# ===========================================================================
# OP04-040 クイーン
#   【ドン!!×1】【アタック時】ライフ+手札合計≤4なら1ドロー。
#   コスト8以上キャラがいれば「ドローの代わりに」デッキ上1枚までをライフへ(任意)
# ===========================================================================

def test_op04_040_attack_draws_when_total_le4():
    """OP04-040 アタック時: ライフ+手札の合計4以下、コスト8以上なしで1ドロー。"""
    gm, p1, p2, L = build("OP04-040")
    set_life(p1, 2)
    p1.hand = p1.hand[:2]   # ライフ2+手札2=4
    clear_field(p1)
    _attach_don(p1, L, 1)
    before = zone_counts(p1)
    gm.resolve_ability(p1, get_ability(L.master, "ON_ATTACK"), L)
    auto_resolve(gm, p1)
    after = zone_counts(p1)
    assert after["hand"] == before["hand"] + 1   # 1ドロー


def test_op04_040_no_effect_when_total_ge5():
    """OP04-040 アタック時: ライフ+手札の合計5以上なら何も起きない。"""
    gm, p1, p2, L = build("OP04-040")
    # 既定: ライフ5+手札5=10
    clear_field(p1)
    _attach_don(p1, L, 1)
    before = zone_counts(p1)
    gm.resolve_ability(p1, get_ability(L.master, "ON_ATTACK"), L)
    auto_resolve(gm, p1)
    after = zone_counts(p1)
    assert after["hand"] == before["hand"]
    assert after["life"] == before["life"]


@pytest.mark.xfail(strict=True, reason="OP04-040バグ: 条件がLIFE_COUNTのみで手札枚数が合算されない。ライフ2+手札5=7(>4)でもドローしてしまう")
def test_op04_040_condition_counts_hand_too():
    """OP04-040 アタック時: ライフ+手札の合計が5以上(ライフ2+手札5=7)ならドローしない。"""
    gm, p1, p2, L = build("OP04-040")
    set_life(p1, 2)   # ライフ2 + 手札5 = 7 > 4
    clear_field(p1)
    _attach_don(p1, L, 1)
    before = zone_counts(p1)
    gm.resolve_ability(p1, get_ability(L.master, "ON_ATTACK"), L)
    auto_resolve(gm, p1)
    after = zone_counts(p1)
    assert after["hand"] == before["hand"]   # 合計>4なのでドローしないのが正


@pytest.mark.xfail(strict=True, reason="OP04-040バグ: 『代わりに』が独立2branchに分解され、ドローとHEALが両方発動する(本来は択一)")
def test_op04_040_heal_replaces_draw_not_both():
    """OP04-040 アタック時: コスト8以上がいれば『ドローの代わりに』ライフ追加（両方は起きない）。"""
    gm, p1, p2, L = build("OP04-040")
    set_life(p1, 2)
    p1.hand = p1.hand[:2]   # ライフ2+手札2=4
    clear_field(p1)
    add_char(p1, cost=8, power=8000)
    _attach_don(p1, L, 1)
    before = zone_counts(p1)
    gm.resolve_ability(p1, get_ability(L.master, "ON_ATTACK"), L)
    auto_resolve(gm, p1)
    after = zone_counts(p1)
    # 「代わりに」＝択一: HEAL(ライフ+1)のみで、デッキ消費は1枚、手札は増えない
    assert after["life"] == before["life"] + 1
    assert after["hand"] == before["hand"]
    assert after["deck"] == before["deck"] - 1


# ===========================================================================
# OP04-058 クロコダイル
#   【相手のターン中】【ターン1回】自効果でドンが戻された時、ドンデッキから1枚アクティブ追加
# ===========================================================================

def test_op04_058_ramp_one_active_don():
    """OP04-058 相手ターン: ドンデッキからドン!!1枚をアクティブで追加する。"""
    gm, p1, p2, L = build("OP04-058")
    total_before = len(p1.don_active) + len(p1.don_rested)
    deck_before = len(p1.don_deck)
    gm.resolve_ability(p1, get_ability(L.master, "OPPONENT_TURN"), L)
    auto_resolve(gm, p1)
    assert len(p1.don_active) + len(p1.don_rested) == total_before + 1
    assert len(p1.don_deck) == deck_before - 1


# ===========================================================================
# OP05-001 サボ
#   【ドン!!×1】【相手のターン中】【ターン1回】自パワー5000以上キャラがKOされる場合、
#   KOの代わりにそのキャラをTHIS_TURNパワー-1000できる
# ===========================================================================

@pytest.mark.xfail(strict=False, reason="OP05-001要確認: KO置換がリーダーPASSIVEで発火せず、5000以上のキャラが通常KOされる")
def test_op05_001_replaces_ko_with_power_down():
    """OP05-001 相手ターン: パワー5000以上のキャラがKOされる代わりにパワー-1000で生存。"""
    gm, p1, p2, L = build("OP05-001")
    clear_field(p1)
    _attach_don(p1, L, 1)
    victim = add_char(p1, name="5k", power=5000)
    gm.apply_action_to_engine(p2, action(ActionType.KO), [victim], 0)
    auto_resolve(gm, p1)
    assert victim in p1.field            # KO置換で生存
    assert victim.get_power(True) == 4000   # THIS_TURN パワー-1000


def test_op05_001_no_replace_for_low_power():
    """OP05-001 相手ターン: パワー4000のキャラは5000以上条件未達で通常どおりKOされる。"""
    gm, p1, p2, L = build("OP05-001")
    clear_field(p1)
    _attach_don(p1, L, 1)
    victim = add_char(p1, name="4k", power=4000)
    gm.apply_action_to_engine(p2, action(ActionType.KO), [victim], 0)
    auto_resolve(gm, p1)
    assert victim not in p1.field
    assert victim in p1.trash


# ===========================================================================
# OP05-002 ベロ・ベティ
#   【起動メイン】【ターン1回】コスト：手札の《革命軍》1枚を捨てる：
#   《革命軍》か【トリガー】を持つキャラ3枚までをTHIS_TURNパワー+3000
# ===========================================================================

def test_op05_002_buffs_revolutionary_chars():
    """OP05-002 起動メイン: 《革命軍》キャラにTHIS_TURN +3000、コストで革命軍1枚捨て。"""
    gm, p1, p2, L = build("OP05-002")
    clear_field(p1)
    rg = add_char(p1, name="革命", traits=["革命軍"], power=5000)
    discard = make_char(p1, name="革手", traits=["革命軍"])
    p1.hand.append(discard)
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    auto_resolve(gm, p1, plan=[select_uuids([rg.uuid])])
    assert rg.get_power(True) == 8000
    assert discard in p1.trash   # コストで革命軍カードを捨てた


@pytest.mark.xfail(strict=True, reason="OP05-002バグ: 対象『か【トリガー】を持つ』キャラ条件が欠落(traits=革命軍のみ)。トリガー持ちが強化されない")
def test_op05_002_buffs_trigger_keyword_chars():
    """OP05-002 起動メイン: 【トリガー】を持つキャラ（革命軍でなくても）も+3000対象。"""
    gm, p1, p2, L = build("OP05-002")
    clear_field(p1)
    trig = Ability(trigger=TriggerType.TRIGGER,
                   effect=GameAction(type=ActionType.DRAW, value=ValueSource(base=1)))
    tc = add_char(p1, name="トリ", traits=["その他"], power=5000, abilities=(trig,))
    discard = make_char(p1, name="革手", traits=["革命軍"])
    p1.hand.append(discard)
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    auto_resolve(gm, p1, plan=[select_uuids([tc.uuid])])
    assert tc.get_power(True) == 8000   # トリガー持ちも+3000されるのが正


# ===========================================================================
# OP05-022 ドンキホーテ・ロシナンテ
#   【自分のターン終了時】手札6枚以下ならこのリーダーをアクティブにする
# ===========================================================================

def test_op05_022_activates_leader_when_hand_le6():
    """OP05-022 ターン終了時: 手札6枚以下、レストのリーダーをアクティブにする。"""
    gm, p1, p2, L = build("OP05-022")
    L.is_rest = True
    gm.resolve_ability(p1, get_ability(L.master, "TURN_END"), L)
    auto_resolve(gm, p1)
    assert L.is_rest is False


def test_op05_022_no_op_when_hand_gt6():
    """OP05-022 ターン終了時: 手札7枚なら発火せずリーダーはレストのまま。"""
    gm, p1, p2, L = build("OP05-022")
    L.is_rest = True
    for _ in range(2):
        p1.hand.append(make_char(p1, name="x"))
    assert len(p1.hand) == 7
    gm.resolve_ability(p1, get_ability(L.master, "TURN_END"), L)
    auto_resolve(gm, p1)
    assert L.is_rest is True


# ===========================================================================
# OP05-041 サカズキ
#   能力0【起動メイン】【ターン1回】コスト：手札1枚捨てる：1ドロー
#   能力1【アタック時】相手キャラ1枚までをTHIS_TURNコスト-1
# ===========================================================================

def test_op05_041_discard_then_draw():
    """OP05-041 起動メイン: 手札1枚捨て→1ドロー（手札枚数±0、デッキ-1/トラッシュ+1）。"""
    gm, p1, p2, L = build("OP05-041")
    before = zone_counts(p1)
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    auto_resolve(gm, p1)
    after = zone_counts(p1)
    assert after["hand"] == before["hand"]        # 捨て1・引き1で±0
    assert after["trash"] == before["trash"] + 1
    assert after["deck"] == before["deck"] - 1


def test_op05_041_attack_reduces_opponent_cost():
    """OP05-041 アタック時: 相手キャラ1枚をTHIS_TURNコスト-1。"""
    gm, p1, p2, L = build("OP05-041")
    p2.field = []
    e = add_char(p2, cost=3)
    gm.resolve_ability(p1, get_ability(L.master, "ON_ATTACK"), L)
    auto_resolve(gm, p1, plan=[select_uuids([e.uuid])])
    assert e.current_cost == 2


# ===========================================================================
# OP05-060 モンキー・D・ルフィ
#   【起動メイン】【ターン1回】コスト：ライフ上1枚を手札：
#   自場のドン!!が0枚か3枚以上ならドンデッキから1枚アクティブ追加
# ===========================================================================

def test_op05_060_ramp_when_zero_don():
    """OP05-060 起動メイン: 自場のドン0枚ならドン!!1枚をアクティブ追加。"""
    gm, p1, p2, L = build("OP05-060")
    # 場のドンを全てドンデッキへ戻す
    while p1.don_active:
        p1.don_deck.append(p1.don_active.pop())
    while p1.don_rested:
        p1.don_deck.append(p1.don_rested.pop())
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    auto_resolve(gm, p1)
    assert len(p1.don_active) + len(p1.don_rested) == 1   # 1枚追加


def test_op05_060_ramp_when_three_or_more_don():
    """OP05-060 起動メイン: 自場のドン3枚以上でもドン!!1枚をアクティブ追加すべき。"""
    gm, p1, p2, L = build("OP05-060")
    # 既定で場にアクティブドン10枚（>=3）
    total_before = len(p1.don_active) + len(p1.don_rested)
    deck_before = len(p1.don_deck)
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    auto_resolve(gm, p1)
    assert len(p1.don_active) + len(p1.don_rested) == total_before + 1
    assert len(p1.don_deck) == deck_before - 1


# ===========================================================================
# OP05-098 エネル
#   【相手のターン中】【ターン1回】ライフが0になった時、デッキ上1枚をライフへ→手札1枚捨て
# ===========================================================================

@pytest.mark.xfail(strict=True, reason="OP05-098バグ: HEAL value=0 でデッキ上1枚のライフ加算が機能しない(手札捨てのみ実行)")
def test_op05_098_adds_life_from_deck():
    """OP05-098 相手ターン: デッキ上1枚をライフに加える（その後手札1枚を捨てる）。"""
    gm, p1, p2, L = build("OP05-098")
    before = zone_counts(p1)
    gm.resolve_ability(p1, get_ability(L.master, "OPPONENT_TURN"), L)
    auto_resolve(gm, p1)
    after = zone_counts(p1)
    assert after["life"] == before["life"] + 1    # デッキ上1枚がライフへ
    assert after["deck"] == before["deck"] - 1


def test_op05_098_discards_one_hand():
    """OP05-098 相手ターン: 手札1枚を捨てる部分は実行される（HEAL有無に依らず）。"""
    gm, p1, p2, L = build("OP05-098")
    before = zone_counts(p1)
    gm.resolve_ability(p1, get_ability(L.master, "OPPONENT_TURN"), L)
    auto_resolve(gm, p1)
    after = zone_counts(p1)
    assert after["hand"] == before["hand"] - 1
    assert after["trash"] == before["trash"] + 1
