"""リーダーカード効果テスト — EB（エクストラブースター）/ PRB（プレミアムブースター）。

仕様書 docs/leader_specs/EB-PRB.md のテストケースを pytest 化したもの。
テキスト準拠の「正しい挙動」をアサートする（現実装にバグがある場合は
@pytest.mark.xfail でバグ検知器として固定する）。

対象7枚: EB01-001 / EB01-021 / EB01-040 / EB02-010 / EB03-001 / EB04-001 / PRB01-001

実行:
    OPCG_LOG_SILENT=1 python -m pytest tests/test_leader_eb_prb.py -q -s -p no:cacheprovider
"""
import pytest

from leader_test_helpers import (
    build, get_ability, abilities_of, auto_resolve,
    select_uuids, confirm,
    add_char, clear_field, set_life,
    leader_power,
)
from engine_helpers import action
from opcg_sim.src.models.effect_types import Ability, GameAction, ValueSource
from opcg_sim.src.models.enums import TriggerType, ActionType


# ---------------------------------------------------------------------------
# 共通補助
# ---------------------------------------------------------------------------

def _attach_don_to_leader(player, n=1):
    """リーダーに付与ドンを n 枚つけて【ドン!!×N】条件を満たす。"""
    L = player.leader
    for _ in range(n):
        if not player.don_active:
            break
        d = player.don_active.pop()
        d.is_rest = False
        d.attached_to = L.uuid
        player.don_attached_cards.append(d)
    L.attached_don = (L.attached_don or 0) + n


def _on_play_ability():
    """【登場時】効果を持つキャラ用のダミー能力。"""
    return Ability(trigger=TriggerType.ON_PLAY,
                   effect=GameAction(type=ActionType.DRAW, value=ValueSource(base=1)))


def _on_attack_ability():
    """【アタック時】効果を持つキャラ用のダミー能力。"""
    return Ability(trigger=TriggerType.ON_ATTACK,
                   effect=GameAction(type=ActionType.DRAW, value=ValueSource(base=1)))


# ===========================================================================
# EB01-001 光月おでん
# 能力0【PASSIVE】ルール: 自分《ワノ国》かつカウンター非所持キャラにカウンター+1000。
# 能力1【ドン!!×1】【ON_ATTACK】自分にコスト5以上《ワノ国》キャラがいる場合、
#   このリーダーは次の自分ターン開始時までパワー+1000。
# ===========================================================================

def test_eb01_001_on_attack_buff_with_cost5_wano():
    """EB01-001 ON_ATTACK: DON1付与・コスト5《ワノ国》キャラ有 → リーダー +1000。"""
    gm, p1, p2, L = build("EB01-001")
    clear_field(p1)
    add_char(p1, name="侍", cost=5, power=6000, traits=["ワノ国"])
    _attach_don_to_leader(p1, 1)
    gm.resolve_ability(p1, get_ability(L.master, "ON_ATTACK"), L)
    auto_resolve(gm, p1)
    assert L.timed_power == 1000


def test_eb01_001_on_attack_cost4_wano_no_buff():
    """EB01-001 ON_ATTACK: 《ワノ国》キャラがコスト4のみ → 条件未達で不発。"""
    gm, p1, p2, L = build("EB01-001")
    clear_field(p1)
    add_char(p1, name="侍", cost=4, power=6000, traits=["ワノ国"])
    _attach_don_to_leader(p1, 1)
    gm.resolve_ability(p1, get_ability(L.master, "ON_ATTACK"), L)
    auto_resolve(gm, p1)
    assert L.timed_power == 0


def test_eb01_001_on_attack_non_wano_no_buff():
    """EB01-001 ON_ATTACK: コスト5でも《ワノ国》以外特徴のみ → 不発。"""
    gm, p1, p2, L = build("EB01-001")
    clear_field(p1)
    add_char(p1, name="他", cost=5, power=6000, traits=["その他"])
    _attach_don_to_leader(p1, 1)
    gm.resolve_ability(p1, get_ability(L.master, "ON_ATTACK"), L)
    auto_resolve(gm, p1)
    assert L.timed_power == 0


def test_eb01_001_on_attack_requires_don():
    """EB01-001 ON_ATTACK: 【ドン!!×1】未付与 → コスト5《ワノ国》がいても不発。"""
    gm, p1, p2, L = build("EB01-001")
    clear_field(p1)
    add_char(p1, name="侍", cost=5, power=6000, traits=["ワノ国"])
    gm.resolve_ability(p1, get_ability(L.master, "ON_ATTACK"), L)
    auto_resolve(gm, p1)
    assert L.timed_power == 0


def test_eb01_001_passive_grants_counter_to_wano():
    """EB01-001 PASSIVE: 自分の《ワノ国》カウンター非所持キャラは手札でカウンター+1000を持つ。

    実ルール: カウンターを持たない《ワノ国》キャラを、手札からカウンターとして使える
    +1000札にする（PASSIVE BUFF(COUNTER)→passive_counter、手札ゾーン対象）。
    """
    from leader_test_helpers import make_char
    gm, p1, p2, L = build("EB01-001")
    clear_field(p1)
    wano0 = make_char(p1, name="ワノ国兵", cost=3, power=3000, counter=0, traits=["ワノ国"])
    wano1k = make_char(p1, name="ワノ国剣士", cost=3, power=3000, counter=1000, traits=["ワノ国"])
    other0 = make_char(p1, name="余所者", cost=3, power=3000, counter=0, traits=["その他"])
    p1.hand = [wano0, wano1k, other0]
    gm._apply_passive_effects(p1)
    # カウンター非所持の《ワノ国》→ +1000 を得る
    assert wano0.current_counter == 1000
    # 既にカウンターを持つ《ワノ国》→ 据え置き（NO_COUNTER で対象外・二重加算しない）
    assert wano1k.current_counter == 1000
    # 非《ワノ国》→ 付与されない
    assert other0.current_counter == 0
    # 戦闘のカウンター候補（current_counter>0）に +1000 化した非所持キャラが現れる
    counters = [c for c in p1.hand if c.current_counter > 0]
    assert wano0 in counters


# ===========================================================================
# EB01-021 ハンニャバル
# 【TURN_END】自分のコスト2以上《インペルダウン》キャラ1枚を持ち主の手札に戻すことが
#   できる：ドンデッキからドン1枚までをアクティブで追加する。
# ===========================================================================

def test_eb01_021_turn_end_bounce_then_ramp():
    """EB01-021 ターン終了時: コスト2以上《インペルダウン》を手札へ戻し→ドン1枚アクティブ追加。"""
    gm, p1, p2, L = build("EB01-021")
    clear_field(p1)
    c = add_char(p1, name="囚人", cost=2, power=2000, traits=["インペルダウン"])
    hand_before = len(p1.hand)
    active_before = len(p1.don_active)
    deck_before = len(p1.don_deck)
    gm.resolve_ability(p1, get_ability(L.master, "TURN_END"), L)
    auto_resolve(gm, p1)
    assert c not in p1.field and c in p1.hand          # 持ち主(自分)の手札へ
    assert len(p1.hand) == hand_before + 1
    assert len(p1.don_active) == active_before + 1     # アクティブで追加
    assert len(p1.don_deck) == deck_before - 1


def test_eb01_021_turn_end_no_cost2_target_no_ramp():
    """EB01-021: コスト2以上《インペルダウン》不在（コスト1のみ）→ コスト払えず不発。"""
    gm, p1, p2, L = build("EB01-021")
    clear_field(p1)
    add_char(p1, name="囚人", cost=1, power=1000, traits=["インペルダウン"])
    active_before = len(p1.don_active)
    gm.resolve_ability(p1, get_ability(L.master, "TURN_END"), L)
    auto_resolve(gm, p1)
    assert len(p1.don_active) == active_before          # ドン追加なし


# ===========================================================================
# EB01-040 キュロス
# 【起動メイン】【ターン1回】自分のライフ上1枚を表向きにできる：
#   相手のコスト0のキャラ1枚までをKOする。
# ===========================================================================

def test_eb01_040_face_up_life_then_ko_cost0():
    """EB01-040 起動メイン: ライフ上1枚を表向き→相手コスト0キャラを1枚KO。"""
    gm, p1, p2, L = build("EB01-040")
    clear_field(p2)
    victim = add_char(p2, name="敵", cost=0, power=1000)
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    auto_resolve(gm, p1, plan=[select_uuids([victim.uuid])])
    assert victim not in p2.field and victim in p2.trash
    assert p1.life[0].is_face_up is True               # ライフ上が表向きに


def test_eb01_040_cost1_not_targetable():
    """EB01-040: 相手キャラがコスト1以上のみ → 対象なし（is_up_te で0枚KO）。"""
    gm, p1, p2, L = build("EB01-040")
    clear_field(p2)
    v = add_char(p2, name="敵", cost=1, power=1000)
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    auto_resolve(gm, p1)
    assert v in p2.field                               # KOされない


def test_eb01_040_turn_once_blocks_second():
    """EB01-040: 【ターン1回】同一ターン2回目の起動メインは不発。"""
    gm, p1, p2, L = build("EB01-040")
    clear_field(p2)
    v1 = add_char(p2, name="敵1", cost=0, power=1000)
    v2 = add_char(p2, name="敵2", cost=0, power=1000)
    ab = get_ability(L.master, "ACTIVATE_MAIN")
    gm.resolve_ability(p1, ab, L)
    auto_resolve(gm, p1, plan=[select_uuids([v1.uuid])])
    assert v1 not in p2.field
    gm.resolve_ability(p1, ab, L)                      # 2回目
    auto_resolve(gm, p1, plan=[select_uuids([v2.uuid])])
    assert v2 in p2.field                              # ターン1回制限で2枚目はKOされない


# ===========================================================================
# EB02-010 モンキー・D・ルフィ
# 【起動メイン】【ターン1回】ドン!!-2：自分の場が《麦わらの一味》のみの場合、
#   自分のドン2枚までをアクティブにする。その後、このリーダーは次の相手ターン終了時まで
#   パワー+1000。
# ===========================================================================

def test_eb02_010_all_mugiwara_active_don_and_buff():
    """EB02-010 起動メイン: 場が《麦わらの一味》のみ → ドン2返却→2枚アクティブ化→リーダー+1000。"""
    gm, p1, p2, L = build("EB02-010")
    clear_field(p1)
    add_char(p1, name="麦", cost=2, power=2000, traits=["麦わらの一味"])
    for _ in range(4):                                  # レストドンを用意（アクティブ化対象）
        d = p1.don_active.pop(); d.is_rest = True; p1.don_rested.append(d)
    active_before = len(p1.don_active)                  # 6
    rested_before = len(p1.don_rested)                  # 4
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    auto_resolve(gm, p1)
    # コスト ドン-2（active から2返却）＋ レスト2枚をアクティブ化 → active は差引同数、rested は-2
    assert len(p1.don_active) == active_before
    assert len(p1.don_rested) == rested_before - 2
    assert L.timed_power == 1000


def test_eb02_010_mixed_field_no_active_don_but_buff():
    """EB02-010: 非《麦わらの一味》混在 → アクティブ化なし、ただし buff は無条件で +1000。"""
    gm, p1, p2, L = build("EB02-010")
    clear_field(p1)
    add_char(p1, name="麦", cost=2, power=2000, traits=["麦わらの一味"])
    add_char(p1, name="他", cost=2, power=2000, traits=["海軍"])
    for _ in range(4):
        d = p1.don_active.pop(); d.is_rest = True; p1.don_rested.append(d)
    rested_before = len(p1.don_rested)
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    auto_resolve(gm, p1)
    assert len(p1.don_rested) == rested_before          # アクティブ化は発生しない
    assert L.timed_power == 1000                         # buff は無条件


# ===========================================================================
# EB03-001 ネフェルタリ・ビビ
# 能力0【PASSIVE】【ターン1回】自分の元々コスト4以上のキャラがKOされる場合、
#   代わりに自分の手札1枚を捨てることができる（KO置換）。
# 能力1【起動メイン】このリーダーをレストにできる：相手キャラ1枚まで-2000。その後、
#   自分の【アタック時】効果を持たないキャラ1枚まで【速攻】を得る。
# ===========================================================================

def test_eb03_001_replace_ko_of_cost4_char():
    """EB03-001 PASSIVE: 元々コスト4以上キャラのKОを置換し、代わりに手札1枚を捨てる。"""
    gm, p1, p2, L = build("EB03-001")
    clear_field(p1)
    c = add_char(p1, name="重要", cost=4, power=5000)
    gm._apply_passive_effects(p1)
    hand_before = len(p1.hand)
    gm.apply_action_to_engine(p2, action(ActionType.KO), [c], 0)
    auto_resolve(gm, p1)
    assert c in p1.field                                # KO回避
    assert len(p1.hand) == hand_before - 1              # 代わりに手札1枚捨て


def test_eb03_001_cost3_char_not_replaced():
    """EB03-001 PASSIVE: 元々コスト3以下のキャラがKOされる場合は置換対象外（通常KO）。"""
    gm, p1, p2, L = build("EB03-001")
    clear_field(p1)
    c = add_char(p1, name="軽い", cost=3, power=3000)
    gm._apply_passive_effects(p1)
    gm.apply_action_to_engine(p2, action(ActionType.KO), [c], 0)
    auto_resolve(gm, p1)
    assert c not in p1.field and c in p1.trash          # コスト4未満は通常通りKO


def test_eb03_001_main_debuff_and_grant_to_plain_char():
    """EB03-001 起動メイン: リーダーをレスト→相手-2000→アタック時効果なしキャラに速攻付与。"""
    gm, p1, p2, L = build("EB03-001")
    clear_field(p1); clear_field(p2)
    plain = add_char(p1, name="無効果", cost=3, power=3000)
    victim = add_char(p2, name="敵", cost=3, power=5000)
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    auto_resolve(gm, p1, plan=[select_uuids([victim.uuid]), select_uuids([plain.uuid])])
    assert L.is_rest is True
    assert victim.get_power(False) == 3000             # -2000
    assert plain.has_keyword("速攻")


def test_eb03_001_grant_excludes_on_attack_char():
    """EB03-001 起動メイン: 【アタック時】効果持ちキャラは速攻付与の対象外であるべき。"""
    gm, p1, p2, L = build("EB03-001")
    clear_field(p1); clear_field(p2)
    atker = add_char(p1, name="攻撃時持ち", cost=3, power=3000,
                     abilities=(_on_attack_ability(),))
    victim = add_char(p2, name="敵", cost=3, power=5000)
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    # 相手-2000 の対象、続いて速攻の対象としてアタック時効果持ちを「選ぼうとする」
    gm.resolve_interaction(p1, {"selected_uuids": [victim.uuid]})
    gm.resolve_interaction(p1, {"selected_uuids": [atker.uuid]})
    assert not atker.has_keyword("速攻")                # 本来は対象外＝速攻が付かない


# ===========================================================================
# EB04-001 ジュエリー・ボニー
# 能力0【相手のターン中】自分のライフが1枚以下の場合、このリーダーのパワー+2000。
# 能力1【起動メイン】【ターン1回】相手キャラ1枚まで-1000。その後、自分のライフが2枚以上の
#   場合、自分のライフ上1枚を手札に加えることができる。
# ===========================================================================

def test_eb04_001_opponent_turn_buff_when_life_le1():
    """EB04-001 相手のターン中: 自分ライフ1枚以下 → リーダー+2000。"""
    gm, p1, p2, L = build("EB04-001")
    set_life(p1, 1)
    gm.resolve_ability(p1, get_ability(L.master, "OPPONENT_TURN"), L)
    auto_resolve(gm, p1)
    assert L.get_power(False) == leader_power(p1, my_turn=False)
    assert L.power_buff == 2000


def test_eb04_001_opponent_turn_no_buff_when_life_ge2():
    """EB04-001 相手のターン中: 自分ライフ2枚以上 → バフなし（条件未達）。"""
    gm, p1, p2, L = build("EB04-001")
    set_life(p1, 3)
    gm.resolve_ability(p1, get_ability(L.master, "OPPONENT_TURN"), L)
    auto_resolve(gm, p1)
    assert L.power_buff == 0


def test_eb04_001_main_debuff_then_life_to_hand_when_life_ge2():
    """EB04-001 起動メイン: ライフ2枚以上 → 相手-1000→ライフ上1枚を手札へ。"""
    gm, p1, p2, L = build("EB04-001")
    clear_field(p2)
    v = add_char(p2, name="敵", cost=3, power=5000)
    set_life(p1, 3)
    hand_before = len(p1.hand)
    life_before = len(p1.life)
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    auto_resolve(gm, p1, plan=[select_uuids([v.uuid]), confirm(True)])
    assert v.get_power(False) == 4000                  # -1000
    assert len(p1.hand) == hand_before + 1
    assert len(p1.life) == life_before - 1


def test_eb04_001_main_no_life_to_hand_when_life_le1():
    """EB04-001 起動メイン: ライフ1枚以下 → 相手-1000は実行、ライフ追加は不発。"""
    gm, p1, p2, L = build("EB04-001")
    clear_field(p2)
    v = add_char(p2, name="敵", cost=3, power=5000)
    set_life(p1, 1)
    hand_before = len(p1.hand)
    life_before = len(p1.life)
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    auto_resolve(gm, p1, plan=[select_uuids([v.uuid])])
    assert v.get_power(False) == 4000                  # -1000 は実行
    assert len(p1.hand) == hand_before                 # ライフ追加なし
    assert len(p1.life) == life_before


# ===========================================================================
# PRB01-001 サンジ
# 【起動メイン】【ターン1回】自分のコスト8以下の【登場時】効果を持たないキャラ1枚までは、
#   このターン中、【速攻】を得る。
# ===========================================================================

def test_prb01_001_trigger_should_be_activate_main():
    """PRB01-001: 速攻付与能力のトリガーは【起動メイン】(ACTIVATE_MAIN)であるべき。"""
    assert abilities_of(build("PRB01-001")[3].master, "ACTIVATE_MAIN"), \
        "ACTIVATE_MAIN 能力が存在しない（ON_PLAY と誤解釈）"


def test_prb01_001_grant_haste_to_plain_char():
    """PRB01-001: コスト8以下・登場時効果なしキャラに速攻を付与（本来トリガーで発動）。"""
    gm, p1, p2, L = build("PRB01-001")
    clear_field(p1)
    plain = add_char(p1, name="無効果", cost=5, power=5000)
    # 本来は ACTIVATE_MAIN だが現状 ON_PLAY としてパースされる能力を発動
    ab = abilities_of(L.master, "ACTIVATE_MAIN") or abilities_of(L.master, "ON_PLAY")
    gm.resolve_ability(p1, ab[0], L)
    auto_resolve(gm, p1, plan=[select_uuids([plain.uuid])])
    assert plain.has_keyword("速攻")


def test_prb01_001_cost9_not_targetable():
    """PRB01-001: コスト9以上キャラは対象外（cost_max=8 フィルタ）。"""
    gm, p1, p2, L = build("PRB01-001")
    clear_field(p1)
    big = add_char(p1, name="重い", cost=9, power=9000)
    ab = abilities_of(L.master, "ACTIVATE_MAIN") or abilities_of(L.master, "ON_PLAY")
    gm.resolve_ability(p1, ab[0], L)
    auto_resolve(gm, p1)
    assert not big.has_keyword("速攻")


def test_prb01_001_grant_excludes_on_play_char():
    """PRB01-001: 【登場時】効果を持つキャラは速攻付与の対象外であるべき。"""
    gm, p1, p2, L = build("PRB01-001")
    clear_field(p1)
    nyu = add_char(p1, name="登場時持ち", cost=5, power=5000,
                   abilities=(_on_play_ability(),))
    ab = abilities_of(L.master, "ACTIVATE_MAIN") or abilities_of(L.master, "ON_PLAY")
    gm.resolve_ability(p1, ab[0], L)
    # 登場時効果持ちを「選ぼうとする」
    gm.resolve_interaction(p1, {"selected_uuids": [nyu.uuid]})
    assert not nyu.has_keyword("速攻")                  # 本来は対象外＝速攻が付かない
