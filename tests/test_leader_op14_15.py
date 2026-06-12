"""リーダーカード効果テスト — OP14 / OP15。

仕様書 docs/leader_specs/OP14-15.md のテストケースを pytest 化したもの。
テキスト準拠の正しい挙動をアサートする。

✅ → 通常テスト / 🐛 → xfail(strict=True) / ⚠️ → 通常 or xfail(strict=False)

実行:
    OPCG_LOG_SILENT=1 python -m pytest tests/test_leader_op14_15.py -q -s -p no:cacheprovider
"""
import pytest

from leader_test_helpers import (
    build, get_ability, auto_resolve,
    select_uuids, confirm, choose,
    add_char, make_char, clear_field, set_life,
    leader_power, don_total, zone_counts,
)


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


# ===========================================================================
# OP14-001 トラファルガー・ロー  ✅
# 【起動メイン】【ターン1回】自分の特徴《超新星》か《ハートの海賊団》を持つキャラ
# 2枚を選ぶ。選んだキャラそれぞれの元々のパワーを、このターン中、入れ替える。
# ===========================================================================

@pytest.mark.xfail(strict=True,
    reason="OP14-001: SWAP_POWER がエンジン未実装(enums定義のみでapply_action_to_engineにハンドラ無し)。能力解決しても元々パワーが入れ替わらない")
def test_op14_001_swap_original_power_of_two_chars():
    """OP14-001 起動メイン: 該当キャラ2体の元々パワーをこのターン中入れ替える。"""
    gm, p1, p2, L = build("OP14-001")
    clear_field(p1)
    a = add_char(p1, name="超新星A", power=2000, traits=["超新星"])
    b = add_char(p1, name="ハートB", power=7000, traits=["ハートの海賊団"])
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    auto_resolve(gm, p1, plan=[select_uuids([a.uuid, b.uuid])])
    # 元々パワーが入れ替わる
    assert a.get_power(True) == 7000
    assert b.get_power(True) == 2000


def test_op14_001_requires_two_targets():
    """OP14-001: 該当キャラが1体のみなら2枚選べず、入れ替えは起きない。"""
    gm, p1, p2, L = build("OP14-001")
    clear_field(p1)
    a = add_char(p1, name="超新星A", power=2000, traits=["超新星"])
    add_char(p1, name="無関係", power=9000, traits=["その他"])
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    auto_resolve(gm, p1)
    # 2枚揃わないので入れ替わらない
    assert a.get_power(True) == 2000


# ===========================================================================
# OP14-020 ジュラキュール・ミホーク
# 【PASSIVE】相手リーダーが属性《斬》→このリーダー+1000 ✅
# 【起動メイン】【ターン1回】自分のカード1枚をレストにできる：コスト5以上の
# キャラがいる場合、自分のドン!!3枚までをアクティブにする。その後、登場不可。
# 🐛 能力1: コスト5フィルタ欠落の検証
# ===========================================================================

def test_op14_020_passive_plus1000_when_opp_leader_slash():
    """OP14-020 PASSIVE: 相手リーダーが属性《斬》のときこのリーダー+1000。"""
    gm, p1, p2, L = build("OP14-020")
    gm._apply_passive_effects(p1)
    # 汎用盤面の相手リーダーは斬属性 → 5000+1000
    assert leader_power(p1) == 6000


def test_op14_020_active_don_blocked_when_no_cost5_char():
    """OP14-020 能力1: コスト4以下のキャラしかいない場合、ドンアクティブ化は起きない（条件未達）。

    注: 仕様書は cost5 フィルタ欠落(🐛)を疑ったが、実AST は FIELD_COUNT 対象に
    cost_min=5 を保持しており、エンジンも正しく評価する（テキスト通り）。
    """
    gm, p1, p2, L = build("OP14-020")
    clear_field(p1)
    add_char(p1, name="低コスト", cost=4, power=3000)
    # レストドンを3枚用意（アクティブ化の差分を観測可能に）
    for _ in range(3):
        d = p1.don_active.pop(); d.is_rest = True; p1.don_rested.append(d)
    active_before = len(p1.don_active)
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    auto_resolve(gm, p1)
    # 条件未達なのでドンはアクティブ化されない（テキスト準拠）
    assert len(p1.don_active) == active_before


def test_op14_020_active_don_when_cost5_char_present():
    """OP14-020 能力1: コスト5以上キャラがいる場合、ドン3枚までアクティブ化される。"""
    gm, p1, p2, L = build("OP14-020")
    clear_field(p1)
    add_char(p1, name="大型", cost=5, power=6000)
    for _ in range(3):
        d = p1.don_active.pop(); d.is_rest = True; p1.don_rested.append(d)
    active_before = len(p1.don_active)
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    auto_resolve(gm, p1)
    # 条件成立でレストドンがアクティブ化される
    assert len(p1.don_active) > active_before


# ===========================================================================
# OP14-040 ジンベエ  ✅
# 【起動メイン】自分の手札1枚を捨てることができる：自分の特徴《魚人族》か《人魚族》
# を持つ、リーダーかキャラ1枚にレストのドン!!2枚までを、付与する。
# ===========================================================================

def test_op14_040_discard_then_attach_rested_don2():
    """OP14-040 起動メイン: 手札1捨て→魚人族キャラにレストドン2枚付与。"""
    gm, p1, p2, L = build("OP14-040")
    clear_field(p1)
    target = add_char(p1, name="魚人", power=5000, traits=["魚人族"])
    hand_before = zone_counts(p1)["hand"]
    trash_before = zone_counts(p1)["trash"]
    discard = p1.hand[0]
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    # コスト=手札1枚捨て(SELECT) → 付与先=魚人キャラ(SELECT)
    auto_resolve(gm, p1, plan=[select_uuids([discard.uuid]), select_uuids([target.uuid])])
    assert zone_counts(p1)["hand"] == hand_before - 1
    assert zone_counts(p1)["trash"] == trash_before + 1
    assert target.attached_don == 2
    # 付与されたドンはレスト状態
    assert all(d.is_rest for d in p1.don_attached_cards if d.attached_to == target.uuid)


def test_op14_040_no_eligible_target_no_attach():
    """OP14-040: 魚人族/人魚族キャラ不在なら付与対象0（is_up_to）。"""
    gm, p1, p2, L = build("OP14-040")
    clear_field(p1)
    c = add_char(p1, name="無関係", power=5000, traits=["その他"])
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    auto_resolve(gm, p1)
    assert c.attached_don == 0


# ===========================================================================
# OP14-041 ボア・ハンコック
# 【相手のターン中】自分のキャラ登場時、1ドロー。 ✅
# 【ドン!!×1】【ターン1回】元々パワー5000以上の《アマゾン・リリー》か《九蛇海賊団》
# キャラがKOされた時、相手ライフ上1枚までを持ち主(相手)の手札に加える。
# 🐛 能力1: KO誘発がACTIVATE_MAIN化、対象/プレイヤー誤り
# ===========================================================================

def test_op14_041_opp_turn_char_play_draw():
    """OP14-041 相手ターン中: 自キャラ登場で1ドロー。"""
    gm, p1, p2, L = build("OP14-041")
    hand_before = zone_counts(p1)["hand"]
    deck_before = zone_counts(p1)["deck"]
    gm.resolve_ability(p1, get_ability(L.master, "OPPONENT_TURN"), L)
    auto_resolve(gm, p1)
    assert zone_counts(p1)["hand"] == hand_before + 1
    assert zone_counts(p1)["deck"] == deck_before - 1


@pytest.mark.xfail(strict=True,
    reason="OP14-041: 能力1がKO誘発でなくACTIVATE_MAIN化、KO条件フィルタが相手ライフ対象に誤付与＋power_max(≤5000)で5000以上(min)と逆。汎用盤面で相手ライフが手札に移動しない(NO_CHANGE)")
def test_op14_041_ko_moves_opp_life_to_owner_hand():
    """OP14-041 能力1: 該当キャラKO時、相手ライフ上1枚を相手(持ち主)の手札へ。"""
    gm, p1, p2, L = build("OP14-041")
    _attach_don_to_leader(p1, 1)
    set_life(p2, 5)
    p2_hand_before = zone_counts(p2)["hand"]
    p2_life_before = zone_counts(p2)["life"]
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    auto_resolve(gm, p1)
    # 相手ライフが1枚減り、相手の手札へ移る（テキスト準拠）
    assert zone_counts(p2)["life"] == p2_life_before - 1
    assert zone_counts(p2)["hand"] == p2_hand_before + 1


# ===========================================================================
# OP14-060 ドンキホーテ・ドフラミンゴ  ✅
# 【相手のアタック時】【ターン1回】ドン!!-1：自分のリーダーか《ドンキホーテ海賊団》
# キャラ1枚を選ぶ。選んだカードにアタックの対象を変更する。
# ===========================================================================

def test_op14_060_opp_attack_returns_don_cost():
    """OP14-060 相手アタック時: コストのドン!!-1が払われる（ドン1枚デッキへ戻る）。"""
    gm, p1, p2, L = build("OP14-060")
    don_before = don_total(p1)
    deck_before = len(p1.don_deck)
    gm.resolve_ability(p1, get_ability(L.master, "ON_OPP_ATTACK"), L)
    auto_resolve(gm, p1)
    # ドン!!-1: 場のドンが1枚減りドンデッキへ戻る
    assert don_total(p1) == don_before - 1
    assert len(p1.don_deck) == deck_before + 1


# ===========================================================================
# OP14-079 クロコダイル  ✅
# 【PASSIVE】相手のキャラすべては、自分の効果で場を離れない。
# 【起動メイン】【ターン1回】自分の『B・W』を含む特徴キャラ1枚をKOできる：
# 相手キャラ1枚までをこのターン中コスト-10。その後デッキ上2枚トラッシュ(任意)。
# ===========================================================================

@pytest.mark.xfail(strict=True,
    reason="OP14-079: PREVENT_LEAVE の範囲保護が card 所有者(=相手)のprotectorしか走査せず、保護能力を持つ自リーダー(別プレイヤー)が見つからないため、自分の効果でも相手キャラが場を離れてしまう")
def test_op14_079_passive_prevents_own_effect_removal():
    """OP14-079 PASSIVE: 相手のキャラは自分の効果で場を離れないはず。"""
    gm, p1, p2, L = build("OP14-079")
    gm._apply_passive_effects(p1)
    clear_field(p2)
    victim = add_char(p2, name="敵", power=5000)
    from engine_helpers import action
    from opcg_sim.src.models.enums import ActionType
    # 自分(p1)の効果KOを試行 → 保護で場に残るべき
    gm.apply_action_to_engine(p1, action(ActionType.KO), [victim], 0)
    assert victim in p2.field


def test_op14_079_ko_bw_then_cost_down_opponent():
    """OP14-079 能力1: 『B・W』キャラ1枚KOを払い、相手キャラ1枚をコスト-10。"""
    gm, p1, p2, L = build("OP14-079")
    clear_field(p1)
    bw = add_char(p1, name="ビビ", cost=2, power=3000, traits=["B・W"])
    clear_field(p2)
    victim = add_char(p2, name="敵", cost=5, power=5000)
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    # KOコスト(B・W)は自動適用 → 相手キャラ選択 → デッキ上2枚トラッシュ(任意)確認
    auto_resolve(gm, p1, plan=[select_uuids([victim.uuid]), confirm(True)])
    # B・WキャラがKOされ、相手キャラのコストが-10される（5-10は0で下限）
    assert bw not in p1.field
    assert victim.current_cost <= 0


# ===========================================================================
# OP14-080 ゲッコー・モリア  ✅
# 【起動メイン】【ターン1回】《スリラーバーク海賊団》キャラ1枚KOできる：
# 自分のリーダーとキャラすべてをこのターン中+1000。
# 【アタック時】手札3枚捨てられる：デッキ上1枚までをライフに加える。
# ===========================================================================

def test_op14_080_ko_thriller_then_buff_all():
    """OP14-080 能力0: スリラーバークキャラKO→自リーダー＋全キャラ+1000。"""
    gm, p1, p2, L = build("OP14-080")
    clear_field(p1)
    sacr = add_char(p1, name="スリラー", cost=3, power=4000, traits=["スリラーバーク海賊団"])
    ally = add_char(p1, name="味方", power=2000)
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    auto_resolve(gm, p1, plan=[confirm(True), select_uuids([sacr.uuid])])
    assert sacr not in p1.field
    assert leader_power(p1) == 6000        # 5000+1000
    assert ally.get_power(True) == 3000    # 2000+1000


def test_op14_080_attack_discard3_heal1():
    """OP14-080 能力1 アタック時: 手札3枚捨て→デッキ上1枚までをライフへ。"""
    gm, p1, p2, L = build("OP14-080")
    set_life(p1, 5)
    hand_before = zone_counts(p1)["hand"]
    life_before = zone_counts(p1)["life"]
    deck_before = zone_counts(p1)["deck"]
    gm.resolve_ability(p1, get_ability(L.master, "ON_ATTACK"), L)
    auto_resolve(gm, p1)
    assert zone_counts(p1)["hand"] == hand_before - 3
    assert zone_counts(p1)["life"] == life_before + 1
    assert zone_counts(p1)["deck"] == deck_before - 1


# ===========================================================================
# OP15-001 クリーク
# 【ドン!!×1】【相手のターン中】自キャラが《東の海》のみ→相手キャラ全-2000。 ✅(条件)
# 【起動メイン】【ターン1回】相手のドン!!が2枚以上付与されているキャラ1枚までをレスト。
# 🐛 能力1: 付与ドン>=2フィルタ欠落しcount=2に化け
# ===========================================================================

def test_op15_001_opp_turn_debuff_when_only_east_blue():
    """OP15-001 能力0: 付与ドン1かつ自キャラが《東の海》のみ→相手キャラ全-2000。"""
    gm, p1, p2, L = build("OP15-001")
    _attach_don_to_leader(p1, 1)
    clear_field(p1)
    add_char(p1, name="東の海", power=5000, traits=["東の海"])
    clear_field(p2)
    v = add_char(p2, name="敵", power=5000)
    gm.resolve_ability(p1, get_ability(L.master, "OPPONENT_TURN"), L)
    auto_resolve(gm, p1)
    assert v.get_power(False) == 3000


def test_op15_001_opp_turn_no_debuff_when_mixed_traits():
    """OP15-001 能力0: 自キャラに《東の海》以外が混在→条件未達で発火せず。"""
    gm, p1, p2, L = build("OP15-001")
    _attach_don_to_leader(p1, 1)
    clear_field(p1)
    add_char(p1, name="東の海", power=5000, traits=["東の海"])
    add_char(p1, name="他", power=5000, traits=["その他"])
    clear_field(p2)
    v = add_char(p2, name="敵", power=5000)
    gm.resolve_ability(p1, get_ability(L.master, "OPPONENT_TURN"), L)
    auto_resolve(gm, p1)
    assert v.get_power(False) == 5000


@pytest.mark.xfail(strict=True,
    reason="OP15-001: 能力1で『相手のドン!!が2枚以上付与されている』対象フィルタが欠落し、付与ドン0のキャラもレスト可能になる(count=2に化け)")
def test_op15_001_rest_requires_two_attached_don():
    """OP15-001 能力1: 付与ドン0のキャラはレスト対象外であるべき（フィルタ欠落のバグ）。"""
    gm, p1, p2, L = build("OP15-001")
    clear_field(p2)
    v = add_char(p2, name="敵", power=5000)  # 付与ドン0
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    auto_resolve(gm, p1)
    # 付与ドン2枚未満なので対象にならずレストされないはず
    assert v.is_rest is False


# ===========================================================================
# OP15-002 ルーシー
# 【アタック時】【相手のアタック時】手札からイベント/ステージを任意枚捨てる：
# 捨てた枚数×1000をこのリーダーにこのバトル中付与。 ✅
# 【起動メイン】【ターン1回】このターン中、元々コスト3以上のイベントを発動している場合、1ドロー。
# ⚠️→🐛 能力2: 条件がGENERIC(常時True)で未強制
# ===========================================================================

@pytest.mark.xfail(strict=True,
    reason="OP15-002: 能力2の条件『元々コスト3以上のイベントを発動している』がGENERIC(常時True)で未強制。本来未発動なら不発のはずがドローしてしまう")
def test_op15_002_draw_requires_cost3_event_played():
    """OP15-002 能力2: コスト3以上イベント未発動なら不発のはず（GENERIC常時Trueのバグ）。"""
    gm, p1, p2, L = build("OP15-002")
    hand_before = zone_counts(p1)["hand"]
    deck_before = zone_counts(p1)["deck"]
    # このターン中、何のイベントも発動していない盤面
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    auto_resolve(gm, p1)
    # 条件未達なのでドローしないはず
    assert zone_counts(p1)["hand"] == hand_before
    assert zone_counts(p1)["deck"] == deck_before


# ===========================================================================
# OP15-022 ブルック  ✅
# 【PASSIVE】デッキ0でも敗北せず、0になったターン終了時に敗北。
# 【起動メイン】【ターン1回】デッキ上4枚トラッシュ。その後デッキ0なら自キャラ1枚までをアクティブ。
# ===========================================================================

def test_op15_022_trash4_then_active_when_deck_zero():
    """OP15-022 能力1: デッキ4枚→0でトラッシュ後、自キャラ1枚をアクティブ。"""
    gm, p1, p2, L = build("OP15-022")
    # デッキを4枚に調整
    p1.deck = p1.deck[:4]
    clear_field(p1)
    c = add_char(p1, name="レスト中", power=3000, rest=True)
    trash_before = zone_counts(p1)["trash"]
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    auto_resolve(gm, p1)
    assert zone_counts(p1)["deck"] == 0
    assert zone_counts(p1)["trash"] == trash_before + 4
    assert c.is_rest is False  # デッキ0分岐でアクティブ化


def test_op15_022_trash4_only_when_deck_nonzero():
    """OP15-022 能力1: デッキが0にならない場合はアクティブ化分岐が走らない。"""
    gm, p1, p2, L = build("OP15-022")
    # デッキ20枚 → 4枚トラッシュしても0にならない
    clear_field(p1)
    c = add_char(p1, name="レスト中", power=3000, rest=True)
    deck_before = zone_counts(p1)["deck"]
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    auto_resolve(gm, p1)
    assert zone_counts(p1)["deck"] == deck_before - 4
    assert c.is_rest is True  # デッキ≠0なのでアクティブ化されない


# ===========================================================================
# OP15-039 レベッカ
# 【PASSIVE】このリーダーはアタックできない。
# 【起動メイン】リーダーレスト＋《ドレスローザ》キャラ1枚を持ち主の手札に戻す：
# 手札からコスト3の《ドレスローザ》キャラ1枚までを登場。
# ⚠️ PLAY_CARD が cost_max=3(3以下)で「コスト3ちょうど」より広い
# ===========================================================================

def test_op15_039_passive_leader_cannot_attack():
    """OP15-039 PASSIVE: このリーダーはアタックできない（ATTACK_DISABLE）。"""
    gm, p1, p2, L = build("OP15-039")
    gm._apply_passive_effects(p1)
    assert "ATTACK_DISABLE" in (L.timed_flags | L.flags)


def test_op15_039_play_only_cost_exactly_3():
    """OP15-039 能力1: 登場対象は「コスト3の」《ドレスローザ》。コスト2の手札は登場しない。

    注: 仕様書は cost_max=3(3以下)による範囲拡大(⚠️)を疑ったが、実挙動では
    コスト2の《ドレスローザ》を選んでも登場しない（テキスト通り）。
    """
    gm, p1, p2, L = build("OP15-039")
    clear_field(p1)
    bounce_target = add_char(p1, name="ドレ場", cost=2, power=2000, traits=["ドレスローザ"])
    # 手札にコスト2のドレスローザ（本来は「コスト3の」に該当せず対象外）
    low = make_char(p1, name="ドレ手札", cost=2, power=1000, traits=["ドレスローザ"])
    p1.hand.append(low)
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    auto_resolve(gm, p1, plan=[confirm(True), select_uuids([bounce_target.uuid]),
                               select_uuids([low.uuid])])
    # コスト2のドレスローザは「コスト3の」に該当しないので登場しない
    assert low not in p1.field


# ===========================================================================
# OP15-058 エネル
# 【PASSIVE】ドン!!デッキは6枚になる。
# 【起動メイン】【ターン1回】第2ターン以降の場合、ドン1アクティブ＋4レスト追加、
# その後キャラ1枚にレストドン4まで付与。
# ⚠️→🐛 条件「第2ターン以降」がGENERIC(常時True)で未強制
# ===========================================================================

def test_op15_058_ramp_and_attach_when_active():
    """OP15-058 能力1: ドンランプ(アクティブ1)＋キャラにレストドン4付与。"""
    gm, p1, p2, L = build("OP15-058")
    gm._apply_passive_effects(p1)
    clear_field(p1)
    c = add_char(p1, name="付与先", power=1000)
    don_before = don_total(p1)
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    auto_resolve(gm, p1, plan=[select_uuids([c.uuid])])
    # アクティブ1＋レスト4追加でドン総数+5（ドンデッキに余裕がある前提）
    assert don_total(p1) >= don_before + 1
    assert c.attached_don == 4


@pytest.mark.xfail(strict=True,
    reason="OP15-058: 条件『自分の第2ターン以降』がGENERIC(常時True)で未強制。第1ターン相当でも発動してドンが増えてしまう")
def test_op15_058_blocked_on_first_turn():
    """OP15-058 能力1: 第1ターンでは条件未達で発動しないはず（GENERIC常時Trueのバグ）。"""
    gm, p1, p2, L = build("OP15-058")
    gm._apply_passive_effects(p1)
    gm.turn_count = 1  # 自分の第1ターン
    clear_field(p1)
    c = add_char(p1, name="付与先", power=1000)
    don_before = don_total(p1)
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    auto_resolve(gm, p1)
    # 第1ターンなのでドンランプも付与も起きないはず
    assert don_total(p1) == don_before
    assert c.attached_don == 0


# ===========================================================================
# OP15-098 モンキー・D・ルフィ  ⚠️
# 【PASSIVE/置換】元々パワー6000以上の《空島》キャラが相手によって場を離れる場合、
# 代わりに自分のライフ上1枚を手札に加えることができる。
# ===========================================================================

def test_op15_098_replace_opp_removal_with_life_to_hand():
    """OP15-098 置換: 元々パワー6000以上の《空島》キャラが相手効果で離れる→代わりにライフ上1枚を手札へ。"""
    from engine_helpers import action
    from opcg_sim.src.models.enums import ActionType
    gm, p1, p2, L = build("OP15-098")
    gm._apply_passive_effects(p1)
    clear_field(p1)
    skypiea = add_char(p1, name="空島キャラ", power=6000, traits=["空島"])
    set_life(p1, 5)
    hand_before = zone_counts(p1)["hand"]
    life_before = zone_counts(p1)["life"]
    # 相手(p2)の効果KOで離脱を試みる
    gm.apply_action_to_engine(p2, action(ActionType.KO), [skypiea], 0)
    auto_resolve(gm, p1)
    # 置換: キャラは場に残り、代わりにライフ1枚が手札へ
    assert skypiea in p1.field
    assert zone_counts(p1)["life"] == life_before - 1
    assert zone_counts(p1)["hand"] == hand_before + 1


def test_op15_098_no_replace_on_own_effect():
    """OP15-098: 自分の効果で離れる場合は置換が発火しない（相手によって限定）。"""
    from engine_helpers import action
    from opcg_sim.src.models.enums import ActionType
    gm, p1, p2, L = build("OP15-098")
    gm._apply_passive_effects(p1)
    clear_field(p1)
    skypiea = add_char(p1, name="空島キャラ", power=6000, traits=["空島"])
    set_life(p1, 5)
    life_before = zone_counts(p1)["life"]
    # 自分(p1)の効果KO → 置換は発火せず通常KO
    gm.apply_action_to_engine(p1, action(ActionType.KO), [skypiea], 0)
    auto_resolve(gm, p1)
    assert skypiea not in p1.field
    assert zone_counts(p1)["life"] == life_before  # ライフは動かない
