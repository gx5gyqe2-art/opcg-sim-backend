"""リーダーカード効果テスト — OP08 / OP09。

仕様書 docs/leader_specs/OP08-09.md のテストケースを pytest 化したもの。
テキスト準拠の正しい挙動をアサートする。

実行:
    OPCG_LOG_SILENT=1 python -m pytest tests/test_leader_op08_09.py -q -s -p no:cacheprovider
"""
import pytest

from leader_test_helpers import (
    build, get_ability, auto_resolve,
    select_uuids, confirm, choose,
    add_char, clear_field, set_life,
    leader_power, don_total, zone_counts,
)


# ===========================================================================
# OP08-001 トニートニー・チョッパー
# 【起動メイン】【ターン1回】自分の特徴《動物》か《ドラム王国》を持つキャラ
# 3枚までにレストのドン!!1枚ずつまでを、付与する。
# ===========================================================================

def test_op08_001_attach_rested_don_to_animal_chars():
    """OP08-001 起動メイン: 《動物》キャラ3体それぞれにレストのドン1枚ずつ付与。"""
    gm, p1, p2, L = build("OP08-001")
    clear_field(p1)
    c1 = add_char(p1, name="動物1", traits=["動物"])
    c2 = add_char(p1, name="動物2", traits=["動物"])
    c3 = add_char(p1, name="動物3", traits=["ドラム王国"])
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    # 「3枚までに」は is_up_to=True → constraints.min=0 のため auto_resolve 既定は
    # 1枚しか選ばない。3枚すべてに付与されることを検証するため明示選択。
    auto_resolve(gm, p1, plan=[select_uuids([c1.uuid, c2.uuid, c3.uuid])])
    assert c1.attached_don == 1
    assert c2.attached_don == 1
    assert c3.attached_don == 1
    # 付与されたドンはレスト状態
    assert all(d.is_rest for d in p1.don_attached_cards)


def test_op08_001_no_eligible_target_no_change():
    """OP08-001: 対象（動物/ドラム王国）キャラ0体なら付与なし（is_up_to で0枚）。"""
    gm, p1, p2, L = build("OP08-001")
    clear_field(p1)
    add_char(p1, name="無関係", traits=["麦わらの一味"])
    before = sum(c.attached_don for c in p1.field)
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    auto_resolve(gm, p1)
    assert sum(c.attached_don for c in p1.field) == before


# ===========================================================================
# OP08-002 マルコ
# 【ドン!!×1】【起動メイン】【ターン1回】カード1枚を引き、自分の手札1枚を
# デッキの上か下に置く。その後、相手のキャラ1枚までを、このターン中、パワー-2000。
# ===========================================================================

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


def test_op08_002_draw_deckback_then_debuff():
    """OP08-002 起動メイン: DON1付与下で1ドロー→手札1枚デッキ戻し→相手キャラ-2000。"""
    gm, p1, p2, L = build("OP08-002")
    _attach_don_to_leader(p1, 1)
    clear_field(p2)
    victim = add_char(p2, name="敵", power=5000)
    hand_before = zone_counts(p1)["hand"]
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    # 手順: SELECT_TARGET(戻す手札) → ARRANGE_DECK(上/下) → SELECT_TARGET(相手キャラ)
    auto_resolve(gm, p1)
    # ドロー+デッキ戻しで手札枚数は不変（+1引いて-1戻す）
    assert zone_counts(p1)["hand"] == hand_before
    assert victim.get_power(True) == 3000


def test_op08_002_no_don_no_activation():
    """OP08-002: 【ドン!!×1】未充足（DON未付与）なら発動不可。"""
    gm, p1, p2, L = build("OP08-002")
    clear_field(p2)
    victim = add_char(p2, name="敵", power=5000)
    hand_before = zone_counts(p1)["hand"]
    deck_before = zone_counts(p1)["deck"]
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    auto_resolve(gm, p1)
    # 条件未達 → ドローもデバフも起きない
    assert zone_counts(p1)["hand"] == hand_before
    assert zone_counts(p1)["deck"] == deck_before
    assert victim.get_power(True) == 5000


# ===========================================================================
# OP08-021 キャロット
# 【起動メイン】【ターン1回】自分の特徴《ミンク族》を持つキャラがいる場合、
# 相手のコスト5以下のキャラ1枚までを、レストにする。
# ===========================================================================

def test_op08_021_rest_opponent_when_mink_present():
    """OP08-021: 自分場に《ミンク族》がいる→相手コスト5以下キャラをレスト。"""
    gm, p1, p2, L = build("OP08-021")
    clear_field(p1)
    add_char(p1, name="ミンク", traits=["ミンク族"])
    clear_field(p2)
    victim = add_char(p2, name="敵", cost=5, power=5000)
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    auto_resolve(gm, p1, plan=[select_uuids([victim.uuid])])
    assert victim.is_rest is True


def test_op08_021_no_mink_no_effect():
    """OP08-021: 自分場に《ミンク族》がいない→不発（条件未達）。"""
    gm, p1, p2, L = build("OP08-021")
    clear_field(p1)
    add_char(p1, name="無関係", traits=["その他"])
    clear_field(p2)
    victim = add_char(p2, name="敵", cost=5, power=5000)
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    auto_resolve(gm, p1)
    assert victim.is_rest is False


def test_op08_021_cost6_not_targetable():
    """OP08-021: 相手がコスト6以上のみ→対象なし（is_up_to で0枚）。"""
    gm, p1, p2, L = build("OP08-021")
    clear_field(p1)
    add_char(p1, name="ミンク", traits=["ミンク族"])
    clear_field(p2)
    victim = add_char(p2, name="敵", cost=6, power=7000)
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    auto_resolve(gm, p1)
    assert victim.is_rest is False


# ===========================================================================
# OP08-057 キング
# 【起動メイン】【ターン1回】ドン!!-2：以下から1つを選ぶ。
# ・自分の手札が5枚以下の場合、カード1枚を引く。
# ・相手のキャラ1枚までを、このターン中、コスト-2。
# ===========================================================================

def test_op08_057_choice_draw_when_hand_le5():
    """OP08-057: 選択A（手札5枚以下）→ドン2返却の上で1ドロー。"""
    gm, p1, p2, L = build("OP08-057")
    set_hand = zone_counts(p1)["hand"]
    assert set_hand <= 5
    deck_before = zone_counts(p1)["deck"]
    don_before = don_total(p1)
    don_uuids = [d.uuid for d in p1.don_active[:2]]
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    # 手順: SELECT_RESOURCE(ドン2返却) → CHOICE(選択A) → DRAW
    auto_resolve(gm, p1, plan=[select_uuids(don_uuids), choose(0)])
    assert zone_counts(p1)["hand"] == set_hand + 1
    assert zone_counts(p1)["deck"] == deck_before - 1
    # コスト ドン!!-2 が支払われている
    assert don_total(p1) == don_before - 2


def test_op08_057_choice_cost_down_opponent():
    """OP08-057: 選択B→相手キャラ1枚を今ターン中コスト-2。"""
    gm, p1, p2, L = build("OP08-057")
    clear_field(p2)
    victim = add_char(p2, name="敵", cost=5, power=5000)
    don_uuids = [d.uuid for d in p1.don_active[:2]]
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    # 手順: SELECT_RESOURCE(ドン2返却) → CHOICE(選択B) → SELECT_TARGET(相手キャラ)
    auto_resolve(gm, p1, plan=[select_uuids(don_uuids), choose(1), select_uuids([victim.uuid])])
    assert victim.current_cost == 3


# ===========================================================================
# OP08-058 シャーロット・プリン
# 【アタック時】自分のライフの上から2枚を表向きにできる：
# ドン!!デッキからドン!!1枚までを、レストで追加する。
# ===========================================================================

def test_op08_058_face_up_life_then_ramp_rested_don():
    """OP08-058 アタック時: ライフ上2枚を表向き→ドン1枚レストで追加。"""
    gm, p1, p2, L = build("OP08-058")
    set_life(p1, 5)
    don_before = don_total(p1)
    rested_before = len(p1.don_rested)
    gm.resolve_ability(p1, get_ability(L.master, "ON_ATTACK"), L)
    auto_resolve(gm, p1)
    assert don_total(p1) == don_before + 1
    assert len(p1.don_rested) == rested_before + 1
    # ライフ上2枚が表向きに
    assert sum(1 for c in p1.life[:2] if c.is_face_up) == 2


# ===========================================================================
# OP08-098 カルガラ
# 【ドン!!×1】【アタック時】自分の手札から自分の場のドン!!の枚数以下のコストを持ち、
# 特徴《シャンドラの戦士》を持つキャラカード1枚までを、登場させる。
# 登場させた場合、自分のライフの上から1枚を手札に加える。
# ===========================================================================

def test_op08_098_play_shandora_then_life_to_hand():
    """OP08-098 アタック時: 《シャンドラの戦士》登場→『登場させた場合』ライフ上1枚を手札へ。"""
    from leader_test_helpers import make_char
    gm, p1, p2, L = build("OP08-098")
    _attach_don_to_leader(p1, 1)
    play = make_char(p1, name="戦士", cost=2, power=3000,
                     traits=["シャンドラの戦士"])
    p1.hand.append(play)
    set_life(p1, 5)
    life_before = zone_counts(p1)["life"]
    gm.resolve_ability(p1, get_ability(L.master, "ON_ATTACK"), L)
    auto_resolve(gm, p1, plan=[select_uuids([play.uuid])])
    assert play in p1.field            # 該当キャラが登場
    assert zone_counts(p1)["life"] == life_before - 1  # ライフ上1枚が手札へ


def test_op08_098_no_play_no_life_gain():
    """OP08-098: 登場させなかった場合（is_up_to で0枚）はライフ加算なし。"""
    gm, p1, p2, L = build("OP08-098")
    _attach_don_to_leader(p1, 1)
    set_life(p1, 5)
    life_before = zone_counts(p1)["life"]
    gm.resolve_ability(p1, get_ability(L.master, "ON_ATTACK"), L)
    # 手札に《シャンドラの戦士》が無い → 登場対象なし → 分岐不成立
    auto_resolve(gm, p1)
    assert zone_counts(p1)["life"] == life_before


# ===========================================================================
# OP09-001 シャンクス
# 【ターン1回】相手がアタックした時、発動できる。
# 相手のリーダーかキャラ1枚までを、このターン中、パワー-1000。
# ===========================================================================

def test_op09_001_opp_attack_debuff_leader_or_char():
    """OP09-001 相手アタック時: 相手リーダーかキャラ1枚を今ターン-1000。"""
    gm, p1, p2, L = build("OP09-001")
    clear_field(p2)
    victim = add_char(p2, name="敵", power=4000)
    gm.resolve_ability(p1, get_ability(L.master, "ON_OPP_ATTACK"), L)
    auto_resolve(gm, p1, plan=[confirm(True), select_uuids([victim.uuid])])
    assert victim.get_power(True) == 3000


# ===========================================================================
# OP09-022 リム
# 自分のキャラカードはレストで登場する。
# 【起動メイン】【ターン1回】自分のドン!!3枚をレストにできる：
# ドン!!デッキからドン!!1枚までをレストで追加し、自分の手札からコスト5以下の
# 特徴《ODYSSEY》を持つキャラカード1枚までを、登場させる。
# ===========================================================================

def test_op09_022_activate_rest3_ramp_and_play():
    """OP09-022 起動メイン: ドン3枚レスト→ドン1枚レスト追加→《ODYSSEY》登場。"""
    from leader_test_helpers import make_char
    gm, p1, p2, L = build("OP09-022")
    play = make_char(p1, name="オデッセイ", cost=4, power=3000, traits=["ODYSSEY"])
    p1.hand.append(play)
    don_before = don_total(p1)
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    auto_resolve(gm, p1)
    # ドンランプで +1
    assert don_total(p1) == don_before + 1
    assert play in p1.field


# ===========================================================================
# OP09-042 バギー
# 【起動メイン】自分のドン!!5枚をレストにし、自分の手札1枚を捨てることができる：
# 自分の手札から特徴《クロスギルド》を持つキャラカード1枚までを、登場させる。
# ===========================================================================

def test_op09_042_rest5_discard_then_play_crossguild():
    """OP09-042 起動メイン: ドン5レスト＋手札1捨て→《クロスギルド》登場。"""
    from leader_test_helpers import make_char
    gm, p1, p2, L = build("OP09-042")
    play = make_char(p1, name="クロスギルド員", cost=4, power=4000, traits=["クロスギルド"])
    p1.hand.append(play)
    rested_before = len(p1.don_rested)
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    auto_resolve(gm, p1)
    assert len(p1.don_rested) == rested_before + 5
    assert play in p1.field


# ===========================================================================
# OP09-061 モンキー・D・ルフィ
# 【ドン!!×1】自分のキャラすべてを、コスト+1。
# 【自分のターン中】【ターン1回】自分の場のドン!!が2枚以上ドン!!デッキに戻された時、
# ドン!!デッキからドン!!1枚までをアクティブで追加し、さらに1枚までをレストで追加する。
# ===========================================================================

def test_op09_061_all_char_cost_plus1():
    """OP09-061 能力0（ドン!!×1）: 自分のキャラすべてコスト+1。"""
    gm, p1, p2, L = build("OP09-061")
    _attach_don_to_leader(p1, 1)
    clear_field(p1)
    c1 = add_char(p1, name="味方1", cost=3)
    c2 = add_char(p1, name="味方2", cost=5)
    gm._apply_passive_effects(p1)   # 【ドン!!×1】常在効果は passive 再計算で適用
    assert c1.current_cost == 4
    assert c2.current_cost == 6


def test_op09_061_ramp_requires_two_don_returned():
    """OP09-061 能力1: 本来『2枚以上戻された時』が前提だが、条件欠落で無条件発火する（バグ）。"""
    gm, p1, p2, L = build("OP09-061")
    don_before = don_total(p1)
    # ドン返却イベントなしで能力1を直接発動 → 本来は不発であるべき
    gm.resolve_ability(p1, get_ability(L.master, "YOUR_TURN"), L)
    auto_resolve(gm, p1)
    # 条件が正しく実装されていれば don は増えない（=このアサートが通る）
    assert don_total(p1) == don_before


# ===========================================================================
# OP09-062 ニコ・ロビン
# 【バニッシュ】... 【アタック時】自分の手札から【トリガー】を持つカード1枚を
# 捨てることができる：ドン!!デッキからドン!!1枚までを、レストで追加する。
# ===========================================================================

def test_op09_062_attack_discard_trigger_ramp_rested_don():
    """OP09-062 アタック時: 【トリガー】持ち手札1枚を捨て→ドン1枚レスト追加。"""
    from leader_test_helpers import make_char
    from opcg_sim.src.models.effect_types import Ability, GameAction, ValueSource
    from opcg_sim.src.models.enums import TriggerType, ActionType, CardType
    from engine_helpers import make_master, make_instance
    gm, p1, p2, L = build("OP09-062")
    # 【トリガー】持ちカードを手札に用意
    trig = Ability(trigger=TriggerType.TRIGGER,
                   effect=GameAction(type=ActionType.DRAW, value=ValueSource(base=1)))
    tcard = make_instance(make_master(card_id="TRG-1", name="トリガー持ち",
                                      type=CardType.EVENT, abilities=(trig,)), owner=p1.name)
    p1.hand.append(tcard)
    don_before = don_total(p1)
    rested_before = len(p1.don_rested)
    gm.resolve_ability(p1, get_ability(L.master, "ON_ATTACK"), L)
    auto_resolve(gm, p1)
    assert don_total(p1) == don_before + 1
    assert len(p1.don_rested) == rested_before + 1


# ===========================================================================
# OP09-081 マーシャル・D・ティーチ
# 自分の【登場時】効果は無効になる。
# 【起動メイン】自分の手札1枚を捨てることができる：次の相手のターン終了時まで、
# 相手の【登場時】効果は無効になる。
# ===========================================================================

def test_op09_081_activate_discard_disable_opp_onplay():
    """OP09-081 起動メイン: 手札1枚捨て→相手の【登場時】無効（次相手ターン終了まで）。"""
    gm, p1, p2, L = build("OP09-081")
    hand_before = zone_counts(p1)["hand"]
    trash_before = zone_counts(p1)["trash"]
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    auto_resolve(gm, p1)
    # コストの手札1枚捨てが行われた
    assert zone_counts(p1)["hand"] == hand_before - 1
    assert zone_counts(p1)["trash"] == trash_before + 1
