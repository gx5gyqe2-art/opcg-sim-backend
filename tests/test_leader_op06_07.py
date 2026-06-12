"""リーダーカード効果テスト — OP06 / OP07（仕様書 docs/leader_specs/OP06-07.md）。

実行（-s 必須）:
    OPCG_LOG_SILENT=1 python -m pytest tests/test_leader_op06_07.py -q -s -p no:cacheprovider

判定ラベルに応じてマーカーを付与:
  ✅ → 通常テスト / 🐛 → xfail(strict=True) / ⚠️ → 通常 or xfail(strict=False)

検出サマリ（仕様書末尾）:
  ✅ OP06-001 / OP06-021 / OP06-022 / OP07-019 / OP07-059 / OP07-079 / OP07-097
  🐛 OP06-020 / OP06-042 / OP06-080 / OP07-001 / OP07-038
"""
import pytest

from leader_test_helpers import (  # noqa: F401
    build, get_ability, abilities_of, auto_resolve,
    select_uuids, confirm, choose,
    add_char, make_char, clear_field, set_life,
    leader_power, don_total, zone_counts, leader_master,
)
from opcg_sim.src.models.models import DonInstance


# ---------------------------------------------------------------------------
# OP06-001 ウタ（✅）
# ---------------------------------------------------------------------------

def test_op06_001_on_attack_discard_film_debuff_and_ramp_rested_don():
    """OP06-001【アタック時】FILM手札を捨て→相手キャラpw-2000＋ドンデッキからレストで1枚追加。"""
    gm, p1, p2, L = build("OP06-001")
    clear_field(p2)
    victim = add_char(p2, name="V", power=5000)
    film = make_char(p1, name="FILMC", traits=["FILM"])
    p1.hand.append(film)
    rested_before = len(p1.don_rested)

    gm.resolve_ability(p1, get_ability(L.master, "ON_ATTACK"), L)
    auto_resolve(gm, p1)

    assert victim.get_power(False) == 3000          # 5000 - 2000(THIS_TURN)
    assert film in p1.trash and film not in p1.hand  # コスト=FILM1枚捨て
    assert len(p1.don_rested) == rested_before + 1   # レストで1枚追加


def test_op06_001_no_film_card_cannot_pay_cost():
    """OP06-001: 手札に《FILM》が無いとコスト未達で発火せず盤面不変（任意なのでスキップ）。"""
    gm, p1, p2, L = build("OP06-001")
    clear_field(p2)
    victim = add_char(p2, name="V", power=5000)
    p1.hand = []                                      # FILM手札なし
    rested_before = len(p1.don_rested)

    gm.resolve_ability(p1, get_ability(L.master, "ON_ATTACK"), L)
    auto_resolve(gm, p1)

    assert victim.get_power(False) == 5000           # 変化なし
    assert len(p1.don_rested) == rested_before


# ---------------------------------------------------------------------------
# OP06-020 ホーディ・ジョーンズ（🐛 対象「コスト3以下キャラ」選択肢＆コスト上限欠落）
# ---------------------------------------------------------------------------

def test_op06_020_rest_opponent_cost3_character():
    """OP06-020【起動メイン】相手のコスト3以下のキャラをレストにできるべき（バグで不可）。"""
    gm, p1, p2, L = build("OP06-020")
    clear_field(p2)
    p2.don_active = []                               # ドン側の選択肢を排除しキャラ経路を検証
    c3 = add_char(p2, name="C3", cost=3, power=5000)

    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    auto_resolve(gm, p1)

    assert c3.is_rest is True                         # コスト3キャラがレストになるはず


def test_op06_020_cost4_character_cannot_be_rested():
    """OP06-020: 相手キャラがコスト4のみならレスト不可（コスト≤3制限）。盤面不変。"""
    gm, p1, p2, L = build("OP06-020")
    clear_field(p2)
    p2.don_active = []
    c4 = add_char(p2, name="C4", cost=4, power=5000)

    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    auto_resolve(gm, p1)

    assert c4.is_rest is False                        # コスト4はレストできない


# ---------------------------------------------------------------------------
# OP06-021 ペローナ（✅ choice 2択）
# ---------------------------------------------------------------------------

def test_op06_021_choice_rest_opponent_cost4_or_less():
    """OP06-021【起動メイン】選択肢①: 相手のコスト4以下のキャラをレストにする。"""
    gm, p1, p2, L = build("OP06-021")
    clear_field(p2)
    c4 = add_char(p2, name="C4", cost=4, power=5000)

    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    auto_resolve(gm, p1, plan=[choose(0), select_uuids([c4.uuid])])

    assert c4.is_rest is True


def test_op06_021_choice_cost_reduction_this_turn():
    """OP06-021【起動メイン】選択肢②: 相手のキャラ1枚をこのターン中コスト-1。"""
    gm, p1, p2, L = build("OP06-021")
    clear_field(p2)
    ch = add_char(p2, name="X", cost=5, power=5000)

    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    auto_resolve(gm, p1, plan=[choose(1), select_uuids([ch.uuid])])

    assert ch.current_cost == 4                       # 5 - 1(THIS_TURN)


# ---------------------------------------------------------------------------
# OP06-022 ヤマト（✅ 正: 1キャラに2ドン）— OP07-001 の鏡像
# ---------------------------------------------------------------------------

def test_op06_022_attaches_two_don_to_one_character():
    """OP06-022【起動メイン】相手ライフ≤3で、自分の『1枚の』キャラにレストドン『2枚』付与。"""
    gm, p1, p2, L = build("OP06-022")
    clear_field(p1)
    clear_field(p2)
    set_life(p2, 3)                                   # 条件: 相手ライフ≤3
    for _ in range(2):                                # レストドンを2枚用意
        d = p1.don_active.pop(); d.is_rest = True; p1.don_rested.append(d)
    ch = add_char(p1, name="A", power=3000)

    ab = get_ability(L.master, "ACTIVATE_MAIN")
    # 鏡像差の厳密化: 付与対象は最大『1枚』のキャラ（OP07-001 は max=2 の誤り）
    gm.resolve_ability(p1, ab, L)
    cons = (gm.active_interaction or {}).get("constraints") or {}
    assert cons.get("max") == 1                       # 対象キャラは1枚まで
    auto_resolve(gm, p1)

    assert ch.attached_don == 2                        # 1キャラに2ドン付与
    assert len(p1.don_rested) == 0


def test_op06_022_condition_not_met_when_opponent_life_high():
    """OP06-022: 相手ライフ4（>3）なら条件未達で発動せず付与なし。"""
    gm, p1, p2, L = build("OP06-022")
    clear_field(p1)
    set_life(p2, 4)
    for _ in range(2):
        d = p1.don_active.pop(); d.is_rest = True; p1.don_rested.append(d)
    ch = add_char(p1, name="A", power=3000)

    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    auto_resolve(gm, p1)

    assert ch.attached_don == 0                        # 条件未達で付与されない


# ---------------------------------------------------------------------------
# OP06-042 ヴィンスモーク・レイジュ（🐛 発火条件「ドン返却時」欠落→自ターン無条件ドロー）
# ---------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason=(
    "OP06-042: 発火条件『自分の場のドンがドンデッキに戻された時』が欠落し、"
    "ドン返却イベントが無くても自ターンに無条件ドローしてしまう。"))
def test_op06_042_no_draw_without_don_returned_event():
    """OP06-042【自分のターン中】ドン返却イベント無しではドローしないべき（バグで引く）。"""
    gm, p1, p2, L = build("OP06-042")
    hand_before, deck_before = len(p1.hand), len(p1.deck)

    gm.resolve_ability(p1, get_ability(L.master, "YOUR_TURN"), L)
    auto_resolve(gm, p1)

    assert len(p1.hand) == hand_before                 # 発火条件未達→ドローしない
    assert len(p1.deck) == deck_before


# ---------------------------------------------------------------------------
# OP06-080 ゲッコー・モリア（🐛 起動コスト➁=ドン2枚レストが value=1 に取りこぼし）
# ---------------------------------------------------------------------------

def test_op06_080_activation_cost_rests_two_don():
    """OP06-080【アタック時】コストの➁はドン2枚レスト（value=2）であるべき（バグで1）。"""
    ab = get_ability(leader_master("OP06-080"), "ON_ATTACK")
    rest_don = ab.cost.actions[0]                       # Sequence[REST_DON, DISCARD]
    assert rest_don.type.name == "REST_DON"
    assert rest_don.value.base == 2                     # ➁＝2枚


# ---------------------------------------------------------------------------
# OP07-001 モンキー・D・ドラゴン（🐛 count↔value 逆転）— OP06-022 の鏡像
# ---------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason=(
    "OP07-001: 『ドン合計2枚まで／キャラ1枚』が『キャラ2枚まで／各1枚』に逆転"
    "（count↔value）。対象は1枚・付与ドンは最大2であるべき。"))
def test_op07_001_attaches_up_to_two_don_to_single_character():
    """OP07-001【起動メイン】付与済みドン合計2枚までを『キャラ1枚』に付与すべき。

    バグでは対象キャラが最大2枚（max=2）・各1枚付与になる。対象上限=1 をアサート。
    """
    gm, p1, p2, L = build("OP07-001")
    clear_field(p1)
    for _ in range(2):                                  # リーダーに付与済みドン2枚
        p1.don_active.pop()
        p1.don_attached_cards.append(DonInstance(owner_id=p1.name, attached_to=L.uuid))
    L.attached_don = 2
    c1 = add_char(p1, name="A", power=1000)
    add_char(p1, name="B", power=1000)

    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    cons = (gm.active_interaction or {}).get("constraints") or {}
    # 正しくは対象キャラ1枚まで（OP06-022 と同じ max=1）。バグでは max=2。
    assert cons.get("max") == 1


@pytest.mark.xfail(strict=True, reason=(
    "OP07-001: ATTACH_DON value=1 のため、キャラが1体でも付与されるドンは1枚に留まる"
    "（合計2枚まで集約されるべき）。鏡像 OP06-022 は同盤面で2枚付与。"))
def test_op07_001_single_character_receives_both_don():
    """OP07-001: 自分キャラが1体だけなら、その1体にドン2枚が集約されるべき（バグで1枚）。"""
    gm, p1, p2, L = build("OP07-001")
    clear_field(p1)
    for _ in range(2):
        p1.don_active.pop()
        p1.don_attached_cards.append(DonInstance(owner_id=p1.name, attached_to=L.uuid))
    L.attached_don = 2
    ch = add_char(p1, name="A", power=1000)

    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    auto_resolve(gm, p1)

    assert ch.attached_don == 2                         # 2枚集約されるべき


# ---------------------------------------------------------------------------
# OP07-019 ジュエリー・ボニー（✅）
# ---------------------------------------------------------------------------

def test_op07_019_rest_don_then_rest_opponent_leader_or_character():
    """OP07-019【相手のアタック時】ドン1枚レスト→相手のリーダーかキャラ1枚をレスト。"""
    gm, p1, p2, L = build("OP07-019")
    clear_field(p2)
    victim = add_char(p2, name="V", power=5000)
    active_before = len(p1.don_active)

    gm.resolve_ability(p1, get_ability(L.master, "ON_OPP_ATTACK"), L)
    auto_resolve(gm, p1)

    assert victim.is_rest is True                        # 対象がレスト
    assert len(p1.don_active) == active_before - 1       # コスト: ドン1枚レスト
    assert len(p1.don_rested) >= 1


# ---------------------------------------------------------------------------
# OP07-038 ボア・ハンコック（🐛 発火条件「自効果でキャラ退場時」欠落→無条件ドロー）
# ---------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason=(
    "OP07-038: 発火条件『キャラが自分の効果で場を離れた時』が欠落し、退場イベントが"
    "無くても自ターン・手札≤5で無条件ドローしてしまう。"))
def test_op07_038_no_draw_without_leave_event():
    """OP07-038【自分のターン中】退場イベント無しでは手札≤5でもドローしないべき。"""
    gm, p1, p2, L = build("OP07-038")
    hand_before, deck_before = len(p1.hand), len(p1.deck)   # 既定手札5枚（≤5）

    gm.resolve_ability(p1, get_ability(L.master, "YOUR_TURN"), L)
    auto_resolve(gm, p1)

    assert len(p1.hand) == hand_before                      # 発火条件未達→ドローしない
    assert len(p1.deck) == deck_before


def test_op07_038_hand_over_five_does_not_draw():
    """OP07-038: 手札が6枚（>5）なら条件未達でドローしない（手札条件は機能している）。"""
    gm, p1, p2, L = build("OP07-038")
    p1.hand.append(make_char(p1, name="extra"))             # 手札6枚に
    hand_before = len(p1.hand)

    gm.resolve_ability(p1, get_ability(L.master, "YOUR_TURN"), L)
    auto_resolve(gm, p1)

    assert len(p1.hand) == hand_before                      # 手札>5でドローしない


# ---------------------------------------------------------------------------
# OP07-059 フォクシー（✅）
# ---------------------------------------------------------------------------

def test_op07_059_freeze_when_foxy_chars_ge_three():
    """OP07-059【アタック時】ドン3戻し→《フォクシー海賊団》3体以上で相手レストカードをFREEZE。"""
    gm, p1, p2, L = build("OP07-059")
    clear_field(p1)
    clear_field(p2)
    for i in range(3):
        add_char(p1, name=f"F{i}", traits=["フォクシー海賊団"], power=1000)
    victim = add_char(p2, name="V", power=5000, rest=True)   # 相手のレストキャラ

    gm.resolve_ability(p1, get_ability(L.master, "ON_ATTACK"), L)
    auto_resolve(gm, p1)

    assert "FREEZE" in (victim.flags | victim.timed_flags)   # FREEZE 付与


@pytest.mark.xfail(strict=False, reason=(
    "要確認: OP07-059 FREEZE は ref_id=selected_card が未保存（branch条件未達で SELECT スキップ）"
    "でも相手キャラへフォールバック適用される。条件未達なら FREEZE されないのが正。"))
def test_op07_059_no_freeze_when_foxy_chars_below_three():
    """OP07-059: 《フォクシー海賊団》が2体以下なら条件未達でFREEZEされないべき。"""
    gm, p1, p2, L = build("OP07-059")
    clear_field(p1)
    clear_field(p2)
    for i in range(2):                                       # 2体のみ
        add_char(p1, name=f"F{i}", traits=["フォクシー海賊団"], power=1000)
    victim = add_char(p2, name="V", power=5000, rest=True)

    gm.resolve_ability(p1, get_ability(L.master, "ON_ATTACK"), L)
    auto_resolve(gm, p1)

    assert "FREEZE" not in (victim.flags | victim.timed_flags)


# ---------------------------------------------------------------------------
# OP07-079 ロブ・ルッチ（✅）
# ---------------------------------------------------------------------------

def test_op07_079_mill_two_then_cost_reduction():
    """OP07-079【アタック時】デッキ上2枚トラッシュ→相手キャラをこのターン中コスト-1。"""
    gm, p1, p2, L = build("OP07-079")
    clear_field(p2)
    victim = add_char(p2, name="V", cost=5, power=5000)
    deck_before, trash_before = len(p1.deck), len(p1.trash)

    gm.resolve_ability(p1, get_ability(L.master, "ON_ATTACK"), L)
    auto_resolve(gm, p1)

    assert len(p1.deck) == deck_before - 2               # デッキ上2枚
    assert len(p1.trash) == trash_before + 2             # トラッシュへ
    assert victim.current_cost == 4                       # 5 - 1(THIS_TURN)


# ---------------------------------------------------------------------------
# OP07-097 ベガパンク（✅ PASSIVE アタック不可 + choice）
# ---------------------------------------------------------------------------

def test_op07_097_passive_disables_leader_attack():
    """OP07-097【PASSIVE】このリーダーはアタックできない（ATTACK_DISABLE 付与）。"""
    gm, p1, p2, L = build("OP07-097")
    gm.resolve_ability(p1, get_ability(L.master, "PASSIVE"), L)
    assert "ATTACK_DISABLE" in (L.flags | L.timed_flags)


def test_op07_097_activate_main_choice_play_egghead():
    """OP07-097【起動メイン】ドン1レスト→選択肢②: コスト5以下《エッグヘッド》を登場。"""
    gm, p1, p2, L = build("OP07-097")
    egg = make_char(p1, name="EGG", cost=3, traits=["エッグヘッド"])
    p1.hand.append(egg)
    active_before = len(p1.don_active)

    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    auto_resolve(gm, p1, plan=[choose(1), select_uuids([egg.uuid])])

    assert egg in p1.field and egg not in p1.hand          # 登場
    assert len(p1.don_active) == active_before - 1         # コスト: ドン1枚レスト


@pytest.mark.xfail(strict=False, reason=(
    "要確認: OP07-097 選択肢①の FACE_UP_LIFE target.zone が LIFE と解釈され、手札の"
    "《エッグヘッド》が候補にならず移動しない（手札→ライフ上の移動が表面化しない）。"))
def test_op07_097_activate_main_choice_face_up_life():
    """OP07-097【起動メイン】選択肢①: コスト5以下《エッグヘッド》をライフ上に表向きで加える。"""
    gm, p1, p2, L = build("OP07-097")
    egg = make_char(p1, name="EGG", cost=3, traits=["エッグヘッド"])
    p1.hand.append(egg)
    life_before = len(p1.life)

    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    auto_resolve(gm, p1, plan=[choose(0), select_uuids([egg.uuid])])

    assert egg not in p1.hand                              # 手札から移動
    assert len(p1.life) == life_before + 1                 # ライフ上に加わる
    assert egg in p1.life and egg.is_face_up is True       # 表向き
