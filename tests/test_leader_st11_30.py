"""リーダーカード効果テスト — ST11〜ST30（10枚）。

仕様書 docs/leader_specs/ST11-30.md のテストケースを pytest 化。
**テキスト準拠の正しい挙動**をアサートする。現実装にバグがある（🐛）能力は
@pytest.mark.xfail(strict=True) を付ける（修正されると xpass→strict で赤になり
マーカー除去を促す）。

実行:
    OPCG_LOG_SILENT=1 python -m pytest tests/test_leader_st11_30.py -q -s -p no:cacheprovider

対象: ST11-001 ST12-001 ST13-001 ST13-002 ST13-003 ST14-001
      ST21-001 ST22-001 ST29-001 ST30-001
"""
import pytest

from leader_test_helpers import (
    build, get_ability, auto_resolve,
    add_char, make_char, clear_field, set_life,
    leader_power, zone_counts,
)
from opcg_sim.src.core.effects.matcher import get_target_cards
from opcg_sim.src.models.effect_types import TargetQuery
from opcg_sim.src.models.enums import Player, Zone


# ---------------------------------------------------------------------------
# 共通補助
# ---------------------------------------------------------------------------

def _attach_don(player, host, n):
    """コストエリアのアクティブドン!! n 枚を host に付与（【ドン!!×N】条件を満たす）。"""
    for _ in range(n):
        d = player.don_active.pop()
        d.is_rest = False
        d.attached_to = host.uuid
        player.don_attached_cards.append(d)
    host.attached_don = getattr(host, "attached_don", 0) + n


def _rest_active_don(player, n):
    """コストエリアのアクティブドン!! n 枚をレストにする。"""
    for _ in range(n):
        d = player.don_active.pop()
        d.is_rest = True
        player.don_rested.append(d)


# ===========================================================================
# ST11-001 ウタ — 【ドン!!×1】【アタック時】【ターン1回】
#   デッキ上1枚公開→FILM1枚までを手札→残りをデッキ下
# ===========================================================================

def test_st11_001_attack_film_top_added_to_hand():
    """ST11-001 アタック時: デッキ上が《FILM》なら公開→手札に加わる（ドン!!×1成立）。"""
    gm, p1, p2, L = build("ST11-001")
    _attach_don(p1, L, 1)
    film = make_char(p1, name="FilmCard", traits=["FILM"])
    p1.deck.insert(0, film)
    hand0 = len(p1.hand)
    gm.resolve_ability(p1, get_ability(L.master, "ON_ATTACK"), L)
    auto_resolve(gm, p1)
    assert film in p1.hand
    assert len(p1.hand) == hand0 + 1


def test_st11_001_attack_non_film_goes_to_deck_bottom():
    """ST11-001 アタック時: デッキ上が非FILMなら手札に加えず（0枚=まで）デッキ下へ。"""
    gm, p1, p2, L = build("ST11-001")
    _attach_don(p1, L, 1)
    non = make_char(p1, name="NonFilm", traits=["その他"])
    p1.deck.insert(0, non)
    hand0 = len(p1.hand)
    gm.resolve_ability(p1, get_ability(L.master, "ON_ATTACK"), L)
    auto_resolve(gm, p1)
    assert non not in p1.hand
    assert len(p1.hand) == hand0          # まで=0枚で手札に加わらない
    assert p1.deck[-1] is non             # 残りはデッキの下へ


def test_st11_001_attack_no_don_does_not_fire():
    """ST11-001 アタック時: ドン!!×1未達（付与0）なら未発動（条件 HAS_DON>=1）。"""
    gm, p1, p2, L = build("ST11-001")
    film = make_char(p1, name="FilmCard", traits=["FILM"])
    p1.deck.insert(0, film)
    hand0 = len(p1.hand)
    gm.resolve_ability(p1, get_ability(L.master, "ON_ATTACK"), L)
    auto_resolve(gm, p1)
    assert film not in p1.hand
    assert len(p1.hand) == hand0


# ===========================================================================
# ST12-001 ロロノア・ゾロ&サンジ — 【ドン!!×1】【アタック時】【ターン1回】
#   任意コスト: 自コスト2以上キャラ1枚を手札に戻す → パワー7000以下1枚までアクティブ
# ===========================================================================

def test_st12_001_bounce_then_activate_le7000():
    """ST12-001 アタック時: コスト2以上を手札に戻し→パワー7000以下のレストキャラをアクティブ。"""
    gm, p1, p2, L = build("ST12-001")
    clear_field(p1)
    _attach_don(p1, L, 1)
    cost_char = add_char(p1, name="C2", cost=3, power=4000)         # コスト2以上→バウンス
    rested = add_char(p1, name="R", cost=1, power=5000, rest=True)  # 7000以下→アクティブ
    gm.resolve_ability(p1, get_ability(L.master, "ON_ATTACK"), L)
    auto_resolve(gm, p1)
    assert cost_char in p1.hand        # 持ち主（自分）の手札へ戻る
    assert rested.is_rest is False     # アクティブ化


def test_st12_001_power8000_not_activated():
    """ST12-001: パワー8000のレストキャラのみ→アクティブ0枚（7000以下＋まで）。"""
    gm, p1, p2, L = build("ST12-001")
    clear_field(p1)
    _attach_don(p1, L, 1)
    cost_char = add_char(p1, name="C2", cost=3, power=4000)
    strong = add_char(p1, name="P8000", cost=1, power=8000, rest=True)  # 7000超→対象外
    gm.resolve_ability(p1, get_ability(L.master, "ON_ATTACK"), L)
    auto_resolve(gm, p1)
    assert cost_char in p1.hand
    assert strong.is_rest is True      # 7000以下条件＋まで でアクティブにならない


# ===========================================================================
# ST13-001 サボ — 🐛 コスト対象「パワー7000以上」が power_max=7000（以下）に反転
#   起動メイン: コスト3以上かつパワー7000以上のキャラをライフ上表向き→キャラ1枚+2000
# ===========================================================================

def _st13_001_cost_query():
    """ST13-001 のコスト(FACE_UP_LIFE)の TargetQuery を FIELD 参照に直して返す。

    パース上 zone=LIFE だが対象は場のキャラなので、パワー条件の検証用に
    zone=FIELD へ写した同等クエリ（コスト/パワー条件は parser 由来をそのまま使用）。
    """
    gm, p1, p2, L = build("ST13-001")
    cost_tq = get_ability(L.master, "ACTIVATE_MAIN").cost.target
    return TargetQuery(
        player=Player.SELF, zone=Zone.FIELD, card_type=["CHARACTER"],
        cost_min=cost_tq.cost_min, power_min=cost_tq.power_min, power_max=cost_tq.power_max,
    )


def test_st13_001_cost_target_power_ge_includes_8000():
    """ST13-001 コスト対象: パワー7000以上=8000のキャラが選択可能であるべき（条件成立盤面）。"""
    gm, p1, p2, L = build("ST13-001")
    clear_field(p1)
    strong = add_char(p1, name="P8000", cost=4, power=8000)   # コスト3以上/パワー7000以上=対象
    weak = add_char(p1, name="P6000", cost=4, power=6000)
    matched = get_target_cards(gm, _st13_001_cost_query(), L)
    assert strong in matched           # 8000は対象（テキスト: 7000以上）
    assert weak not in matched         # 6000は対象外


def test_st13_001_cost_target_power_lt_7000_excluded():
    """ST13-001 コスト対象: パワー6000のキャラのみ→コスト対象不在であるべき（条件不成立盤面）。"""
    gm, p1, p2, L = build("ST13-001")
    clear_field(p1)
    add_char(p1, name="P6000", cost=4, power=6000)   # 7000未満=テキストでは対象外
    add_char(p1, name="P5000", cost=5, power=5000)
    matched = get_target_cards(gm, _st13_001_cost_query(), L)
    assert matched == []               # 7000以上のキャラがいない=対象0枚


# ===========================================================================
# ST13-002 ポートガス・D・エース
#   能力0 起動メイン(ドン!!×2): デッキ上5枚→コスト5キャラ1枚までライフ上表向き→残りデッキ下
#   能力1 ターン終了時: 自分のライフの🐛「表向き」のカードすべてをトラッシュ
# ===========================================================================

def test_st13_002_a0_look5_arrange_completes_cleanly():
    """ST13-002 能力0: デッキ上5枚を見て→並べ替えデッキ下、デッキ枚数が復元され temp リーク無し。"""
    gm, p1, p2, L = build("ST13-002")
    _attach_don(p1, L, 2)
    c5 = make_char(p1, name="C5", cost=5)
    p1.deck = [c5] + p1.deck
    deck0, life0 = len(p1.deck), len(p1.life)
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    auto_resolve(gm, p1)
    assert len(p1.deck) == deck0        # 見た5枚はデッキ下へ戻り枚数復元
    assert len(p1.life) == life0        # まで=0枚（自動選択で未取得でも破綻しない）
    assert len(p1.temp_zone) == 0       # temp リーク無し


def test_st13_002_a1_turn_end_trashes_only_face_up_life():
    """ST13-002 能力1: ターン終了時、ライフの表向きカードのみをトラッシュ（裏向きは残す）。"""
    gm, p1, p2, L = build("ST13-002")
    set_life(p1, 5)
    p1.life[0].is_face_up = True
    p1.life[1].is_face_up = True       # 表向き2枚 / 裏向き3枚
    gm.resolve_ability(p1, get_ability(L.master, "TURN_END"), L)
    auto_resolve(gm, p1)
    assert len(p1.life) == 3           # 表向き2枚のみトラッシュ→残3枚


# ===========================================================================
# ST13-003 モンキー・D・ルフィ — 🐛 取得元が手札(HAND)のみで「トラッシュ」欠落
#   起動メイン(ドン!!×2): 手札1枚捨て可→ライフ0枚の場合、手札かトラッシュの
#   コスト5キャラ2枚までをライフ上表向きで加える
# ===========================================================================

def test_st13_003_life_not_zero_does_not_fire():
    """ST13-003: ライフ1枚以上なら不発（LIFE_COUNT EQ0 未達）。"""
    gm, p1, p2, L = build("ST13-003")
    set_life(p1, 1)
    _attach_don(p1, L, 2)
    p1.hand = [make_char(p1, name="Filler", cost=1)]
    c5 = make_char(p1, name="HandC5", cost=5)
    p1.hand.append(c5)
    life0 = len(p1.life)
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    auto_resolve(gm, p1)
    assert len(p1.life) == life0       # 条件未達→ライフ追加なし


def test_st13_003_adds_cost5_char_from_trash():
    """ST13-003: ライフ0かつコスト5キャラがトラッシュにのみある場合でもライフ上に加えられるべき。"""
    gm, p1, p2, L = build("ST13-003")
    set_life(p1, 0)
    _attach_don(p1, L, 2)
    p1.hand = [make_char(p1, name="Filler", cost=1)]   # 捨てコスト用
    trash_c5 = make_char(p1, name="TrashC5", cost=5)
    p1.trash.append(trash_c5)
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    auto_resolve(gm, p1)
    assert trash_c5 in p1.life          # トラッシュからもライフ上に加わるべき


# ===========================================================================
# ST14-001 モンキー・D・ルフィ — 🐛 分岐条件「コスト8以上」が FIELD_COUNT GE1 に縮退
#   【ドン!!×1】自分のキャラ全てコスト+1。コスト8以上がいればリーダー+1000
# ===========================================================================

def test_st14_001_cost8_present_buffs_leader():
    """ST14-001（条件成立）: コスト8以上のキャラがいる→全キャラコスト+1＋リーダー+1000。"""
    gm, p1, p2, L = build("ST14-001")
    clear_field(p1)
    _attach_don(p1, L, 1)
    c2 = add_char(p1, name="A", cost=2, power=2000)
    c8 = add_char(p1, name="B", cost=8, power=9000)
    base = leader_power(p1)            # 付与ドン1枚込みの発動前パワー(6000)
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    auto_resolve(gm, p1)
    assert c2.current_cost == 3        # 全キャラコスト+1
    assert c8.current_cost == 9
    assert leader_power(p1) == base + 1000   # コスト8以上いる→リーダー+1000


@pytest.mark.xfail(strict=True,
                   reason="ST14-001: 分岐条件『コスト8以上のキャラがいる場合』が FIELD_COUNT GE1 に"
                          "縮退（cost_min=8 欠落）。コスト7以下のみでもリーダー+1000してしまう")
def test_st14_001_only_cost7_no_leader_buff():
    """ST14-001（条件不成立）: コスト7以下のキャラのみ→全キャラコスト+1のみ、リーダー+1000なし。"""
    gm, p1, p2, L = build("ST14-001")
    clear_field(p1)
    _attach_don(p1, L, 1)
    c7 = add_char(p1, name="A", cost=7, power=7000)
    base = leader_power(p1)            # 付与ドン1枚込みの発動前パワー(6000)
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    auto_resolve(gm, p1)
    assert c7.current_cost == 8        # コスト+1 は適用
    assert leader_power(p1) == base    # コスト8以上不在→リーダーは増えない（テキスト）


# ===========================================================================
# ST21-001 モンキー・D・ルフィ — 【ドン!!×1】【起動メイン】【ターン1回】
#   自分のキャラ1枚にレストのドン!!2枚までを付与
# ===========================================================================

def test_st21_001_attaches_up_to_2_rested_don():
    """ST21-001 起動メイン: 自キャラ1枚にレストのドン!!最大2枚を付与する。"""
    gm, p1, p2, L = build("ST21-001")
    clear_field(p1)
    target = add_char(p1, name="Tgt", power=5000)
    _attach_don(p1, L, 1)              # ドン!!×1 条件
    _rest_active_don(p1, 2)            # 付与可能なレストドン2枚
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    auto_resolve(gm, p1)
    assert target.attached_don == 2


def test_st21_001_attached_don_are_rested():
    """ST21-001: 付与されるドン!!は「レスト」状態である。"""
    gm, p1, p2, L = build("ST21-001")
    clear_field(p1)
    target = add_char(p1, name="Tgt", power=5000)
    _attach_don(p1, L, 1)
    _rest_active_don(p1, 2)
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    auto_resolve(gm, p1)
    assert target.attached_don == 2
    attached = [d for d in p1.don_attached_cards if d.attached_to == target.uuid]
    assert len(attached) == 2
    assert all(d.is_rest for d in attached)   # 付与ドンはレスト状態


# ===========================================================================
# ST22-001 エース&ニューゲート — 【起動メイン】【ターン1回】
#   任意コスト: 手札の『白ひげ海賊団』を含む特徴カード1枚公開→1ドロー→公開カードをデッキ上
# ===========================================================================

def test_st22_001_reveal_draw_put_on_top():
    """ST22-001 起動メイン: 白ひげ海賊団特徴を公開→1ドロー→公開カードをデッキの上に置く。"""
    gm, p1, p2, L = build("ST22-001")
    wb = make_char(p1, name="WB", traits=["白ひげ海賊団"])
    p1.hand.append(wb)
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    auto_resolve(gm, p1)
    assert wb not in p1.hand           # 公開カードは手札を離れ
    assert p1.deck[0] is wb            # デッキの上へ


def test_st22_001_compound_trait_revealable():
    """ST22-001: 複合特徴（『白ひげ海賊団/〇〇』）でも"含む特徴"として公開対象になる。"""
    gm, p1, p2, L = build("ST22-001")
    wb = make_char(p1, name="WBcombo", traits=["白ひげ海賊団", "四皇"])
    p1.hand.append(wb)
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    auto_resolve(gm, p1)
    assert wb not in p1.hand
    assert p1.deck[0] is wb


# ===========================================================================
# ST29-001 モンキー・D・ルフィ — 🐛 DISCARD が条件分岐の外（無条件化）
#   【アタック時】自分のライフが2枚以下の場合、カード1枚を引き、自分の手札1枚を捨てる
# ===========================================================================

def test_st29_001_life_le2_draws_and_discards():
    """ST29-001（条件成立）: ライフ2枚→1ドロー＋手札1枚捨て（両方実行）。"""
    gm, p1, p2, L = build("ST29-001")
    set_life(p1, 2)
    z0 = zone_counts(p1)
    gm.resolve_ability(p1, get_ability(L.master, "ON_ATTACK"), L)
    auto_resolve(gm, p1)
    z1 = zone_counts(p1)
    assert z1["deck"] == z0["deck"] - 1    # 1ドロー
    assert z1["trash"] == z0["trash"] + 1  # 1枚捨て
    assert z1["hand"] == z0["hand"]        # +1ドロー -1捨て=±0


def test_st29_001_life0_draws_and_discards():
    """ST29-001（条件成立・境界下）: ライフ0枚でも LE2 成立→1ドロー＋手札1枚捨て。"""
    gm, p1, p2, L = build("ST29-001")
    set_life(p1, 0)
    z0 = zone_counts(p1)
    gm.resolve_ability(p1, get_ability(L.master, "ON_ATTACK"), L)
    auto_resolve(gm, p1)
    z1 = zone_counts(p1)
    assert z1["deck"] == z0["deck"] - 1
    assert z1["trash"] == z0["trash"] + 1


def test_st29_001_life_ge3_does_nothing():
    """ST29-001（条件不成立）: ライフ3枚以上→ドローも捨ても起きない（何もしない）。"""
    gm, p1, p2, L = build("ST29-001")
    set_life(p1, 4)
    z0 = zone_counts(p1)
    gm.resolve_ability(p1, get_ability(L.master, "ON_ATTACK"), L)
    auto_resolve(gm, p1)
    z1 = zone_counts(p1)
    assert z1["deck"] == z0["deck"]        # ドローなし
    assert z1["trash"] == z0["trash"]      # 捨てなし
    assert z1["hand"] == z0["hand"]


# ===========================================================================
# ST30-001 ルフィ&エース
#   能力0 PASSIVE: 🐛 自分の元々のパワー7000以上のキャラがいる場合、リーダー-2000
#   能力1 相手ターン中: 🐛「ポートガス・D・エース」と「モンキー・D・ルフィ」すべて+3000
# ===========================================================================

def test_st30_001_passive_power_ge7000_reduces_leader():
    """ST30-001 能力0（条件成立）: 元々パワー7000以上のキャラがいる→リーダー-2000(6000→4000)。"""
    gm, p1, p2, L = build("ST30-001")
    clear_field(p1)
    add_char(p1, name="Strong", power=8000)   # 元々パワー7000以上
    gm._apply_passive_effects(p1)
    assert leader_power(p1) == 4000            # 6000-2000


def test_st30_001_passive_power_lt7000_no_reduction():
    """ST30-001 能力0（条件不成立）: 元々パワー6000以下のキャラのみ→リーダー減算なし(6000)。"""
    gm, p1, p2, L = build("ST30-001")
    clear_field(p1)
    add_char(p1, name="Weak", power=6000)     # 7000未満=条件不成立
    gm._apply_passive_effects(p1)
    assert leader_power(p1) == 6000            # 減算されない


def test_st30_001_passive_no_char_no_reduction():
    """ST30-001 能力0: 自場にキャラ0枚→リーダー減算なし(6000)。"""
    gm, p1, p2, L = build("ST30-001")
    clear_field(p1)
    gm._apply_passive_effects(p1)
    assert leader_power(p1) == 6000


def test_st30_001_opponent_turn_buffs_both_names():
    """ST30-001 能力1（相手ターン中）: 自場の『エース』と『ルフィ』両名すべて+3000。"""
    gm, p1, p2, L = build("ST30-001")
    clear_field(p1)
    ace = add_char(p1, name="ポートガス・D・エース", power=5000)
    luffy = add_char(p1, name="モンキー・D・ルフィ", power=5000)
    gm.resolve_ability(p2, get_ability(L.master, "OPPONENT_TURN"), L)
    auto_resolve(gm, p2)
    assert ace.get_power(False) == 8000     # エース+3000
    assert luffy.get_power(False) == 8000   # ルフィ+3000（欠落でここが失敗）


def test_st30_001_opponent_turn_buffs_ace():
    """ST30-001 能力1: 自場の『ポートガス・D・エース』は相手ターン中+3000される。"""
    gm, p1, p2, L = build("ST30-001")
    clear_field(p1)
    ace = add_char(p1, name="ポートガス・D・エース", power=5000)
    other = add_char(p1, name="ゾロ", power=5000)
    gm.resolve_ability(p2, get_ability(L.master, "OPPONENT_TURN"), L)
    auto_resolve(gm, p2)
    assert ace.get_power(False) == 8000     # エースは+3000
    assert other.get_power(False) == 5000   # 対象外は不変
