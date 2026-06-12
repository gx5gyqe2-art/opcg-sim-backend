"""リーダーカード効果テスト — スターターデッキ ST01〜ST10（12枚）。

仕様書: docs/leader_specs/ST01-10.md / ガイド: docs/leader_specs/_TEST_GUIDE.md

各テストはテキスト準拠の「正しい挙動」をアサートする。現実装にバグがある
ケースは @pytest.mark.xfail(strict=True) を付け、修正で xpass→赤になりマーカー
除去を促す。要確認(⚠️)ケースは通常テストで書き、検証不能/不安定なものは
xfail(strict=False) とする。

実行:
    OPCG_LOG_SILENT=1 python -m pytest tests/test_leader_st01_10.py -q -s -p no:cacheprovider
"""
import pytest

from leader_test_helpers import (
    build, get_ability, auto_resolve,
    select_uuids, confirm,
    add_char, clear_field, set_life,
    leader_power, don_total,
)


# ---------------------------------------------------------------------------
# ヘルパ: 場のドン(active+rested)を正確に n 枚に再構成する
# ---------------------------------------------------------------------------

def _set_field_don(player, n_active, n_rested=0):
    """player の場のドンを active=n_active / rested=n_rested 枚に再構成する。"""
    pool = list(player.don_active) + list(player.don_rested) + list(player.don_deck)
    player.don_active = []
    player.don_rested = []
    for _ in range(n_active):
        d = pool.pop(0)
        d.is_rest = False
        player.don_active.append(d)
    for _ in range(n_rested):
        d = pool.pop(0)
        d.is_rest = True
        player.don_rested.append(d)
    player.don_deck = pool


def _drive(gm, player, prefer_uuids=()):
    """active_interaction を駆動する。SELECT 系で prefer_uuids が候補にあれば
    それを優先選択し、無ければ min 枚を先頭から選ぶ。"""
    prefer = set(prefer_uuids)
    steps = 0
    while gm.active_interaction and steps < 20:
        ia = gm.active_interaction
        at = ia.get("action_type", "")
        if at in ("CONFIRM_OPTIONAL", "CONFIRM_TRIGGER"):
            gm.resolve_interaction(player, {"accepted": True})
        elif at in ("SELECT_TARGET", "SELECT_RESOURCE"):
            cands = [c.uuid for c in ia.get("candidates", [])]
            cons = ia.get("constraints") or {}
            mn = cons.get("min", 0)
            mx = cons.get("max", 1)
            chosen = [u for u in cands if u in prefer]
            if not chosen:
                n = max(mn, 1) if cands else 0
                if mx:
                    n = min(n, mx)
                chosen = cands[:n]
            gm.resolve_interaction(player, {"selected_uuids": chosen})
        else:
            gm.resolve_interaction(player, {"selected_uuids": [], "index": 0})
        steps += 1
    return steps


# ===========================================================================
# ST01-001 モンキー・D・ルフィ ✅
#   【起動メイン】【ターン1回】このリーダーか自分のキャラ1枚にレストのドン1枚までを付与
# ===========================================================================

def test_st01_001_attach_rested_don_to_leader():
    """ST01-001 起動メイン: このリーダーにレストのドン1枚付与でパワー+1000、ドンはレスト。"""
    gm, p1, p2, L = build("ST01-001")
    ab = get_ability(L.master, "ACTIVATE_MAIN")
    don_before = don_total(p1)
    gm.resolve_ability(p1, ab, L)
    gm.resolve_interaction(p1, select_uuids([L.uuid]))
    assert leader_power(p1) == 6000           # +1000
    assert L.attached_don == 1
    assert don_total(p1) == don_before - 1     # 場のドンが1枚付与へ移動
    assert all(d.is_rest for d in p1.don_attached_cards)  # レスト状態で付与


def test_st01_001_attach_zero_is_allowed():
    """ST01-001 起動メイン: 「1枚まで」(is_up_to)なので0枚付与（付与先を選ばない）も可。"""
    gm, p1, p2, L = build("ST01-001")
    ab = get_ability(L.master, "ACTIVATE_MAIN")
    don_before = don_total(p1)
    gm.resolve_ability(p1, ab, L)
    gm.resolve_interaction(p1, select_uuids([]))   # 0枚選択
    assert leader_power(p1) == 5000                # 変化なし
    assert L.attached_don == 0
    assert don_total(p1) == don_before


def test_st01_001_turn_once_blocks_second():
    """ST01-001 起動メイン: 【ターン1回】なので同一ターンの2回目は不発。"""
    gm, p1, p2, L = build("ST01-001")
    ab = get_ability(L.master, "ACTIVATE_MAIN")
    gm.resolve_ability(p1, ab, L)
    gm.resolve_interaction(p1, select_uuids([L.uuid]))
    assert leader_power(p1) == 6000
    gm.resolve_ability(p1, ab, L)                  # 2回目
    assert gm.active_interaction is None           # 対話すら開始しない
    assert leader_power(p1) == 6000                # 上昇は1回分のまま


# ===========================================================================
# ST02-001 ユースタス・キッド ⚠️
#   【起動メイン】【ターン1回】③，手札1枚捨て：このリーダーをアクティブにする
# ===========================================================================

def test_st02_001_activates_leader_with_cost():
    """ST02-001 起動メイン: コスト(ドンレスト＋手札1捨て)を払い、レストのリーダーをアクティブに。"""
    gm, p1, p2, L = build("ST02-001")
    L.is_rest = True
    ab = get_ability(L.master, "ACTIVATE_MAIN")
    hand_before = len(p1.hand)
    trash_before = len(p1.trash)
    gm.resolve_ability(p1, ab, L)
    _drive(gm, p1)
    assert L.is_rest is False                       # リーダーがアクティブに戻った
    assert len(p1.hand) == hand_before - 1          # 手札1枚捨て
    assert len(p1.trash) == trash_before + 1


def test_st02_001_no_hand_cannot_activate():
    """ST02-001 起動メイン: 手札0枚では捨てコストを払えず発動不可（リーダーはレストのまま）。"""
    gm, p1, p2, L = build("ST02-001")
    L.is_rest = True
    p1.hand = []
    ab = get_ability(L.master, "ACTIVATE_MAIN")
    gm.resolve_ability(p1, ab, L)
    _drive(gm, p1)
    assert L.is_rest is True                         # 払えないのでアクティブにならない


def test_st02_001_cost_rests_three_don():
    """ST02-001 起動メイン: コスト③はドン3枚をレストにするはず（テキスト準拠）。"""
    gm, p1, p2, L = build("ST02-001")
    L.is_rest = True
    rested_before = len(p1.don_rested)
    ab = get_ability(L.master, "ACTIVATE_MAIN")
    gm.resolve_ability(p1, ab, L)
    _drive(gm, p1)
    assert len(p1.don_rested) == rested_before + 3   # ③＝3枚レスト


# ===========================================================================
# ST03-001 クロコダイル 🐛
#   【起動メイン】【ターン1回】ドン-4：コスト5以下のキャラ1枚までを持ち主の手札に戻す
#   原文は「キャラ」で自分・相手両方が対象。実装は OPPONENT 固定（対象範囲欠落）。
# ===========================================================================

def test_st03_001_bounce_opponent_character():
    """ST03-001 起動メイン: ドン4戻し→相手のコスト5以下キャラを持ち主(相手)の手札へ戻す。"""
    gm, p1, p2, L = build("ST03-001")
    clear_field(p2)
    victim = add_char(p2, name="相手キャラ", cost=3, power=2000)
    ab = get_ability(L.master, "ACTIVATE_MAIN")
    don_before = don_total(p1)
    gm.resolve_ability(p1, ab, L)
    _drive(gm, p1)
    assert victim not in p2.field
    assert victim in p2.hand                         # 持ち主(相手)の手札へ
    assert don_total(p1) == don_before - 4           # ドン-4コスト


@pytest.mark.xfail(strict=True, reason="ST03-001 BUG: BOUNCE対象が player=OPPONENT 固定。原文「コスト5以下のキャラ」は自分のキャラも対象であるべき")
def test_st03_001_bounce_can_target_own_character():
    """ST03-001 起動メイン: 自分のコスト5以下キャラも対象に取れ、持ち主(自分)の手札へ戻るべき。"""
    gm, p1, p2, L = build("ST03-001")
    clear_field(p1)
    clear_field(p2)                                  # 相手キャラを除き、自分のキャラのみ
    mine = add_char(p1, name="自分キャラ", cost=3, power=2000)
    ab = get_ability(L.master, "ACTIVATE_MAIN")
    gm.resolve_ability(p1, ab, L)
    # 自分キャラを対象に取れるなら、それを選んでバウンスする
    _drive(gm, p1, prefer_uuids=[mine.uuid])
    assert mine not in p1.field
    assert mine in p1.hand                           # 持ち主(自分)の手札へ


# ===========================================================================
# ST04-001 カイドウ ✅
#   【起動メイン】【ターン1回】ドン-7：相手のライフ1枚までをトラッシュに置く
# ===========================================================================

def test_st04_001_trash_opponent_life():
    """ST04-001 起動メイン: ドン7戻し→相手ライフ1枚をトラッシュへ（トリガー処理なし）。"""
    gm, p1, p2, L = build("ST04-001")
    life_before = len(p2.life)
    trash_before = len(p2.trash)
    don_before = don_total(p1)
    ab = get_ability(L.master, "ACTIVATE_MAIN")
    gm.resolve_ability(p1, ab, L)
    auto_resolve(gm, p1)
    assert len(p2.life) == life_before - 1
    assert len(p2.trash) == trash_before + 1         # ライフが直接トラッシュへ
    assert don_total(p1) == don_before - 7           # ドン-7コスト


def test_st04_001_insufficient_don_cannot_activate():
    """ST04-001 起動メイン: 場のドンが6枚以下ではコスト(ドン-7)を払えず不発。"""
    gm, p1, p2, L = build("ST04-001")
    _set_field_don(p1, n_active=6)
    life_before = len(p2.life)
    ab = get_ability(L.master, "ACTIVATE_MAIN")
    gm.resolve_ability(p1, ab, L)
    assert gm.active_interaction is None             # コスト不成立で対話開始せず
    assert len(p2.life) == life_before               # ライフ不変


# ===========================================================================
# ST05-001 シャンクス ⚠️
#   【起動メイン】【ターン1回】ドン-3：自分の特徴《FILM》キャラ全てをこのターン+2000
# ===========================================================================

def test_st05_001_buffs_only_self_film_characters():
    """ST05-001 起動メイン: ドン3戻し→自分のFILMキャラ全て+2000。非FILM/相手は対象外。"""
    gm, p1, p2, L = build("ST05-001")
    clear_field(p1)
    film = add_char(p1, name="FILMキャラ", cost=3, power=4000, traits=["FILM"])
    non_film = add_char(p1, name="非FILM", cost=3, power=4000, traits=["麦わらの一味"])
    opp_film = add_char(p2, name="相手FILM", cost=3, power=4000, traits=["FILM"])
    don_before = don_total(p1)
    ab = get_ability(L.master, "ACTIVATE_MAIN")
    gm.resolve_ability(p1, ab, L)
    auto_resolve(gm, p1)
    assert film.get_power(True) == 6000              # +2000
    assert non_film.get_power(True) == 4000          # FILM以外は不変
    assert opp_film.get_power(True) == 4000          # 相手は不変
    assert don_total(p1) == don_before - 3


def test_st05_001_no_film_character_no_effect():
    """ST05-001 起動メイン: FILMキャラ不在なら対象0枚で盤面のパワーは変化しない。"""
    gm, p1, p2, L = build("ST05-001")
    clear_field(p1)
    plain = add_char(p1, name="非FILM", cost=3, power=4000, traits=["麦わらの一味"])
    ab = get_ability(L.master, "ACTIVATE_MAIN")
    gm.resolve_ability(p1, ab, L)
    auto_resolve(gm, p1)
    assert plain.get_power(True) == 4000             # 対象外、変化なし


# ===========================================================================
# ST06-001 サカズキ ⚠️
#   【起動メイン】【ターン1回】③，手札1枚捨て：相手のコスト0のキャラ1枚までをKO
# ===========================================================================

def test_st06_001_ko_opponent_cost0_character():
    """ST06-001 起動メイン: 手札1捨て等のコスト→相手のコスト0キャラをKO（コスト1以上は対象外）。"""
    gm, p1, p2, L = build("ST06-001")
    clear_field(p2)
    victim = add_char(p2, name="コスト0", cost=0, power=1000)
    safe = add_char(p2, name="コスト1", cost=1, power=1000)
    hand_before = len(p1.hand)
    ab = get_ability(L.master, "ACTIVATE_MAIN")
    gm.resolve_ability(p1, ab, L)
    _drive(gm, p1, prefer_uuids=[victim.uuid])
    assert victim in p2.trash                        # コスト0キャラがKO
    assert victim not in p2.field
    assert safe in p2.field                          # コスト1キャラは対象外
    assert len(p1.hand) == hand_before - 1           # 手札1枚捨てコスト


def test_st06_001_no_cost0_target_no_ko():
    """ST06-001 起動メイン: 相手にコスト1以上のキャラしかいなければKO対象は0枚。"""
    gm, p1, p2, L = build("ST06-001")
    clear_field(p2)
    safe = add_char(p2, name="コスト2", cost=2, power=1000)
    ab = get_ability(L.master, "ACTIVATE_MAIN")
    gm.resolve_ability(p1, ab, L)
    _drive(gm, p1)
    assert safe in p2.field                          # KOされない


# ===========================================================================
# ST07-001 シャーロット・リンリン ⚠️
#   【ドン×2】【アタック時】ライフ上下1枚を手札に加える：ライフ2枚以下の場合、手札1枚まで
#   をライフの上に加える
# ===========================================================================

def test_st07_001_life_swap_when_life_le_2():
    """ST07-001 アタック時: ドン2＋ライフ2枚以下で、ライフ操作(コスト)と手札→ライフ(効果)が解決。"""
    gm, p1, p2, L = build("ST07-001")
    set_life(p1, 2)
    for _ in range(2):                               # リーダーにドン2付与（【ドン×2】条件）
        d = p1.don_active.pop()
        p1.don_attached_cards.append(d)
    L.attached_don = 2
    ab = get_ability(L.master, "ON_ATTACK")
    hand_before = len(p1.hand)
    life_before = len(p1.life)
    gm.resolve_ability(p1, ab, L)
    _drive(gm, p1)
    # コストでライフ1枚を手札へ(+1)、効果で手札1枚をライフへ(-1) → 差し引きゼロ
    assert len(p1.hand) == hand_before
    assert len(p1.life) == life_before


@pytest.mark.xfail(strict=False, reason="要確認(ST07-001): 「ライフ2枚以下」が効果側条件だがコスト(ライフ上下→手札)にも掛かり、ライフ3枚以上だとコストも不発の疑い")
def test_st07_001_cost_is_unconditional_when_life_ge_3():
    """ST07-001 アタック時: ライフ3枚でも先のコスト(ライフ上下→手札)は無条件で発生するはず。"""
    gm, p1, p2, L = build("ST07-001")
    set_life(p1, 3)
    for _ in range(2):
        d = p1.don_active.pop()
        p1.don_attached_cards.append(d)
    L.attached_don = 2
    ab = get_ability(L.master, "ON_ATTACK")
    hand_before = len(p1.hand)
    gm.resolve_ability(p1, ab, L)
    _drive(gm, p1)
    # コスト(ライフ→手札)は条件非依存なので、ライフ3枚でも手札が1枚増えるはず
    assert len(p1.hand) == hand_before + 1


# ===========================================================================
# ST08-001 モンキー・D・ルフィ ⚠️
#   【自分のターン中】キャラがKOされた時、このリーダーにレストのドン1枚までを付与
# ===========================================================================

def test_st08_001_attach_rested_don_to_leader():
    """ST08-001: 自分ターン中のKO連動で、このリーダーにレストのドン1枚付与（+1000、レスト）。"""
    gm, p1, p2, L = build("ST08-001")
    don_before = don_total(p1)
    ab = get_ability(L.master, "YOUR_TURN")
    gm.resolve_ability(p1, ab, L)
    _drive(gm, p1, prefer_uuids=[L.uuid])
    assert leader_power(p1) == 6000                  # +1000
    assert L.attached_don == 1
    assert don_total(p1) == don_before - 1
    assert all(d.is_rest for d in p1.don_attached_cards)  # レスト状態


# ===========================================================================
# ST09-001 ヤマト ⚠️
#   【ドン×1】【相手のターン中】自分のライフが2枚以下の場合、このリーダーはパワー+1000
# ===========================================================================

def test_st09_001_buff_when_life_le_2():
    """ST09-001: 相手ターン・ドン1・自分ライフ2枚以下で、このリーダーがパワー+1000。"""
    gm, p1, p2, L = build("ST09-001")
    set_life(p1, 2)
    d = p1.don_active.pop()
    p1.don_attached_cards.append(d)
    L.attached_don = 1
    base = leader_power(p1, my_turn=False)
    ab = get_ability(L.master, "OPPONENT_TURN")
    gm.resolve_ability(p1, ab, L)
    auto_resolve(gm, p1)
    assert leader_power(p1, my_turn=False) == base + 1000


def test_st09_001_no_buff_when_life_ge_3():
    """ST09-001: 自分ライフ3枚以上では条件未達で+1000は乗らない。"""
    gm, p1, p2, L = build("ST09-001")
    set_life(p1, 3)
    d = p1.don_active.pop()
    p1.don_attached_cards.append(d)
    L.attached_don = 1
    base = leader_power(p1, my_turn=False)
    ab = get_ability(L.master, "OPPONENT_TURN")
    gm.resolve_ability(p1, ab, L)
    auto_resolve(gm, p1)
    assert leader_power(p1, my_turn=False) == base   # 変化なし


# ===========================================================================
# ST10-001 トラファルガー・ロー ✅
#   【起動メイン】【ターン1回】ドン-3：相手のパワー3000以下キャラ1枚までを持ち主の
#   デッキの下へ、自分の手札からコスト4以下キャラ1枚までを登場
# ===========================================================================

def test_st10_001_deckbottom_opp_and_play_self():
    """ST10-001 起動メイン: ドン3戻し→相手パワー3000以下キャラを相手デッキ下へ、自分のコスト4以下キャラを登場。"""
    gm, p1, p2, L = build("ST10-001")
    clear_field(p2)
    victim = add_char(p2, name="弱キャラ", cost=2, power=3000)   # power_max=3000 ちょうど含む
    p1.hand = []
    play_me = add_char(p1, name="登場キャラ", cost=4, power=5000)
    p1.field.remove(play_me)                          # 場でなく手札に置く
    p1.hand.append(play_me)
    p1_field_before = len(p1.field)
    deck_before = len(p2.deck)
    don_before = don_total(p1)
    ab = get_ability(L.master, "ACTIVATE_MAIN")
    gm.resolve_ability(p1, ab, L)
    _drive(gm, p1, prefer_uuids=[victim.uuid, play_me.uuid])
    assert victim not in p2.field
    assert p2.deck[-1] is victim                      # 持ち主(相手)のデッキの下へ
    assert len(p2.deck) == deck_before + 1
    assert play_me in p1.field                        # 自分の手札キャラが登場
    assert len(p1.field) == p1_field_before + 1
    assert don_total(p1) == don_before - 3


def test_st10_001_no_valid_bounce_target_still_plays():
    """ST10-001 起動メイン: 相手にパワー3001以上キャラしかいなくてもデッキ下は0枚、登場は実行可。"""
    gm, p1, p2, L = build("ST10-001")
    clear_field(p2)
    tough = add_char(p2, name="高パワー", cost=2, power=4000)   # 3000超で対象外
    p1.hand = []
    play_me = add_char(p1, name="登場キャラ", cost=4, power=5000)
    p1.field.remove(play_me)
    p1.hand.append(play_me)
    field_before = len(p1.field)
    ab = get_ability(L.master, "ACTIVATE_MAIN")
    gm.resolve_ability(p1, ab, L)
    _drive(gm, p1, prefer_uuids=[play_me.uuid])
    assert tough in p2.field                          # デッキ下は0枚
    assert play_me in p1.field                        # 登場のみ実行
    assert len(p1.field) == field_before + 1


# ===========================================================================
# ST10-002 モンキー・D・ルフィ 🐛
#   【起動メイン】【ターン1回】自分の場のドンが0枚、または8枚以上ある場合、
#   ドンデッキからドン1枚までをアクティブで追加
#   実装は条件が DON==0 のみで「または8枚以上」のOR節が欠落。
# ===========================================================================

def test_st10_002_ramp_when_don_is_zero():
    """ST10-002 起動メイン: 場のドンが0枚なら条件成立でドン1枚アクティブ追加。"""
    gm, p1, p2, L = build("ST10-002")
    _set_field_don(p1, n_active=0)
    field_before = don_total(p1)
    ab = get_ability(L.master, "ACTIVATE_MAIN")
    gm.resolve_ability(p1, ab, L)
    auto_resolve(gm, p1)
    assert don_total(p1) == field_before + 1


def test_st10_002_no_ramp_when_don_in_middle():
    """ST10-002 起動メイン: 場のドンが1〜7枚(ここでは3枚)では条件未達で追加なし。"""
    gm, p1, p2, L = build("ST10-002")
    _set_field_don(p1, n_active=3)
    field_before = don_total(p1)
    ab = get_ability(L.master, "ACTIVATE_MAIN")
    gm.resolve_ability(p1, ab, L)
    auto_resolve(gm, p1)
    assert don_total(p1) == field_before             # 変化なし


def test_st10_002_ramp_when_don_ge_8():
    """ST10-002 起動メイン: 場のドンが8枚以上でもOR条件成立でドン1枚アクティブ追加されるべき。"""
    gm, p1, p2, L = build("ST10-002")
    _set_field_don(p1, n_active=8)
    field_before = don_total(p1)
    ab = get_ability(L.master, "ACTIVATE_MAIN")
    gm.resolve_ability(p1, ab, L)
    auto_resolve(gm, p1)
    assert don_total(p1) == field_before + 1


# ===========================================================================
# ST10-003 ユースタス・キッド ✅
#   [0]【自分のターン中】ライフ4枚以上の場合、このリーダーはパワー-1000
#   [1]【アタック時】ドン-1：このリーダーはこのターン中パワー+2000
# ===========================================================================

def test_st10_003_demerit_minus_1000_when_life_ge_4():
    """ST10-003 能力0: 自分ターン・ライフ4枚以上で、このリーダーはパワー-1000（デメリット）。"""
    gm, p1, p2, L = build("ST10-003")
    set_life(p1, 5)
    ab = get_ability(L.master, "YOUR_TURN")
    gm.resolve_ability(p1, ab, L)
    auto_resolve(gm, p1)
    assert leader_power(p1) == 4000                  # 5000 - 1000


def test_st10_003_no_demerit_when_life_le_3():
    """ST10-003 能力0: 自分ライフ3枚以下では -1000 は乗らない。"""
    gm, p1, p2, L = build("ST10-003")
    set_life(p1, 3)
    ab = get_ability(L.master, "YOUR_TURN")
    gm.resolve_ability(p1, ab, L)
    auto_resolve(gm, p1)
    assert leader_power(p1) == 5000                  # 変化なし


def test_st10_003_attack_buff_plus_2000():
    """ST10-003 能力1: アタック時にドン1戻し→このリーダーはこのターン中パワー+2000。"""
    gm, p1, p2, L = build("ST10-003")
    don_before = don_total(p1)
    ab = get_ability(L.master, "ON_ATTACK")
    gm.resolve_ability(p1, ab, L)
    auto_resolve(gm, p1)
    assert leader_power(p1) == 7000                  # 5000 + 2000
    assert don_total(p1) == don_before - 1           # ドン-1コスト
