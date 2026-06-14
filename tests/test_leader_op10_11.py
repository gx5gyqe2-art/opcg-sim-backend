"""リーダー効果テスト — OP10 / OP11（12枚）。

仕様書: docs/leader_specs/OP10-11.md / ガイド: docs/leader_specs/_TEST_GUIDE.md
テキスト準拠の「正しい挙動」をアサートする。現実装にバグがある（🐛）ケースは
@pytest.mark.xfail(strict=True) を付け、修正されると xpass→strict で赤になり
マーカー除去を促す。

実行:
    OPCG_LOG_SILENT=1 python -m pytest tests/test_leader_op10_11.py -q -s -p no:cacheprovider
"""
import pytest

from leader_test_helpers import (
    build, get_ability, auto_resolve,
    add_char, make_char, clear_field, zone_counts, leader_power,
)


# ---------------------------------------------------------------------------
# 小道具
# ---------------------------------------------------------------------------

def _rest_two_don(p):
    """アクティブドン2枚をレストにする。"""
    for _ in range(2):
        d = p.don_active.pop()
        d.is_rest = True
        p.don_rested.append(d)


def _rest_one_don(p):
    d = p.don_active.pop()
    d.is_rest = True
    p.don_rested.append(d)


def _fill_hand_to(p, n):
    while len(p.hand) < n:
        p.hand.append(p.deck.pop())
    p.hand = p.hand[:n]


# ===========================================================================
# OP10-001 スモーカー
#   能力1【起動メイン】: 自分のパワー7000以上のキャラがいる場合ドン2枚アクティブ。
#   🐛 条件が power_max=7000（GE→LE反転）。条件成立/不成立の両盤面で反転を捉える。
# ===========================================================================

def test_op10_001_active_don_condition_met_pw8000():
    """OP10-001 能力1: パワー8000のキャラがいれば（7000以上成立）ドン2枚アクティブになるべき。"""
    gm, p1, p2, L = build("OP10-001")
    clear_field(p1)
    add_char(p1, power=8000)
    _rest_two_don(p1)
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    auto_resolve(gm, p1)
    # 条件成立 → レストドン2枚がアクティブに復帰
    assert len(p1.don_active) == 10 and len(p1.don_rested) == 0


def test_op10_001_active_don_condition_unmet_pw3000():
    """OP10-001 能力1: パワー3000のキャラのみ（7000以上を満たさない）なら発動しないべき。"""
    gm, p1, p2, L = build("OP10-001")
    clear_field(p1)
    add_char(p1, power=3000)
    _rest_two_don(p1)
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    auto_resolve(gm, p1)
    # 条件不成立 → レストドンはそのまま
    assert len(p1.don_rested) == 2 and len(p1.don_active) == 8


def test_op10_001_passive_buff_navy_punkhazard():
    """OP10-001 能力0【相手のターン中】: 自分の《海軍》《パンクハザード》キャラ全部にパワー+1000。"""
    gm, p1, p2, L = build("OP10-001")
    clear_field(p1)
    navy = add_char(p1, name="海軍兵", power=4000, traits=["海軍"])
    gm.resolve_ability(p1, get_ability(L.master, "OPPONENT_TURN"), L)
    auto_resolve(gm, p1)
    assert navy.get_power(True) == 5000


# ===========================================================================
# OP10-002 シーザー・クラウン
#   【ドン!!×2】【アタック時】PHコスト2以上1枚を手札に戻す：相手pw4000以下1枚までKO。
#   ✅ テキストと整合（ドン2枚付与＝attached_don で条件成立）。
# ===========================================================================

def test_op10_002_bounce_cost_then_ko():
    """OP10-002【アタック時】: ドン2付与下、自PHを手札に戻し相手pw4000以下キャラをKO。"""
    gm, p1, p2, L = build("OP10-002")
    clear_field(p1)
    ph = add_char(p1, name="PH", cost=3, traits=["パンクハザード"])
    p2.field = []
    victim = make_char(p2, name="V", power=4000)
    p2.field.append(victim)
    L.attached_don = 2  # 【ドン!!×2】
    gm.resolve_ability(p1, get_ability(L.master, "ON_ATTACK"), L)
    auto_resolve(gm, p1)
    assert ph in p1.hand          # バウンス成立
    assert victim not in p2.field  # KO 成立


def test_op10_002_no_fire_without_two_don():
    """OP10-002: ドン2付与が無い（【ドン!!×2】未充足）なら発動しない。"""
    gm, p1, p2, L = build("OP10-002")
    clear_field(p1)
    ph = add_char(p1, name="PH", cost=3, traits=["パンクハザード"])
    p2.field = []
    victim = make_char(p2, name="V", power=4000)
    p2.field.append(victim)
    L.attached_don = 0
    gm.resolve_ability(p1, get_ability(L.master, "ON_ATTACK"), L)
    auto_resolve(gm, p1)
    assert ph in p1.field and victim in p2.field  # 何も起きない


# ===========================================================================
# OP10-003 シュガー
#   能力0【ターン終了時】: パワー6000以上のDQ海賊団キャラがいる場合ドン1枚アクティブ。
#   🐛 条件 power_max=6000（GE→LE反転）。成立/不成立の両盤面で反転を捉える。
# ===========================================================================

def test_op10_003_active_don_condition_met_pw7000():
    """OP10-003 能力0: pw7000のDQ海賊団キャラ（6000以上成立）でドン1枚アクティブになるべき。"""
    gm, p1, p2, L = build("OP10-003")
    clear_field(p1)
    add_char(p1, power=7000, traits=["ドンキホーテ海賊団"])
    _rest_one_don(p1)
    gm.resolve_ability(p1, get_ability(L.master, "TURN_END"), L)
    auto_resolve(gm, p1)
    assert len(p1.don_rested) == 0  # 成立 → アクティブ復帰


def test_op10_003_active_don_condition_unmet_pw5000():
    """OP10-003 能力0: pw5000のDQ海賊団キャラのみ（6000以上を満たさない）なら発動しないべき。"""
    gm, p1, p2, L = build("OP10-003")
    clear_field(p1)
    add_char(p1, power=5000, traits=["ドンキホーテ海賊団"])
    _rest_one_don(p1)
    gm.resolve_ability(p1, get_ability(L.master, "TURN_END"), L)
    auto_resolve(gm, p1)
    assert len(p1.don_rested) == 1  # 不成立 → そのまま


# ===========================================================================
# OP10-022 トラファルガー・ロー
#   【起動メイン】自分のキャラのコスト合計が5以上の場合、キャラ1枚を手札へ：…
#   🐛 「コスト合計5以上」条件が AST から欠落。成立/不成立の両盤面で捉える。
# ===========================================================================

def test_op10_022_cost_sum_ge5_fires():
    """OP10-022: コスト合計6（5以上成立）でキャラ1枚を持ち主の手札に戻せる。"""
    gm, p1, p2, L = build("OP10-022")
    clear_field(p1)
    ch = add_char(p1, name="C", cost=6, traits=["x"])
    L.attached_don = 1  # 【ドン!!×1】
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    auto_resolve(gm, p1)
    assert ch in p1.hand and ch not in p1.field  # バウンス成立


def test_op10_022_cost_sum_lt5_does_not_fire():
    """OP10-022: コスト合計4（5未満）では発動せず、キャラはバウンスされないべき。"""
    gm, p1, p2, L = build("OP10-022")
    clear_field(p1)
    ch = add_char(p1, name="C", cost=4, traits=["x"])
    L.attached_don = 1
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    auto_resolve(gm, p1)
    assert ch in p1.field and ch not in p1.hand  # 不成立 → バウンスされない


# ===========================================================================
# OP10-042 ウソップ
#   能力0【PASSIVE】: 自《ドレスローザ》コスト2以上キャラ全部にコスト+1。
#   能力1【相手のターン中】: 手札5枚以下なら1枚ドロー。✅
# ===========================================================================

def test_op10_042_passive_cost_plus_one():
    """OP10-042 能力0: 自《ドレスローザ》コスト2以上キャラのコストが+1される。"""
    gm, p1, p2, L = build("OP10-042")
    clear_field(p1)
    c = add_char(p1, name="DR", cost=3, traits=["ドレスローザ"])
    gm._apply_passive_effects(p1)
    assert c.current_cost == 4


def test_op10_042_draw_when_hand_le5():
    """OP10-042 能力1: 手札5枚以下なら相手ターン中に1枚ドロー。"""
    gm, p1, p2, L = build("OP10-042")
    _fill_hand_to(p1, 5)
    deck_before = len(p1.deck)
    gm.turn_player = p2   # 【相手のターン中】＝相手のターン（CONTEXT 条件を満たす）
    gm.resolve_ability(p1, get_ability(L.master, "ON_KO"), L)
    auto_resolve(gm, p1)
    assert len(p1.hand) == 6 and len(p1.deck) == deck_before - 1


def test_op10_042_no_draw_when_hand_gt5():
    """OP10-042 能力1: 手札6枚（5枚超）ならドローしない。"""
    gm, p1, p2, L = build("OP10-042")
    _fill_hand_to(p1, 6)
    gm.turn_player = p2   # 【相手のターン中】＝相手のターン
    gm.resolve_ability(p1, get_ability(L.master, "ON_KO"), L)
    auto_resolve(gm, p1)
    assert len(p1.hand) == 6


# ===========================================================================
# OP10-099 ユースタス・キッド
#   【ターン終了時】ライフ表向き：自コスト3〜8の《超新星》1枚までアクティブ＋ブロッカー。
#   🐛 対象 cost_max=3 のみ（下限3欠落・上限が3に誤設定）。
#   コスト範囲: 成立(5)/不成立(2,9) の両方をアサート。観測は付与された【ブロッカー】。
# ===========================================================================

def _op10_099_run(cost):
    gm, p1, p2, L = build("OP10-099")
    clear_field(p1)
    c = add_char(p1, name="SN", cost=cost, traits=["超新星"], rest=True)
    gm.resolve_ability(p1, get_ability(L.master, "TURN_END"), L)
    auto_resolve(gm, p1)
    return c


def test_op10_099_cost5_in_range_grants_blocker():
    """OP10-099: コスト5の《超新星》（範囲3〜8内）は対象になりブロッカーを得る。"""
    c = _op10_099_run(5)
    assert c.has_keyword("ブロッカー")


def test_op10_099_cost2_below_range_no_effect():
    """OP10-099: コスト2の《超新星》（下限3未満）は対象外でブロッカーを得ないべき。"""
    c = _op10_099_run(2)
    assert not c.has_keyword("ブロッカー")


def test_op10_099_cost9_above_range_no_effect():
    """OP10-099: コスト9の《超新星》（上限8超）は対象外でブロッカーを得ないべき。"""
    c = _op10_099_run(9)
    assert not c.has_keyword("ブロッカー")


# ===========================================================================
# OP11-001 コビー
#   能力0【PASSIVE】: 自《SWORD》キャラに速攻系（登場ターンにキャラへアタック可）付与。✅
#   能力1【ターン1回】: 元々pw7000以下の《海軍》が相手効果で離れる→置換でトラッシュ3枚デッキ下。✅
# ===========================================================================

def test_op11_001_sword_gains_rush():
    """OP11-001 能力0: 自《SWORD》キャラに速攻（登場ターンにキャラへアタック）が付与される。"""
    gm, p1, p2, L = build("OP11-001")
    clear_field(p1)
    c = add_char(p1, name="S", traits=["SWORD"])
    gm._apply_passive_effects(p1)
    assert "速攻" in (c.current_keywords | c.timed_keywords)


def _opponent_removes(gm, opponent, victim, action_type="KO"):
    """相手(opponent)の効果で victim を場から除去する（KO/BOUNCE 等）。

    汎用盤面では「相手効果で場を離れる」契機を直接イベント化しづらいため、
    除去アクションを apply_action_to_engine で相手側プレイヤーから適用する。
    OPPONENT 側からの _LEAVE_ACTIONS 適用は _active_replacement/_active_protection を
    走査するので、置換効果（REPLACE_EFFECT status=LEAVE）の発火経路を再現できる。
    """
    from opcg_sim.src.models.effect_types import GameAction, TargetQuery
    from opcg_sim.src.models.enums import ActionType, Player as _P
    act = GameAction(type=ActionType[action_type], target=TargetQuery(player=_P.OPPONENT),
                     raw_text=f"相手のキャラを{action_type}")
    return gm.apply_action_to_engine(opponent, act, [victim], 0)


def _setup_navy_with_trash(power):
    """OP11-001 リーダー下で、p1 場に《海軍》(指定パワー)＋トラッシュ3枚を用意する。"""
    gm, p1, p2, L = build("OP11-001")
    clear_field(p1); clear_field(p2)
    navy = add_char(p1, name="海軍兵", power=power, traits=["海軍"])
    p1.trash[:] = [make_char(p1, name=f"trash{i}", power=1000) for i in range(3)]
    return gm, p1, p2, L, navy


def test_op11_001_replace_leave_navy_pw7000_or_less():
    """OP11-001 能力1: 元々pw7000以下の《海軍》が相手効果で場を離れる→置換成立。

    本来の除去（KO）は行われず、代わりにトラッシュ3枚がデッキの下へ置かれる。
    """
    gm, p1, p2, L, navy = _setup_navy_with_trash(power=6000)
    deck_before = len(p1.deck)
    assert _opponent_removes(gm, p2, navy, "KO") is True
    # 置換成立: 海軍は KO されず場に残る
    assert navy in p1.field and navy not in p1.trash
    # 代わりにトラッシュ3枚 → デッキ下（トラッシュ -3 / デッキ +3）
    assert len(p1.trash) == 0
    assert len(p1.deck) == deck_before + 3


def test_op11_001_replace_leave_navy_via_bounce():
    """OP11-001 能力1: 除去が BOUNCE（手札に戻す）でも置換が成立し、場に残る。"""
    gm, p1, p2, L, navy = _setup_navy_with_trash(power=7000)  # 7000ちょうどは対象
    deck_before = len(p1.deck)
    assert _opponent_removes(gm, p2, navy, "BOUNCE") is True
    assert navy in p1.field and navy not in p1.hand
    assert len(p1.deck) == deck_before + 3


def test_op11_001_no_replace_when_power_above_7000():
    """OP11-001 能力1: 元々pw8000（7000超）の《海軍》は対象外 → 置換せず通常どおりKO。"""
    gm, p1, p2, L, navy = _setup_navy_with_trash(power=8000)
    trash_before = len(p1.trash); deck_before = len(p1.deck)
    assert _opponent_removes(gm, p2, navy, "KO") is True
    # 置換不成立: 海軍は KO されトラッシュへ。トラッシュ→デッキの移動は起きない。
    assert navy not in p1.field and navy in p1.trash
    assert len(p1.deck) == deck_before  # デッキ下送りは発生しない


# ===========================================================================
# OP11-101 カポネ・ベッジ（キャラ）
#   【ターン1回】「カポネ・ベッジ」以外の自分の《超新星》キャラが相手の効果で場を
#   離れる場合、代わりに自分のライフの上に裏向きで加えることができる。
#   ✅ REPLACE_EFFECT(LEAVE): 離れるカード自身を持ち主のライフ(上/裏向き)へ移す。
# ===========================================================================

def _build_with_bedge_and_supernova():
    """任意リーダー下で、p1 場に OP11-101（カポネ・ベッジ）＋《超新星》キャラを置く。"""
    from opcg_sim.src.utils.loader import CardLoader
    import os
    db = CardLoader(os.path.join(os.path.dirname(__file__), "..", "opcg_sim", "data", "opcg_cards.json"))
    db.load()
    gm, p1, p2, L = build("OP01-001")
    clear_field(p1); clear_field(p2)
    bedge = make_char(p1, name="カポネ・ベッジ", cost=4)
    bedge.master = db.get_card("OP11-101")
    p1.field.append(bedge)
    star = add_char(p1, name="超新星キャラ", power=5000, traits=["超新星"])
    return gm, p1, p2, bedge, star


def test_op11_101_replace_leave_supernova_to_life():
    """OP11-101: 自分の《超新星》が相手効果で場を離れる→代わりに自ライフ上(裏向き)へ。"""
    gm, p1, p2, bedge, star = _build_with_bedge_and_supernova()
    life_before = len(p1.life)
    assert _opponent_removes(gm, p2, star, "KO") is True
    # 置換成立: 超新星キャラは KO されず、自分のライフへ裏向きで加わる。
    assert star not in p1.field and star not in p1.trash
    assert star in p1.life and len(p1.life) == life_before + 1
    assert star.is_face_up is False
    # 保護者（カポネ・ベッジ）は場に残る。
    assert bedge in p1.field


# ===========================================================================
# OP11-021 ジンベエ
#   【ターン終了時】手札6枚以下なら《魚人族》or《人魚族》1枚まで＋ドン1枚までアクティブ。
#   🐛 キャラアクティブ部分が欠落しドンのアクティブのみ。
# ===========================================================================

def test_op11_021_don_active_when_hand_le6():
    """OP11-021: 手札6枚以下ならドン1枚がアクティブになる（実装済みのドン部分）。"""
    gm, p1, p2, L = build("OP11-021")
    _fill_hand_to(p1, 5)
    clear_field(p1)
    _rest_one_don(p1)
    gm.resolve_ability(p1, get_ability(L.master, "TURN_END"), L)
    auto_resolve(gm, p1)
    assert len(p1.don_rested) == 0  # ドンはアクティブ化


def test_op11_021_no_fire_when_hand_gt6():
    """OP11-021: 手札7枚（6枚超）なら発動せずドンもレストのまま。"""
    gm, p1, p2, L = build("OP11-021")
    _fill_hand_to(p1, 7)
    clear_field(p1)
    _rest_one_don(p1)
    gm.resolve_ability(p1, get_ability(L.master, "TURN_END"), L)
    auto_resolve(gm, p1)
    assert len(p1.don_rested) == 1  # 不発


def test_op11_021_fishman_char_should_be_activated():
    """OP11-021: 手札6枚以下なら《魚人族》レストキャラもアクティブになるべき。"""
    gm, p1, p2, L = build("OP11-021")
    _fill_hand_to(p1, 5)
    clear_field(p1)
    c = add_char(p1, name="Fish", traits=["魚人族"], rest=True)
    _rest_one_don(p1)
    gm.resolve_ability(p1, get_ability(L.master, "TURN_END"), L)
    auto_resolve(gm, p1)
    assert c.is_rest is False  # キャラもアクティブ化されるべき


# ===========================================================================
# OP11-022 しらほし
#   能力1【起動メイン】ドン1レスト＋ライフ表向き：手札から(場ドン以下コスト)《海王類》か
#   「メガロ」1枚までを登場。
#   🐛 「《海王類》か「メガロ」」の OR が trait∧name の AND になり対象が常に空。
#   成立すべき盤面（海王類）/ 不成立すべき盤面（無関係キャラ）の両方をアサート。
# ===========================================================================

def test_op11_022_kaiou_char_should_be_playable():
    """OP11-022: 手札のコスト≤場ドンの《海王類》キャラは登場できるべき（OR の片方）。"""
    gm, p1, p2, L = build("OP11-022")
    seaking = make_char(p1, name="シーキング", cost=3, traits=["海王類"])
    p1.hand.append(seaking)
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    auto_resolve(gm, p1)
    assert seaking in p1.field  # 海王類は登場対象であるべき


def test_op11_022_unrelated_card_not_playable():
    """OP11-022: 《海王類》でも「メガロ」でもないキャラは登場対象外（不成立盤面）。"""
    gm, p1, p2, L = build("OP11-022")
    other = make_char(p1, name="ただのキャラ", cost=3, traits=["その他"])
    p1.hand.append(other)
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    auto_resolve(gm, p1)
    assert other not in p1.field  # 対象外なので登場しない


def test_op11_022_leader_cannot_attack():
    """OP11-022 能力0【PASSIVE】: このリーダーはアタックできない（ATTACK_DISABLE 付与）。"""
    gm, p1, p2, L = build("OP11-022")
    gm.resolve_ability(p1, get_ability(L.master, "PASSIVE"), L)
    auto_resolve(gm, p1)
    assert "ATTACK_DISABLE" in (L.flags | L.timed_flags)


# ===========================================================================
# OP11-040 モンキー・D・ルフィ
#   【ターン開始時】場ドン8枚以上なら上5枚を見て《麦わらの一味》1枚まで手札→残りを並べ替え。
#   🐛 「ドン8枚以上」条件が LOOK のみに掛かり MOVE_CARD/DECK_BOTTOM が条件外（スコープ欠落）。
# ===========================================================================

def test_op11_040_don8_looks_and_arranges():
    """OP11-040: 場ドン8枚以上なら上5枚を見て並べ替え（ARRANGE_DECK 対話に入る）。"""
    gm, p1, p2, L = build("OP11-040")
    while len(p1.don_active) < 8 and p1.don_rested:
        p1.don_active.append(p1.don_rested.pop())
    while len(p1.don_active) > 8:
        p1.don_active.pop()
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    assert gm.active_interaction is not None
    assert gm.active_interaction.get("action_type") == "ARRANGE_DECK"
    auto_resolve(gm, p1)


def test_op11_040_don7_no_look():
    """OP11-040: 場ドン7枚（8枚未満）なら上5枚を見ない＝手札・デッキ枚数は不変。"""
    gm, p1, p2, L = build("OP11-040")
    while len(p1.don_active) > 7:
        p1.don_active.pop()
    zb = zone_counts(p1)
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    auto_resolve(gm, p1)
    za = zone_counts(p1)
    # LOOK が走らないので手札は増えない（条件外に出た MOVE_CARD/DECK_BOTTOM は temp 空で no-op）
    assert za["hand"] == zb["hand"] and za["deck"] == zb["deck"]


# ===========================================================================
# OP11-041 ナミ
#   能力0【自分のターン中/ライフが離れた時】手札7枚以下なら1枚ドロー。
#   能力1【ドン!!×1/相手のアタック時】手札1枚捨て：このリーダーにこのターン+2000。✅
# ===========================================================================

def test_op11_041_draw_on_life_decrease_hand_le7():
    """OP11-041 能力0: 自ターン中ライフが離れ、手札7枚以下なら1枚ドロー。"""
    gm, p1, p2, L = build("OP11-041")
    _fill_hand_to(p1, 5)
    deck_before = len(p1.deck)
    gm.resolve_ability(p1, get_ability(L.master, "ON_LIFE_DECREASE"), L)
    auto_resolve(gm, p1)
    assert len(p1.hand) == 6 and len(p1.deck) == deck_before - 1


def test_op11_041_no_draw_when_hand_gt7():
    """OP11-041 能力0: 手札8枚（7枚超）ならドローしない。"""
    gm, p1, p2, L = build("OP11-041")
    _fill_hand_to(p1, 8)
    gm.resolve_ability(p1, get_ability(L.master, "ON_LIFE_DECREASE"), L)
    auto_resolve(gm, p1)
    assert len(p1.hand) == 8


def test_op11_041_discard_buffs_leader_this_turn():
    """OP11-041 能力1: ドン1付与下、相手アタック時に手札1枚捨ててリーダー+2000。"""
    gm, p1, p2, L = build("OP11-041")
    L.attached_don = 1  # 【ドン!!×1】
    base = leader_power(p1)
    gm.resolve_ability(p1, get_ability(L.master, "ON_OPP_ATTACK"), L)
    auto_resolve(gm, p1)
    assert leader_power(p1) == base + 2000


# ===========================================================================
# OP11-062 シャーロット・カタクリ
#   【アタック時】/【相手のアタック時】ドン!!-1：相手デッキ上1枚を見る→リーダーこのバトル+1000。✅
# ===========================================================================

def test_op11_062_attack_returns_don_and_buffs():
    """OP11-062 能力0【アタック時】: ドン-1して相手デッキ上1枚閲覧、リーダー+1000。"""
    gm, p1, p2, L = build("OP11-062")
    base = leader_power(p1)
    don_before = len(p1.don_active)
    gm.resolve_ability(p1, get_ability(L.master, "ON_ATTACK"), L)
    auto_resolve(gm, p1)
    assert leader_power(p1) == base + 1000
    assert len(p1.don_active) == don_before - 1


def test_op11_062_opp_attack_returns_don_and_buffs():
    """OP11-062 能力1【相手のアタック時】: 同じくドン-1してリーダー+1000。"""
    gm, p1, p2, L = build("OP11-062")
    base = leader_power(p1)
    don_before = len(p1.don_active)
    gm.resolve_ability(p1, get_ability(L.master, "ON_OPP_ATTACK"), L)
    auto_resolve(gm, p1)
    assert leader_power(p1) == base + 1000
    assert len(p1.don_active) == don_before - 1


def test_op11_062_no_fire_without_don():
    """OP11-062: コストのドン!!-1 が払えない（ドン0）なら発動しない。"""
    gm, p1, p2, L = build("OP11-062")
    p1.don_active = []
    p1.don_rested = []
    base = leader_power(p1)
    gm.resolve_ability(p1, get_ability(L.master, "ON_ATTACK"), L)
    auto_resolve(gm, p1)
    assert leader_power(p1) == base  # 不発


# ===========================================================================
# OP11-102 ケイミー
#   【相手のイベント/トリガー発動時】相手ライフ2枚以上なら
#   「お互いのライフの上から1枚をトラッシュに置く」= 両者のライフが各1枚減る。
#   （旧: Player.ALL の単一選択で相手側のみ解決されていた＝SPEC §6.1 の既知制約）
# ===========================================================================

def test_op11_102_mutual_life_to_trash_both_sides():
    """OP11-102: 「お互いのライフの上から1枚をトラッシュに置く」→ 両者のライフが各1枚減る。"""
    from leader_test_helpers import set_life, db
    from opcg_sim.src.models.models import CardInstance
    gm, p1, p2, L = build("OP10-001")  # 任意リーダー（OP11-102 はキャラ）
    src = CardInstance(db().get_card("OP11-102"), p1.name)
    p1.field.append(src)
    set_life(p1, 3)
    set_life(p2, 3)
    t1, t2 = len(p1.trash), len(p2.trash)
    gm.resolve_ability(p1, get_ability(src.master, "YOUR_TURN"), src)
    auto_resolve(gm, p1)
    # 両側がそれぞれ 1 枚ずつライフを失う（片側のみではない）
    assert len(p1.life) == 2, f"自分ライフが減っていない: {len(p1.life)}"
    assert len(p2.life) == 2, f"相手ライフが減っていない: {len(p2.life)}"
    assert len(p1.trash) == t1 + 1
    assert len(p2.trash) == t2 + 1
