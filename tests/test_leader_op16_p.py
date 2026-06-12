"""リーダーカード効果テスト — OP16 / プロモ P（docs/leader_specs/OP16-P.md 準拠）。

対象11枚: OP16-001, OP16-022, OP16-041, OP16-060, OP16-079, OP16-080,
          P-011, P-047, P-076, P-086, P-117

方針（_TEST_GUIDE.md）:
  - 常にテキスト準拠の「正しい挙動」をアサートする。
  - ✅ → 通常テスト / 🐛 → xfail(strict=True) / ⚠️ → 通常、失敗/不安定なら xfail(strict=False)。
  - 条件分岐は成立/不成立を別ケースに。

実行:
  OPCG_LOG_SILENT=1 python -m pytest tests/test_leader_op16_p.py -q -s -p no:cacheprovider
"""
import pytest

from leader_test_helpers import (
    build, get_ability, abilities_of, auto_resolve,
    select_uuids, confirm,
    add_char, make_char, clear_field, set_life,
)


# ---------------------------------------------------------------------------
# 共通ヘルパ
# ---------------------------------------------------------------------------

def _drive(gm, p1, *, resource=None, targets=None, limit=10):
    """active_interaction を逐次解決する。

    resource: SELECT_RESOURCE で選ぶ uuid のリスト（None なら候補先頭から min 枚）。
    targets : SELECT_TARGET で選ぶ uuid のリスト（候補に含まれるものだけ採用）。
    """
    targets = list(targets or [])
    steps = 0
    while gm.active_interaction and steps < limit:
        ia = gm.active_interaction
        at = ia["action_type"]
        cands = [c.uuid for c in ia.get("candidates", [])]
        if at == "SELECT_RESOURCE":
            cons = ia.get("constraints") or {}
            mn = cons.get("min", 1) or 1
            sel = resource if resource is not None else cands[:mn]
            gm.resolve_interaction(p1, {"selected_uuids": sel})
        elif at == "SELECT_TARGET":
            pick = [u for u in targets if u in cands]
            gm.resolve_interaction(p1, {"selected_uuids": pick})
        else:
            gm.resolve_interaction(p1, {"accepted": True, "selected_uuids": cands[:1]})
        steps += 1


def _has_keyword(inst, kw):
    return kw in (inst.current_keywords | inst.timed_keywords)


# ===========================================================================
# OP16-001 ポートガス・D・エース
#   【起動メイン】【ターン1回】自分のパワー8000以上の「モンキー・D・ルフィ」か
#   《白ひげ海賊団》キャラ1枚までに、このターン中【速攻】を付与。
#   🐛 「パワー8000以上」が power_max=8000（=8000以下）に反転（matcher.py:209/216）。
#   ※ 現実装は names と traits を AND 適用するため、対象は name=「モンキー・D・ルフィ」
#      かつ trait《白ひげ海賊団》のキャラに限られる。本テストはパワー条件の反転に
#      焦点を当てるため、その両方を満たすキャラで検証する。
# ===========================================================================

def test_op16_001_grant_haste_power_at_threshold():
    """OP16-001: パワー8000ちょうどの対象キャラは【速攻】を得る（境界・成立）。"""
    gm, p1, p2, L = build("OP16-001")
    clear_field(p1)
    c = add_char(p1, name="モンキー・D・ルフィ", power=8000, traits=["白ひげ海賊団"])
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    _drive(gm, p1, targets=[c.uuid])
    assert _has_keyword(c, "速攻")


def test_op16_001_grant_haste_power_above_threshold():
    """OP16-001: パワー9000（8000以上）のキャラは【速攻】を得るべき（テキスト準拠）。

    現実装は power_max=8000 と誤パースするため 8000 超は対象外となり付与されない → xfail。
    """
    gm, p1, p2, L = build("OP16-001")
    clear_field(p1)
    c = add_char(p1, name="モンキー・D・ルフィ", power=9000, traits=["白ひげ海賊団"])
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    _drive(gm, p1, targets=[c.uuid])
    assert _has_keyword(c, "速攻")


def test_op16_001_grant_haste_power_below_threshold_excluded():
    """OP16-001: パワー7000（8000未満）のキャラは対象外で【速攻】を得ないべき。

    現実装は power_max=8000（8000以下を対象）と誤パースし、7000 にも付与してしまう → xfail。
    """
    gm, p1, p2, L = build("OP16-001")
    clear_field(p1)
    c = add_char(p1, name="モンキー・D・ルフィ", power=7000, traits=["白ひげ海賊団"])
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    _drive(gm, p1, targets=[c.uuid])
    assert not _has_keyword(c, "速攻")


# ===========================================================================
# OP16-022 モンキー・D・ルフィ
#   【起動メイン】【ターン1回】自分のキャラが《インペルダウン》のみの場合、
#   自分のドン!!2枚までをアクティブにする。  ⚠️ 整合（要確認）。
# ===========================================================================

def _rest_don(player, n):
    for _ in range(n):
        d = player.don_active.pop()
        d.is_rest = True
        player.don_rested.append(d)


def test_op16_022_active_don_when_all_impeldown():
    """OP16-022: 自分キャラが全て《インペルダウン》なら、レストのドン2枚をアクティブ化。"""
    gm, p1, p2, L = build("OP16-022")
    clear_field(p1)
    add_char(p1, name="囚人", traits=["インペルダウン"])
    _rest_don(p1, 2)
    before_active = len(p1.don_active)
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    auto_resolve(gm, p1)
    assert len(p1.don_active) == before_active + 2
    assert len(p1.don_rested) == 0


def test_op16_022_no_active_when_non_impeldown_present():
    """OP16-022: 《インペルダウン》以外のキャラが混在する場合は不発（条件未達）。"""
    gm, p1, p2, L = build("OP16-022")
    clear_field(p1)
    add_char(p1, name="囚人", traits=["インペルダウン"])
    add_char(p1, name="仲間", traits=["麦わらの一味"])
    _rest_don(p1, 2)
    before_rested = len(p1.don_rested)
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    auto_resolve(gm, p1)
    assert len(p1.don_rested) == before_rested  # アクティブ化されない


# ===========================================================================
# OP16-041 バギー
#   【ドン!!×1】【ターン1回】自分の《インペルダウン》キャラが場を離れた時、
#   手札の「インペルダウンの囚人」1枚までを登場させる。
#   🐛 誘発「場を離れた時」が欠落し ACTIVATE_MAIN 化（トリガー種別の取りこぼし）。
# ===========================================================================

@pytest.mark.xfail(strict=True,
                   reason="OP16-041: 誘発『自分の《インペルダウン》キャラが場を離れた時』が欠落し ACTIVATE_MAIN 化。本来は自動誘発であって起動メインではない")
def test_op16_041_should_be_leave_triggered_not_activate_main():
    """OP16-041: 本来「場を離れた時」の自動誘発であり、起動メインであってはならない。

    現実装は ACTIVATE_MAIN として登録されており、テキストの発動契機を満たさない → xfail。
    """
    gm, p1, p2, L = build("OP16-041")
    main_abilities = abilities_of(L.master, "ACTIVATE_MAIN")
    assert not main_abilities, "場を離れた時の誘発が ACTIVATE_MAIN に化けている"


# ===========================================================================
# OP16-060 センゴク
#   【起動メイン】アクティブのドン!!8枚をドン!!デッキに戻す：手札からカード名の
#   異なる《大将》キャラ3枚までを登場させる。  ⚠️ distinct-name 制約の欠落疑い。
# ===========================================================================

def test_op16_060_return_8_don_play_three_distinct_generals():
    """OP16-060: ドン8枚返却 → カード名の異なる《大将》3枚を登場させる。"""
    gm, p1, p2, L = build("OP16-060")
    g1 = make_char(p1, name="青雉", traits=["大将"])
    g2 = make_char(p1, name="赤犬", traits=["大将"])
    g3 = make_char(p1, name="黄猿", traits=["大将"])
    p1.hand += [g1, g2, g3]
    before_active = len(p1.don_active)
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    _drive(gm, p1, targets=[g1.uuid, g2.uuid, g3.uuid])
    assert len(p1.don_active) == before_active - 8
    assert g1 in p1.field and g2 in p1.field and g3 in p1.field


@pytest.mark.xfail(strict=False,
                   reason="OP16-060: 『カード名の異なる』distinct 制約がパース/実装に無く、同名の《大将》を複数登場できてしまう（要確認）")
def test_op16_060_distinct_name_constraint_limits_same_name():
    """OP16-060: 「カード名の異なる」制約により、同名の《大将》は1枚しか登場できないべき。

    現実装は distinct 制約が無く同名2枚とも登場するため、テキスト準拠の本アサートは失敗 → xfail。
    """
    gm, p1, p2, L = build("OP16-060")
    a = make_char(p1, name="青雉", traits=["大将"])
    b = make_char(p1, name="青雉", traits=["大将"])  # 同名
    p1.hand += [a, b]
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    _drive(gm, p1, targets=[a.uuid, b.uuid])
    # 同名は1枚まで → 場に出るのは高々1枚であるべき
    assert sum(1 for c in (a, b) if c in p1.field) <= 1


# ===========================================================================
# OP16-079 ヤマト
#   自分のトラッシュから《ワノ国》キャラが登場した時、そのキャラはこのターン【速攻】。
#   ⚠️ target zone=TRASH のまま「登場した場のキャラ」へ正しく付与されるか要確認。
# ===========================================================================

def test_op16_079_grant_haste_to_wano_from_trash():
    """OP16-079: トラッシュ由来で登場した《ワノ国》キャラに、このターン【速攻】を付与。"""
    gm, p1, p2, L = build("OP16-079")
    wano = make_char(p1, name="ワノ侍", traits=["ワノ国"])
    p1.trash.append(wano)
    gm.resolve_ability(p1, get_ability(L.master, "PASSIVE"), L)
    auto_resolve(gm, p1)
    assert _has_keyword(wano, "速攻")


# ===========================================================================
# OP16-080 マーシャル・D・ティーチ
#   能力0【相手のターン中】自分のキャラすべてをコスト+1。
#   能力1【相手のアタック時】【ターン1回】手札の【トリガー】持ち1枚を捨てて、
#         アタック対象をリーダー/《黒ひげ海賊団》キャラに変更。  ⚠️ 整合（要確認）。
# ===========================================================================

def test_op16_080_opponent_turn_cost_plus_one():
    """OP16-080 能力0: 相手ターン中、自分の全キャラのコストが+1される。"""
    gm, p1, p2, L = build("OP16-080")
    c = add_char(p1, name="手下", cost=3)
    before = c.current_cost
    gm.resolve_ability(p1, get_ability(L.master, "OPPONENT_TURN"), L)
    auto_resolve(gm, p1)
    assert c.current_cost == before + 1


def test_op16_080_redirect_discard_cost_requires_trigger_card():
    """OP16-080 能力1: 捨てコストは手札の【トリガー】を持つカードに限定される。

    汎用フィラー手札（【トリガー】非保持）では捨てコスト候補が無く、コストを払えず
    REDIRECT_ATTACK が成立しない（手札・盤面が変化しない）。
    """
    gm, p1, p2, L = build("OP16-080")
    hand_before = len(p1.hand)
    ab1 = get_ability(L.master, "ON_OPP_ATTACK")
    assert ab1.cost is not None and "HAS_TRIGGER" in ab1.cost.target.flags
    gm.resolve_ability(p1, ab1, L)
    auto_resolve(gm, p1)
    # 【トリガー】持ちが手札に無いため捨てられず、手札枚数は不変
    assert len(p1.hand) == hand_before


# ===========================================================================
# P-011 ウタ
#   【起動メイン】【ターン1回】ドン!!1枚をレスト：自分の元々の効果がないキャラ1枚までを
#   このターン中パワー+2000。  ✅ 整合。
#   注: P-011 はバニラではなく効果（能力）を持つリーダーなので skip 対象ではない。
# ===========================================================================

def test_p011_buff_vanilla_character_plus_2000():
    """P-011: 元々の効果がない（バニラ）キャラ1枚にこのターン中パワー+2000。"""
    gm, p1, p2, L = build("P-011")
    clear_field(p1)
    van = add_char(p1, name="バニラ", power=3000)  # effect_text 空 = バニラ
    before = van.get_power(True)
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    _drive(gm, p1, targets=[van.uuid])
    assert van.get_power(True) == before + 2000


def test_p011_effect_holding_character_not_targetable():
    """P-011: 効果（テキスト）を持つキャラは「元々の効果がない」対象外で強化されない。"""
    gm, p1, p2, L = build("P-011")
    clear_field(p1)
    eff = add_char(p1, name="効果持ち", power=3000,
                   effect_text="【登場時】カード1枚を引く。")
    before = eff.get_power(True)
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    _drive(gm, p1, targets=[eff.uuid])
    assert eff.get_power(True) == before  # 強化されない


# ===========================================================================
# P-047 モンキー・D・ルフィ
#   【ドン!!×1】【アタック時】自分の手札が3枚以下の場合、カード1枚を引く。
#   ⚠️ パース完全整合。条件下のドローを個別検証。
#   注: 【ドン!!×1】はリーダーへの付与ドン(attached_don)で判定される。
# ===========================================================================

def test_p047_draw_when_hand_le_3_and_don_attached():
    """P-047: 手札3枚以下＋【ドン!!×1】付与時、アタック時に1ドロー（条件成立）。"""
    gm, p1, p2, L = build("P-047")
    p1.hand = p1.hand[:3]
    L.attached_don = 1
    before = len(p1.hand)
    gm.resolve_ability(p1, get_ability(L.master, "ON_ATTACK"), L)
    auto_resolve(gm, p1)
    assert len(p1.hand) == before + 1


def test_p047_no_draw_when_hand_ge_4():
    """P-047: 手札4枚以上では条件未達でドローしない（境界: 4枚=不発）。"""
    gm, p1, p2, L = build("P-047")
    p1.hand = (p1.hand + p1.hand)[:4]
    L.attached_don = 1
    before = len(p1.hand)
    gm.resolve_ability(p1, get_ability(L.master, "ON_ATTACK"), L)
    auto_resolve(gm, p1)
    assert len(p1.hand) == before  # 不発


def test_p047_no_draw_when_no_don_attached():
    """P-047: 【ドン!!×1】未付与（attached_don=0）ではドローできない。"""
    gm, p1, p2, L = build("P-047")
    p1.hand = p1.hand[:3]
    L.attached_don = 0
    before = len(p1.hand)
    gm.resolve_ability(p1, get_ability(L.master, "ON_ATTACK"), L)
    auto_resolve(gm, p1)
    assert len(p1.hand) == before  # 不発


# ===========================================================================
# P-076 サカズキ
#   【起動メイン】【ターン1回】手札の《海軍》カード1枚を捨てて：相手キャラ1枚までを
#   このターン中コスト-1。  ⚠️ 方向(相手)・数値・コスト整合。連鎖を検証。
# ===========================================================================

def test_p076_discard_navy_then_opp_char_cost_minus_one():
    """P-076: 《海軍》1枚を捨てて、相手キャラ1枚のコストをこのターン-1にする。"""
    gm, p1, p2, L = build("P-076")
    navy = make_char(p1, name="海兵", traits=["海軍"])
    p1.hand.append(navy)
    opp = add_char(p2, name="相手", cost=5)
    before = opp.current_cost
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    _drive(gm, p1, resource=[navy.uuid], targets=[opp.uuid])
    assert navy in p1.trash
    assert opp.current_cost == before - 1


def test_p076_targets_opponent_not_self():
    """P-076: 強化（コスト-1）の対象は相手キャラで、自分キャラには及ばない。"""
    gm, p1, p2, L = build("P-076")
    navy = make_char(p1, name="海兵", traits=["海軍"])
    p1.hand.append(navy)
    my_char = add_char(p1, name="自分キャラ", cost=5)
    opp = add_char(p2, name="相手", cost=5)
    my_before = my_char.current_cost
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    _drive(gm, p1, resource=[navy.uuid], targets=[opp.uuid])
    assert my_char.current_cost == my_before  # 自分キャラは不変


# ===========================================================================
# P-086 トラファルガー・ロー
#   【起動メイン】【ターン1回】ドン!!-3, 自分のパワー3000以上のキャラ1枚をデッキ下に置く：
#   手札のコスト4以下《ハートの海賊団》キャラ1枚までを登場させる。
#   🐛 コスト側「パワー3000以上」が power_max=3000（=3000以下）に反転（同正規表現不具合）。
# ===========================================================================

def test_p086_deck_bottom_cost_target_at_threshold():
    """P-086: パワー3000ちょうどのキャラはデッキ下コストの対象になる（境界・成立）。"""
    gm, p1, p2, L = build("P-086")
    clear_field(p1)
    victim = add_char(p1, name="犠牲", power=3000, traits=["ハートの海賊団"])
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    _drive(gm, p1, resource=None, targets=[victim.uuid])
    assert victim in p1.deck and victim not in p1.field


def test_p086_deck_bottom_cost_target_above_threshold():
    """P-086: パワー5000（3000以上）のキャラはデッキ下コストの対象になるべき（テキスト準拠）。

    現実装は power_max=3000 と誤パースし 3000 超を対象外にするため、5000 は選べず場に残る → xfail。
    """
    gm, p1, p2, L = build("P-086")
    clear_field(p1)
    victim = add_char(p1, name="犠牲", power=5000, traits=["ハートの海賊団"])
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    _drive(gm, p1, resource=None, targets=[victim.uuid])
    assert victim in p1.deck and victim not in p1.field


def test_p086_deck_bottom_cost_target_below_threshold_excluded():
    """P-086: パワー2000（3000未満）のキャラはデッキ下コストの対象外であるべき。

    現実装は power_max=3000（3000以下を対象）と誤パースし、2000 を誤ってデッキ下に置ける → xfail。
    """
    gm, p1, p2, L = build("P-086")
    clear_field(p1)
    victim = add_char(p1, name="犠牲", power=2000, traits=["ハートの海賊団"])
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    _drive(gm, p1, resource=None, targets=[victim.uuid])
    assert victim in p1.field and victim not in p1.deck


# ===========================================================================
# P-117 ナミ
#   能力0【常在】自分デッキ0枚なら敗北の代わりに勝利（REPLACE_DECKOUT_LOSS）。
#   能力1【ドン!!×1】このリーダーのアタックでライフにダメージを与えた時、デッキ上1枚を
#         トラッシュへ置いてもよい。  🐛 能力1の誘発が ACTIVATE_MAIN 化。
# ===========================================================================

def test_p117_deckout_win_replacement():
    """P-117 能力0: 自分のデッキが0枚になっても、敗北の代わりに自分が勝利する。"""
    gm, p1, p2, L = build("P-117")
    p1.deck = []
    p2.deck = [p2.deck[0]] if p2.deck else []
    gm.check_victory()
    assert gm.winner == p1.name


@pytest.mark.xfail(strict=True,
                   reason="P-117: 能力1の誘発『リーダーのアタックでライフにダメージを与えた時』(ON_DAMAGE_DEALT_TO_LIFE)が取りこぼされ ACTIVATE_MAIN 化")
def test_p117_trash_trigger_should_be_damage_dealt_not_activate_main():
    """P-117 能力1: 誘発は「ライフにダメージを与えた時」であり、起動メインではないべき。

    現実装は当該の TRASH_FROM_DECK 能力を ACTIVATE_MAIN として登録しているため、
    ON_DAMAGE_DEALT_TO_LIFE トリガーの能力が存在しない → xfail。
    """
    triggers = [
        (a.trigger.name if hasattr(a.trigger, "name") else str(a.trigger))
        for a in (L.master.abilities or [])
    ]
    assert "ON_DAMAGE_DEALT_TO_LIFE" in triggers
    assert "ACTIVATE_MAIN" not in triggers
