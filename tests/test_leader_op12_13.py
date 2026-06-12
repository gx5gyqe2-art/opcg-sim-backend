"""リーダー効果 pytest（OP12 / OP13 の12枚）。

仕様書: docs/leader_specs/OP12-13.md
ガイド : docs/leader_specs/_TEST_GUIDE.md

方針:
  - 常に **テキスト準拠の正しい挙動** をアサートする（現実装に合わせない）。
  - ✅ → 通常テスト / 🐛 → xfail(strict=True) / ⚠️ → 通常 or xfail(strict=False)。
  - 誘発トリガー化け・条件反転・対象取り違えは「本来の発火条件/対象」を
    アサートして現実装との差を出す。

実行（-s 必須）:
  OPCG_LOG_SILENT=1 python -m pytest tests/test_leader_op12_13.py -q -s -p no:cacheprovider
"""
import pytest

import conftest  # noqa: F401  (google スタブ & sys.path)

from leader_test_helpers import (
    build, get_ability, auto_resolve,
    add_char, clear_field, set_life, leader_power, don_total,
)
from opcg_sim.src.models.enums import CardType, ActionType
from opcg_sim.src.models.models import CardInstance, DonInstance
from engine_helpers import make_master, action


# ---------------------------------------------------------------------------
# 小ヘルパ
# ---------------------------------------------------------------------------

def _hand_events(player, n):
    """player の手札を n 枚のイベントカードに差し替える。"""
    player.hand = [
        CardInstance(make_master(card_id=f"EV-{i}", type=CardType.EVENT), player.name)
        for i in range(n)
    ]
    return player.hand


def _attach_don_to_leader(player, leader, n):
    """リーダーに active ドンを n 枚付与する（【ドン!!×N】条件用）。"""
    for _ in range(n):
        d = player.don_active.pop()
        d.attached_to = leader.uuid
        player.don_attached_cards.append(d)
        leader.attached_don += 1


# ===========================================================================
# OP12-001 シルバーズ・レイリー  ⚠️ コスト公開が is_up_to:true で「2枚ちょうど」が緩和
# ===========================================================================

def test_op12_001_buff_with_two_revealed_events():
    """OP12-001 起動メイン: イベント2枚公開で、元々パワー4000以下のキャラ1枚に+2000。"""
    gm, p1, p2, L = build("OP12-001")
    clear_field(p1)
    c = add_char(p1, power=4000)
    _hand_events(p1, 2)
    ab = get_ability(L.master, "ACTIVATE_MAIN")
    gm.resolve_ability(p1, ab, L)
    auto_resolve(gm, p1)
    assert c.get_power(True) == 6000          # +2000
    assert len(p1.hand) == 2                  # 公開なので手札は残る


def test_op12_001_one_event_cannot_pay_cost():
    """OP12-001: 手札イベントが1枚のみ＝2枚公開コストを払えず、バフは発生しない（正しい挙動）。"""
    gm, p1, p2, L = build("OP12-001")
    clear_field(p1)
    c = add_char(p1, power=4000)
    _hand_events(p1, 1)                        # イベント1枚のみ
    ab = get_ability(L.master, "ACTIVATE_MAIN")
    gm.resolve_ability(p1, ab, L)
    auto_resolve(gm, p1)
    assert c.get_power(True) == 4000          # コスト払えず未バフが正


def test_op12_001_no_target_when_power_over_4000():
    """OP12-001: 元々パワー5000のキャラのみ＝対象不在（1枚までなので0枚で成立、バフ無し）。"""
    gm, p1, p2, L = build("OP12-001")
    clear_field(p1)
    c = add_char(p1, power=5000)              # 元々5000 → 対象外
    _hand_events(p1, 2)
    ab = get_ability(L.master, "ACTIVATE_MAIN")
    gm.resolve_ability(p1, ab, L)
    auto_resolve(gm, p1)
    assert c.get_power(True) == 5000          # 対象外なので不変


# ===========================================================================
# OP12-020 ロロノア・ゾロ  🐛 branch条件が「バトル中」でなく「相手キャラ存在」
#                              ＆ ATTACK_DISABLE が自リーダーでなく相手キャラ1枚選択
# ===========================================================================

@pytest.mark.xfail(strict=True,
    reason="OP12-020: branch条件が FIELD_COUNT(OPPONENT)>=1（相手がキャラを持つ）に緩和。"
           "テキストは『このリーダーが相手キャラとバトルしている場合』＝バトル中限定。"
           "バトルしていないのに相手がキャラを持つだけでリーダーがアクティブ化してしまう。")
def test_op12_020_no_active_when_not_in_battle():
    """OP12-020 起動メイン: 相手キャラはいるがバトルしていない場合、リーダーはアクティブ化しない（正しい挙動）。"""
    gm, p1, p2, L = build("OP12-020")
    _attach_don_to_leader(p1, L, 3)
    L.is_rest = True
    clear_field(p2)
    add_char(p2, cost=3, power=1000)          # 相手キャラはいるが「バトル中」ではない
    ab = get_ability(L.master, "ACTIVATE_MAIN")
    gm.resolve_ability(p1, ab, L)
    auto_resolve(gm, p1)
    assert L.is_rest is True                   # バトルしていないのでアクティブ化しないのが正


@pytest.mark.xfail(strict=True,
    reason="OP12-020: ATTACK_DISABLE が相手キャラ1枚を選択して付与する形。"
           "テキストは『このリーダーが相手のコスト7以下キャラへアタックできない』＝自リーダーへの制限。"
           "制限フラグは相手キャラではなく自リーダーに乗るのが正。")
def test_op12_020_attack_restriction_on_self_leader():
    """OP12-020 起動メイン: アタック制限は自リーダーに付く（相手キャラに付くのは誤り）。"""
    gm, p1, p2, L = build("OP12-020")
    _attach_don_to_leader(p1, L, 3)
    clear_field(p2)
    oc = add_char(p2, cost=3, power=1000)
    ab = get_ability(L.master, "ACTIVATE_MAIN")
    gm.resolve_ability(p1, ab, L)
    auto_resolve(gm, p1)
    # 自リーダーにアタック制限が乗るのが正。現実装は相手キャラ側に乗るため失敗する。
    assert "ATTACK_DISABLE" in getattr(L, "timed_flags", set())
    assert "ATTACK_DISABLE" not in getattr(oc, "timed_flags", set())


def test_op12_020_insufficient_don_no_activation():
    """OP12-020 起動メイン: ドン!!付与が3枚未満（条件 HAS_DON>=3 未達）なら不発。"""
    gm, p1, p2, L = build("OP12-020")
    _attach_don_to_leader(p1, L, 2)            # 2枚のみ → 条件未達
    L.is_rest = True
    clear_field(p2)
    add_char(p2, cost=3, power=1000)
    ab = get_ability(L.master, "ACTIVATE_MAIN")
    gm.resolve_ability(p1, ab, L)
    auto_resolve(gm, p1)
    assert L.is_rest is True                    # 条件未達でアクティブ化されない


# ===========================================================================
# OP12-040 クザン  🐛 誘発条件(《海軍》効果起因の捨て)欠落＋固定1ドローで PASSIVE 常時ドロー化
# ===========================================================================

def test_op12_040_no_draw_without_navy_discard_trigger():
    """OP12-040: 《海軍》効果での手札捨てが発生していない場面では引かない（正しい挙動）。"""
    gm, p1, p2, L = build("OP12-040")
    ab = get_ability(L.master, "PASSIVE")
    before = len(p1.hand)
    gm.resolve_ability(p1, ab, L)
    auto_resolve(gm, p1)
    # 捨てが無いので引かないのが正。PASSIVE常時ドロー化のため現実装は +1 になり失敗する。
    assert len(p1.hand) == before


# ===========================================================================
# OP12-041 サンジ  ⚠️ 能力1のドン比較条件成立時のレスト追加発火を確認
# ===========================================================================

def test_op12_041_execute_event_returns_don():
    """OP12-041 起動メイン: ドン!!-1を払って手札の《麦わら》コスト3以下イベントを発動できる。"""
    gm, p1, p2, L = build("OP12-041")
    ab = get_ability(L.master, "ACTIVATE_MAIN")
    don_before = don_total(p1)
    gm.resolve_ability(p1, ab, L)
    auto_resolve(gm, p1)
    assert don_total(p1) == don_before - 1     # ドン!!-1 を支払った


def test_op12_041_ramp_when_self_don_le_opponent():
    """OP12-041 アタック時: 自場ドン<=相手場ドンの場合、ドン!!デッキからレストで1枚追加。"""
    gm, p1, p2, L = build("OP12-041")
    # 自分の active ドンを2枚に減らす（相手は5枚）→ 自<=相手で条件成立
    for _ in range(8):
        p1.don_active.pop()
    ab = get_ability(L.master, "ON_ATTACK")
    rested_before = len(p1.don_rested)
    deck_before = len(p1.don_deck)
    gm.resolve_ability(p1, ab, L)
    auto_resolve(gm, p1)
    assert len(p1.don_rested) == rested_before + 1   # レストで1枚追加
    assert len(p1.don_deck) == deck_before - 1


def test_op12_041_no_ramp_when_self_don_gt_opponent():
    """OP12-041 アタック時: 自場ドン>相手場ドン（10 vs 5）なら条件未達で追加なし。"""
    gm, p1, p2, L = build("OP12-041")
    ab = get_ability(L.master, "ON_ATTACK")
    total_before = don_total(p1)
    gm.resolve_ability(p1, ab, L)
    auto_resolve(gm, p1)
    assert don_total(p1) == total_before        # 条件未達で増えない


# ===========================================================================
# OP12-061 ドンキホーテ・ロシナンテ  ⚠️ 能力0の置換が「ロー」名称限定か
# ===========================================================================

def test_op12_061_replacement_only_for_law():
    """OP12-061 能力0: 「トラファルガー・ロー」以外のキャラがKOされても置換しない（正しい挙動）。"""
    gm, p1, p2, L = build("OP12-061")
    clear_field(p1)
    victim = add_char(p1, name="ロロノア・ゾロ", power=5000)   # ロー以外
    gm.apply_action_to_engine(p2, action(ActionType.KO), [victim], 0)
    auto_resolve(gm, p1)
    # ロー以外なので置換せず、通常どおりKOされるのが正。
    assert victim not in p1.field
    assert victim in p1.trash


def test_op12_061_cost_reduction_returns_don():
    """OP12-061 起動メイン: ドン!!-1を払い、次に登場する「ロー」のコストを軽減する（ドン返却を確認）。"""
    gm, p1, p2, L = build("OP12-061")
    ab = get_ability(L.master, "ACTIVATE_MAIN")
    don_before = don_total(p1)
    gm.resolve_ability(p1, ab, L)
    auto_resolve(gm, p1)
    assert don_total(p1) == don_before - 1


# ===========================================================================
# OP12-081 コアラ  🐛 能力0の条件プレイヤー逆(OPPONENT→本来SELF)、能力1がACTIVATE_MAIN化
# ===========================================================================

def test_op12_081_draw_when_self_has_two_cost8():
    """OP12-081 アタック時: 自分のコスト8以上キャラが2枚以上で1枚引く（正しい挙動）。"""
    gm, p1, p2, L = build("OP12-081")
    clear_field(p1)
    clear_field(p2)
    add_char(p1, cost=8, power=10000)
    add_char(p1, cost=8, power=10000)          # 自分のコスト8以上が2枚
    ab = get_ability(L.master, "ON_ATTACK")
    before = len(p1.hand)
    gm.resolve_ability(p1, ab, L)
    auto_resolve(gm, p1)
    assert len(p1.hand) == before + 1          # 自分側2枚で引くのが正（現実装は相手側を見るため引かない）


def test_op12_081_no_draw_when_self_has_one_cost8():
    """OP12-081 アタック時: 自分のコスト8以上キャラが1枚のみなら条件未達で引かない。"""
    gm, p1, p2, L = build("OP12-081")
    clear_field(p1)
    clear_field(p2)
    add_char(p1, cost=8, power=10000)          # 1枚のみ
    ab = get_ability(L.master, "ON_ATTACK")
    before = len(p1.hand)
    gm.resolve_ability(p1, ab, L)
    auto_resolve(gm, p1)
    assert len(p1.hand) == before              # 1枚では引かない（自他いずれでも未達）


def test_op12_081_ability1_is_triggered_not_activate_main():
    """OP12-081 能力1: 相手のキャラ登場に反応する誘発（ON_OPP_PLAY）であるべき（ACTIVATE_MAIN化は誤り）。"""
    gm, p1, p2, L = build("OP12-081")
    trig = get_ability(L.master, "ON_OPP_PLAY", n=0)  # 相手登場の誘発種別
    assert trig is not None
    trigs = [a.trigger.name for a in (L.master.abilities or [])]
    assert "ACTIVATE_MAIN" not in trigs


# ===========================================================================
# OP13-001 モンキー・D・ルフィ  ⚠️ branch条件 DON_COUNT が「アクティブのドン」限定か
# ===========================================================================

@pytest.mark.xfail(strict=False,
    reason="OP13-001 要確認: 【相手のアタック時】レスト枚数×2000バフ。ON_OPP_ATTACK は"
           "バトル文脈に依存し汎用盤面では安定起動しないため strict=False。"
           "branch条件 DON_COUNT が『アクティブのドン5枚以下』限定かも要確認。")
def test_op13_001_rest_don_buffs_leader():
    """OP13-001 相手のアタック時: ドンをレストし、レスト枚数×2000をリーダー等に付与。"""
    gm, p1, p2, L = build("OP13-001")
    # ドン!!1付与（HAS_DON>=1）。active を絞ってアクティブドン<=5。
    for _ in range(7):
        p1.don_active.pop()                    # active 3 枚
    _attach_don_to_leader(p1, L, 1)            # leader に1枚付与（active 2）
    ab = get_ability(L.master, "ON_OPP_ATTACK")
    pb = leader_power(p1, my_turn=False)
    gm.resolve_ability(p1, ab, L)
    auto_resolve(gm, p1)
    assert leader_power(p1, my_turn=False) > pb   # 1枚以上レストでバフが乗る


# ===========================================================================
# OP13-002 ポートガス・D・エース  🐛 能力1がACTIVATE_MAIN化（ダメージ被弾/6000以上KO誘発が消失）
# ===========================================================================

def test_op13_002_debuff_opponent_on_opp_attack():
    """OP13-002 能力0: 相手のアタック時、手札1枚を捨てて相手キャラ1枚に-2000（このバトル）。"""
    gm, p1, p2, L = build("OP13-002")
    clear_field(p2)
    v = add_char(p2, power=5000)
    ab = get_ability(L.master, "ON_OPP_ATTACK")
    hand_before = len(p1.hand)
    gm.resolve_ability(p1, ab, L)
    auto_resolve(gm, p1)
    assert v.get_power(False) == 3000          # -2000
    assert len(p1.hand) == hand_before - 1     # 手札1枚捨てた


@pytest.mark.xfail(strict=True,
    reason="OP13-002: 能力1のトリガーが ACTIVATE_MAIN 化。"
           "テキストは『自分がダメージを受けた時/元々パワー6000以上のキャラがKOされた時』の誘発ドロー。"
           "本来はダメージ被弾/KO誘発（ON_DAMAGE系/ON_KO等）であって起動メインではない。")
def test_op13_002_ability1_draw_is_triggered_not_main():
    """OP13-002 能力1: ドローはダメージ被弾/KOの誘発であるべきで、ACTIVATE_MAIN化は誤り。"""
    gm, p1, p2, L = build("OP13-002")
    from opcg_sim.src.models.enums import TriggerType
    triggered = (TriggerType.ON_DAMAGE_DEALT_TO_LIFE, TriggerType.ON_LIFE_DECREASE,
                 TriggerType.ON_KO)
    # ドローを持つ能力のトリガーが誘発種別であることを期待。現実装は ACTIVATE_MAIN なので失敗。
    draw_ab = get_ability(L.master, "ACTIVATE_MAIN")
    assert draw_ab.trigger in triggered


# ===========================================================================
# OP13-003 ゴール・D・ロジャー  🐛 能力0の条件が DON_COUNT EQ 0 で反転（本来 ドン>=1）
# ===========================================================================

def test_op13_003_attach_requires_don_present():
    """OP13-003 能力0: 場にドンがある（>=1）場合のみドン付与が発火する（正しい条件）。"""
    gm, p1, p2, L = build("OP13-003")
    # 場にドンが10枚ある（>=1）→ 本来は条件成立。
    ab = get_ability(L.master, "PASSIVE", n=0)
    assert ab.condition is not None
    cond = ab.condition
    # 反転していなければ operator は GE（>=1）であるべき。EQ 0 なら失敗。
    op = cond.operator.name if cond.operator and hasattr(cond.operator, "name") else cond.operator
    assert (op, cond.value) == ("GE", 1)


def test_op13_003_leader_minus_2000_when_don_le_9():
    """OP13-003 能力1: 場のドンが9枚以下ならリーダーのパワー-2000、10枚なら-2000なし。"""
    gm, p1, p2, L = build("OP13-003")
    # 既定 active ドン10枚 → -2000なし（7000のまま）
    gm._apply_passive_effects(p1)
    assert leader_power(p1) == 7000
    # ドンを9枚に減らす → -2000
    p1.don_active.pop()
    gm._apply_passive_effects(p1)
    assert leader_power(p1) == 5000


# ===========================================================================
# OP13-004 サボ  🐛 能力1がACTIVATE_MAIN化（常在条件付き+1000であるべき）
# ===========================================================================

def test_op13_004_leader_minus_1000_when_life_ge_4():
    """OP13-004 能力0: 自ライフ4枚以上でリーダー-1000、3枚で解除。"""
    gm, p1, p2, L = build("OP13-004")
    gm._apply_passive_effects(p1)
    assert leader_power(p1) == 4000            # ライフ5 → -1000
    set_life(p1, 3)
    gm._apply_passive_effects(p1)
    assert leader_power(p1) == 5000            # ライフ3 → 解除


@pytest.mark.xfail(strict=True,
    reason="OP13-004: 能力1のトリガーが ACTIVATE_MAIN 化。"
           "テキストは【ドン‼×1】の常在条件付き効果（コスト8以上キャラがいれば自リーダー・キャラ全体+1000）。"
           "passive 再計算では適用されず、起動メイン発動が必要になってしまう。")
def test_op13_004_all_buff_is_continuous_when_cost8_present():
    """OP13-004 能力1: コスト8以上キャラ＋ドン!!1で、起動操作なしに全体+1000が継続適用されるべき。"""
    gm, p1, p2, L = build("OP13-004")
    set_life(p1, 3)                            # 能力0(-1000)を切って計測を分離
    clear_field(p1)
    c = add_char(p1, cost=8, power=10000)
    _attach_don_to_leader(p1, L, 1)
    gm._apply_passive_effects(p1)              # 常在効果なら passive 再計算で +1000 が乗るはず
    assert leader_power(p1) == 6000            # 5000 +1000
    assert c.get_power(True) == 11000          # 10000 +1000


# ===========================================================================
# OP13-079 イム  ✅ ゲーム開始時ステージ登場・2択コスト・1ドローともテキスト整合
# ===========================================================================

def test_op13_079_game_start_plays_marijoa_stage():
    """OP13-079 能力0: ゲーム開始時、デッキの《聖地マリージョア》ステージを登場させる。"""
    gm, p1, p2, L = build("OP13-079")
    stage = CardInstance(
        make_master(card_id="ST-MJ", type=CardType.STAGE, traits=["聖地マリージョア"]),
        p1.name,
    )
    p1.deck.insert(0, stage)
    ab = get_ability(L.master, "GAME_START")
    gm.resolve_ability(p1, ab, L)
    auto_resolve(gm, p1)
    assert p1.stage is stage                   # ステージゾーンへ登場


def test_op13_079_activate_main_choice_cost_draws():
    """OP13-079 能力1: 《天竜人》キャラか手札1枚をトラッシュして1枚引く（2択コスト）。"""
    gm, p1, p2, L = build("OP13-079")
    add_char(p1, name="天竜人A", traits=["天竜人"])
    ab = get_ability(L.master, "ACTIVATE_MAIN")
    hand_before = len(p1.hand)
    trash_before = len(p1.trash)
    gm.resolve_ability(p1, ab, L)
    auto_resolve(gm, p1)
    assert len(p1.hand) == hand_before + 1     # 1枚引く
    assert len(p1.trash) == trash_before + 1   # コストでトラッシュ1枚増


# ===========================================================================
# OP13-100 ジュエリー・ボニー  🐛 誘発条件「【トリガー】持ちキャラ登場時」欠落、YOUR_TURN単純起動化
# ===========================================================================

def test_op13_100_no_attach_without_trigger_char_play():
    """OP13-100: 【トリガー】持ちキャラの登場が無い場面では付与が発火しない（正しい挙動）。"""
    gm, p1, p2, L = build("OP13-100")
    clear_field(p1)                            # 【トリガー】持ちキャラの登場イベントは無い
    ab = get_ability(L.master, "YOUR_TURN")
    don_before = don_total(p1)
    gm.resolve_ability(p1, ab, L)
    auto_resolve(gm, p1)
    # 登場誘発が無いので付与されない（ドン総数が変わらない）のが正。
    # 現実装は YOUR_TURN 単純起動で2枚付与してしまい don_total が減るため失敗する。
    assert don_total(p1) == don_before
