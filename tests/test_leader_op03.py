"""OP03 リーダーカード効果の pytest 化（docs/leader_specs/OP03.md 準拠）。

対象8枚: OP03-001 / OP03-021 / OP03-022 / OP03-040 / OP03-058 /
         OP03-076 / OP03-077 / OP03-099

判定ラベルとマーカーの対応（_TEST_GUIDE.md）:
  ✅ → 通常テスト / 🐛 → xfail(strict=True) / ⚠️ → 通常（不安定なら xfail(strict=False)）

実行: OPCG_LOG_SILENT=1 python -m pytest tests/test_leader_op03.py -q -s -p no:cacheprovider
"""
import pytest

import conftest  # noqa: F401  (google スタブ & sys.path)

from leader_test_helpers import (
    build, get_ability, abilities_of, leader_power,
    add_char, clear_field, set_life,
)
from engine_helpers import make_master, make_instance
from opcg_sim.src.models.models import CardInstance, DonInstance
from opcg_sim.src.models.enums import CardType


# ---------------------------------------------------------------------------
# 共通ヘルパ
# ---------------------------------------------------------------------------

def _drive(gm, player, limit=20):
    """active_interaction を賢い既定で解決まで進める（任意=受諾、SELECT=最大数選択）。

    auto_resolve は SELECT を min(最低1) しか取らないため、複数枚捨てる/選ぶ
    ケースを正しく駆動できるよう独自に max まで選択する。
    """
    steps = 0
    while gm.active_interaction and steps < limit:
        ia = gm.active_interaction
        at = ia.get("action_type", "")
        if at in ("CONFIRM_OPTIONAL", "CONFIRM_TRIGGER"):
            gm.resolve_interaction(player, {"accepted": True})
        elif at in ("SELECT_TARGET", "SELECT_RESOURCE"):
            cands = ia.get("selectable_uuids") or [c.uuid for c in ia.get("candidates", [])]
            cons = ia.get("constraints") or {}
            mn = cons.get("min", 0)
            mx = cons.get("max", 1)
            if mx is not None and mx < 0:
                n = len(cands)
            else:
                n = max(mn, 1) if cands else 0
                if mx:
                    n = min(n, mx)
            gm.resolve_interaction(player, {"selected_uuids": cands[:n], "index": 0})
        else:  # CHOICE 等
            gm.resolve_interaction(player, {"selected_uuids": [], "index": 0})
        steps += 1
    return steps


def _attach_don(leader, player, n):
    """リーダーにドン!!を n 枚付与する（【ドン!!×N】条件 HAS_DON を満たすため）。"""
    leader.attached_don = n
    for _ in range(n):
        player.don_attached_cards.append(
            DonInstance(owner_id=player.name, attached_to=leader.uuid)
        )


def _trig_names(master):
    return [a.trigger.name if hasattr(a.trigger, "name") else str(a.trigger)
            for a in (master.abilities or [])]


# ===========================================================================
# OP03-001 ポートガス・D・エース ⚠️
# ===========================================================================

def test_op03_001_attack_discard_two_events_buffs_2000():
    """OP03-001 アタック時: 手札のイベント2枚を捨てると、このバトル中パワー+2000。"""
    gm, p1, p2, L = build("OP03-001")
    p1.hand = [CardInstance(make_master(card_id=f"EV-{i}", type=CardType.EVENT), p1.name)
               for i in range(2)]
    assert leader_power(p1) == 5000
    gm.resolve_ability(p1, get_ability(L.master, "ON_ATTACK"), L)
    # 任意確認を受諾 → 捨てる対象選択で手札のイベント2枚を全選択
    gm.resolve_interaction(p1, {"accepted": True})
    ia = gm.active_interaction
    cands = ia.get("selectable_uuids") or [c.uuid for c in ia.get("candidates", [])]
    assert len(cands) == 2
    gm.resolve_interaction(p1, {"selected_uuids": cands, "index": 0})
    _drive(gm, p1)
    assert leader_power(p1) == 7000   # 5000 + 1000 * 2
    assert len(p1.hand) == 0
    assert len(p1.trash) == 12        # 既定10 + 捨てた2


def test_op03_001_attack_no_discard_target_keeps_power():
    """OP03-001 アタック時: 捨てる対象（イベント/ステージ）が手札に無ければ
    任意効果をスキップでき、パワーは増えずキャラは捨てられない。"""
    gm, p1, p2, L = build("OP03-001")
    ch = CardInstance(make_master(card_id="CH-1", type=CardType.CHARACTER), p1.name)
    p1.hand = [ch]
    gm.resolve_ability(p1, get_ability(L.master, "ON_ATTACK"), L)
    # 任意効果は no で拒否（キャラは対象外なので捨てられない）
    while gm.active_interaction:
        gm.resolve_interaction(p1, {"accepted": False, "selected_uuids": [], "index": 0})
    assert leader_power(p1) == 5000
    assert ch in p1.hand   # キャラは捨て対象外


@pytest.mark.xfail(strict=False, reason="要確認: OP03-001 テキストは「アタックされた時」も誘発するが"
                                        "指紋は ON_ATTACK 単独で ON_OPP_ATTACK が欠落（被アタック側未実装の疑い）")
def test_op03_001_also_triggers_when_attacked():
    """OP03-001: 「アタックされた時」にも能力が誘発するべき（被アタック側トリガーの存在確認）。"""
    gm, p1, p2, L = build("OP03-001")
    assert "ON_OPP_ATTACK" in _trig_names(L.master)


# ===========================================================================
# OP03-021 クロ
# ===========================================================================

def test_op03_021_activate_rests_opponent_and_activates_leader():
    """OP03-021 起動メイン: ドン3+《東の海》2枚レストを払い、自リーダーをアクティブ化、
    相手コスト5以下キャラ1枚をレストにする。✅"""
    gm, p1, p2, L = build("OP03-021")
    clear_field(p1)
    e1 = add_char(p1, name="eb1", traits=["東の海"], cost=2)
    e2 = add_char(p1, name="eb2", traits=["東の海"], cost=2)
    clear_field(p2)
    victim = add_char(p2, name="victim", cost=3)
    L.is_rest = True   # レスト状態からアクティブ化を観測
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    _drive(gm, p1)
    assert L.is_rest is False          # リーダーがアクティブに
    assert e1.is_rest and e2.is_rest   # コストで《東の海》2枚レスト
    assert victim.is_rest is True      # 相手コスト3キャラがレスト


def test_op03_021_opponent_cost6_not_targetable():
    """OP03-021: 相手がコスト6キャラのみの場合、対象不可（1枚まで=0）。
    リーダーはアクティブ化するがコスト6キャラはレストにならない。✅"""
    gm, p1, p2, L = build("OP03-021")
    clear_field(p1)
    add_char(p1, name="eb1", traits=["東の海"], cost=2)
    add_char(p1, name="eb2", traits=["東の海"], cost=2)
    clear_field(p2)
    big = add_char(p2, name="big", cost=6)
    L.is_rest = True
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    _drive(gm, p1)
    assert L.is_rest is False     # リーダーはアクティブ化
    assert big.is_rest is False   # コスト6は対象外


@pytest.mark.xfail(strict=True, reason="OP03-021: コスト《東の海》2枚レストが strict でなく、"
                                       "1枚しか無くてもコスト未達のまま効果が実行されてしまう疑い")
def test_op03_021_insufficient_east_blue_chars_does_not_execute():
    """OP03-021: 《東の海》キャラが1枚しか無い→コスト未達で発動不可。
    本来は相手キャラがレストにならないはず。🐛(疑い)"""
    gm, p1, p2, L = build("OP03-021")
    clear_field(p1)
    add_char(p1, name="eb1", traits=["東の海"], cost=2)   # 1枚のみ（要求2枚）
    clear_field(p2)
    victim = add_char(p2, name="victim", cost=3)
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    _drive(gm, p1)
    assert victim.is_rest is False   # コスト未達なので効果は走らないべき


# ===========================================================================
# OP03-022 アーロン 🐛
# ===========================================================================

@pytest.mark.xfail(strict=True, reason="OP03-022: 対象条件「【トリガー】を持つ」が指紋に欠落し"
                                       "(target.traits/flags 空)、トリガー無しキャラも登場できてしまう")
def test_op03_022_non_trigger_char_not_playable():
    """OP03-022 アタック時: 登場対象は「コスト4以下かつ【トリガー】を持つ」キャラのみ。
    トリガーを持たないコスト3キャラは登場できないべき。🐛"""
    gm, p1, p2, L = build("OP03-022")
    _attach_don(L, p1, 2)   # 【ドン!!×2】条件
    no_trigger = CardInstance(
        make_master(card_id="NOTRIG", type=CardType.CHARACTER, cost=3), p1.name)
    p1.hand = [no_trigger]
    gm.resolve_ability(p1, get_ability(L.master, "ON_ATTACK"), L)
    _drive(gm, p1)
    assert no_trigger not in p1.field   # トリガー非保持は登場不可であるべき


def test_op03_022_requires_two_attached_don():
    """OP03-022: ドン!!付与が1枚のみ（条件【ドン!!×2】未達）なら能力は不発。✅"""
    gm, p1, p2, L = build("OP03-022")
    _attach_don(L, p1, 1)   # 1枚のみ
    ch = CardInstance(
        make_master(card_id="C3", type=CardType.CHARACTER, cost=3), p1.name)
    p1.hand = [ch]
    field_before = len(p1.field)
    gm.resolve_ability(p1, get_ability(L.master, "ON_ATTACK"), L)
    _drive(gm, p1)
    assert ch in p1.hand                 # 登場せず手札に残る
    assert len(p1.field) == field_before


# ===========================================================================
# OP03-040 ナミ
# ===========================================================================

def test_op03_040_deckout_win_replacement():
    """OP03-040 能力0: 自分のデッキが0枚→敗北の代わりに勝利する。✅"""
    gm, p1, p2, L = build("OP03-040")
    p1.deck = []
    p2.deck = [make_instance(make_master(card_id="D"), owner=p2.name)]
    gm.check_victory()
    assert gm.winner == p1.name


def test_op03_040_deckout_win_only_when_deck_zero():
    """OP03-040 能力0: デッキが0枚でなければ勝利化しない（相手が0枚なら相手が敗北）。✅"""
    gm, p1, p2, L = build("OP03-040")
    p1.deck = [make_instance(make_master(card_id="D1"), owner=p1.name)]
    p2.deck = []
    gm.check_victory()
    assert gm.winner == p1.name   # 相手(p2)がデッキアウトで p1 勝利（置換は発動しない）


def test_op03_040_life_damage_trigger_type():
    """OP03-040 能力1: 「相手のライフにダメージを与えた時」のデッキトップトラッシュは
    ダメージ誘発トリガーであるべき（起動メインではない）。🐛"""
    gm, p1, p2, L = build("OP03-040")
    abs1 = abilities_of(L.master, "ON_DAMAGE_DEALT_TO_LIFE")
    assert abs1, "能力1 は ON_DAMAGE_DEALT_TO_LIFE 誘発であるべき"


# ===========================================================================
# OP03-058 アイスバーグ ⚠️
# ===========================================================================

def test_op03_058_leader_cannot_attack():
    """OP03-058 能力0: このリーダーはアタックできない（PASSIVE で ATTACK_DISABLE 付与）。✅"""
    gm, p1, p2, L = build("OP03-058")
    gm._apply_passive_effects(p1)
    assert "ATTACK_DISABLE" in (L.timed_flags | getattr(L, "flags", set()))


def test_op03_058_activate_plays_gc_char_and_rests_leader():
    """OP03-058 能力1: ドン!!-1返却＋リーダーレストを払い、手札のコスト5以下《GC》キャラを登場。✅"""
    gm, p1, p2, L = build("OP03-058")
    gc = CardInstance(
        make_master(card_id="GC-1", type=CardType.CHARACTER, cost=5, traits=["GC"]), p1.name)
    p1.hand = [gc]
    don_before = len(p1.don_active)
    field_before = len(p1.field)
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    _drive(gm, p1)
    assert L.is_rest is True            # コストでリーダーがレスト
    assert gc in p1.field              # 《GC》キャラ登場
    assert len(p1.field) == field_before + 1
    assert len(p1.don_active) == don_before - 1   # ドン!!-1


def test_op03_058_non_gc_char_not_playable():
    """OP03-058: 手札が《GC》以外のキャラのみの場合、登場対象が無く登場は起きない。✅"""
    gm, p1, p2, L = build("OP03-058")
    non_gc = CardInstance(
        make_master(card_id="NGC", type=CardType.CHARACTER, cost=5, traits=["W7"]), p1.name)
    p1.hand = [non_gc]
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    _drive(gm, p1)
    assert non_gc not in p1.field   # 非《GC》は登場不可


# ===========================================================================
# OP03-076 ロブ・ルッチ 🐛
# ===========================================================================

@pytest.mark.xfail(strict=True, reason="OP03-076: 誘発条件「相手のキャラがKOされた時」が欠落し"
                                       "YOUR_TURN の単純起動になっている（KO 無しでも手札2枚捨てでアクティブ化できる疑い）")
def test_op03_076_requires_opponent_ko_trigger():
    """OP03-076: 効果は「相手のキャラがKOされた時」に誘発するべきで、
    KO に依存しない YOUR_TURN 起動になっていてはいけない。🐛"""
    gm, p1, p2, L = build("OP03-076")
    trigs = _trig_names(L.master)
    # KO 誘発（ON_KO 等）であるべき。YOUR_TURN 単独はバグ。
    assert "ON_KO" in trigs


def test_op03_076_activates_leader_after_ko_with_discard():
    """OP03-076: （現実装の YOUR_TURN 起動として）手札2枚を捨てるとレストのリーダーが
    アクティブになる。コスト・効果自体の動作確認。✅"""
    gm, p1, p2, L = build("OP03-076")
    L.is_rest = True
    p1.hand = [CardInstance(make_master(card_id=f"H{i}"), p1.name) for i in range(3)]
    gm.resolve_ability(p1, get_ability(L.master, "YOUR_TURN"), L)
    _drive(gm, p1)
    assert L.is_rest is False        # リーダーがアクティブに
    assert len(p1.hand) == 1         # 手札2枚を捨てた


def test_op03_076_turn_limit_blocks_second_activation():
    """OP03-076 【ターン1回】: 同一ターンの2回目は不発。✅"""
    gm, p1, p2, L = build("OP03-076")
    L.is_rest = True
    p1.hand = [CardInstance(make_master(card_id=f"H{i}"), p1.name) for i in range(5)]
    ab = get_ability(L.master, "YOUR_TURN")
    gm.resolve_ability(p1, ab, L)
    _drive(gm, p1)
    assert L.is_rest is False and len(p1.hand) == 3
    # 2回目: ターン1回済みで不発（リーダーを再度レストにして観測）
    L.is_rest = True
    gm.resolve_ability(p1, ab, L)
    _drive(gm, p1)
    assert L.is_rest is True          # 2回目は発動せずレストのまま
    assert len(p1.hand) == 3          # 手札も減らない


# ===========================================================================
# OP03-077 シャーロット・リンリン ⚠️
# ===========================================================================

def test_op03_077_life_le_1_heals_one():
    """OP03-077 アタック時: ドン!!×2＋手札1枚捨てを払い、自ライフ1枚以下なら
    デッキトップ1枚をライフ上へ加える（ライフ+1）。✅"""
    gm, p1, p2, L = build("OP03-077")
    _attach_don(L, p1, 2)
    set_life(p1, 1)
    p1.hand = [CardInstance(make_master(card_id="H1"), p1.name)]
    deck_before = len(p1.deck)
    gm.resolve_ability(p1, get_ability(L.master, "ON_ATTACK"), L)
    _drive(gm, p1)
    assert len(p1.life) == 2              # ライフ +1
    assert len(p1.deck) == deck_before - 1
    assert len(p1.hand) == 0              # 手札1枚を捨てた


def test_op03_077_life_above_1_no_heal():
    """OP03-077: 自ライフが2枚以上（条件 LIFE<=1 未達）なら効果は走らずライフは増えない。✅"""
    gm, p1, p2, L = build("OP03-077")
    _attach_don(L, p1, 2)
    set_life(p1, 3)
    p1.hand = [CardInstance(make_master(card_id="H1"), p1.name)]
    gm.resolve_ability(p1, get_ability(L.master, "ON_ATTACK"), L)
    _drive(gm, p1)
    assert len(p1.life) == 3   # 条件未達で回復なし


def test_op03_077_up_to_allows_zero():
    """OP03-077: 「1枚まで」なので0枚（加えない＝確認を拒否）選択でライフは据え置きにできる。✅"""
    gm, p1, p2, L = build("OP03-077")
    _attach_don(L, p1, 2)
    set_life(p1, 1)
    p1.hand = [CardInstance(make_master(card_id="H1"), p1.name)]
    life_before = len(p1.life)
    gm.resolve_ability(p1, get_ability(L.master, "ON_ATTACK"), L)
    # すべての確認/選択で「拒否/0枚」を選ぶ
    steps = 0
    while gm.active_interaction and steps < 20:
        ia = gm.active_interaction
        at = ia.get("action_type", "")
        if at in ("CONFIRM_OPTIONAL", "CONFIRM_TRIGGER"):
            gm.resolve_interaction(p1, {"accepted": False})
        elif at in ("SELECT_TARGET", "SELECT_RESOURCE"):
            # コストは払う必要があるため最低限のみ、HEAL は対象不要なので空
            cands = ia.get("selectable_uuids") or [c.uuid for c in ia.get("candidates", [])]
            cons = ia.get("constraints") or {}
            mn = cons.get("min", 0)
            gm.resolve_interaction(p1, {"selected_uuids": cands[:max(mn, 0)], "index": 0})
        else:
            gm.resolve_interaction(p1, {"selected_uuids": [], "index": 0})
        steps += 1
    # 加えない選択ができれば life は据え置きであるべき
    assert len(p1.life) == life_before


# ===========================================================================
# OP03-099 シャーロット・カタクリ ⚠️
# ===========================================================================

def test_op03_099_attack_with_don_buffs_1000():
    """OP03-099 アタック時(ドン!!×1): 自/相手ライフ上1枚を見て上下に置き、その後
    このバトル中パワー+1000。✅"""
    gm, p1, p2, L = build("OP03-099")
    _attach_don(L, p1, 1)
    base = leader_power(p1)        # 付与ドン込みのベース（5000+1000=6000）
    gm.resolve_ability(p1, get_ability(L.master, "ON_ATTACK"), L)
    _drive(gm, p1)
    assert leader_power(p1) == base + 1000


def test_op03_099_no_don_does_not_fire():
    """OP03-099: ドン!!付与が無い（条件【ドン!!×1】未達）なら能力は発動せずパワー据え置き。✅"""
    gm, p1, p2, L = build("OP03-099")
    gm.resolve_ability(p1, get_ability(L.master, "ON_ATTACK"), L)
    _drive(gm, p1)
    assert leader_power(p1) == 5000
    assert gm.active_interaction is None
