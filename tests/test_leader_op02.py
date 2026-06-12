"""OP02 リーダーカード効果の pytest 化。

仕様書: docs/leader_specs/OP02.md
対象8枚: OP02-001 / OP02-002 / OP02-025 / OP02-026 / OP02-049 /
        OP02-071 / OP02-072 / OP02-093

方針（_TEST_GUIDE.md）:
  - 常にテキスト準拠の「正しい挙動」をアサートする。
  - ✅ → 通常テスト / 🐛 → xfail(strict=True) / ⚠️ → 通常（不安定なら xfail(strict=False)）。

実行:
    OPCG_LOG_SILENT=1 python -m pytest tests/test_leader_op02.py -q -s -p no:cacheprovider
"""
import pytest

from leader_test_helpers import (
    build, get_ability, auto_resolve,
    select_uuids,
    add_char, make_char, clear_field, set_life,
    leader_power, zone_counts,
)
from opcg_sim.src.models.models import DonInstance


def _attach_don_to_leader(player, leader, n=1):
    """【ドン!!×N】条件用に、リーダーへドン!!を n 枚付与する。"""
    for _ in range(n):
        d = player.don_active.pop()
        att = DonInstance(owner_id=player.name, attached_to=leader.uuid)
        player.don_attached_cards.append(att)
    leader.attached_don = (leader.attached_don or 0) + n


def _drive(gm, player, victim_uuid=None):
    """active_interaction を解決まで駆動。victim 指定時は対象選択をそのカードに固定。"""
    steps = 0
    while gm.active_interaction and steps < 10:
        ia = gm.active_interaction
        at = ia.get("action_type", "")
        cands = ia.get("selectable_uuids") or [c.uuid for c in ia.get("candidates", [])]
        if at in ("SELECT_TARGET", "SELECT_RESOURCE") and victim_uuid and victim_uuid in cands:
            sel = [victim_uuid]
        else:
            sel = cands[:1]
        gm.resolve_interaction(player, {"selected_uuids": sel, "accepted": True, "index": 0})
        steps += 1
    return steps


# ---------------------------------------------------------------------------
# OP02-001 エドワード・ニューゲート ✅
# 【自分のターン終了時】自分のライフの上から1枚を手札に加える。
# ---------------------------------------------------------------------------

def test_op02_001_turn_end_moves_top_life_to_hand():
    """OP02-001 【自ターン終了時】: ライフ上1枚が手札へ（ライフ-1・手札+1）。"""
    gm, p1, p2, L = build("OP02-001")
    set_life(p1, 5)
    before = zone_counts(p1)
    ab = get_ability(L.master, "TURN_END")
    gm.resolve_ability(p1, ab, L)
    auto_resolve(gm, p1)
    after = zone_counts(p1)
    assert after["life"] == before["life"] - 1
    assert after["hand"] == before["hand"] + 1


def test_op02_001_turn_end_no_life_is_noop():
    """OP02-001 【自ターン終了時】: ライフ0枚なら対象不在で移動なし。"""
    gm, p1, p2, L = build("OP02-001")
    set_life(p1, 0)
    hand_before = zone_counts(p1)["hand"]
    ab = get_ability(L.master, "TURN_END")
    gm.resolve_ability(p1, ab, L)
    auto_resolve(gm, p1)
    assert zone_counts(p1)["life"] == 0
    assert zone_counts(p1)["hand"] == hand_before


# ---------------------------------------------------------------------------
# OP02-002 モンキー・D・ガープ 🐛
# 【自分のターン中】このリーダーか自分のキャラにドン‼が付与された時、
# 相手のコスト7以下のキャラ1枚までを、このターン中、コスト-1。
# → 実装は ATTACH_DON（相手キャラへドン付与＝パワー+1000）に誤実装。
# ---------------------------------------------------------------------------

def test_op02_002_reduces_opponent_cost_by_one():
    """OP02-002: 相手コスト7以下キャラのコストが-1される（テキスト準拠の正しい挙動）。"""
    gm, p1, p2, L = build("OP02-002")
    clear_field(p2)
    victim = add_char(p2, cost=5, power=5000)
    ab = get_ability(L.master, "YOUR_TURN")
    gm.resolve_ability(p1, ab, L)
    auto_resolve(gm, p1)
    # 正しくは cost 5→4。誤実装ではコスト不変かつ attached_don で強化される。
    assert victim.current_cost == 4
    assert victim.attached_don == 0


def test_op02_002_cost8_out_of_range_untouched():
    """OP02-002: コスト8キャラ（7以下でない）は対象外で何も起きない。"""
    gm, p1, p2, L = build("OP02-002")
    clear_field(p2)
    big = add_char(p2, cost=8, power=8000)
    ab = get_ability(L.master, "YOUR_TURN")
    gm.resolve_ability(p1, ab, L)
    auto_resolve(gm, p1)
    assert big.current_cost == 8
    assert big.attached_don == 0


# ---------------------------------------------------------------------------
# OP02-025 錦えもん ⚠️
# 【起動メイン】【ターン1回】自分のキャラが1枚以下の場合、このターン中、次に
# 自分が手札から登場させるコスト3以上《ワノ国》キャラの支払うコストは1少なくなる。
# ---------------------------------------------------------------------------

def test_op02_025_discounts_next_wano_in_hand():
    """OP02-025: 自キャラ≤1枚で起動 → 手札のコスト3以上《ワノ国》のコストが-1。"""
    gm, p1, p2, L = build("OP02-025")
    clear_field(p1)  # 0枚 ≤ 1 → 条件成立
    wano = make_char(p1, name="ワノ国子", cost=4, power=4000, traits=["ワノ国"])
    p1.hand.append(wano)
    ab = get_ability(L.master, "ACTIVATE_MAIN")
    gm.resolve_ability(p1, ab, L)
    auto_resolve(gm, p1)
    assert wano.current_cost == 3  # 4 - 1


def test_op02_025_condition_not_met_when_two_chars():
    """OP02-025: 自キャラ2枚（>1）なら条件未達で割引されない。"""
    gm, p1, p2, L = build("OP02-025")
    clear_field(p1)
    add_char(p1, cost=1)
    add_char(p1, cost=1)
    wano = make_char(p1, name="ワノ国子", cost=4, power=4000, traits=["ワノ国"])
    p1.hand.append(wano)
    ab = get_ability(L.master, "ACTIVATE_MAIN")
    gm.resolve_ability(p1, ab, L)
    auto_resolve(gm, p1)
    assert wano.current_cost == 4  # 未割引


def test_op02_025_cost2_wano_below_min_no_discount():
    """OP02-025: コスト2《ワノ国》（コスト3未満）は割引対象外。"""
    gm, p1, p2, L = build("OP02-025")
    clear_field(p1)
    wano = make_char(p1, name="安ワノ", cost=2, power=2000, traits=["ワノ国"])
    p1.hand.append(wano)
    ab = get_ability(L.master, "ACTIVATE_MAIN")
    gm.resolve_ability(p1, ab, L)
    auto_resolve(gm, p1)
    assert wano.current_cost == 2  # 未割引


# ---------------------------------------------------------------------------
# OP02-026 サンジ 🐛
# 【ターン1回】自分が元々の効果のないキャラを手札から登場させた時、自分のキャラが
# 3枚以下の場合、自分のドン‼2枚までを、アクティブにする。
# → 条件が HAND_COUNT(手札枚数)に誤実装（本来 FIELD_COUNT 場のキャラ枚数）。
#   トリガーも登場時→ACTIVATE_MAIN。
# ---------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason="OP02-026: 条件『自分のキャラ3枚以下』が HAND_COUNT(手札枚数)に誤実装。場のキャラ枚数(FIELD_COUNT)であるべき")
def test_op02_026_activates_two_don_when_field_le3():
    """OP02-026: 場のキャラ≤3枚なら、起動でレストドン2枚がアクティブになる（テキスト準拠）。"""
    gm, p1, p2, L = build("OP02-026")
    clear_field(p1)
    add_char(p1, cost=1)
    add_char(p1, cost=1)  # 場2枚 ≤ 3 → 条件成立（手札は既定5枚のまま=HAND誤実装なら未達）
    # レストドン3枚を用意
    for _ in range(3):
        d = p1.don_active.pop()
        d.is_rest = True
        p1.don_rested.append(d)
    active_before = len(p1.don_active)
    ab = get_ability(L.master, "ACTIVATE_MAIN")
    gm.resolve_ability(p1, ab, L)
    auto_resolve(gm, p1)
    assert len(p1.don_active) == active_before + 2  # 2枚アクティブ化


def test_op02_026_condition_not_met_when_field_over3():
    """OP02-026: 場のキャラ4枚（>3）なら条件未達でアクティブ化されない。"""
    gm, p1, p2, L = build("OP02-026")
    clear_field(p1)
    for _ in range(4):
        add_char(p1, cost=1)  # 場4枚 > 3 → 条件未達
    for _ in range(3):
        d = p1.don_active.pop()
        d.is_rest = True
        p1.don_rested.append(d)
    active_before = len(p1.don_active)
    ab = get_ability(L.master, "ACTIVATE_MAIN")
    gm.resolve_ability(p1, ab, L)
    auto_resolve(gm, p1)
    assert len(p1.don_active) == active_before  # アクティブ化なし


# ---------------------------------------------------------------------------
# OP02-049 エンポリオ・イワンコフ ✅
# 【自分のターン終了時】自分の手札が0枚の場合、カード2枚を引く。
# ---------------------------------------------------------------------------

def test_op02_049_draws_two_when_hand_empty():
    """OP02-049 【自ターン終了時】: 手札0枚なら2枚ドロー。"""
    gm, p1, p2, L = build("OP02-049")
    p1.hand = []
    deck_before = zone_counts(p1)["deck"]
    ab = get_ability(L.master, "TURN_END")
    gm.resolve_ability(p1, ab, L)
    auto_resolve(gm, p1)
    assert zone_counts(p1)["hand"] == 2
    assert zone_counts(p1)["deck"] == deck_before - 2


def test_op02_049_no_draw_when_hand_nonempty():
    """OP02-049 【自ターン終了時】: 手札1枚以上なら条件未達でドローなし。"""
    gm, p1, p2, L = build("OP02-049")
    hand_before = zone_counts(p1)["hand"]  # 既定5枚
    assert hand_before >= 1
    ab = get_ability(L.master, "TURN_END")
    gm.resolve_ability(p1, ab, L)
    auto_resolve(gm, p1)
    assert zone_counts(p1)["hand"] == hand_before


# ---------------------------------------------------------------------------
# OP02-071 マゼラン ⚠️
# 【自分のターン中】【ターン1回】場のドン‼がドン‼デッキに戻された時、
# このリーダーは、このターン中、パワー+1000。
# ---------------------------------------------------------------------------

def test_op02_071_leader_gets_plus_1000():
    """OP02-071: 発動でこのリーダーがこのターン中パワー+1000。"""
    gm, p1, p2, L = build("OP02-071")
    base = leader_power(p1)
    ab = get_ability(L.master, "YOUR_TURN")
    gm.resolve_ability(p1, ab, L)
    auto_resolve(gm, p1)
    assert leader_power(p1) == base + 1000


def test_op02_071_turn_once_blocks_second_buff():
    """OP02-071 【ターン1回】: 同一ターンの2回目は発動せず、+1000は重ねられない。"""
    gm, p1, p2, L = build("OP02-071")
    base = leader_power(p1)
    ab = get_ability(L.master, "YOUR_TURN")
    gm.resolve_ability(p1, ab, L)
    auto_resolve(gm, p1)
    gm.resolve_ability(p1, ab, L)  # 2回目
    auto_resolve(gm, p1)
    assert leader_power(p1) == base + 1000  # 2000 にはならない


# ---------------------------------------------------------------------------
# OP02-072 ゼット ✅
# 【アタック時】ドン‼-4(自分の場のドン‼を戻せる)：相手のコスト3以下のキャラ1枚までを、
# KOする。その後、このリーダーは、このターン中、パワー+1000。
# ---------------------------------------------------------------------------

def test_op02_072_ko_opponent_and_buff_leader():
    """OP02-072 【アタック時】: ドン4返却・相手コスト3以下KO・リーダー+1000。"""
    gm, p1, p2, L = build("OP02-072")
    clear_field(p2)
    victim = add_char(p2, cost=2, power=2000)
    don_before = len(p1.don_active)
    base = leader_power(p1)
    ab = get_ability(L.master, "ON_ATTACK")
    gm.resolve_ability(p1, ab, L)
    auto_resolve(gm, p1)
    assert len(p1.don_active) == don_before - 4   # ドン-4
    assert victim not in p2.field                 # KO
    assert victim in p2.trash
    assert leader_power(p1) == base + 1000         # 後続バフ


def test_op02_072_cost4_not_ko_but_buff_still_applies():
    """OP02-072: 相手がコスト4のみ（KO対象外）でも、その後の+1000は適用される。"""
    gm, p1, p2, L = build("OP02-072")
    clear_field(p2)
    safe = add_char(p2, cost=4, power=4000)
    base = leader_power(p1)
    ab = get_ability(L.master, "ON_ATTACK")
    gm.resolve_ability(p1, ab, L)
    auto_resolve(gm, p1)
    assert safe in p2.field                # コスト4はKOされない
    assert leader_power(p1) == base + 1000  # +1000は適用


def test_op02_072_no_target_still_buffs():
    """OP02-072: 相手キャラ0枚でもKO0枚、その後リーダー+1000。"""
    gm, p1, p2, L = build("OP02-072")
    clear_field(p2)
    base = leader_power(p1)
    don_before = len(p1.don_active)
    ab = get_ability(L.master, "ON_ATTACK")
    gm.resolve_ability(p1, ab, L)
    auto_resolve(gm, p1)
    assert len(p1.don_active) == don_before - 4
    assert leader_power(p1) == base + 1000


# ---------------------------------------------------------------------------
# OP02-093 スモーカー 🐛
# 【ドン‼×1】【起動メイン】【ターン1回】相手のキャラ1枚までを、このターン中、コスト-1。
# その後、コスト0のキャラがいる場合、このリーダーは、このターン中、パワー+1000。
# → 後続バフ条件「コスト0のキャラがいる場合」が「自分の場のキャラ数≥1」に誤実装。
# ---------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason="OP02-093: 後続+1000の条件『コスト0のキャラがいる』が『自分の場のキャラ(cost0)』に誤実装。相手のコスト0キャラを数えないため発動しない")
def test_op02_093_reduces_cost_and_buffs_when_cost0_exists():
    """OP02-093: 相手コスト1→0で『コスト0キャラ存在』成立→相手-1かつリーダー+1000（テキスト準拠）。"""
    gm, p1, p2, L = build("OP02-093")
    clear_field(p1)  # 自場は空（自場のコスト0誤判定を排除）
    clear_field(p2)
    victim = add_char(p2, cost=1, power=1000)
    _attach_don_to_leader(p1, L, 1)  # 【ドン!!×1】を満たす
    base = leader_power(p1)           # 付与済みドンの +1000 を含む
    ab = get_ability(L.master, "ACTIVATE_MAIN")
    gm.resolve_ability(p1, ab, L)
    _drive(gm, p1, victim_uuid=victim.uuid)
    assert victim.current_cost == 0          # コスト-1
    # コスト0キャラ（相手）が存在 → 本来 +1000。誤実装では相手のコスト0を数えず加算なし。
    assert leader_power(p1) == base + 1000


def test_op02_093_no_buff_when_no_cost0_char():
    """OP02-093: 相手コスト3→2でコスト0キャラ不在→リーダー+1000なし（前半のコスト-1は適用）。"""
    gm, p1, p2, L = build("OP02-093")
    clear_field(p1)  # 自場にキャラ無し（コスト0キャラもどこにも無い）
    clear_field(p2)
    victim = add_char(p2, cost=3, power=3000)
    _attach_don_to_leader(p1, L, 1)
    base = leader_power(p1)
    ab = get_ability(L.master, "ACTIVATE_MAIN")
    gm.resolve_ability(p1, ab, L)
    _drive(gm, p1, victim_uuid=victim.uuid)
    assert victim.current_cost == 2           # コスト-1（前半は正しい）
    assert leader_power(p1) == base            # コスト0キャラ不在 → 加算なし


def test_op02_093_no_don_cannot_activate():
    """OP02-093 【ドン!!×1】: リーダーへのドン付与が無ければ発動できず効果なし。"""
    gm, p1, p2, L = build("OP02-093")
    clear_field(p2)
    victim = add_char(p2, cost=1, power=1000)
    base = leader_power(p1)
    # ドン未付与（attached_don=0）→ 条件未達
    ab = get_ability(L.master, "ACTIVATE_MAIN")
    gm.resolve_ability(p1, ab, L)
    _drive(gm, p1, victim_uuid=victim.uuid)
    assert victim.current_cost == 1           # コスト不変
    assert leader_power(p1) == base            # バフなし
