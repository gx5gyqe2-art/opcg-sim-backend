"""OP01 リーダーカード効果の挙動テスト（docs/leader_specs/OP01.md 準拠）。

実行:
    OPCG_LOG_SILENT=1 python -m pytest tests/test_leader_op01.py -q -s -p no:cacheprovider

方針（docs/leader_specs/_TEST_GUIDE.md）:
  - 常に「テキスト準拠の正しい挙動」をアサートする。
  - ✅ → 通常テスト / 🐛 → xfail(strict=True) / ⚠️ → 通常 or xfail(strict=False)。
"""
import pytest

from engine_helpers import make_master, make_instance
from leader_test_helpers import (
    build, get_ability, auto_resolve,
    add_char, clear_field,
)
from opcg_sim.src.models.enums import CardType, Color


# ---------------------------------------------------------------------------
# OP01-001 ロロノア・ゾロ
#   【ドン!!×1】【自分のターン中】自分のキャラすべてを、パワー+1000。
# ---------------------------------------------------------------------------

def test_op01_001_buff_all_own_chars_when_don_attached():
    """OP01-001 / 常在バフ(ドン!!×1) / 条件成立: ドン!!付与中は自分キャラ全てに+1000。"""
    gm, p1, p2, L = build("OP01-001")
    clear_field(p1)
    c1 = add_char(p1, power=2000)
    c2 = add_char(p1, power=3000)
    L.attached_don = 1  # 【ドン!!×1】= リーダーに付与ドン1枚
    gm.resolve_ability(p1, get_ability(L.master, "YOUR_TURN"), L)
    auto_resolve(gm, p1)
    assert c1.get_power(True) == 3000
    assert c2.get_power(True) == 4000


def test_op01_001_no_buff_without_don():
    """OP01-001 / 条件不成立: ドン!!が付与されていなければバフは乗らない。"""
    gm, p1, p2, L = build("OP01-001")
    clear_field(p1)
    c1 = add_char(p1, power=2000)
    # attached_don は 0（既定）
    gm.resolve_ability(p1, get_ability(L.master, "YOUR_TURN"), L)
    auto_resolve(gm, p1)
    assert c1.get_power(True) == 2000


def test_op01_001_does_not_buff_opponent():
    """OP01-001 / 対象範囲: 相手のキャラには乗らない（target=SELF）。"""
    gm, p1, p2, L = build("OP01-001")
    clear_field(p1)
    clear_field(p2)
    mine = add_char(p1, power=2000)
    foe = add_char(p2, power=2000)
    L.attached_don = 1
    gm.resolve_ability(p1, get_ability(L.master, "YOUR_TURN"), L)
    auto_resolve(gm, p1)
    assert mine.get_power(True) == 3000
    assert foe.get_power(True) == 2000


# ---------------------------------------------------------------------------
# OP01-002 トラファルガー・ロー  🐛
#   【起動メイン】【ターン1回】②：自分のキャラが5枚いる場合、自分のキャラ1枚を
#   持ち主の手札に戻し、自分の手札から、戻したキャラと異なる色のコスト5以下の
#   キャラカード1枚までを、登場させる。
# ---------------------------------------------------------------------------

def _op01_002_setup():
    gm, p1, p2, L = build("OP01-002")
    clear_field(p1)
    for i in range(5):  # 自分キャラ5枚（全て赤）
        add_char(p1, name=f"C{i}", cost=2, power=2000, colors=["赤"])
    return gm, p1, p2, L


def test_op01_002_play_excludes_same_color_as_bounced():
    """OP01-002 / 起動メイン: 戻したキャラ(赤)と同色のキャラは登場候補に出ない（異色制約）。"""
    gm, p1, p2, L = _op01_002_setup()
    # 手札に異色(青)1枚と同色(赤)1枚を用意。異色制約が効けば登場候補は青のみ。
    blue = make_instance(make_master(card_id="BLUEPLAY", cost=3, type=CardType.CHARACTER),
                         owner=p1.name)
    blue.master.colors[:] = [Color.BLUE]
    red_hand = make_instance(make_master(card_id="REDPLAY", cost=3, type=CardType.CHARACTER),
                             owner=p1.name)  # make_master 既定で赤
    p1.hand += [blue, red_hand]
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    # 1段目: バウンス対象(赤キャラ)を選択
    assert gm.active_interaction["action_type"] == "SELECT_TARGET"
    bounce = [c for c in gm.active_interaction["candidates"]
              if Color.RED in (c.master.colors or [])][0]
    gm.resolve_interaction(p1, {"selected_uuids": [bounce.uuid]})
    # 2段目: 登場候補。赤(=戻したキャラと同色)は候補から除外され、青のみが残るべき。
    assert gm.active_interaction["action_type"] == "SELECT_TARGET"
    cands = gm.active_interaction["candidates"]
    red_cands = [c for c in cands if Color.RED in (c.master.colors or [])]
    assert red_cands == []   # 同色は登場できない（テキスト準拠の正しい挙動）
    assert blue in cands     # 異色は登場できる


def test_op01_002_bounce_then_play_changes_field_by_one():
    """OP01-002 / 起動メイン: キャラ5枚で発動すると 1枚戻し+1枚登場で場の増減は実質+1。"""
    gm, p1, p2, L = _op01_002_setup()
    # 異色（青）コスト5以下キャラを手札に用意し、登場可能にする。
    blue = make_instance(make_master(card_id="BLUEPLAY", cost=3, type=CardType.CHARACTER),
                         owner=p1.name)
    blue.master.colors[:] = [Color.BLUE]   # make_master 既定の赤を上書きし純粋な異色にする
    p1.hand.append(blue)
    before = len(p1.field)
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    # バウンスは赤キャラ、登場は青キャラを明示選択
    assert gm.active_interaction["action_type"] == "SELECT_TARGET"
    red = [c for c in gm.active_interaction["candidates"]
           if Color.RED in (c.master.colors or [])][0]
    gm.resolve_interaction(p1, {"selected_uuids": [red.uuid]})
    assert gm.active_interaction["action_type"] == "SELECT_TARGET"
    gm.resolve_interaction(p1, {"selected_uuids": [blue.uuid]})
    auto_resolve(gm, p1)
    assert blue in p1.field
    assert len(p1.field) == before  # 5→(戻し4)→(登場5)


# ---------------------------------------------------------------------------
# OP01-003 モンキー・D・ルフィ  ✅
#   【起動メイン】【ターン1回】④：自分のコスト5以下の特徴《超新星》か《麦わらの一味》
#   を持つキャラ1枚までを、アクティブにし、そのキャラを、このターン中、パワー+1000。
# ---------------------------------------------------------------------------

def test_op01_003_buff_supernova_char_this_turn():
    """OP01-003 / 起動メイン: 対象特徴・コスト5以下のキャラに +1000(このターン)。"""
    gm, p1, p2, L = build("OP01-003")
    clear_field(p1)
    c = add_char(p1, cost=3, power=4000, traits=["超新星"])
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    auto_resolve(gm, p1)
    assert c.get_power(True) == 5000
    # このターン中のバフ → ターン終了で失効する
    gm.continuous.expire("TURN_END", gm.turn_count)
    assert c.get_power(True) == 4000


def test_op01_003_no_active_candidate_when_trait_absent():
    """OP01-003 / 対象条件: 対象特徴を持たないキャラは ACTIVE 対象の候補にならない。"""
    gm, p1, p2, L = build("OP01-003")
    clear_field(p1)
    add_char(p1, cost=3, power=4000, traits=["その他"])  # 対象特徴を持たない
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    # ACTIVE 段の対象選択（SELECT_TARGET）が立ち上がらない＝選べる対象が無い。
    ia = gm.active_interaction
    assert ia is None or ia.get("action_type") != "SELECT_TARGET" \
        or len(ia.get("candidates", [])) == 0


def test_op01_003_no_active_candidate_for_cost6():
    """OP01-003 / コスト条件: コスト6の麦わらキャラは ACTIVE 対象の候補にならない(コスト5以下)。"""
    gm, p1, p2, L = build("OP01-003")
    clear_field(p1)
    add_char(p1, cost=6, power=4000, traits=["麦わらの一味"])
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    ia = gm.active_interaction
    assert ia is None or ia.get("action_type") != "SELECT_TARGET" \
        or len(ia.get("candidates", [])) == 0


# ---------------------------------------------------------------------------
# OP01-031 光月おでん  ⚠️
#   【起動メイン】【ターン1回】自分の手札から特徴《ワノ国》を持つカード1枚を
#   捨てることができる：自分のドン!!2枚までをアクティブにする。
# ---------------------------------------------------------------------------

def _rest_active_don(player, n):
    for _ in range(n):
        d = player.don_active.pop()
        d.is_rest = True
        player.don_rested.append(d)


def test_op01_031_discard_wano_actives_two_don():
    """OP01-031 / 起動メイン: ワノ国1枚を捨て、レストのドン!!2枚をアクティブにする。"""
    gm, p1, p2, L = build("OP01-031")
    _rest_active_don(p1, 2)  # レストのドン!!を2枚用意
    wano = make_instance(make_master(card_id="WANO", traits=["ワノ国"]), owner=p1.name)
    p1.hand.append(wano)
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    auto_resolve(gm, p1)
    assert wano in p1.trash
    assert len(p1.don_rested) == 0
    assert len(p1.don_active) == 10


def test_op01_031_actives_only_one_when_one_rested():
    """OP01-031 / is_up_to: レストのドン!!が1枚しかなくても発動でき、1枚だけアクティブ。"""
    gm, p1, p2, L = build("OP01-031")
    _rest_active_don(p1, 1)  # レスト1枚のみ
    wano = make_instance(make_master(card_id="WANO", traits=["ワノ国"]), owner=p1.name)
    p1.hand.append(wano)
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    auto_resolve(gm, p1)
    assert len(p1.don_rested) == 0
    assert len(p1.don_active) == 10  # 9 active + 1 復帰


def test_op01_031_no_wano_cannot_pay_cost():
    """OP01-031 / コスト未達: 手札に《ワノ国》が無ければ捨てられず、ドン!!はアクティブにならない。"""
    gm, p1, p2, L = build("OP01-031")
    _rest_active_don(p1, 2)
    # 既定の手札フィラーは特徴を持たない（《ワノ国》無し）
    gm.resolve_ability(p1, get_ability(L.master, "ACTIVATE_MAIN"), L)
    auto_resolve(gm, p1)
    assert len(p1.don_rested) == 2  # 復帰していない
    assert len(p1.don_active) == 8


# ---------------------------------------------------------------------------
# OP01-060 ドンキホーテ・ドフラミンゴ  ⚠️
#   【ドン!!×2】【アタック時】①：自分のデッキの一番上を公開し、そのカードが
#   コスト4以下の特徴《王下七武海》を持つキャラカードの場合、レストで登場させてもよい。
# ---------------------------------------------------------------------------

def _deck_top(player, **master_kw):
    top = make_instance(make_master(type=CardType.CHARACTER, **master_kw), owner=player.name)
    player.deck.insert(0, top)
    return top


def test_op01_060_reveal_and_play_when_match():
    """OP01-060 / アタック時: デッキトップが王下七武海・コスト4以下キャラなら、レストで登場可。"""
    gm, p1, p2, L = build("OP01-060")
    L.attached_don = 2  # 【ドン!!×2】
    top = _deck_top(p1, card_id="SHICHI", cost=4, traits=["王下七武海"], power=3000)
    gm.resolve_ability(p1, get_ability(L.master, "ON_ATTACK"), L)
    auto_resolve(gm, p1)  # 任意効果を受諾→対象選択
    assert top in p1.field
    assert top.is_rest is True


def test_op01_060_cost5_not_played():
    """OP01-060 / 条件不成立: デッキトップがコスト5なら登場できない（公開のみ）。"""
    gm, p1, p2, L = build("OP01-060")
    L.attached_don = 2
    top = _deck_top(p1, card_id="SHICHI5", cost=5, traits=["王下七武海"], power=3000)
    gm.resolve_ability(p1, get_ability(L.master, "ON_ATTACK"), L)
    auto_resolve(gm, p1)
    assert top not in p1.field
    assert top in p1.deck


def test_op01_060_wrong_trait_not_played():
    """OP01-060 / 条件不成立: 王下七武海でないキャラは登場できない（公開のみ）。"""
    gm, p1, p2, L = build("OP01-060")
    L.attached_don = 2
    top = _deck_top(p1, card_id="NOPE", cost=3, traits=["麦わらの一味"], power=3000)
    gm.resolve_ability(p1, get_ability(L.master, "ON_ATTACK"), L)
    auto_resolve(gm, p1)
    assert top not in p1.field
    assert top in p1.deck


# ---------------------------------------------------------------------------
# OP01-061 カイドウ  🐛
#   【ドン!!×1】【自分のターン中】【ターン1回】相手のキャラがKOされた時、
#   ドン!!デッキからドン!!1枚までを、アクティブで追加する。
# ---------------------------------------------------------------------------

def test_op01_061_has_on_ko_trigger():
    """OP01-061 / 誘発タイミング: 能力は『相手キャラKO時』=ON_KO で誘発するべき。"""
    gm, p1, p2, L = build("OP01-061")
    triggers = [a.trigger.name if hasattr(a.trigger, "name") else str(a.trigger)
                for a in (L.master.abilities or [])]
    # 正しくは KO 起因の誘発（ON_KO）が存在するはず。
    assert "ON_KO" in triggers


# ---------------------------------------------------------------------------
# OP01-062 クロコダイル  🐛
#   【ドン!!×1】自分がイベントを発動した時、自分の手札が4枚以下でかつ、
#   このターン中、このリーダーの効果でカードを引いていない場合、カード1枚を引くことができる。
# ---------------------------------------------------------------------------

def test_op01_062_triggered_by_event_not_activate_main():
    """OP01-062 / 誘発タイミング: 能力は『イベント発動時』に誘発すべきで、起動メインではない。"""
    gm, p1, p2, L = build("OP01-062")
    ab = L.master.abilities[0]
    trig = ab.trigger.name if hasattr(ab.trigger, "name") else str(ab.trigger)
    assert trig != "ACTIVATE_MAIN"  # イベント誘発であるべき


def test_op01_062_has_not_drawn_this_turn_condition():
    """OP01-062 / 重複防止条件: 『このターン中このリーダー効果で未ドロー』条件が AST に存在すべき。"""
    gm, p1, p2, L = build("OP01-062")
    ab = L.master.abilities[0]
    cond = ab.condition

    def _flatten(c):
        if c is None:
            return []
        out = [c]
        for sub in (getattr(c, "args", None) or []):
            out.extend(_flatten(sub))
        return out

    types = {getattr(getattr(c, "type", None), "name", "")
             for c in _flatten(cond)}
    # 現状の条件型は {AND, HAS_DON, HAND_COUNT} のみで、
    # 「このターン中このリーダー効果で引いていない」を表す条件型
    # (TURN_LIMIT / CONTEXT / PREV_ACTION / SOURCE_STATE 等) が存在しない。
    # 「このターン中…引いていない」は EVENT_THIS_TURN(LEADER_DREW_BY_EFFECT, LT) で表す。
    draw_limit_types = {"TURN_LIMIT", "CONTEXT", "PREV_ACTION", "SOURCE_STATE", "GENERIC", "EVENT_THIS_TURN"}
    assert types & draw_limit_types


# ---------------------------------------------------------------------------
# OP01-091 キング  ✅
#   【自分のターン中】自分の場にドン!!が10枚ある場合、相手のキャラすべてを、パワー-1000。
# ---------------------------------------------------------------------------

def test_op01_091_debuff_all_opponent_chars_when_don_10():
    """OP01-091 / 常在: 自分の場のドン!!が10枚なら、相手キャラ全てを -1000。"""
    gm, p1, p2, L = build("OP01-091")
    clear_field(p2)
    e1 = add_char(p2, power=2000)
    e2 = add_char(p2, power=3000)
    assert len(p1.don_active) + len(p1.don_rested) == 10  # 既定盤面はドン10枚
    gm.resolve_ability(p1, get_ability(L.master, "YOUR_TURN"), L)
    auto_resolve(gm, p1)
    assert e1.get_power(True) == 1000
    assert e2.get_power(True) == 2000


def test_op01_091_no_debuff_when_don_below_10():
    """OP01-091 / 条件不成立: 自分の場のドン!!が9枚なら相手キャラに変化なし(=10 厳密)。"""
    gm, p1, p2, L = build("OP01-091")
    clear_field(p2)
    e1 = add_char(p2, power=2000)
    # ドン!!を1枚ドンデッキへ戻して 9 枚にする
    d = p1.don_active.pop()
    p1.don_deck.append(d)
    assert len(p1.don_active) + len(p1.don_rested) == 9
    gm.resolve_ability(p1, get_ability(L.master, "YOUR_TURN"), L)
    auto_resolve(gm, p1)
    assert e1.get_power(True) == 2000


def test_op01_091_does_not_debuff_own_chars():
    """OP01-091 / 対象範囲: 自分のキャラには -1000 が乗らない（target=OPPONENT）。"""
    gm, p1, p2, L = build("OP01-091")
    clear_field(p1)
    clear_field(p2)
    mine = add_char(p1, power=2000)
    foe = add_char(p2, power=2000)
    gm.resolve_ability(p1, get_ability(L.master, "YOUR_TURN"), L)
    auto_resolve(gm, p1)
    assert mine.get_power(True) == 2000
    assert foe.get_power(True) == 1000
