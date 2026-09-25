"""`search_price.py`（T68・探す能力の価格＝取れる札の `max(ΔH, ΔG)` の期待値）の算術と配線を固める。

**絞り込み**（特徴・「名前かイベント」＝`NAME_OR_TYPE`・コスト上限）・**空振り**（合う札が無ければ 0）・
**ユーザの 3 条件**（手札が小物やバニラだけなら上がる／最終盤まで使う札が揃っていれば下がる／カウンターが要るなら
カウンター札で上がる）・**`effect_value` の切替**（`plan` は状態が在るときだけ・無ければ `sel` に落ちる・`sel(k)` と μ は
二重に足さない）。
"""
import os
import sys

import pytest

pytestmark = pytest.mark.cpu_infra

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bootstrap  # noqa: F401,E402

_SCRIPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import effect_value as EV  # noqa: E402
import search_price as SP  # noqa: E402
from theory_order import KO_P, MU  # noqa: E402


class _Cards:
    """試験用のカード表（`plan_labels.Cards.info` と同じ欄）。"""

    def __init__(self, table):
        self._t = table

    def info(self, cid):
        return self._t.get(cid)


_TABLE = {
    "SMALL": {"cost": 2, "power": 3000, "counter": 1000, "event": False, "stage": False, "leader": False},
    "MID": {"cost": 4, "power": 5000, "counter": 1000, "event": False, "stage": False, "leader": False},
    "BIG": {"cost": 7, "power": 8000, "counter": 0, "event": False, "stage": False, "leader": False},
    "CNT": {"cost": 1, "power": 0, "counter": 2000, "event": False, "stage": False, "leader": False},
    "EVT": {"cost": 1, "power": 0, "counter": 0, "event": True, "stage": False, "leader": False},
    "LDR": {"cost": 0, "power": 5000, "counter": 0, "event": False, "stage": False, "leader": True},
}
_IDENT = {"SMALL": {"traits": ["赤髪海賊団"], "names": ["ヤソップ"], "colors": ["RED"]},
          "MID": {"traits": ["赤髪海賊団"], "names": ["ベックマン"], "colors": ["RED"]},
          "BIG": {"traits": ["赤髪海賊団"], "names": ["シャンクス"], "colors": ["RED"]},
          "CNT": {"traits": ["麦わらの一味"], "names": ["サンジ"], "colors": ["RED"]},
          "EVT": {"traits": [], "names": ["ゴムゴムの銃"], "colors": ["RED"]},
          "LDR": {"traits": ["赤髪海賊団"], "names": ["シャンクス"], "colors": ["RED"]}}


@pytest.fixture(autouse=True)
def _stub(monkeypatch):
    """素性は同梱 DB ではなく表から・`v` はコストだけで決まる簡単な形（出せる札ほど価値がある）。"""
    monkeypatch.setattr(SP, "card_identity", lambda cid: _IDENT.get(cid))
    monkeypatch.setattr(SP, "use_value", lambda cid, info, olp, r: (0.0 if (info or {}).get("event") else 0.02 * float((info or {}).get("cost") or 0)))
    SP._GAIN.clear()
    before = EV.SEARCH_PRICE_MODE
    yield
    EV.set_search_price_mode(before)


def _item(cid, v=None):
    info = _TABLE[cid]
    return {"cid": cid, "cost": float(info["cost"]), "v": (0.02 * info["cost"] if v is None else v),
            "counter": float(info["counter"]), "event": bool(info["event"])}


def _ctx(hand, caps, xs=(), take=0.087, deck=None):
    return {"hand_items": [_item(c) for c in hand], "caps": list(caps), "xs": list(xs), "take": take,
            "deck": list(deck if deck is not None else ["SMALL", "MID", "BIG", "CNT", "EVT"] * 4), "olp": 5000.0, "r": 4.0, "field": []}


def test_eligible_cards_follow_the_filter_and_never_include_the_leader():
    cards = _Cards(_TABLE)
    deck = ["SMALL", "MID", "BIG", "CNT", "EVT", "LDR"]
    assert SP.eligible_deck_cards({"traits": ["赤髪海賊団"]}, deck, cards) == ["SMALL", "MID", "BIG"]
    # 「サンジ」かイベント（`NAME_OR_TYPE`）
    assert SP.eligible_deck_cards({"names": ["サンジ"], "card_type": ["EVENT"], "flags": ["NAME_OR_TYPE"]}, deck, cards) == ["CNT", "EVT"]
    # 旗が無ければ両方を満たす札だけ（イベントの「サンジ」は無い）
    assert SP.eligible_deck_cards({"names": ["サンジ"], "card_type": ["EVENT"]}, deck, cards) == []
    assert SP.eligible_deck_cards({"traits": ["赤髪海賊団"], "cost_max": 4}, deck, cards) == ["SMALL", "MID"]
    assert SP.eligible_deck_cards({}, deck, cards) == ["SMALL", "MID", "BIG", "CNT", "EVT"]


def test_search_actions_finds_the_look_and_the_move_to_hand():
    look = {"type": "LOOK", "value": {"base": 5}, "target": None}
    take = {"type": "MOVE_CARD", "destination": "HAND", "target": {"zone": "TEMP", "traits": ["赤髪海賊団"], "count": 1}}
    rest = {"type": "DECK_BOTTOM", "target": {"zone": "TEMP", "count": -1}}
    k, target, act = SP.search_actions([look, take, rest])
    assert k == 5 and target["traits"] == ["赤髪海賊団"] and act is take
    assert SP.search_actions([look, rest]) is None                                            # 手札に加えない
    life = {"type": "MOVE_CARD", "destination": "LIFE", "target": {"zone": "TEMP", "count": 1}}
    assert SP.search_actions([look, life, rest]) is None                                      # ライフに加えるのは従来の価格


def test_the_search_value_is_zero_when_nothing_eligible_and_the_expected_best_gain_otherwise():
    cards = _Cards(_TABLE)
    ctx = _ctx(["BIG"], [4, 6, 7, 10])
    assert SP.search_value(ctx, 5, {"traits": ["無い特徴"]}, cards) == 0.0
    assert SP.search_value({**ctx, "deck": []}, 5, {}, cards) == 0.0
    # 4 コストが 1 枚だけ・k = デッキ全部なら必ず取れる＝その札の ΔH（t=1 に出せる → s·v）
    deck = ["EVT"] * 9 + ["MID"]
    v = SP.search_value(_ctx(["BIG"], [4, 6, 7, 10], deck=deck), 10, {"card_type": ["CHARACTER"]}, cards)
    assert v == pytest.approx(0.02 * 4, abs=1e-9)                                            # 今のターン（cap 4）に出せる → 満額
    # k = 1 なら 1/10 でしか当たらない＝0.08 × 0.1（サンプル 400 の平均・幅を取る）
    v1 = SP.search_value(_ctx(["BIG"], [4, 6, 7, 10], deck=deck), 1, {"card_type": ["CHARACTER"]}, cards, samples=400)
    assert 0.003 < v1 < 0.02


def test_a_hand_of_vanillas_or_small_bodies_raises_the_search_value_and_a_complete_hand_lowers_it():
    """**ユーザの条件（2026-09-17）**: 出せる札が無い・小さい体しか無い手札では上がり、最終盤まで使う札が揃っていれば下がる。"""
    cards = _Cards(_TABLE)
    deck = ["SMALL", "MID", "BIG"] * 4
    tgt = {"traits": ["赤髪海賊団"]}
    caps = [4, 5, 6, 10]                                     # R = 4 → 後ろ枠 10（1 ターン）
    poor = SP.search_value(_ctx(["EVT", "EVT"], caps, deck=deck), 5, tgt, cards)               # 出せる札が無い（イベントは v 0）
    small = SP.search_value(_ctx(["SMALL"], caps, deck=deck), 5, tgt, cards)                   # 2 コストの小物だけ
    full = SP.search_value(_ctx(["MID"] * 4 + ["SMALL"] * 3 + ["BIG"], caps, deck=deck), 5, tgt, cards)   # 枠が埋まっている
    assert poor > full and small > full                      # 出せる札が無い／小物だけ → 高い・揃っていれば低い
    assert full < 0.5 * poor                                 # 押し出しの分だけ残る（0 にはならない）
    assert poor > 0.06 and small > 0.06                      # ほぼ毎回 4 コスト（0.08）か 2 コスト（0.04）が取れる


def test_a_counter_card_is_worth_the_guard_saving_when_attacks_are_coming():
    """カウンター値を増やす札（`CNT`・v が低くカウンター 2000）は来る攻撃が在れば `ΔG` で立つ。"""
    cards = _Cards(_TABLE)
    deck = ["CNT"] * 10
    tgt = {"names": ["サンジ"], "card_type": ["EVENT"], "flags": ["NAME_OR_TYPE"]}
    calm = SP.search_value(_ctx(["BIG"], [0, 0, 0, 0], xs=[], deck=deck), 4, tgt, cards)
    under_fire = SP.search_value(_ctx(["BIG"], [0, 0, 0, 0], xs=[1000.0], deck=deck), 4, tgt, cards)
    assert calm == 0.0                                       # 出せず（ドン 0）攻撃も来ない → 0
    assert under_fire == pytest.approx(0.087 - 0.02, abs=1e-9)   # 受ける損 − 切る札の v（t=0 は割引なし）


def test_playing_the_searcher_removes_it_from_the_hand_and_spends_its_cost():
    ctx = _ctx(["SMALL", "MID"], [4, 6, 7, 10])
    after = SP.ctx_after_play(ctx, "SMALL", 2)
    assert [it["cid"] for it in after["hand_items"]] == ["MID"] and after["caps"] == [2, 6, 7, 10]
    assert SP.ctx_after_play(ctx, None, 0)["caps"] == [4, 6, 7, 10]
    assert sorted(SP.remaining_deck(["A", "A", "B"], ["A"], ["B"])) == ["A"]


def _search_ability():
    take = {"type": "MOVE_CARD", "destination": "HAND", "target": {"zone": "TEMP", "traits": ["赤髪海賊団"], "count": 1, "player": "SELF"}}
    rest = {"type": "DECK_BOTTOM", "target": {"zone": "TEMP", "count": -1, "player": "SELF"}}
    return {"trigger": "ON_PLAY", "effect": {"type": "LOOK", "value": {"base": 5}, "target": None, "sub_effect": [take, rest]}}


def test_effect_value_uses_the_plan_price_only_when_the_state_carries_a_search_context():
    """`plan` は `st["search_ctx"]` が在るときだけ・無ければ `sel(k)`（旧）に落ちる・`sel` なら常に旧。"""
    ab = _search_ability()
    old = MU + EV._sel_premium(5)
    EV.set_search_price_mode("plan")
    v, unp = EV.ability_value(ab, st=None)
    assert unp == [] and v == pytest.approx(old)
    v, unp = EV.ability_value(ab, st={"my_life": 4})
    assert v == pytest.approx(old)                                                            # 状態はあるが探す状態は無い
    cards = _Cards(_TABLE)
    ctx = _ctx(["EVT"], [4, 6, 7, 10], deck=["SMALL", "MID", "BIG"] * 4)
    ctx["cards"] = cards
    v, unp = EV.ability_value(ab, st={"search_ctx": ctx}, card={"card_id": "EVT", "cost": 1})
    assert unp == []
    want = SP.search_value(ctx, 5, ab["effect"]["sub_effect"][0]["target"], cards, played_cid="EVT", played_cost=1)
    assert v == pytest.approx(want) and v != pytest.approx(old)                              # `sel(k)` も μ も足さない
    EV.set_search_price_mode("sel")
    v, _ = EV.ability_value(ab, st={"search_ctx": ctx}, card={"card_id": "EVT", "cost": 1})
    assert v == pytest.approx(old)
    with pytest.raises(ValueError):
        EV.set_search_price_mode("guess")


def test_the_gain_cache_and_the_draw_baseline():
    cards = _Cards(_TABLE)
    ctx = _ctx(["EVT"], [4, 6, 7, 10], deck=["SMALL"] * 10)
    assert SP.draw_value(ctx, cards) == pytest.approx(0.02 * 2, abs=1e-9)                     # 全部 SMALL → 今出せる → 満額
    assert len(SP._GAIN) == 1                                                                 # 同じ手札・同じ札は 1 回だけ
    assert 0.0 < 1.0 - KO_P < 1.0


def test_play_from_hand_is_priced_by_the_matching_card_in_hand_now():
    """**T70**: 「手札から出す」効果は今の手札に合う札が在ればその値（体 ＋ 登場時効果 − μ・負なら出さない＝0）・無ければ 0・
    出す札そのものは除く・`full` なら旧（満額）。"""
    import hand_spend
    cards = _Cards(_TABLE)
    act = {"type": "PLAY_CARD", "target": {"zone": "HAND", "card_type": ["CHARACTER"], "cost_max": 4, "count": 1, "player": "SELF"}}
    look = {"type": "LOOK", "value": {"base": 5}, "target": None}
    assert SP.play_from_hand_target([look, act]) is act["target"] and SP.play_from_hand_target([look]) is None
    deck_act = {"type": "PLAY_CARD", "target": {"zone": "DECK", "count": 1}}
    assert SP.play_from_hand_target([deck_act]) is None
    before = EV.PLAY_NOW_MODE
    try:
        old = EV.action_value(act)                                                          # 状態なし＝旧（満額）
        assert old == pytest.approx(EV.field_value(act["target"]) - MU)
        ctx = _ctx(["MID", "BIG"], [4, 6, 7, 10]); ctx["cards"] = cards
        st = {"search_ctx": ctx}
        fv = hand_spend.free_value("MID", _TABLE["MID"], 5000.0, 4.0)
        assert fv > MU
        assert EV.action_value(act, st=st) == pytest.approx(fv - MU)                        # 4 コストの体が在る → その値
        assert EV.action_value(act, st={"search_ctx": {**ctx, "hand_items": [_item("BIG")]}}) == 0.0   # 合う札が無い → 0
        # 体が μ に届かない小物（3000）は出しても損＝「1 枚まで」なので 0
        assert EV.action_value(act, st={"search_ctx": {**ctx, "hand_items": [_item("SMALL")]}}) == 0.0
        # 出す札そのもの（MID を出してその効果で MID を出す）は除く
        assert EV.action_value(act, st=st, card={"card_id": "MID", "cost": 4}) == 0.0
        assert SP.eligible_hand_cards({"cost_max": 4}, ctx["hand_items"], cards, skip_cid="MID") == []
        EV.set_play_now_mode("full")
        assert EV.action_value(act, st=st) == pytest.approx(old)
        with pytest.raises(ValueError):
            EV.set_play_now_mode("guess")
    finally:
        EV.set_play_now_mode(before)


def test_enabler_target_is_read_from_the_on_play_ability(monkeypatch):
    SP._ENABLER.clear()
    fake = {"E": {"abilities": [{"trigger": "ON_PLAY", "effect": {"type": "PLAY_CARD", "target": {"zone": "HAND", "cost_max": 2}}}]},
            "N": {"abilities": [{"trigger": "ON_PLAY", "effect": {"type": "DRAW", "value": {"base": 1}}}]}}
    assert SP.enabler_target("E", fake)["cost_max"] == 2 and SP.enabler_target("N", fake) is None and SP.enabler_target("?", fake) is None
    SP._ENABLER.clear()


def test_an_on_play_ability_with_a_cost_can_be_declined_and_needs_a_payable_card():
    """**T70**: 登場時のコスト付き能力は「〜できる」＝効果がコストに届かなければ払わない（0）・コストを払える札が場／手札に
    無ければ効果は起きない（0）・起動メイン（`offered`）は負のまま・状態が無ければ払えるとして読む。"""
    cards = _Cards(_TABLE)
    bounce = {"type": "BOUNCE", "target": {"zone": "FIELD", "player": "SELF", "card_type": ["CHARACTER"], "cost_min": 2, "count": 1}}
    play = {"type": "PLAY_CARD", "target": {"zone": "HAND", "card_type": ["CHARACTER"], "cost_max": 2, "count": 1, "player": "SELF"}}
    ab = {"trigger": "ON_PLAY", "cost": bounce, "effect": play}
    before = EV.PLAY_NOW_MODE
    try:
        v_full, unp = EV.ability_value(ab)                                            # 状態なし: 旧＝満額の効果 − 戻す費用（≥ 0 に床）
        assert unp == [] and v_full == pytest.approx(max(0.0, (EV.field_value(play["target"]) - MU) - abs(EV.action_value(bounce))))
        ctx = _ctx(["BIG"], [4, 6, 7, 10]); ctx["cards"] = cards; ctx["field"] = ["MID"]
        v, _ = EV.ability_value(ab, st={"search_ctx": ctx})                           # 相方が手札に無い → 効果 0 → 払わない → 0
        assert v == 0.0
        assert EV._cost_unpayable([bounce], None, {"search_ctx": {**ctx, "field": ["CNT"]}}) is True     # 戻せる札（コスト 2 以上）が無い（CNT は 1）
        assert EV._cost_unpayable([bounce], None, {"search_ctx": ctx}) is False
        assert EV._cost_unpayable([bounce], None, None) is False
        v0, _ = EV.ability_value(ab, st={"search_ctx": {**ctx, "field": ["CNT"], "hand_items": [_item("SMALL")]}})
        assert v0 == 0.0                                                              # 相方は在るが戻せる札が無い → 起きない
        # 起動メイン（エンジンが候補に出した＝払う前提）は負のまま
        act = {"trigger": "ACTIVATE_MAIN", "cost": bounce, "effect": play}
        va, _ = EV.ability_value(act, st={"search_ctx": ctx}, offered=True)
        assert va < 0.0
        # 手札から捨てるコストも同じ（合う札が無ければ払えない）
        discard = {"type": "TRASH", "target": {"zone": "HAND", "player": "SELF", "card_type": ["EVENT"], "count": 1}}
        assert EV._cost_unpayable([discard], None, {"search_ctx": ctx}) is True
        assert EV._cost_unpayable([discard], None, {"search_ctx": {**ctx, "hand_items": [_item("EVT")]}}) is False
    finally:
        EV.set_play_now_mode(before)


def test_a_don_attach_requirement_is_a_choice_that_costs_n_active_don_for_a_turn():
    """**T74**: 【ドン!!×N】は付ける選択＝効果から `N × δ / R` を引く（付けない自由＝0 に床）。付けられなければ（アクティブ < N）条件は偽で 0。
    候補に出た起動メイン（`offered`）は付いている前提で費用なし。状態が無ければ費用なし（上限）。"""
    import condition_value as CV
    draw = {"type": "DRAW", "value": {"base": 1}, "target": None}
    cond = {"type": "HAS_DON", "operator": "GE", "value": 2, "player": "SELF", "args": []}
    ab = {"trigger": "ON_ATTACK", "effect": draw, "condition": cond}
    assert CV.has_don_requirement(cond) == 2 and CV.has_don_requirement({"type": "AND", "args": [cond, {"type": "HAS_DON", "value": 3, "args": []}]}) == 3
    assert CV.has_don_requirement({"type": "LIFE_COUNT", "value": 2, "args": []}) is None
    full, _ = EV.ability_value(ab)                                                     # 状態なし＝費用なし
    assert full == pytest.approx(MU)
    st = {"my_don_active": 3, "r_turns": 4.0}
    v, _ = EV.ability_value(ab, st=st)
    assert v == pytest.approx(MU - 2 * EV.DELTA / 4.0)
    assert EV.ability_value(ab, st={"my_don_active": 1, "r_turns": 4.0})[0] == 0.0     # 付けられない → 条件が偽
    assert EV.ability_value(ab, st=st, offered=True)[0] == pytest.approx(MU)           # 付いている前提
    tiny = {"trigger": "ON_ATTACK", "effect": {"type": "DRAW", "value": {"base": 1}, "target": None},
            "condition": {"type": "HAS_DON", "operator": "GE", "value": 9, "player": "SELF", "args": []}}
    assert EV.ability_value(tiny, st={"my_don_active": 10, "r_turns": 1.0})[0] == 0.0  # 費用 9δ > μ → 付けない


def test_a_return_don_cost_is_unpayable_when_there_is_not_enough_active_don():
    """**D-2**: `RETURN_DON`（`target=None`・【メイン】ドン‼️-N のイベント本文の形・OP15-074〜078 等）は
    元々 `_cost_unpayable` の場・手札の判定（`target` 必須）を素通りして常に「払える」扱いだった
    （D-1 の未確定点(i)）。`COST_AFFORD_MODE="check"` なら `st["my_don_active"]`
    （`condition_value.state_from_scalars` が既に積む・T72／`_don_attach_cost` と同じ場所）と比べる。
    既定 `off` は旧のまま・状態が無ければ払えるとして読む（上限）。"""
    ret_don = {"type": "RETURN_DON", "target": None, "value": {"base": 1}}
    draw = {"type": "DRAW", "value": {"base": 1}, "target": None}
    ab = {"trigger": "ON_PLAY", "cost": ret_don, "effect": draw}
    assert EV.COST_AFFORD_MODE == "off"                                                # 既定
    assert EV._cost_unpayable([ret_don], None, None) is False                          # 状態なし＝上限
    assert EV._cost_unpayable([ret_don], None, {"my_don_active": 0}) is False           # 既定 off は場・手札コストと同じ従来どおり
    v_off, _ = EV.ability_value(ab, st={"my_don_active": 0})
    assert v_off == pytest.approx(MU - EV.DELTA)                                       # 従来どおり払える前提で値付け
    try:
        EV.set_cost_afford_mode("check")
        assert EV._cost_unpayable([ret_don], None, None) is False                      # 状態なし＝上限（check でも変わらない）
        assert EV._cost_unpayable([ret_don], None, {}) is False                        # 場のドンの合計が無ければ上限
        assert EV._cost_unpayable([ret_don], None, {"my_don_total": 0}) is True         # 場にドンが無い → 払えない
        assert EV._cost_unpayable([ret_don], None, {"my_don_total": 1}) is False        # ちょうど払える
        # **見直し（2026-09-25）**: ドン‼️−N はレストのドンも戻せる（エンジン `can_satisfy_node_on` は
        # アクティブ＋レスト＋付与中の合計で判定）。D-2 はアクティブだけと比べていた＝厳しすぎた。
        assert EV._cost_unpayable([ret_don], None, {"my_don_active": 0, "my_don_total": 3}) is False
        assert EV._cost_unpayable([ret_don], None, {"my_don_active": 0}) is False       # 合計が読めない＝上限（アクティブは下限にすぎない）
        two_don = {"type": "RETURN_DON", "target": None, "value": {"base": 2}}
        assert EV._cost_unpayable([two_don], None, {"my_don_total": 1}) is True         # 2 枚要るのに 1 枚しか無い
        # 出す札のコストはレストで払うだけ＝場のドンの合計は減らない
        assert EV._cost_unpayable([ret_don], {"cost": 5}, {"my_don_active": 5, "my_don_total": 5}) is False
        v0, _ = EV.ability_value(ab, st={"my_don_total": 0})
        assert v0 == 0.0                                                               # 払えない → 効果は起きない（登場時）
        va, _ = EV.ability_value(ab, st={"my_don_total": 0}, offered=True)
        assert va == pytest.approx(MU - EV.DELTA)                                      # 起動メイン（候補に出た＝払う前提）はゲートを通らない
        # 場・手札の判定は独立に効き続ける（既存の T70 の道は素通りしない）
        bounce = {"type": "BOUNCE", "target": {"zone": "FIELD", "player": "SELF", "card_type": ["CHARACTER"], "cost_min": 2, "count": 1}}
        cards = _Cards(_TABLE)
        ctx = _ctx(["BIG"], [4, 6, 7, 10]); ctx["cards"] = cards; ctx["field"] = ["CNT"]
        assert EV._cost_unpayable([bounce], None, {"search_ctx": {**ctx, "field": ["CNT"]}, "my_don_active": 5}) is True
    finally:
        EV.set_cost_afford_mode("off")
    with pytest.raises(ValueError):
        EV.set_cost_afford_mode("guess")


def test_the_cli_arg_actually_reaches_the_mode():
    """**D-2/D-3**: `add_cost_afford_arg` を足しても `apply_cost_afford` を呼び忘れると
    CLI で切替を渡しても何も変わらない（`price_realised.py` の最初の計測で実際に踏んだ配線漏れ・
    `off`/`check` の出力が1バイトも変わらなかった）。`apply_cost_afford(a)` 自体がその橋渡しを
    正しく行うことを固定する。"""
    import argparse
    before = EV.COST_AFFORD_MODE
    try:
        ap = argparse.ArgumentParser()
        EV.add_cost_afford_arg(ap)
        a = ap.parse_args([])
        assert EV.apply_cost_afford(a) == "off"                      # 省略時は不動
        assert EV.COST_AFFORD_MODE == "off"
        a = ap.parse_args(["--cost-afford", "check"])
        assert EV.apply_cost_afford(a) == "check"                    # 渡せば実際に切り替わる
        assert EV.COST_AFFORD_MODE == "check"
    finally:
        EV.set_cost_afford_mode(before)


def test_a_rest_don_cost_needs_active_don_too_not_just_return_don():
    """**D-4**: `REST_DON`（ドン‼️をレストにするだけのコスト・恒久には失わない）も `RETURN_DON` と同じく
    アクティブなドンが N 枚要る。D-2 は `RETURN_DON` だけ見ていて、この型は同じ穴が残っていた。"""
    rest_don = {"type": "REST_DON", "target": None, "value": {"base": 2}}
    assert EV._cost_unpayable([rest_don], None, {"my_don_active": 1}) is False          # off は不変（旧どおり払える）
    try:
        EV.set_cost_afford_mode("check")
        assert EV._cost_unpayable([rest_don], None, {"my_don_active": 1}) is True       # 2 枚要るのに 1 枚
        assert EV._cost_unpayable([rest_don], None, {"my_don_active": 2}) is False       # ちょうど払える
        # **見直し（2026-09-25）**: 手札から出す札はコストをアクティブなドンのレストで**先に**払ってから
        # 登場時・【メイン】を解決する（エンジン `rules/actions.rs` の PLAY）＝残りのアクティブで判定する。
        assert EV._cost_unpayable([rest_don], {"cost": 3}, {"my_don_active": 4}) is True   # 4 − 3 = 1 < 2
        assert EV._cost_unpayable([rest_don], {"cost": 3}, {"my_don_active": 5}) is False  # 5 − 3 = 2
        assert EV._cost_unpayable([rest_don], {"cost": 3}, {"my_don_total": 9}) is False   # アクティブが読めない＝上限
        # レストにするコストは場の合計ではなくアクティブで見る（レストのドンはもうレストにできない）
        assert EV._cost_unpayable([rest_don], None, {"my_don_active": 1, "my_don_total": 9}) is True
    finally:
        EV.set_cost_afford_mode("off")


def test_a_self_targeting_cost_is_always_payable_under_check_but_not_off():
    """**D-4**: 「このカードをレストにする」等（`ref_id=="self"`）は能力を持つカード自身が対象——
    判定時点で必ず場に在る（登場時なら出た札はアクティブ）ので常に払える。`check` 以前は場の一覧の中から
    探しており、他に場の札が無いと常に「払えない」誤判定だった（実測で確認）。`off` はその旧挙動を保つ。"""
    self_rest = {"type": "REST", "target": {"zone": "FIELD", "player": "SELF", "count": 1, "ref_id": "self"}}
    cards = _Cards(_TABLE)
    ctx = {"cards": cards, "field": [], "hand_items": []}                              # 場に他の札が無い
    assert EV._cost_unpayable([self_rest], None, {"search_ctx": ctx}) is True           # off（旧）＝誤って払えない
    try:
        EV.set_cost_afford_mode("check")
        assert EV._cost_unpayable([self_rest], None, {"search_ctx": ctx}) is False      # check＝自分自身は常に在る
    finally:
        EV.set_cost_afford_mode("off")


def test_a_leader_targeting_cost_is_payable_even_with_an_empty_field():
    """**D-4**: 場の一覧を絞り込む関数（`eligible_deck_cards`）はリーダーを最初から除くので、
    `card_type` に`"LEADER"`を含む対象は場が空だと`check`以前は常に「払えない」誤判定だった
    （実測で確認・「自分のリーダーのパワー-5000」等）。リーダーは常に場に在る。"""
    leader_cost = {"type": "BUFF", "target": {"zone": "FIELD", "player": "SELF", "count": 1, "card_type": ["LEADER"]},
                   "value": {"base": -5000}}
    cards = _Cards(_TABLE)
    ctx = {"cards": cards, "field": [], "hand_items": []}
    assert EV._cost_unpayable([leader_cost], None, {"search_ctx": ctx}) is True         # off（旧）＝誤って払えない
    try:
        EV.set_cost_afford_mode("check")
        assert EV._cost_unpayable([leader_cost], None, {"search_ctx": ctx}) is False    # check＝リーダーは常に在る
        # 素性が読めなければ上限（払えるとして読む）——st["my_leader"] が無いのと同じ扱い
        assert EV._cost_unpayable([leader_cost], None, {"search_ctx": ctx, "my_leader": None}) is False
    finally:
        EV.set_cost_afford_mode("off")


def test_a_leader_targeting_cost_with_a_trait_filter_checks_the_actual_leader():
    """**D-4（実装レビューで確認した実害）**: `card_type` に`"LEADER"`が在るだけで常に払える、では
    「特徴《ドレスローザ》のリーダーかステージ」（OP10-043等15枚）のような絞り込みを無視してしまう。
    `st["my_leader"]`（`condition_value.leader_info`と同じ素性の形）に実際に合うかを見る。"""
    dressrosa_leader_or_stage = {"type": "REST", "target": {
        "zone": "FIELD", "player": "SELF", "count": 1, "card_type": ["LEADER", "STAGE"], "traits": ["ドレスローザ"]}}
    cards = _Cards(_TABLE)
    ctx = {"cards": cards, "field": [], "hand_items": []}
    try:
        EV.set_cost_afford_mode("check")
        other_leader = {"names": ["だれか"], "traits": ["麦わらの一味"], "colors": [], "attribute": None}
        assert EV._cost_unpayable([dressrosa_leader_or_stage], None,
                                   {"search_ctx": ctx, "my_leader": other_leader}) is True    # 特徴が合わない
        dressrosa_leader = {"names": ["だれか"], "traits": ["ドレスローザ"], "colors": [], "attribute": None}
        assert EV._cost_unpayable([dressrosa_leader_or_stage], None,
                                   {"search_ctx": ctx, "my_leader": dressrosa_leader}) is False  # 特徴が合う
    finally:
        EV.set_cost_afford_mode("off")


def test_attaching_ones_own_active_don_as_a_cost_also_needs_active_don():
    """**D-4（実装レビューで確認した実害）**: `ATTACH_DON` を自分のキャラ等へ「アクティブなドン‼️」を
    付与するコストとして使うカード（EB04-009等5枚）は、`target` を持つので場の判定は掛かるが、
    アクティブなドンの枚数チェック（`RETURN_DON`／`REST_DON`だけ見ていた）の対象外だった。"""
    attach_own_active = {"type": "ATTACH_DON", "raw_text": "自分の「Xレイリー」1枚にアクティブのドン!!1枚を付与する",
                          "target": {"zone": "FIELD", "player": "SELF", "count": 1, "card_type": ["LEADER", "CHARACTER"],
                                     "names": ["Xレイリー"]}, "value": {"base": 1}}
    cards = _Cards(_TABLE)
    ctx = _ctx(["SMALL"], [4, 6, 7, 10]); ctx["cards"] = cards; ctx["field"] = []
    other_leader = {"names": ["だれか"], "traits": [], "colors": [], "attribute": None}  # card_type にLEADERも
    # 在るので、対象の名前(Xレイリー)に合わないリーダーを明示して素性チェックで弾かせる（上限の抜け道を塞ぐ）
    assert EV._cost_unpayable([attach_own_active], None, {"my_don_active": 0}) is False   # off は不変（ドン枚数は見ない）
    try:
        EV.set_cost_afford_mode("check")
        assert EV._cost_unpayable([attach_own_active], None,
                                   {"search_ctx": ctx, "my_don_active": 0, "my_leader": other_leader}) is True   # アクティブなドンが無い
        assert EV._cost_unpayable([attach_own_active], None,
                                   {"search_ctx": ctx, "my_don_active": 1, "my_leader": other_leader}) is True   # ドンは足りるが対象の場札が無い
        opp_attach = dict(attach_own_active); opp_target = dict(attach_own_active["target"]); opp_target["player"] = "OPPONENT"
        opp_attach["target"] = opp_target
        assert EV._cost_unpayable([opp_attach], None, {"my_don_active": 0}) is False       # 相手対象は既存の足切りの外
    finally:
        EV.set_cost_afford_mode("off")


def test_a_count_of_zero_is_always_payable_under_the_new_mode():
    """**D-4（実装レビューで確認した潜在バグ）**: `target.count == 0`（「0枚を…」）は `n or 1` だと
    偽値として `1` に化けて誤って「払えない」判定になりかねない。`None` との比較にして直した
    （実在のカードには `count == 0` の対象コストは無い・将来の保険）。"""
    zero_count = {"type": "TRASH", "target": {"zone": "FIELD", "player": "SELF", "count": 0, "card_type": ["CHARACTER"]}}
    cards = _Cards(_TABLE)
    ctx = {"cards": cards, "field": [], "hand_items": []}
    try:
        EV.set_cost_afford_mode("check")
        assert EV._cost_unpayable([zero_count], None, {"search_ctx": ctx}) is False       # 0 枚要る＝常に払える
    finally:
        EV.set_cost_afford_mode("off")


def test_the_required_count_is_checked_only_under_the_new_mode():
    """**D-4**: 旧は「1 枚合えば払える」で `target.count` を見ていなかった。`check` は要る枚数と比べる。"""
    two_chars = {"type": "TRASH", "target": {"zone": "FIELD", "player": "SELF", "count": 2, "card_type": ["CHARACTER"]}}
    cards = _Cards(_TABLE)
    ctx = {"cards": cards, "field": ["SMALL"], "hand_items": []}                       # 合う札は 1 枚だけ
    assert EV._cost_unpayable([two_chars], None, {"search_ctx": ctx}) is False          # off（旧）＝1 枚在れば払える
    try:
        EV.set_cost_afford_mode("check")
        assert EV._cost_unpayable([two_chars], None, {"search_ctx": ctx}) is True       # check＝2 枚要るのに 1 枚
        ctx2 = {"cards": cards, "field": ["SMALL", "MID"], "hand_items": []}
        assert EV._cost_unpayable([two_chars], None, {"search_ctx": ctx2}) is False     # 2 枚在れば払える
    finally:
        EV.set_cost_afford_mode("off")


def test_life_and_trash_zone_costs_are_checked_by_count_only_under_the_new_mode():
    """**D-4**: 「自分のライフの上から1枚を…」「自分のトラッシュの…N枚を…」のような、枚数だけで
    絞り込みの無いライフ／トラッシュゾーンのコストは、`check` 以前は判定対象にすら入っていなかった。
    個々の札の素性を追う記録が無いので、絞り込み（特徴・名前）は見ず枚数だけ見る（上限として読む規約）。"""
    life_cost = {"type": "TRASH", "target": {"zone": "LIFE", "player": "SELF", "count": 1}}
    trash_cost = {"type": "DECK_BOTTOM", "target": {"zone": "TRASH", "player": "SELF", "count": 3, "traits": ["CP"]}}
    cards = _Cards(_TABLE)
    ctx = {"cards": cards, "field": [], "hand_items": []}
    assert EV._cost_unpayable([life_cost], None, {"search_ctx": ctx, "my_life": 0}) is False    # off はゾーンごと見ない
    assert EV._cost_unpayable([trash_cost], None, {"search_ctx": ctx, "my_trash": 0}) is False
    try:
        EV.set_cost_afford_mode("check")
        assert EV._cost_unpayable([life_cost], None, {"search_ctx": ctx, "my_life": 0}) is True    # ライフが無い
        assert EV._cost_unpayable([life_cost], None, {"search_ctx": ctx, "my_life": 1}) is False
        assert EV._cost_unpayable([trash_cost], None, {"search_ctx": ctx, "my_trash": 2}) is True  # 3 枚要るのに 2 枚
        assert EV._cost_unpayable([trash_cost], None, {"search_ctx": ctx, "my_trash": 3}) is False  # 絞り込みは見ない（上限）
    finally:
        EV.set_cost_afford_mode("off")
