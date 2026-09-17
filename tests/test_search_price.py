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
