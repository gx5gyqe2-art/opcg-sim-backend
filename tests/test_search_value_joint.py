"""**N-4**（2026-10-04）: 足した札の値を 1 枚 1 役の手札の価値（N-1）で読む切替と、デッキのカウンター・イベントの直し。

固めること:

* 既定は `legacy`＋`printed`（旧の数字と 1 ビットも変わらない）・知らない名前は落ちる。
* `joint` の札の値＝`V(手札 ∪ 札) − V(手札)`（`hand_joint` の定義そのもの）・旧の `max(ΔH, ΔG)` との違いは
  守る側を**粗の節約**で数えること（1 枚の手札の例で値を固定）。
* 入った札が複数なら順に足した差（和は順に依らず `V(後) − V(後 − 入った札)`）。
* デッキのカウンター・イベントは `printed` では 0・`rules` では手札の符号化と同じ値（上限 5000）。
* **先読みしない**: 探す能力の価格は判断点の行（と完全情報のデッキの構成）だけで決まる——実記録 2 局で、同じ局の他の行を
  壊しても値が変わらない。物差しの入った札は渡した 2 行（窓の前と後）だけで決まる。
"""
import os
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.cpu_infra

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _bootstrap  # noqa: F401,E402

_SCRIPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import hand_joint as HJ  # noqa: E402
import hand_plan as HP  # noqa: E402
import search_price as SP  # noqa: E402
from theory_order import KO_P  # noqa: E402

_REC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "f_identity", "rec")


@pytest.fixture(autouse=True)
def _modes():
    before = (SP.SEARCH_VALUE_MODE, SP.DECK_COUNTER_MODE)
    SP._GAIN.clear()
    yield
    SP.set_search_value_mode(before[0])
    SP.set_deck_counter_mode(before[1])
    SP._GAIN.clear()


def test_defaults_are_the_old_reading_and_unknown_names_fail():
    assert SP.SEARCH_VALUE_MODE == "legacy" and SP.DECK_COUNTER_MODE == "printed"
    with pytest.raises(ValueError):
        SP.set_search_value_mode("max")
    with pytest.raises(ValueError):
        SP.set_deck_counter_mode("guess")


def _it(cost, v, counter, cid="X", event=False):
    return {"cid": cid, "cost": float(cost), "v": v, "counter": float(counter), "event": event}


def test_joint_gain_is_the_n1_marginal_and_differs_from_t67_by_the_gross_guard_saving():
    """1 枚だけの手札に 2000 カウンター（v 0.03）を足す・来る攻撃 1000・受ける損 0.0872・ドンが無い（出せない）:
    1 枚 1 役は守る役の粗の節約 0.0872。T67 の `max(ΔH, ΔG)` は ΔG を純額（0.0872 − 0.03）で比べる＝0.0572。"""
    take = 0.0872
    card = _it(1, 0.03, 2000, "CNT")
    caps = [0, 0, 0, 0]
    j = HP.joint_gain([], card, caps, [1000.0], take)
    assert j == pytest.approx(take)
    legacy = HP.card_deltas([], card, caps, [1000.0], take)["dtotal"]
    assert legacy == pytest.approx(take - 0.03)
    # 出せるなら出す役（0.03）と守る役（0.0872）の大きい方＝守る役
    assert HP.joint_gain([], card, [1, 1, 1, 10], [1000.0], take) == pytest.approx(take)
    # 攻撃が来なければ出す役だけ
    assert HP.joint_gain([], card, [1, 1, 1, 10], [], take) == pytest.approx(0.03)


def test_joint_gain_equals_the_hand_joint_card_value_on_a_mixed_hand():
    rest = [_it(4, 0.08, 1000, "A"), _it(2, 0.04, 2000, "B"), _it(7, [0.0, 0.1, 0.12], 0, "C")]
    card = _it(3, 0.05, 1000, "D")
    caps, xs, take = [4, 6, 8, 10], [1000.0, 2000.0], 0.08
    items = [(it["cost"], it["v"], it["counter"]) for it in rest + [card]]
    want = HJ.card_value_joint(items, 3, caps, xs, take)
    assert HP.joint_gain(rest, card, caps, xs, take) == pytest.approx(want)
    assert want >= 0.0


def test_gain_of_follows_the_switch():
    rest = [_it(2, 0.04, 1000, "A")]
    card = _it(1, 0.03, 2000, "CNT")
    caps, xs, take = [0, 0, 0, 0], [1000.0], 0.0872
    assert HP.gain_of(rest, card, caps, xs, take) == pytest.approx(HP.card_deltas(rest, card, caps, xs, take)["dtotal"])
    SP.set_search_value_mode("joint")
    assert HP.gain_of(rest, card, caps, xs, take) == pytest.approx(HP.joint_gain(rest, card, caps, xs, take))


def test_added_cards_telescope_to_the_set_value_in_any_order():
    items = [_it(4, 0.08, 1000, "A"), _it(2, 0.04, 2000, "B"), _it(1, 0.03, 2000, "C"), _it(5, 0.1, 0, "D")]
    caps, xs, take = [5, 6, 8, 10], [1000.0, 3000.0], 0.0872
    jv = HJ.valuer_of([(it["cost"], it["v"], it["counter"]) for it in items], caps, xs, take)
    full = frozenset(range(4))
    want = jv.value(full)[0] - jv.value(full - {1, 2})[0]
    g12 = HP.joint_gains_seq(items, [1, 2], caps, xs, take)
    g21 = HP.joint_gains_seq(items, [2, 1], caps, xs, take)
    assert sum(g12) == pytest.approx(want) and sum(g21) == pytest.approx(want)
    assert len(g12) == 2 and min(g12 + g21) >= 0.0
    # 1 枚だけなら `joint_gain` と同じ
    assert HP.joint_gains_seq(items, [3], caps, xs, take)[0] == pytest.approx(HP.joint_gain(items[:3], items[3], caps, xs, take))


class _Cards:
    def __init__(self, table):
        self._t = table

    def info(self, cid):
        return self._t.get(cid)


_TABLE = {
    "BIG": {"cost": 7, "power": 8000, "counter": 0, "event": False, "stage": False, "leader": False},
    "CEV": {"cost": 1, "power": 0, "counter": 0, "event": True, "stage": False, "leader": False},
    "HUGE": {"cost": 1, "power": 0, "counter": 0, "event": True, "stage": False, "leader": False},
}


def test_deck_counter_events_are_zero_when_printed_and_read_by_rule(monkeypatch):
    """デッキの【カウンター】イベント（印字 0・上げ幅 2000）: `printed` では守る役が 0・`rules` では 2000 で立つ。
    上げ幅 6000 の札は手札の符号化と同じく 5000 で打ち切る。"""
    monkeypatch.setattr(SP, "counter_event_of", lambda cid: {"CEV": 2000.0, "HUGE": 6000.0}.get(cid, 0.0))
    monkeypatch.setattr(SP, "card_identity", lambda cid: {"names": [cid], "traits": [], "colors": []})
    monkeypatch.setattr(SP, "use_value", lambda cid, info, olp, r: 0.0)
    cards = _Cards(_TABLE)
    assert SP.deck_counter("CEV", _TABLE["CEV"]) == 0.0
    SP.set_deck_counter_mode("rules")
    assert SP.deck_counter("CEV", _TABLE["CEV"]) == 2000.0
    assert SP.deck_counter("HUGE", _TABLE["HUGE"]) == 5000.0
    assert SP.deck_counter("BIG", _TABLE["BIG"]) == 0.0                 # キャラは印字のまま
    ctx = {"hand_items": [_it(7, 0.1, 0, "BIG")], "caps": [0, 0, 0, 0], "xs": [1000.0], "take": 0.0872,
           "deck": ["CEV"] * 10, "olp": 5000.0, "r": 4.0, "field": []}
    tgt = {"card_type": ["EVENT"]}
    SP.set_deck_counter_mode("printed")
    SP._GAIN.clear()
    assert SP.search_value(ctx, 4, tgt, cards) == 0.0                   # 旧: カウンター 0・出す値 0
    SP.set_deck_counter_mode("rules")
    SP._GAIN.clear()
    assert SP.search_value(ctx, 4, tgt, cards) == pytest.approx(0.0872)  # 受ける損 − v（v = 0）
    SP.set_search_value_mode("joint")
    SP._GAIN.clear()
    assert SP.search_value(ctx, 4, tgt, cards) == pytest.approx(0.0872)
    assert 0.0 < 1.0 - KO_P < 1.0


def test_the_cache_keeps_the_modes_apart(monkeypatch):
    monkeypatch.setattr(SP, "counter_event_of", lambda cid: 2000.0)
    monkeypatch.setattr(SP, "card_identity", lambda cid: {"names": [cid], "traits": [], "colors": []})
    monkeypatch.setattr(SP, "use_value", lambda cid, info, olp, r: 0.03)
    cards = _Cards(_TABLE)
    ctx = {"hand_items": [], "caps": [0, 0, 0, 0], "xs": [1000.0], "take": 0.0872,
           "deck": ["CEV"], "olp": 5000.0, "r": 4.0, "field": []}
    a = SP.card_gain("CEV", ctx, cards)
    SP.set_deck_counter_mode("rules")
    b = SP.card_gain("CEV", ctx, cards)
    SP.set_search_value_mode("joint")
    c = SP.card_gain("CEV", ctx, cards)
    assert a == pytest.approx(0.0) and b == pytest.approx(0.0872 - 0.03) and c == pytest.approx(0.0872)
    SP.set_search_value_mode("legacy"); SP.set_deck_counter_mode("printed")
    assert SP.card_gain("CEV", ctx, cards) == pytest.approx(0.0)


def _fixture_rows():
    from opcg_sim.learned.train import plan_labels as PL
    import guard_afford as GA
    import theory_bridge as TB
    from theory_bridge import POL_COLS, ROW_COLS, _extra
    idx2cid = {i: c for c, i in GA._vocab().items()}
    cards = PL.Cards()
    rec_decks = SP.record_decks([_REC])
    for rows, pol, ex, L, ptr, idx in PL.iter_games([_REC], row_cols=ROW_COLS, pol_cols=POL_COLS, extra_fn=_extra):
        st = {}
        decks = TB._seat_decks(rec_decks, int(rows["seed"][idx[0]]), rows, ex, idx, idx2cid, st)
        yield rows, ex, list(idx), idx2cid, cards, decks


def _broken(ex, keep):
    """`keep` 以外の行を全部壊した `ex` の写し（他の行を読んでいれば値が変わる）。"""
    out = {}
    for k, v in ex.items():
        a = np.array(v, copy=True)
        mask = np.ones(len(a), bool)
        mask[list(keep)] = False
        a[mask] = a[mask][::-1] if a.dtype.kind in "iu" else (a[mask] * 0.0 + 0.37).astype(a.dtype)
        out[k] = a
    return out


def test_the_search_price_reads_only_the_decision_row_and_the_deck():
    """実記録 2 局の自席の行で、探す能力の価格（`joint`＋`rules`）を行 i の状態から作り、同じ局の他の行を壊しても変わらない。"""
    import theory_bridge as TB
    from opcg_sim.learned.train import plan_labels as PL
    SP.set_search_value_mode("joint"); SP.set_deck_counter_mode("rules")
    tgt = {"card_type": ["CHARACTER"]}
    n = changed = 0
    for rows, ex, order, idx2cid, cards, decks in _fixture_rows():
        for i in order[::7]:
            w, t = int(rows["who"][i]), int(rows["turn"][i])
            if t < 1 or not PL.is_own_turn(w, t) or decks.get(w) is None:
                continue
            ctx = TB._search_ctx(ex["sc"][i], ex["tok"][i], ex["ci"][i], idx2cid, cards, decks.get(w))
            v = SP.search_value(ctx, 5, tgt, cards)
            bad = _broken(ex, [i])
            SP._GAIN.clear()
            ctx2 = TB._search_ctx(bad["sc"][i], bad["tok"][i], bad["ci"][i], idx2cid, cards, decks.get(w))
            assert SP.search_value(ctx2, 5, tgt, cards) == pytest.approx(v)
            worse = _broken(ex, [])                                       # 行 i そのものも壊す＝検出器が働くか
            SP._GAIN.clear()
            ctx3 = TB._search_ctx(worse["sc"][i], worse["tok"][i], worse["ci"][i], idx2cid, cards, decks.get(w))
            changed += SP.search_value(ctx3, 5, tgt, cards) != pytest.approx(v)
            n += 1
    assert n >= 5
    assert changed >= 1


def test_the_yardstick_added_cards_read_only_the_two_window_rows():
    """物差しの入った札（`joint`）は窓の前と後の 2 行だけで決まる（他の行を壊しても同じ）・入った札の数は旧と同じ。"""
    from opcg_sim.learned.train import plan_labels as PL
    SP.set_search_value_mode("joint")
    n = 0
    for rows, ex, order, idx2cid, cards, decks in _fixture_rows():
        for a, b in zip(order[:-1:5], order[1::5]):
            if int(rows["who"][a]) != int(rows["who"][b]):
                continue
            w = int(rows["who"][a])
            got = HP.added_card_gains(ex["sc"][b], ex["tok"][b], ex["ci"][a], ex["ci"][b], idx2cid, cards, deck=decks.get(w))
            if not got:
                continue
            bad = _broken(ex, [a, b])
            again = HP.added_card_gains(bad["sc"][b], bad["tok"][b], bad["ci"][a], bad["ci"][b], idx2cid, cards, deck=decks.get(w))
            assert [c for c, _g in again] == [c for c, _g in got]
            assert [g for _c, g in again] == pytest.approx([g for _c, g in got])
            SP.set_search_value_mode("legacy")
            old = HP.added_card_gains(ex["sc"][b], ex["tok"][b], ex["ci"][a], ex["ci"][b], idx2cid, cards, deck=decks.get(w))
            SP.set_search_value_mode("joint")
            assert [c for c, _g in old] == [c for c, _g in got] and all(g >= 0.0 for _c, g in got)
            n += 1
    assert n >= 3
    assert PL is not None
