"""`tests/scripts/guard_afford.py` の純関数（ナップサック・無料/有料の分解・会計）。

基盤健全性（`cpu_infra`）: 読み取り専用の計器の算術を固める。ここが壊れると
`docs/life_budget.md` §5 の会計（「守れたのに受けた」の割合）が静かに間違う——
特に**無料（印字カウンター）と有料（【カウンター】イベント）の切り分け**は
`rules/battle.rs::apply_counter` の裁定そのままなので、ずれたら結論が反転しうる。
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

import guard_afford as G  # noqa: E402


class FakeCards:
    """card_id → info（`plan_labels.Cards.info` の必要な欄だけ）。"""

    def __init__(self, table):
        self.table = table

    def info(self, cid):
        return self.table.get(cid)


def hand_row(slots):
    """手札の枠だけを埋めたトークン行を作る。slots＝{枠 index: (counter_value, is_event)}。"""
    tok = np.zeros((22, 22), np.float32)
    for j, (cv, is_event) in slots.items():
        tok[G.SLOT_HAND.start + j, 0] = 5000.0                 # power_now＝枠が埋まっている印
        tok[G.SLOT_HAND.start + j, G.S_COUNTER] = cv / G.COUNTER_SCALE
        tok[G.SLOT_HAND.start + j, G.S_IS_EVENT] = 1.0 if is_event else 0.0
    return tok


def test_knapsack_respects_the_budget():
    items = [(1, 1000.0), (2, 3000.0), (3, 3500.0)]
    assert G.knapsack(items, 0) == 0.0
    assert G.knapsack(items, 1) == 1000.0
    assert G.knapsack(items, 2) == 3000.0
    assert G.knapsack(items, 3) == 4000.0          # 1+2 を選ぶ（3500 の 1 枚より良い）
    assert G.knapsack(items, 6) == 7500.0          # 全部
    assert G.knapsack([], 5) == 0.0


def test_knapsack_skips_items_over_budget_and_takes_each_once():
    assert G.knapsack([(9, 9000.0)], 4) == 0.0
    assert G.knapsack([(2, 2000.0)], 4) == 2000.0   # 同じ札を 2 回は使えない


def test_hand_counters_splits_free_and_paid():
    """キャラの印字カウンターは無料・イベントの上げ幅は印字を超える分だけ有料。"""
    cards = FakeCards({
        "CHR": {"counter": 2000, "event": False, "cost": 3},
        "EVT": {"counter": 0, "event": True, "cost": 2},
        "EVT_P": {"counter": 1000, "event": True, "cost": 1},   # 印字もある【カウンター】イベント
    })
    idx2cid = {11: "CHR", 12: "EVT", 13: "EVT_P"}
    tok = hand_row({0: (2000.0, False), 1: (4000.0, True), 2: (3000.0, True)})
    ci = np.zeros(22, np.int64)
    ci[G.SLOT_HAND.start + 0] = 11
    ci[G.SLOT_HAND.start + 1] = 12
    ci[G.SLOT_HAND.start + 2] = 13
    free, paid, slots = G.hand_counters(tok, ci, idx2cid, cards)
    assert slots == 3
    assert free == pytest.approx(3000.0)            # 2000（キャラ）＋1000（イベントの印字）
    assert sorted(paid) == [(1.0, 2000.0), (2.0, 4000.0)]


def test_hand_counters_ignores_empty_slots_and_zero_counters():
    cards = FakeCards({"NONE": {"counter": 0, "event": False, "cost": 4}})
    idx2cid = {5: "NONE"}
    tok = hand_row({0: (0.0, False)})
    ci = np.zeros(22, np.int64)
    ci[G.SLOT_HAND.start + 0] = 5
    free, paid, slots = G.hand_counters(tok, ci, idx2cid, cards)
    assert (free, paid, slots) == (0.0, [], 1)      # 枠は埋まっているがカウンターは 0


def test_hand_counters_falls_back_to_the_encoding_when_the_card_is_unknown():
    """vocab を引けない枠（語彙外）はトークンの `is_event` で種別を決める。"""
    tok = hand_row({0: (4000.0, True)})
    ci = np.zeros(22, np.int64)
    free, paid, slots = G.hand_counters(tok, ci, {}, FakeCards({}))
    assert slots == 1 and free == 0.0
    assert paid == [(0.0, 4000.0)]                 # コスト不明＝0 として扱う（過大評価側）


def test_clock_band_edges():
    assert G.clock_band(1.0) == "t<=2"
    assert G.clock_band(2.5) == "t<=2"
    assert G.clock_band(3.0) == "t3-4"
    assert G.clock_band(4.5) == "t3-4"
    assert G.clock_band(5.0) == "t5+"


def test_block_reports_what_the_don_correction_removed():
    recs = [
        # 旧では足りていたが、払えないので足りない行（差の正体）
        {"means_naive": True, "means_don": True, "enough_naive": True, "enough_don": False,
         "blocker": False, "budget": 0, "free": 1000.0, "paid_n": 1, "naive_sum": 5000.0,
         "best_don": 1000.0, "over": 3000.0},
        # 旧も新も足りている行
        {"means_naive": True, "means_don": True, "enough_naive": True, "enough_don": True,
         "blocker": False, "budget": 2, "free": 4000.0, "paid_n": 0, "naive_sum": 4000.0,
         "best_don": 4000.0, "over": 2000.0},
        # 攻撃が来ていない行（over=None）
        {"means_naive": False, "means_don": False, "enough_naive": False, "enough_don": False,
         "blocker": False, "budget": 1, "free": 0.0, "paid_n": 0, "naive_sum": 0.0,
         "best_don": 0.0, "over": None},
    ]
    b = G.block(recs)
    assert b["n"] == 3
    assert b["enough_naive"] == pytest.approx(2 / 3, abs=1e-4)
    assert b["enough_don"] == pytest.approx(1 / 3, abs=1e-4)
    assert b["blocked_by_don_n"] == 1
    assert b["blocked_by_don"] == pytest.approx(1 / 3, abs=1e-4)
    assert b["no_attack"] == pytest.approx(1 / 3, abs=1e-4)
    assert b["over_mean"] == pytest.approx(2500.0)   # None は平均に入れない
    assert G.block([]) is None


def test_cards_info_carries_type_and_cost():
    """`plan_labels.Cards.info` に種別とコストが在る（本計器の前提）。"""
    from opcg_sim.learned.train import plan_labels as PL
    cards = PL.Cards()
    from opcg_sim.loop import decks as D
    db = D.load_db()
    ev = next(m for m in db.cards.values()
              if getattr(getattr(m, "type", None), "name", "") == "EVENT")
    inf = cards.info(ev.card_id)
    assert inf["event"] is True
    assert inf["cost"] == int(ev.cost or 0)
    ch = next(m for m in db.cards.values()
              if getattr(getattr(m, "type", None), "name", "") == "CHARACTER")
    assert cards.info(ch.card_id)["event"] is False
